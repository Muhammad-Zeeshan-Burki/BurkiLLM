import os
import math
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from data.dataset import collate_fn


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Separate weight decay from norm/bias parameters
        no_decay = {"bias", "LayerNorm.weight", "ln_1.weight", "ln_2.weight", "ln_f.weight"}
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.95),
        )

        steps_per_epoch = max(1, len(train_dataset) // config.batch_size)
        total_steps = steps_per_epoch * config.num_epochs
        warmup_steps = min(config.num_warmup_steps, total_steps // 5)

        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - warmup_steps))
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        self.use_amp = config.mixed_precision and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _make_loader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda b: collate_fn(b, self.model.config.pad_token_id),
            pin_memory=torch.cuda.is_available(),
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        loader = self._make_loader(self.train_dataset, shuffle=True)

        pbar = tqdm(loader, desc="Training")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    loss, _ = self.model(input_ids, attention_mask, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, _ = self.model(input_ids, attention_mask, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        loader = self._make_loader(self.val_dataset, shuffle=False)

        for batch in tqdm(loader, desc="Validating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    loss, _ = self.model(input_ids, attention_mask, labels)
            else:
                loss, _ = self.model(input_ids, attention_mask, labels)

            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    def save_checkpoint(self, path, metrics=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.model.config,
            "metrics": metrics or {},
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("metrics", {})

    def train(self, num_epochs=None, save_dir="outputs/checkpoints"):
        num_epochs = num_epochs or self.config.num_epochs
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            perplexity = math.exp(min(val_loss, 20))  # cap to avoid overflow
            lr = self.scheduler.get_last_lr()[0]

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  LR:         {lr:.2e}")

            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"  GPU Memory: {mem:.2f} GB")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    os.path.join(save_dir, "best_model.pt"),
                    {"epoch": epoch, "val_loss": val_loss, "train_loss": train_loss},
                )
                self.patience_counter = 0
                print("  -> New best model saved")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break

            self.save_checkpoint(
                os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
                {"epoch": epoch, "val_loss": val_loss, "train_loss": train_loss},
            )

        return self.best_val_loss
