"""
CLIP-style training script with InfoNCE loss.

Simplified to core arguments:
- --lr: learning rate
- --epochs: number of training epochs
- --batch-size: batch size for training
- --train-subset: limit training to N samples (0 = use all)
- --save-plot: save loss curve plot (default: loss_curves.png)
"""

import argparse
import time
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt

from model import CLIPModel
from dataloader import create_dataloader


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_subset(dataset, subset_size: int, seed: int):
    """Create a random subset of the dataset for quick tuning."""
    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:subset_size])


def simple_tokenize(caption: str, max_length: int = 77, vocab_size: int = 49408):
    """Simple hash-based tokenizer (placeholder - use proper tokenizer in production)."""
    words = caption.strip().lower().split()
    tokens = []
    for w in words[:max_length]:
        bucket = (hash(w) % (vocab_size - 2)) + 2
        tokens.append(bucket)
    tokens += [0] * (max_length - len(tokens))  # pad
    return torch.tensor(tokens, dtype=torch.long)


def get_caption_from_labels(labels, dataset):
    """Convert object labels to pseudo-caption."""
    if not labels:
        return "none"
    names = [dataset.get_category_name(l) for l in labels]
    # Deduplicate
    seen = set()
    filtered = []
    for n in names:
        if n not in seen:
            seen.add(n)
            filtered.append(n)
    return " ".join(filtered)


def clip_loss(image_embeds, text_embeds, logit_scale):
    """InfoNCE contrastive loss (symmetric)."""
    logits = logit_scale * (image_embeds @ text_embeds.T)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2.0


@torch.no_grad()
def validate(model, dataloader, device):
    """Run validation pass."""
    model.eval()
    total_loss = 0.0
    batches = 0
    
    base_dataset = dataloader.dataset.dataset if isinstance(dataloader.dataset, Subset) else dataloader.dataset
    
    for batch in dataloader:
        images = batch['images'].to(device)
        
        # Build text tokens from category labels
        text_tokens = []
        for labels in batch['labels']:
            caption = get_caption_from_labels(labels, base_dataset)
            text_tokens.append(simple_tokenize(caption))
        text_tokens = torch.stack(text_tokens).to(device)
        
        img_embeds, txt_embeds, logit_scale = model(images, text_tokens)
        loss = clip_loss(img_embeds, txt_embeds, logit_scale)
        total_loss += loss.item()
        batches += 1
    
    return total_loss / max(1, batches)


def train(args):
    set_seed(int(time.time()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create datasets first (before DataLoader)
    from dataloader import COCODataset, get_default_transforms
    
    train_transform = get_default_transforms(image_size=(224, 224))
    train_dataset = COCODataset(
        dataset='train',
        transform=train_transform,
        load_annotations=True,
        image_size=(224, 224)
    )
    
    val_transform = get_default_transforms(image_size=(224, 224))
    val_dataset = COCODataset(
        dataset='val',
        transform=val_transform,
        load_annotations=True,
        image_size=(224, 224)
    )
    
    # Apply subset if specified
    if args.train_subset > 0:
        train_dataset = build_subset(train_dataset, args.train_subset, int(time.time()))
        print(f"Training on subset: {len(train_dataset)} images")
    
    if args.val_subset > 0:
        val_dataset = build_subset(val_dataset, args.val_subset, int(time.time()))
        print(f"Validation on subset: {len(val_dataset)} images")
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create DataLoaders with (possibly subsetted) datasets
    from dataloader import collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    # Model
    model = CLIPModel(embedding_dim=512, pretrained_image=True).to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    
    print(f"\nTrainable parameters: {sum(p.numel() for p in params):,}")
    print(f"Config: lr={args.lr} epochs={args.epochs} batch_size={args.batch_size}\n")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    start_time = time.time()
    
    base_dataset = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        batches = 0
        
        for batch in train_loader:
            images = batch['images'].to(device)
            
            # Build text tokens from category labels
            text_tokens = []
            for labels in batch['labels']:
                caption = get_caption_from_labels(labels, base_dataset)
                text_tokens.append(simple_tokenize(caption))
            text_tokens = torch.stack(text_tokens).to(device)
            
            # Forward
            img_embeds, txt_embeds, logit_scale = model(images, text_tokens)
            loss = clip_loss(img_embeds, txt_embeds, logit_scale)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
        
        avg_train = epoch_loss / batches
        train_losses.append(avg_train)
        
        # Validation
        val_start = time.time()
        avg_val = validate(model, val_loader, device)
        val_losses.append(avg_val)
        val_time = time.time() - val_start
        
        # Timing calculations
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        avg_epoch_time = elapsed_time / epoch
        remaining_epochs = args.epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_minutes = eta_seconds / 60
        eta_hours = eta_minutes / 60
        
        # Inference speed (ms/image during validation)
        val_images = len(val_loader.dataset)
        inference_speed = (val_time / val_images * 1000) if val_time > 0 else 0
        
        # Estimated finish time
        import datetime
        finish_time = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
        finish_str = finish_time.strftime("%H:%M:%S")
        
        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        # Format time as hh:mm:ss
        epoch_hours = int(epoch_time // 3600)
        epoch_minutes = int((epoch_time % 3600) // 60)
        epoch_seconds = int(epoch_time % 60)
        
        elapsed_hours = int(elapsed_time // 3600)
        elapsed_minutes = int((elapsed_time % 3600) // 60)
        elapsed_seconds = int(elapsed_time % 60)
        
        eta_hours_int = int(eta_hours)
        eta_minutes_int = int(eta_minutes % 60)
        eta_seconds_int = int(eta_seconds % 60)
        
        print(f"  Time: {epoch_hours:02d}:{epoch_minutes:02d}:{epoch_seconds:02d} | Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d} | ETA: {eta_hours_int:02d}:{eta_minutes_int:02d}:{eta_seconds_int:02d} (finish ~{finish_str}) | Inference: {inference_speed:.1f} ms")

        # Save model checkpoint if validation improves
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            Path("outputs").mkdir(exist_ok=True)
            best_ckpt_path = Path("outputs") / "best_clip.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train,
                'val_loss': avg_val,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': {'lr': args.lr, 'epochs': args.epochs, 'batch_size': args.batch_size}
            }, best_ckpt_path)
            print(f"  ✓ New best model saved (val_loss: {avg_val:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
                break
        
        # Save plot after each epoch
        Path("outputs").mkdir(exist_ok=True)
        plot_path = Path("outputs") / args.save_plot
        
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss", marker='o')
        plt.plot(val_losses, label="Val Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("InfoNCE Loss")
        plt.title(f"CLIP Training Loss Curves (Epoch {epoch}/{args.epochs})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
    
    total_time = time.time() - start_time
    total_hours = int(total_time // 3600)
    total_minutes = int((total_time % 3600) // 60)
    total_seconds = int(total_time % 60)
    
    print(f"\nTotal training time: {total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}")
    print(f"Inference speed: {inference_speed:.1f} ms/image")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final checkpoint
    ckpt_path = Path("outputs") / "final_clip.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': {'lr': args.lr, 'epochs': args.epochs, 'batch_size': args.batch_size}
    }, ckpt_path)
    print(f"Model checkpoint saved to: {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP model with InfoNCE loss")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--train-subset", type=int, default=0, 
                        help="Use only N training samples (0 = all). For tuning hyperparams on small subset.")
    parser.add_argument("--val-subset", type=int, default=0, 
                        help="Use only N validation samples (0 = all). For faster validation during tuning.")
    parser.add_argument("--save-plot", type=str, default="loss_curves.png", 
                        help="Filename for loss curve plot")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)