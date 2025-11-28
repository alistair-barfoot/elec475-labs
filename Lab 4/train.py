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
import datetime
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter warnings
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataloader import COCODataset, collate_fn, create_dataloader, get_default_transforms
from model import CLIPModel


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
    return " ".join(sorted(filtered))


def compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10]):
    """
    Compute Recall@K for retrieval tasks.
    
    Args:
        similarity_matrix: (N, M) matrix where entry [i,j] is similarity between query i and item j
        k_values: List of K values to compute recall for
    
    Returns:
        recalls: Dictionary with recall@k for each k
    """
    recalls = {}
    n_queries = similarity_matrix.shape[0]
    
    # Get top-K predictions for each query
    # Correct match is at index i (diagonal)
    for k in k_values:
        # Get indices of top-k most similar items
        top_k_indices = similarity_matrix.topk(k, dim=1).indices
        
        # Check if correct index (i) is in top-k for each query i
        correct_in_top_k = 0
        for i in range(n_queries):
            if i in top_k_indices[i]:
                correct_in_top_k += 1
        
        recall = correct_in_top_k / n_queries
        recalls[f'R@{k}'] = recall
    
    return recalls


def clip_loss(image_embeds, text_embeds, logit_scale):
    """InfoNCE contrastive loss (symmetric)."""
    logits = logit_scale * (image_embeds @ text_embeds.T)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2.0


@torch.no_grad()
def validate(model, dataloader, device, compute_retrieval_metrics=False):
    """Run validation pass."""
    model.eval()
    total_loss = 0.0
    batches = 0
    
    # For retrieval metrics
    all_image_embeds = []
    all_text_embeds = []
    
    if isinstance(dataloader.dataset, Subset):
        base_dataset = dataloader.dataset.dataset
    else:
        base_dataset = dataloader.dataset
    assert isinstance(base_dataset, COCODataset)
    
    for batch in dataloader:
        images = batch['images'].to(device)
        
        # Build text tokens (prefer real captions; fallback to labels)
        text_tokens = []
        for i, labels in enumerate(batch['labels']):
            caption_list = []
            if 'filenames' in batch:
                fname = batch['filenames'][i]
                get_captions_fn = getattr(base_dataset, 'get_captions', None)
                if callable(get_captions_fn):
                    try:
                        caption_list = get_captions_fn(fname)
                    except Exception:
                        caption_list = []
            if isinstance(caption_list, list) and len(caption_list) > 0:
                caption = caption_list[0]
            else:
                caption = get_caption_from_labels(labels, base_dataset)
            text_tokens.append(simple_tokenize(caption))
        text_tokens = torch.stack(text_tokens).to(device)
        
        img_embeds, txt_embeds, logit_scale = model(images, text_tokens)
        loss = clip_loss(img_embeds, txt_embeds, logit_scale)
        total_loss += loss.item()
        batches += 1
        
        # Store embeddings for retrieval metrics
        if compute_retrieval_metrics:
            all_image_embeds.append(img_embeds.cpu())
            all_text_embeds.append(txt_embeds.cpu())
    
    avg_loss = total_loss / max(1, batches)
    
    # Compute retrieval metrics if requested
    retrieval_metrics = {}
    if compute_retrieval_metrics and len(all_image_embeds) > 0:
        # Concatenate all embeddings
        image_embeds = torch.cat(all_image_embeds, dim=0)
        text_embeds = torch.cat(all_text_embeds, dim=0)
        
        # Compute cosine similarity matrix (embeddings already normalized)
        similarity_matrix = image_embeds @ text_embeds.T
        
        # Image-to-Text retrieval: For each image, find matching text
        i2t_recalls = compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10])
        
        # Text-to-Image retrieval: For each text, find matching image
        t2i_recalls = compute_recall_at_k(similarity_matrix.T, k_values=[1, 5, 10])
        
        retrieval_metrics = {
            'i2t': i2t_recalls,
            't2i': t2i_recalls,
            'similarity_matrix': similarity_matrix
        }
    
    return avg_loss, retrieval_metrics


def train(args):
    set_seed(int(time.time()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.debug_cuda and torch.cuda.is_available():
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("[Debug] CUDA_LAUNCH_BLOCKING=1 enabled")

    if args.deterministic and torch.cuda.is_available():
        # Ensure cuBLAS reproducibility on CUDA >= 10.2
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            print('[Debug] Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic cuBLAS')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[Debug] Deterministic CUDA algorithms enabled")
    
    # Create datasets first (before DataLoader)
    train_transform = get_default_transforms(image_size=(224, 224))
    train_dataset = COCODataset(
        dataset='train',
        transform=train_transform,
        load_annotations=True,
        image_size=(224, 224),
        load_captions=True
    )
    
    val_transform = get_default_transforms(image_size=(224, 224))
    val_dataset = COCODataset(
        dataset='val',
        transform=val_transform,
        load_annotations=True,
        image_size=(224, 224),
        load_captions=True
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0,  # Reduced workers to save memory
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn,
        persistent_workers=False  # Don't keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0,  # Reduced workers to save memory
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn,
        persistent_workers=False  # Don't keep workers alive between epochs
    )
    
    # Model
    model = CLIPModel(embedding_dim=512, pretrained_image=True).to(device)
    if getattr(args, 'train_text', False):
        if hasattr(model, "text_encoder") and hasattr(model.text_encoder, "parameters"):
            for p in model.text_encoder.parameters():
                p.requires_grad = True
            print("[Config] Text encoder parameters set to requires_grad=True")
        else:
            print("[Warning] model has no text_encoder; --train-text ignored")
    
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
    start_epoch = 1
    start_time = time.time()
    inference_speed = 0.0
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from epoch {checkpoint['epoch']}, best val loss: {best_val_loss:.4f}\n")
    
    if isinstance(train_loader.dataset, Subset):
        base_dataset = train_loader.dataset.dataset
    else:
        base_dataset = train_loader.dataset
    assert isinstance(base_dataset, COCODataset)
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        batches = 0
        
        # Clear cache at start of epoch (optional mid-batch clearing removed for stability)
        if torch.cuda.is_available() and not args.no_epoch_cache_clear:
            torch.cuda.empty_cache()
        
        for batch in train_loader:
            images = batch['images'].to(device, non_blocking=True)
            
            # Build text tokens (prefer real captions; fallback to labels)
            text_tokens = []
            for i, labels in enumerate(batch['labels']):
                caption_list = []
                if 'filenames' in batch:
                    fname = batch['filenames'][i]
                    if hasattr(base_dataset, 'get_captions'):
                        try:
                            caption_list = base_dataset.get_captions(fname)
                        except Exception:
                            caption_list = []
                if isinstance(caption_list, list) and len(caption_list) > 0:
                    caption = caption_list[0]
                else:
                    caption = get_caption_from_labels(labels, base_dataset)
                text_tokens.append(simple_tokenize(caption))
            text_tokens = torch.stack(text_tokens).to(device)
            
            try:
                # Forward
                img_embeds, txt_embeds, logit_scale = model(images, text_tokens)
                loss = clip_loss(img_embeds, txt_embeds, logit_scale)

                # Backward
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
            except RuntimeError as e:
                if 'illegal memory access' in str(e).lower():
                    print("\n[Error] CUDA illegal memory access detected. Aborting current run gracefully.")
                    print("Suggestion: \n - Restart Python process to fully reset GPU state\n - Resume from latest checkpoint with --resume\n - Optionally reduce --batch-size or enable --deterministic\n - Run with --debug-cuda for precise backtrace")
                    raise
                else:
                    raise
            
            epoch_loss += loss.item()
            batches += 1
            
            # (Removed periodic empty_cache to avoid potential driver instability)
        
        avg_train = epoch_loss / batches
        train_losses.append(avg_train)
        
        # Validation (compute retrieval metrics every epoch)
        val_start = time.time()
        if torch.cuda.is_available() and not args.no_epoch_cache_clear:
            torch.cuda.empty_cache()  # Clear cache before validation if not disabled
        avg_val, retrieval_metrics = validate(model, val_loader, device, compute_retrieval_metrics=True)
        val_losses.append(avg_val)
        val_time = time.time() - val_start
        
        # Timing calculations (handle resume correctly)
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        epochs_completed_since_resume = max(1, (epoch - start_epoch + 1))
        avg_epoch_time = elapsed_time / epochs_completed_since_resume
        remaining_epochs = max(0, args.epochs - epoch)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_minutes = eta_seconds / 60
        eta_hours = eta_minutes / 60
        
        # Inference speed (ms/image during validation)
        val_images = len(val_dataset)
        inference_speed = (val_time / val_images * 1000) if val_time > 0 else 0
        
        # Estimated finish time
        finish_time = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
        finish_str = finish_time.strftime("%H:%M:%S")
        
        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        # Print retrieval metrics
        if retrieval_metrics:
            i2t = retrieval_metrics['i2t']
            t2i = retrieval_metrics['t2i']
            print(f"  Image→Text: R@1={i2t['R@1']*100:.1f}% R@5={i2t['R@5']*100:.1f}% R@10={i2t['R@10']*100:.1f}%")
            print(f"  Text→Image: R@1={t2i['R@1']*100:.1f}% R@5={t2i['R@5']*100:.1f}% R@10={t2i['R@10']*100:.1f}%")
        
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
    parser.add_argument("--resume", type=str, default="", 
                        help="Path to checkpoint to resume from (e.g., outputs/best_clip.pth)")
    parser.add_argument("--train-text", action="store_true", 
                        help="Unfreeze and train the text encoder as well")
    parser.add_argument("--debug-cuda", action="store_true", help="Enable CUDA_LAUNCH_BLOCKING for debugging")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDA kernels")
    parser.add_argument("--no-epoch-cache-clear", action="store_true", help="Disable emptying CUDA cache each epoch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)