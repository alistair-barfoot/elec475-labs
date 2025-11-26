"""
CLIP Model Evaluation and Visualization Script

Evaluates trained CLIP model on validation set with:
- Recall@K metrics (K=1,5,10) for image-to-text and text-to-image retrieval
- Text-based image retrieval visualization
- Zero-shot image classification
- Cosine similarity analysis
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

from model import CLIPModel
from dataloader import COCODataset, get_default_transforms, collate_fn
from train import simple_tokenize, get_caption_from_labels


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


@torch.no_grad()
def evaluate_retrieval(model, dataloader, device, max_samples=1000):
    """
    Evaluate image-text retrieval performance.
    
    Returns:
        i2t_recalls: Image-to-text recall@k metrics
        t2i_recalls: Text-to-image recall@k metrics
        image_embeds: All image embeddings (N, embed_dim)
        text_embeds: All text embeddings (N, embed_dim)
        similarity_matrix: Cosine similarity matrix (N, N)
    """
    model.eval()
    
    all_image_embeds = []
    all_text_embeds = []
    
    base_dataset = dataloader.dataset.dataset if isinstance(dataloader.dataset, Subset) else dataloader.dataset
    
    print(f"Computing embeddings for {len(dataloader.dataset)} samples...")
    
    sample_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= max_samples:
            break
            
        images = batch['images'].to(device)
        
        # Build text tokens from category labels
        text_tokens = []
        for labels in batch['labels']:
            caption = get_caption_from_labels(labels, base_dataset)
            text_tokens.append(simple_tokenize(caption))
        text_tokens = torch.stack(text_tokens).to(device)
        
        # Get embeddings
        img_embeds, txt_embeds, _ = model(images, text_tokens)
        
        all_image_embeds.append(img_embeds.cpu())
        all_text_embeds.append(txt_embeds.cpu())
        
        sample_count += len(images)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {sample_count} samples...")
    
    # Concatenate all embeddings
    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)
    
    print(f"Computing similarity matrix ({image_embeds.shape[0]}x{text_embeds.shape[0]})...")
    
    # Compute cosine similarity matrix (embeddings already normalized)
    similarity_matrix = image_embeds @ text_embeds.T
    
    print("Computing recall metrics...")
    
    # Image-to-Text retrieval: For each image, find matching text
    i2t_recalls = compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10])
    
    # Text-to-Image retrieval: For each text, find matching image
    t2i_recalls = compute_recall_at_k(similarity_matrix.T, k_values=[1, 5, 10])
    
    return i2t_recalls, t2i_recalls, image_embeds, text_embeds, similarity_matrix


def text_to_image_retrieval(text_query, model, dataloader, device, top_k=5):
    """
    Given a text query, retrieve top-K most similar images.
    
    Args:
        text_query: String text query (e.g., 'sport', 'person playing tennis')
        model: Trained CLIP model
        dataloader: DataLoader with images
        device: torch device
        top_k: Number of top images to retrieve
    
    Returns:
        top_images: List of PIL Images
        top_scores: List of similarity scores
        top_indices: List of dataset indices
    """
    model.eval()
    
    # Encode text query
    text_tokens = simple_tokenize(text_query).unsqueeze(0).to(device)
    with torch.no_grad():
        text_embed = model.text_encoder(text_tokens)  # (1, embed_dim)
    
    # Compute image embeddings and similarities
    all_similarities = []
    all_indices = []
    
    base_dataset = dataloader.dataset.dataset if isinstance(dataloader.dataset, Subset) else dataloader.dataset
    
    print(f"Searching for images matching: '{text_query}'...")
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(device)
        
        with torch.no_grad():
            img_embeds = model.image_encoder(images)
        
        # Compute similarity with query
        similarities = (img_embeds @ text_embed.T).squeeze(-1)
        all_similarities.append(similarities.cpu())
        
        # Track original indices
        batch_size = len(images)
        start_idx = batch_idx * dataloader.batch_size
        batch_indices = list(range(start_idx, start_idx + batch_size))
        all_indices.extend(batch_indices)
    
    # Concatenate all similarities
    all_similarities = torch.cat(all_similarities, dim=0)
    
    # Get top-K
    top_k_scores, top_k_positions = torch.topk(all_similarities, k=min(top_k, len(all_similarities)))
    top_k_indices = [all_indices[pos] for pos in top_k_positions.tolist()]
    
    # Retrieve actual images
    top_images = []
    for idx in top_k_indices:
        if isinstance(base_dataset, Subset):
            actual_idx = base_dataset.indices[idx]
            img_filename = base_dataset.dataset.image_files[actual_idx]
            img_path = os.path.join(base_dataset.dataset.images_path, img_filename)
        else:
            img_filename = base_dataset.image_files[idx]
            img_path = os.path.join(base_dataset.images_path, img_filename)
        
        img = Image.open(img_path).convert('RGB')
        top_images.append(img)
    
    return top_images, top_k_scores.tolist(), top_k_indices


def zero_shot_classification(image_path, class_labels, model, device):
    """
    Classify an image using text prompts (zero-shot classification).
    
    Args:
        image_path: Path to image file
        class_labels: List of class labels (e.g., ['a person', 'an animal', 'a landscape'])
        model: Trained CLIP model
        device: torch device
    
    Returns:
        predictions: Dictionary with class probabilities
        top_class: Most likely class
    """
    model.eval()
    
    # Load and preprocess image
    transform = get_default_transforms(image_size=(224, 224))
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Encode image
    with torch.no_grad():
        img_embed = model.image_encoder(image_tensor)  # (1, embed_dim)
    
    # Encode all class labels
    text_embeds = []
    for label in class_labels:
        text_tokens = simple_tokenize(label).unsqueeze(0).to(device)
        with torch.no_grad():
            txt_embed = model.text_encoder(text_tokens)
        text_embeds.append(txt_embed)
    
    text_embeds = torch.cat(text_embeds, dim=0)  # (n_classes, embed_dim)
    
    # Compute similarities
    similarities = (img_embed @ text_embeds.T).squeeze(0)  # (n_classes,)
    
    # Convert to probabilities with softmax
    probs = F.softmax(similarities * 100, dim=0)  # Scale by temperature
    
    predictions = {label: prob.item() for label, prob in zip(class_labels, probs)}
    top_class = class_labels[probs.argmax().item()]
    
    return predictions, top_class, image


def visualize_text_retrieval(text_query, top_images, top_scores, save_path=None):
    """Visualize top-K retrieved images for a text query."""
    n_images = len(top_images)
    fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))
    
    if n_images == 1:
        axes = [axes]
    
    fig.suptitle(f'Top {n_images} Images for Query: "{text_query}"', fontsize=14, fontweight='bold')
    
    for idx, (img, score, ax) in enumerate(zip(top_images, top_scores, axes)):
        ax.imshow(img)
        ax.set_title(f'Rank {idx+1}\nScore: {score:.3f}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def visualize_classification(image, predictions, save_path=None):
    """Visualize zero-shot classification results."""
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    ax_img.imshow(image)
    ax_img.set_title('Input Image', fontsize=12, fontweight='bold')
    ax_img.axis('off')
    
    # Show predictions as bar chart
    classes = list(predictions.keys())
    probs = list(predictions.values())
    
    colors = ['green' if p == max(probs) else 'steelblue' for p in probs]
    bars = ax_bar.barh(classes, probs, color=colors)
    ax_bar.set_xlabel('Probability', fontsize=11)
    ax_bar.set_title('Zero-Shot Classification', fontsize=12, fontweight='bold')
    ax_bar.set_xlim([0, 1])
    
    # Add probability labels on bars
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax_bar.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{prob:.2%}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = CLIPModel(embedding_dim=512, pretrained_image=True).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!\n")
    
    # Create validation dataloader
    val_transform = get_default_transforms(image_size=(224, 224))
    val_dataset = COCODataset(
        dataset='val',
        transform=val_transform,
        load_annotations=True,
        image_size=(224, 224)
    )
    
    # Use subset if specified
    if args.eval_subset > 0:
        indices = list(range(min(args.eval_subset, len(val_dataset))))
        val_dataset = Subset(val_dataset, indices)
        print(f"Using subset of {len(val_dataset)} validation samples\n")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    # Evaluate retrieval metrics
    print("="*60)
    print("RETRIEVAL EVALUATION")
    print("="*60)
    
    i2t_recalls, t2i_recalls, img_embeds, txt_embeds, sim_matrix = evaluate_retrieval(
        model, val_loader, device, max_samples=args.max_eval_samples
    )
    
    print(f"\n{'='*60}")
    print("IMAGE-TO-TEXT RETRIEVAL")
    print(f"{'='*60}")
    for metric, value in i2t_recalls.items():
        print(f"  {metric}: {value*100:.2f}%")
    
    print(f"\n{'='*60}")
    print("TEXT-TO-IMAGE RETRIEVAL")
    print(f"{'='*60}")
    for metric, value in t2i_recalls.items():
        print(f"  {metric}: {value*100:.2f}%")
    
    # Analyze similarity matrix
    print(f"\n{'='*60}")
    print("SIMILARITY MATRIX STATISTICS")
    print(f"{'='*60}")
    print(f"  Shape: {sim_matrix.shape}")
    print(f"  Mean similarity: {sim_matrix.mean().item():.4f}")
    print(f"  Std similarity: {sim_matrix.std().item():.4f}")
    print(f"  Min similarity: {sim_matrix.min().item():.4f}")
    print(f"  Max similarity: {sim_matrix.max().item():.4f}")
    
    # Diagonal (correct pairs) vs off-diagonal (incorrect pairs)
    diagonal = sim_matrix.diagonal()
    off_diagonal = sim_matrix[~torch.eye(sim_matrix.shape[0], dtype=bool)].view(sim_matrix.shape[0], -1)
    
    print(f"\n  Correct pairs (diagonal):")
    print(f"    Mean: {diagonal.mean().item():.4f}")
    print(f"    Std: {diagonal.std().item():.4f}")
    print(f"\n  Incorrect pairs (off-diagonal):")
    print(f"    Mean: {off_diagonal.mean().item():.4f}")
    print(f"    Std: {off_diagonal.std().item():.4f}")
    
    # Visualize similarity matrix
    print(f"\n{'='*60}")
    print("SIMILARITY MATRIX VISUALIZATION")
    print(f"{'='*60}")
    
    Path("outputs/visualizations").mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot similarity matrix heatmap
    im1 = axes[0].imshow(sim_matrix[:50, :50].numpy(), cmap='viridis', aspect='auto')
    axes[0].set_title('Similarity Matrix (First 50 samples)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Text Index')
    axes[0].set_ylabel('Image Index')
    plt.colorbar(im1, ax=axes[0], label='Cosine Similarity')
    
    # Plot distribution comparison
    axes[1].hist(diagonal.numpy(), bins=30, alpha=0.7, label='Correct Pairs (Diagonal)', color='green')
    axes[1].hist(off_diagonal.flatten().numpy(), bins=30, alpha=0.7, label='Incorrect Pairs', color='red')
    axes[1].set_xlabel('Similarity Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Similarity Score Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    sim_matrix_path = Path("outputs/visualizations") / "similarity_matrix.png"
    plt.savefig(sim_matrix_path, dpi=150, bbox_inches='tight')
    print(f"Saved similarity matrix visualization to: {sim_matrix_path}")
    plt.show()
    
    # Visualize actual image-text pair examples with performance
    print(f"\n{'='*60}")
    print("IMAGE-TEXT PAIR EXAMPLES WITH SIMILARITY SCORES")
    print(f"{'='*60}")
    
    # Get actual dataset for loading images
    base_dataset = val_loader.dataset.dataset if isinstance(val_loader.dataset, Subset) else val_loader.dataset
    
    # Select diverse samples: best matches, worst matches, and average matches
    n_samples = min(12, len(diagonal))
    
    # Get indices for best, worst, and medium performing pairs
    sorted_indices = diagonal.argsort(descending=True)
    best_indices = sorted_indices[:4].tolist()
    worst_indices = sorted_indices[-4:].tolist()
    medium_indices = sorted_indices[len(sorted_indices)//2 - 2:len(sorted_indices)//2 + 2].tolist()
    
    sample_indices = best_indices + medium_indices + worst_indices
    sample_categories = ['Best Match'] * 4 + ['Medium Match'] * 4 + ['Worst Match'] * 4
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Image-Text Pair Examples Ranked by Similarity Score', fontsize=16, fontweight='bold')
    
    for idx, (sample_idx, category) in enumerate(zip(sample_indices, sample_categories)):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        # Get the actual image
        if isinstance(val_loader.dataset, Subset):
            actual_idx = val_loader.dataset.indices[sample_idx]
            img_filename = base_dataset.image_files[actual_idx]
            img_path = os.path.join(base_dataset.images_path, img_filename)
            _, labels, _ = base_dataset._get_image_annotations(img_filename)
        else:
            img_filename = base_dataset.image_files[sample_idx]
            img_path = os.path.join(base_dataset.images_path, img_filename)
            _, labels, _ = base_dataset._get_image_annotations(img_filename)
        
        # Load and display image
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        
        # Get caption
        caption = get_caption_from_labels(labels, base_dataset)
        similarity = diagonal[sample_idx].item()
        
        # Color code by performance
        if category == 'Best Match':
            color = 'green'
        elif category == 'Worst Match':
            color = 'red'
        else:
            color = 'orange'
        
        # Truncate long captions
        if len(caption) > 50:
            caption = caption[:47] + '...'
        
        ax.set_title(f'{category}\nSim: {similarity:.3f}\n"{caption}"', 
                    fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    pair_examples_path = Path("outputs/visualizations") / "image_text_pair_examples.png"
    plt.savefig(pair_examples_path, dpi=150, bbox_inches='tight')
    print(f"Saved image-text pair examples to: {pair_examples_path}")
    plt.show()
    
    # Visualize retrieval performance with examples
    print(f"\n{'='*60}")
    print("RETRIEVAL PERFORMANCE EXAMPLES")
    print(f"{'='*60}")
    
    # Show a few examples of image-to-text retrieval
    n_retrieval_examples = 6
    example_indices = np.linspace(0, min(len(diagonal)-1, 100), n_retrieval_examples, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Image-to-Text Retrieval: Top-5 Retrieved Texts', fontsize=14, fontweight='bold')
    
    for plot_idx, sample_idx in enumerate(example_indices):
        row = plot_idx // 3
        col = plot_idx % 3
        ax = axes[row, col]
        
        # Get the image
        if isinstance(val_loader.dataset, Subset):
            actual_idx = val_loader.dataset.indices[sample_idx]
            img_filename = base_dataset.image_files[actual_idx]
            img_path = os.path.join(base_dataset.images_path, img_filename)
            _, correct_labels, _ = base_dataset._get_image_annotations(img_filename)
        else:
            img_filename = base_dataset.image_files[sample_idx]
            img_path = os.path.join(base_dataset.images_path, img_filename)
            _, correct_labels, _ = base_dataset._get_image_annotations(img_filename)
        
        img = Image.open(img_path).convert('RGB')
        
        # Get top-5 retrieved texts for this image
        similarities = sim_matrix[sample_idx]
        top5_text_indices = similarities.topk(5).indices.tolist()
        top5_scores = similarities.topk(5).values.tolist()
        
        # Create visualization
        ax.imshow(img)
        
        # Build text showing retrieved captions
        correct_caption = get_caption_from_labels(correct_labels, base_dataset)
        retrieved_texts = []
        for rank, (text_idx, score) in enumerate(zip(top5_text_indices, top5_scores), 1):
            # Get labels for this text index
            if isinstance(val_loader.dataset, Subset):
                text_actual_idx = val_loader.dataset.indices[text_idx]
                text_filename = base_dataset.image_files[text_actual_idx]
                _, text_labels, _ = base_dataset._get_image_annotations(text_filename)
            else:
                text_filename = base_dataset.image_files[text_idx]
                _, text_labels, _ = base_dataset._get_image_annotations(text_filename)
            
            caption = get_caption_from_labels(text_labels, base_dataset)
            if len(caption) > 30:
                caption = caption[:27] + '...'
            
            # Mark if it's the correct match
            marker = '✓' if text_idx == sample_idx else '✗'
            retrieved_texts.append(f"{rank}. {marker} {caption} ({score:.3f})")
        
        # Check if correct text is in top-5
        rank_color = 'green' if sample_idx in top5_text_indices else 'red'
        correct_rank = top5_text_indices.index(sample_idx) + 1 if sample_idx in top5_text_indices else '>5'
        
        title_text = f'Query Image (Rank: {correct_rank})\nTrue: "{correct_caption[:35]}..."'
        ax.set_title(title_text, fontsize=8, color=rank_color, fontweight='bold')
        ax.axis('off')
        
        # Add retrieved text as text below image
        text_str = '\n'.join(retrieved_texts)
        ax.text(0.5, -0.15, text_str, transform=ax.transAxes,
               fontsize=7, verticalalignment='top', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    retrieval_examples_path = Path("outputs/visualizations") / "retrieval_performance_examples.png"
    plt.savefig(retrieval_examples_path, dpi=150, bbox_inches='tight')
    print(f"Saved retrieval performance examples to: {retrieval_examples_path}")
    plt.show()
    
    # Automatic text-based image retrieval for common queries
    print(f"\n{'='*60}")
    print("AUTOMATIC TEXT-TO-IMAGE RETRIEVAL EXAMPLES")
    print(f"{'='*60}")
    
    default_queries = ['person', 'animal', 'vehicle', 'food']
    for query_idx, query in enumerate(default_queries):
        print(f"\nQuery {query_idx+1}: '{query}'")
        
        top_images, top_scores, top_indices = text_to_image_retrieval(
            query, model, val_loader, device, top_k=5
        )
        
        print(f"  Top 5 scores: {[f'{s:.3f}' for s in top_scores]}")
        
        save_path = Path("outputs/visualizations") / f"text_retrieval_{query.replace(' ', '_')}.png"
        visualize_text_retrieval(query, top_images, top_scores, save_path=save_path)
    
    # Text-based image retrieval visualization (custom queries)
    if args.text_query:
        print(f"\n{'='*60}")
        print("CUSTOM TEXT-TO-IMAGE RETRIEVAL QUERIES")
        print(f"{'='*60}")
        
        Path("outputs/visualizations").mkdir(parents=True, exist_ok=True)
        
        for query_idx, query in enumerate(args.text_query):
            print(f"\nQuery {query_idx+1}: '{query}'")
            
            top_images, top_scores, top_indices = text_to_image_retrieval(
                query, model, val_loader, device, top_k=5
            )
            
            print(f"  Top 5 scores: {[f'{s:.3f}' for s in top_scores]}")
            
            save_path = Path("outputs/visualizations") / f"text_retrieval_{query.replace(' ', '_')}.png"
            visualize_text_retrieval(query, top_images, top_scores, save_path=save_path)
    
    # Zero-shot classification
    if args.classify_image:
        print(f"\n{'='*60}")
        print("ZERO-SHOT IMAGE CLASSIFICATION")
        print(f"{'='*60}")
        
        Path("outputs/visualizations").mkdir(parents=True, exist_ok=True)
        
        for img_idx, img_path in enumerate(args.classify_image):
            print(f"\nImage {img_idx+1}: {img_path}")
            
            class_labels = args.class_labels if args.class_labels else [
                'a person', 'an animal', 'a landscape', 'a vehicle', 
                'food', 'a building', 'sports', 'indoor scene'
            ]
            
            predictions, top_class, image = zero_shot_classification(
                img_path, class_labels, model, device
            )
            
            print(f"  Predicted class: {top_class}")
            print(f"  Probabilities:")
            for label, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                print(f"    {label}: {prob*100:.2f}%")
            
            img_name = Path(img_path).stem
            save_path = Path("outputs/visualizations") / f"classification_{img_name}.png"
            visualize_classification(image, predictions, save_path=save_path)
    
    print(f"\n{'='*60}")
    print("✓ Evaluation complete!")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CLIP model on retrieval and classification tasks")
    
    parser.add_argument("--checkpoint", type=str, default="outputs/best_clip.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--eval-subset", type=int, default=0,
                        help="Use only N validation samples (0 = all)")
    parser.add_argument("--max-eval-samples", type=int, default=1000,
                        help="Maximum samples for retrieval evaluation")
    
    # Text-to-image retrieval
    parser.add_argument("--text-query", type=str, nargs='+', default=None,
                        help="Text queries for image retrieval (e.g., 'sport' 'person')")
    
    # Zero-shot classification
    parser.add_argument("--classify-image", type=str, nargs='+', default=None,
                        help="Image paths to classify")
    parser.add_argument("--class-labels", type=str, nargs='+', default=None,
                        help="Class labels for classification")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
