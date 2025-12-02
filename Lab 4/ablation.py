"""
CLIP Model Ablation Study Script

Evaluates multiple CLIP models on retrieval tasks and classification.
Provides comparative analysis with tabulated results.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tabulate import tabulate

from dataloader import COCODataset, collate_fn, get_default_transforms
from model import CLIPModel
from train import simple_tokenize, compute_recall_at_k, validate


def load_model(checkpoint_path: str, device: torch.device) -> CLIPModel | None:
    """Load a CLIP model from checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config if available
        config = checkpoint.get('config', {})
        
        # Auto-detect model architecture from state dict if config is missing
        state_dict = checkpoint['model_state_dict']
        use_batchnorm = False
        use_dropout = False
        
        # Check for BatchNorm layers in state dict (look for specific BatchNorm indicators)
        batchnorm_keys = [key for key in state_dict.keys() if 'BatchNorm' in key or ('running_mean' in key and 'projection' in key)]
        if batchnorm_keys:
            use_batchnorm = True
            print(f"  Detected BatchNorm layers in {checkpoint_path}: {len(batchnorm_keys)} BN parameters")
        
        # Check projection layer structure to determine if dropout is present
        projection_weight_keys = [key for key in state_dict.keys() if 'image_encoder.projection' in key and 'weight' in key]
        projection_weight_keys.sort()  # Sort to get consistent ordering
        
        if len(projection_weight_keys) == 2:
            # Standard architecture: just input->hidden and hidden->output
            use_batchnorm = False
            use_dropout = False
            print(f"  Detected standard architecture (no regularization) in {checkpoint_path}")
        elif len(projection_weight_keys) > 2:
            # Extended architecture with regularization layers
            # Analyze the layer indices to determine what's present
            layer_indices = [int(key.split('.')[2]) for key in projection_weight_keys]
            max_idx = max(layer_indices)
            
            # If we have layers beyond index 2, we have regularization
            if max_idx >= 3:
                use_dropout = True
                print(f"  Detected Dropout layers in {checkpoint_path}")
            
            # Check if layer 1 exists (BatchNorm position in regularized architecture)
            if 1 in layer_indices:
                # Check if there are BatchNorm parameters at layer 1
                bn_check_keys = [key for key in state_dict.keys() if 'projection.1.' in key and ('running_mean' in key or 'running_var' in key)]
                if bn_check_keys:
                    use_batchnorm = True
                    print(f"  Detected BatchNorm at layer 1 in {checkpoint_path}")
        
        print(f"  Final detection: BatchNorm={use_batchnorm}, Dropout={use_dropout}")
        
        # Use config values if available, otherwise use detected values
        final_use_batchnorm = config.get('use_batchnorm', use_batchnorm)
        final_use_dropout = config.get('use_dropout', use_dropout)
        final_dropout_rate = config.get('dropout_rate', 0.1)
        
        print(f"  Creating model with BatchNorm={final_use_batchnorm}, Dropout={final_use_dropout}")
        
        # Create model with detected or configured architecture
        model = CLIPModel(
            embedding_dim=512,
            pretrained_image=True,
            use_batchnorm=final_use_batchnorm,
            use_dropout=final_use_dropout,
            dropout_rate=final_dropout_rate
        ).to(device)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")
        print(f"This might be due to architecture mismatch. Check if model was trained with different regularization settings.")
        return None


def compute_embeddings(model: CLIPModel, dataloader: DataLoader, device: torch.device, 
                      max_samples: int = 2000) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Compute image and text embeddings for a dataset."""
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    all_filenames = []
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_processed >= max_samples:
                break
                
            images = batch['images'].to(device)
            filenames = batch['filenames']
            
            # Build text tokens
            text_tokens = []
            for i, labels in enumerate(batch['labels']):
                # Use real captions if available, otherwise use labels
                base_dataset = dataloader.dataset
                
                # Check if dataset has get_captions method and use it safely
                caption_list = []
                try:
                    # Try to access get_captions if it exists (COCODataset should have this)
                    if hasattr(base_dataset, 'get_captions'):
                        caption_list = base_dataset.get_captions(filenames[i])  # type: ignore
                except (AttributeError, Exception) as e:
                    print(f"Warning: Could not get captions for {filenames[i]}: {e}")
                    caption_list = []
                if caption_list:
                    caption = caption_list[0]
                else:
                    # Fallback to label-based caption
                    from train import get_caption_from_labels
                    caption = get_caption_from_labels(labels, base_dataset)
                text_tokens.append(simple_tokenize(caption))
            
            text_tokens = torch.stack(text_tokens).to(device)
            
            # Forward pass
            img_embeds, txt_embeds, _ = model(images, text_tokens)
            
            # Store embeddings
            all_image_embeds.append(img_embeds.cpu())
            all_text_embeds.append(txt_embeds.cpu())
            all_filenames.extend(filenames)
            
            samples_processed += len(filenames)
    
    # Concatenate all embeddings
    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, p=2, dim=1)
    text_embeds = F.normalize(text_embeds, p=2, dim=1)
    
    return image_embeds, text_embeds, all_filenames


def evaluate_retrieval(models: Dict[str, CLIPModel], val_loader: DataLoader, 
                      device: torch.device, compute_recall: bool = False) -> Dict[str, Dict]:
    """Evaluate retrieval performance for all models."""
    results = {}
    
    if compute_recall:
        print("Evaluating retrieval performance...")
    else:
        print("Computing embeddings (skipping recall metrics - use --compute-recall for full evaluation)...")
    
    for model_name, model in models.items():
        print(f"  Processing {model_name}...")
        
        # Compute embeddings
        image_embeds, text_embeds, filenames = compute_embeddings(
            model, val_loader, device, max_samples=2000
        )
        
        # Compute similarity matrix and recall metrics if requested
        similarity_matrix = None
        i2t_recalls = None
        t2i_recalls = None
        
        if compute_recall:
            similarity_matrix = image_embeds @ text_embeds.T
            
            # Compute retrieval metrics
            i2t_recalls = compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10])
            t2i_recalls = compute_recall_at_k(similarity_matrix.T, k_values=[1, 5, 10])
        
        results[model_name] = {
            'i2t': i2t_recalls,
            't2i': t2i_recalls,
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
            'filenames': filenames,
            'similarity_matrix': similarity_matrix
        }
    
    return results


def text_to_image_retrieval(query: str, models: Dict[str, CLIPModel], 
                           embeddings: Dict[str, Dict], device: torch.device,
                           top_k: int = 5) -> Dict[str, Dict[str, List]]:
    """Perform text-to-image retrieval for a given query."""
    results: Dict[str, Dict[str, List]] = {}
    
    print(f"\nText-to-image retrieval for query: '{query}'")
    
    for model_name, model in models.items():
        print(f"  Processing {model_name}...")
        
        # Encode query
        query_tokens = simple_tokenize(query).unsqueeze(0).to(device)
        with torch.no_grad():
            # Use dummy image for text-only encoding
            dummy_image = torch.zeros(1, 3, 224, 224, device=device)
            _, query_embed, _ = model(dummy_image, query_tokens)
            query_embed = F.normalize(query_embed, p=2, dim=1)
        
        # Compute similarities with all images
        image_embeds = embeddings[model_name]['image_embeds']
        similarities = (query_embed.cpu() @ image_embeds.T).squeeze(0)
        
        # Get top-k results
        top_indices = torch.topk(similarities, k=top_k).indices
        top_similarities = similarities[top_indices]
        top_filenames = [embeddings[model_name]['filenames'][idx] for idx in top_indices]
        
        results[model_name] = {
            'filenames': top_filenames,
            'similarities': top_similarities.tolist(),
            'indices': top_indices.tolist()
        }
    
    return results


def classify_image(image_path: str, class_list: List[str], models: Dict[str, CLIPModel],
                  device: torch.device) -> Dict[str, Dict]:
    """Classify an image using zero-shot classification."""
    results = {}
    
    print(f"\nClassifying image: {image_path}")
    print(f"Classes: {class_list}")
    
    # Load and preprocess image
    transform = get_default_transforms(image_size=(224, 224))
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)  # This returns a tensor
        if not isinstance(image_tensor, torch.Tensor):
            raise ValueError("Transform should return a tensor")
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
    except Exception as e:
        print(f"Error loading image: {e}")
        return {}
    
    for model_name, model in models.items():
        print(f"  Processing {model_name}...")
        
        with torch.no_grad():
            # Encode image
            image_features, _, _ = model(image_tensor, torch.zeros(1, 77, dtype=torch.long, device=device))
            image_features = F.normalize(image_features, p=2, dim=1)
            
            # Encode class texts
            class_features = []
            for class_name in class_list:
                class_tokens = simple_tokenize(f"a photo of a {class_name}").unsqueeze(0).to(device)
                dummy_image = torch.zeros(1, 3, 224, 224, device=device)
                _, class_embed, _ = model(dummy_image, class_tokens)
                class_features.append(class_embed)
            
            class_features = torch.cat(class_features, dim=0)
            class_features = F.normalize(class_features, p=2, dim=1)
            
            # Compute similarities
            similarities = (image_features @ class_features.T).squeeze(0)
            probabilities = F.softmax(similarities * 100, dim=0)  # Scale for softmax
            
            # Sort by probability
            sorted_indices = torch.argsort(probabilities, descending=True)
            sorted_probs = probabilities[sorted_indices]
            sorted_classes = [class_list[idx] for idx in sorted_indices]
            
            results[model_name] = {
                'classes': sorted_classes,
                'probabilities': sorted_probs.cpu().tolist(),
                'similarities': similarities[sorted_indices].cpu().tolist()
            }
    
    return results


def create_results_table(retrieval_results: Dict[str, Dict]) -> str:
    """Create a formatted table of retrieval results."""
    # Check if any model has recall results
    has_recall_data = any(results.get('i2t') is not None for results in retrieval_results.values())
    
    if not has_recall_data:
        return "No recall metrics computed. Use --compute-recall flag to enable detailed evaluation."
    
    # Prepare data for table
    table_data = []
    headers = ['Model', 'I->T R@1', 'I->T R@5', 'I->T R@10', 'T->I R@1', 'T->I R@5', 'T->I R@10']
    
    for model_name, results in retrieval_results.items():
        i2t = results.get('i2t')
        t2i = results.get('t2i')
        
        if i2t is None or t2i is None:
            row = [model_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
        else:
            row = [
                model_name,
                f"{i2t['R@1']*100:.1f}%",
                f"{i2t['R@5']*100:.1f}%", 
                f"{i2t['R@10']*100:.1f}%",
                f"{t2i['R@1']*100:.1f}%",
                f"{t2i['R@5']*100:.1f}%",
                f"{t2i['R@10']*100:.1f}%"
            ]
        table_data.append(row)
    
    return tabulate(table_data, headers=headers, tablefmt='grid')


def create_retrieval_table(retrieval_results: Dict[str, Dict[str, List]], query: str) -> str:
    """Create a formatted table of text-to-image retrieval results."""
    headers = ['Rank', 'Model', 'Filename', 'Similarity']
    table_data = []
    
    max_results = max(len(results['filenames']) for results in retrieval_results.values())  # type: ignore
    
    for rank in range(max_results):
        for model_name, results in retrieval_results.items():
            if rank < len(results['filenames']):  # type: ignore
                table_data.append([
                    rank + 1,
                    model_name,
                    results['filenames'][rank],  # type: ignore
                    f"{results['similarities'][rank]:.3f}"  # type: ignore
                ])
    
    return tabulate(table_data, headers=headers, tablefmt='grid')


def create_classification_table(classification_results: Dict[str, Dict]) -> str:
    """Create a formatted table of classification results."""
    if not classification_results:
        return "No classification results available."
    
    # Get all unique classes
    all_classes = set()
    for results in classification_results.values():
        all_classes.update(results['classes'])
    all_classes = sorted(list(all_classes))
    
    headers = ['Class'] + list(classification_results.keys())
    table_data = []
    
    for class_name in all_classes:
        row = [class_name]
        for model_name in classification_results.keys():
            results = classification_results[model_name]
            if class_name in results['classes']:
                idx = results['classes'].index(class_name)
                prob = results['probabilities'][idx]
                row.append(f"{prob*100:.1f}%")
            else:
                row.append("0.0%")
        table_data.append(row)
    
    return tabulate(table_data, headers=headers, tablefmt='grid')


def save_retrieval_images(retrieval_results: Dict[str, Dict[str, List]], query: str, 
                         dataset_path: str, output_dir: str = "ablation_results"):
    """Save retrieved images for visualization."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # Create parent directories
    
    # Create query-specific directory
    query_dir = output_path / f"query_{query.replace(' ', '_')}"
    query_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, results in retrieval_results.items():
        model_dir = query_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        for i, filename in enumerate(results['filenames']):  # type: ignore
            # Find source image
            source_path = Path(dataset_path) / "coco2014" / "images" / "val2014" / filename
            if source_path.exists():
                # Copy image with similarity score in filename
                similarity = results['similarities'][i]  # type: ignore
                dest_name = f"rank_{i+1}_sim_{similarity:.3f}_{filename}"
                dest_path = model_dir / dest_name
                
                try:
                    image = Image.open(source_path)
                    image.save(dest_path)
                except Exception as e:
                    print(f"Error copying {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="CLIP Model Ablation Study")
    parser.add_argument("--models", nargs='+', required=True,
                       help="Paths to model checkpoints")
    parser.add_argument("--model-names", nargs='+', 
                       help="Names for models (default: use checkpoint filenames)")
    parser.add_argument("--text-query", type=str, default="sport",
                       help="Text query for retrieval demo")
    parser.add_argument("--test-image", type=str, 
                       help="Path to test image for classification")
    parser.add_argument("--classes", nargs='+', 
                       default=["cat", "dog", "car", "person", "bicycle"],
                       help="Classes for zero-shot classification")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--max-samples", type=int, default=40000,
                       help="Maximum samples for retrieval evaluation")
    parser.add_argument("--save-images", action="store_true",
                       help="Save retrieved images to disk")
    parser.add_argument("--compute-recall", action="store_true",
                       help="Compute Recall@K metrics (slower but comprehensive)")
    parser.add_argument("--output-dir", type=str, default="ablation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    models = {}
    model_names = args.model_names if args.model_names else [Path(p).stem for p in args.models]
    
    if len(model_names) != len(args.models):
        print("Error: Number of model names must match number of model paths")
        return
    
    print(f"Loading {len(args.models)} models...")
    for model_path, model_name in zip(args.models, model_names):
        model = load_model(model_path, device)
        if model is not None:
            models[model_name] = model
            print(f"  ✓ Loaded {model_name}")
        else:
            print(f"  ✗ Failed to load {model_name}")
    
    if not models:
        print("No models loaded successfully!")
        return
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = COCODataset(
        dataset='val',
        transform=get_default_transforms(image_size=(224, 224)),
        load_annotations=True,
        image_size=(224, 224),
        load_captions=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    print(f"Loaded {len(val_dataset)} validation samples")
    
    # Evaluate retrieval performance
    print(f"\n{'='*60}")
    if args.compute_recall:
        print("RETRIEVAL EVALUATION")
    else:
        print("EMBEDDING COMPUTATION")
    print('='*60)
    
    retrieval_results = evaluate_retrieval(models, val_loader, device, compute_recall=args.compute_recall)
    
    # Display results table only if recall was computed
    if args.compute_recall:
        results_table = create_results_table(retrieval_results)
        print("\nRetrieval Results:")
        print(results_table)
    else:
        print("\nEmbeddings computed successfully. Use --compute-recall for detailed metrics.")
    
    # Text-to-image retrieval demo
    print(f"\n{'='*60}")
    print("TEXT-TO-IMAGE RETRIEVAL DEMO")
    print('='*60)
    
    query_results = text_to_image_retrieval(
        args.text_query, models, retrieval_results, device, top_k=5
    )
    
    retrieval_table = create_retrieval_table(query_results, args.text_query)
    print(f"\nTop 5 retrieved images for query '{args.text_query}':")
    print(retrieval_table)
    
    # Save retrieved images if requested
    if args.save_images:
        print(f"\nSaving retrieved images to {args.output_dir}...")
        try:
            # Read dataset path from path_to_archive.txt
            with open("path_to_archive.txt", "r") as f:
                dataset_path = f.read().strip()
            save_retrieval_images(query_results, args.text_query, dataset_path, args.output_dir)
            print("Images saved successfully!")
        except Exception as e:
            print(f"Error saving images: {e}")
    
    # Zero-shot classification demo
    classification_results = None
    classification_table = ""
    
    if args.test_image:
        print(f"\n{'='*60}")
        print("ZERO-SHOT CLASSIFICATION DEMO")
        print('='*60)
        
        # Check if test image exists
        if not Path(args.test_image).exists():
            print(f"Warning: Test image not found at {args.test_image}")
            print("Skipping classification demo.")
        else:
            classification_results = classify_image(
                args.test_image, args.classes, models, device
            )
            
            if classification_results:
                classification_table = create_classification_table(classification_results)
                print(f"\nClassification results for {args.test_image}:")
                print(classification_table)
    
    # Save detailed results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # Create parent directories
    
    # Save summary report
    report_path = output_path / "ablation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CLIP Model Ablation Study Report\n")
        f.write("="*50 + "\n\n")
        
        f.write("Models Evaluated:\n")
        for i, (model_path, model_name) in enumerate(zip(args.models, model_names)):
            f.write(f"{i+1}. {model_name}: {model_path}\n")
        f.write("\n")
        
        if args.compute_recall:
            results_table = create_results_table(retrieval_results)
            f.write("Retrieval Results:\n")
            f.write(results_table)
            f.write("\n\n")
        else:
            f.write("Retrieval Results:\n")
            f.write("Recall metrics not computed (--compute-recall flag not used)\n\n")
        
        f.write(f"Text-to-Image Retrieval for '{args.text_query}':\n")
        f.write(retrieval_table)
        f.write("\n\n")
        
        if args.test_image and classification_results:
            f.write(f"Classification Results for {args.test_image}:\n")
            f.write(classification_table)
            f.write("\n")
    
    print(f"\nDetailed report saved to: {report_path}")
    print("\nAblation study complete!")


if __name__ == "__main__":
    main()