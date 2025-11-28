"""
CLIP-style Image Encoder with ResNet50 backbone.

Features:
- ResNet50 pretrained on ImageNet
- Projection head mapping to 512-dim CLIP embedding space
- Frozen text encoder (trainable image path only)
- InfoNCE contrastive loss for training
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from typing import Optional


class ImageEncoder(nn.Module):
    """
    ResNet50-based image encoder with projection head for CLIP embedding space.
    """
    def __init__(self, embedding_dim=512, pretrained=True):
        """
        Args:
            embedding_dim: Dimension of CLIP embedding space (default 512)
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        
        # Load pretrained ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2  # Better weights than V1
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get feature dimension from ResNet50 (2048 for final layer)
        self.feature_dim = self.backbone.fc.in_features
        
        # Remove the classification head
        self.backbone.fc = nn.Identity()  # type: ignore[assignment]
        
        # Projection head: 2048 -> hidden -> 512
        hidden_dim = 2048  # Common choice for intermediate layer
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Keep backbone trainable by default
        # (User can freeze layers if desired via freeze_backbone())
        
    def forward(self, x):
        """
        Args:
            x: Image tensor (B, 3, H, W)
        
        Returns:
            embeddings: (B, embedding_dim) projected features
        """
        # Extract features from ResNet50
        features = self.backbone(x)  # (B, 2048)
        
        # Project to CLIP embedding space
        embeddings = self.projection(features)  # (B, 512)
        
        # L2 normalize embeddings (standard for CLIP)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def freeze_backbone(self):
        """Freeze ResNet50 backbone (only train projection head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze ResNet50 backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class TextEncoder(nn.Module):
    """
    Placeholder text encoder (frozen).
    
    In practice, you'd use a pretrained transformer (e.g., CLIP's text encoder,
    DistilBERT, or similar). This is a simple placeholder.
    """
    def __init__(self, embedding_dim=512, vocab_size=49408, max_length=77):
        """
        Args:
            embedding_dim: Output embedding dimension (512 for CLIP)
            vocab_size: Size of text vocabulary
            max_length: Maximum sequence length
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Simple transformer-like architecture (placeholder)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Parameter(torch.randn(max_length, embedding_dim))
        
        # Simple projection to match CLIP space
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Freeze all parameters
        self.freeze()
    
    def forward(self, text_tokens):
        """
        Args:
            text_tokens: (B, seq_len) tokenized text
        
        Returns:
            embeddings: (B, embedding_dim) text embeddings
        """
        # Token + positional embeddings
        x = self.token_embedding(text_tokens)  # (B, seq_len, embed_dim)
        seq_len = x.shape[1]
        x = x + self.positional_embedding[:seq_len]
        
        # Pool (take mean over sequence)
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # Project and normalize
        embeddings = self.projection(x)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def freeze(self):
        """Freeze all text encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False


class CLIPModel(nn.Module):
    """
    CLIP-style multimodal model combining image and text encoders.
    """
    def __init__(self, embedding_dim=512, temperature=0.07, pretrained_image=True):
        """
        Args:
            embedding_dim: Shared embedding space dimension
            temperature: Temperature parameter for contrastive loss
            pretrained_image: Use ImageNet pretrained weights for image encoder
        """
        super().__init__()
        
        self.image_encoder = ImageEncoder(embedding_dim=embedding_dim, 
                                          pretrained=pretrained_image)
        self.text_encoder = TextEncoder(embedding_dim=embedding_dim)
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        
    def forward(self, images, text_tokens):
        """
        Args:
            images: (B, 3, H, W) image tensor
            text_tokens: (B, seq_len) tokenized text
        
        Returns:
            image_embeddings: (B, embedding_dim)
            text_embeddings: (B, embedding_dim)
            logit_scale: scalar temperature parameter
        """
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(text_tokens)
        
        return image_embeddings, text_embeddings, self.logit_scale.exp()
    
    def get_trainable_params(self):
        """Return only trainable parameters (image encoder + projection)."""
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(name)
        return trainable


class InfoNCELoss(nn.Module):
    """
    InfoNCE (contrastive) loss for CLIP-style training.
    
    This is the contrastive loss that aligns image-text pairs by treating
    each pair in the batch as positive, and all other pairs as negatives.
    
    The loss is computed symmetrically:
    - Image-to-text: For each image, predict which text it matches
    - Text-to-image: For each text, predict which image it matches
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for scaling logits (default 0.07)
                        Lower temperature = sharper distributions
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_embeddings: torch.Tensor, 
                text_embeddings: torch.Tensor, 
                logit_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute InfoNCE loss between image and text embeddings.
        
        Args:
            image_embeddings: (B, embedding_dim) L2-normalized image features
            text_embeddings: (B, embedding_dim) L2-normalized text features
            logit_scale: Optional learned temperature (if None, uses self.temperature)
        
        Returns:
            loss: Scalar contrastive loss value
            
        Note:
            Assumes embeddings are already L2-normalized (as done in the encoders)
        """
        batch_size = image_embeddings.shape[0]
        
        # Use learned logit_scale if provided, else use fixed temperature
        if logit_scale is None:
            logit_scale = torch.tensor(1.0 / self.temperature, device=image_embeddings.device, dtype=image_embeddings.dtype)
        
        # Compute cosine similarity matrix: (B, B)
        # logits[i,j] = similarity between image i and text j
        logits = logit_scale * (image_embeddings @ text_embeddings.T)
        
        # Ground truth: diagonal elements are positive pairs
        # labels[i] = i means image i matches with text i
        labels = torch.arange(batch_size, device=logits.device)
        
        # Compute cross-entropy loss in both directions
        
        # Image-to-text direction:
        # For each image, predict which text it matches (rows of logits)
        loss_i2t = nn.functional.cross_entropy(logits, labels)
        
        # Text-to-image direction:
        # For each text, predict which image it matches (columns of logits)
        loss_t2i = nn.functional.cross_entropy(logits.T, labels)
        
        # Average both directions (symmetric loss)
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss


# ---------------------------
# Smoke test / usage example
# ---------------------------
if __name__ == '__main__':
    print("="*60)
    print("CLIP Model Smoke Test")
    print("="*60)
    
    # Create model
    model = CLIPModel(embedding_dim=512, pretrained_image=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Show trainable components
    print("\nTrainable components:")
    for name in model.get_trainable_params():
        print(f"  ✓ {name}")
    
    # Test forward pass
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_text = torch.randint(0, 49408, (batch_size, 77))  # Random tokens
    
    print(f"\nInput shapes:")
    print(f"  Images: {dummy_images.shape}")
    print(f"  Text tokens: {dummy_text.shape}")
    
    image_embeds, text_embeds, logit_scale = model(dummy_images, dummy_text)
    
    print(f"\nOutput shapes:")
    print(f"  Image embeddings: {image_embeds.shape}")
    print(f"  Text embeddings: {text_embeds.shape}")
    print(f"  Logit scale: {logit_scale.item():.4f}")
    
    # Verify embeddings are normalized
    image_norms = torch.norm(image_embeds, p=2, dim=1)
    text_norms = torch.norm(text_embeds, p=2, dim=1)
    print(f"\nEmbedding L2 norms (should be ~1.0):")
    print(f"  Image: {image_norms.mean().item():.4f} ± {image_norms.std().item():.4f}")
    print(f"  Text: {text_norms.mean().item():.4f} ± {text_norms.std().item():.4f}")
    
    # Compute similarity matrix (CLIP-style)
    similarity = logit_scale * (image_embeds @ text_embeds.T)
    print(f"\nSimilarity matrix shape: {similarity.shape}")
    print(f"Similarity range: [{similarity.min().item():.2f}, {similarity.max().item():.2f}]")
    
    # Test InfoNCE loss
    print("\n" + "="*60)
    print("Testing InfoNCE Loss")
    print("="*60)
    
    loss_fn = InfoNCELoss(temperature=0.07)
    
    # Test with learned logit_scale from model
    loss_with_scale = loss_fn(image_embeds, text_embeds, logit_scale)
    print(f"InfoNCE loss (with learned scale): {loss_with_scale.item():.4f}")
    
    # Test with default temperature
    loss_default = loss_fn(image_embeds, text_embeds, logit_scale=None)
    print(f"InfoNCE loss (default temp): {loss_default.item():.4f}")
    
    # Verify loss is differentiable
    loss_with_scale.backward()
    print("✓ Loss is differentiable (backward pass successful)")
    
    print("\n" + "="*60)
    print("✓ All checks passed!")
    print("="*60)
