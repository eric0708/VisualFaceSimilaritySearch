"""
Grad-CAM Visualization Module
Implements Gradient-weighted Class Activation Mapping for explainability
Team Member: Yen-Shuo Su
"""

import os
from typing import Dict, Optional

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import Config


class GradCAM:
    """
    Implements Grad-CAM for visual explanations
    """

    def __init__(
        self, model: torch.nn.Module, target_layer: torch.nn.Module, device: str = None
    ):
        """
        Initialize Grad-CAM

        Args:
            model: Neural network model
            target_layer: Layer to compute gradients for
            device: Device to run on
        """
        self.config = Config()
        self.model = model
        self.target_layer = target_layer
        self.device = device or self.config.DEVICE

        self.gradients = None
        self.activations = None

        # Register hooks
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks"""

        def forward_hook(module, input, output):
            # Save activations and register hook on the output tensor itself
            self.activations = output

            # Register backward hook on the output tensor
            def tensor_backward_hook(grad):
                self.gradients = grad

            if output.requires_grad:
                output.register_hook(tensor_backward_hook)

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))

    def remove_hooks(self):
        """Remove all hooks"""
        for handler in self.handlers:
            handler.remove()

    def generate_cam(
        self, input_image: torch.Tensor, target_embedding: torch.Tensor = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap

        Args:
            input_image: Input image tensor [1, C, H, W]
            target_embedding: Target embedding to compute similarity with (optional)

        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        input_image = input_image.to(self.device)

        # Forward pass
        output = self.model(input_image)

        # If target embedding provided, compute similarity loss
        if target_embedding is not None:
            target_embedding = target_embedding.to(self.device)
            # Cosine similarity
            loss = F.cosine_similarity(output, target_embedding, dim=1).mean()
        else:
            # Use output magnitude
            loss = output.norm()

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Get gradients and activations
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]

        # Weighted combination of activation maps
        # Method 1: Global average pooling on gradients
        # weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        # cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]

        # Method 2: Direct element-wise multiplication (used here)
        cam = (activations * gradients).sum(dim=1, keepdim=True)

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()

        return cam

    def visualize_cam(
        self,
        image_path: str,
        cam: np.ndarray,
        save_path: Optional[str] = None,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Overlay CAM on original image

        Args:
            image_path: Path to original image
            cam: CAM heatmap
            save_path: Path to save visualization (optional)
            alpha: Transparency of overlay
            colormap: Matplotlib colormap name

        Returns:
            Overlayed image as numpy array
        """
        # Load original image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Resize CAM to match image size
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Apply colormap
        cmap = cm.get_cmap(colormap)
        cam_colored = cmap(cam_resized)[:, :, :3]  # RGB only
        cam_colored = (cam_colored * 255).astype(np.uint8)

        # Overlay
        overlayed = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray(overlayed).save(save_path)

        return overlayed

    def generate_pairwise_cam(
        self,
        query_image_path: str,
        reference_image_path: str,
        model_embedder,
        save_dir: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM for pairwise similarity

        Args:
            query_image_path: Path to query image
            reference_image_path: Path to reference image
            model_embedder: Embedder object (CLIP or DINOv2)
            save_dir: Directory to save visualizations

        Returns:
            Dictionary with CAMs and visualizations
        """
        # Get reference embedding
        ref_emb = model_embedder.embed_image(reference_image_path)

        # Prepare query image
        query_img = Image.open(query_image_path).convert("RGB")
        query_tensor = model_embedder.preprocess(query_img).unsqueeze(0)

        # Generate CAM showing what in query is similar to reference
        query_cam = self.generate_cam(
            query_tensor, target_embedding=torch.from_numpy(ref_emb).unsqueeze(0)
        )

        # Visualize
        results = {"cam": query_cam}

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            # Save query CAM
            query_vis_path = os.path.join(save_dir, "query_cam.jpg")
            query_vis = self.visualize_cam(
                query_image_path, query_cam, save_path=query_vis_path
            )
            results["visualization"] = query_vis

            # Save side-by-side comparison
            self._save_comparison(
                query_image_path, reference_image_path, query_cam, save_dir
            )

        return results

    def _save_comparison(
        self, query_path: str, ref_path: str, cam: np.ndarray, save_dir: str
    ):
        """Save side-by-side comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Query image
        query_img = Image.open(query_path)
        axes[0].imshow(query_img)
        axes[0].set_title("Query Image")
        axes[0].axis("off")

        # CAM overlay
        cam_vis = self.visualize_cam(query_path, cam)
        axes[1].imshow(cam_vis)
        axes[1].set_title("Grad-CAM Explanation")
        axes[1].axis("off")

        # Reference image
        ref_img = Image.open(ref_path)
        axes[2].imshow(ref_img)
        axes[2].set_title("Reference Image")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "comparison.jpg"), bbox_inches="tight", dpi=150
        )
        plt.close()


class CLIPGradCAM(GradCAM):
    """Grad-CAM specifically for CLIP model"""

    def __init__(self, clip_model, device: str = None):
        """Initialize with CLIP model"""
        # For CLIP ViT, use the last transformer block
        target_layer = clip_model.visual.transformer.resblocks[-1]
        super().__init__(clip_model.visual, target_layer, device)
        self.full_model = clip_model

    def reshape_transform(self, tensor, height=7, width=7):
        """
        Reshape transformer output to spatial format

        Args:
            tensor: transformer output (can be [seq_len, batch, dim] or [batch, seq_len, dim])
            height: spatial height (default 7 for ViT-B/32 with 224x224 input)
            width: spatial width

        Returns:
            Reshaped tensor [batch, dim, height, width]
        """
        if tensor is None:
            return None

        # Check if tensor is empty
        if tensor.numel() == 0:
            print("Warning: Empty tensor in reshape_transform")
            return None

        # CLIP transformer outputs [seq_len, batch, dim], need to transpose to [batch, seq_len, dim]
        if tensor.dim() == 3:
            # Check if format is [seq_len, batch, dim] (CLIP format)
            if tensor.size(1) == 1 and tensor.size(0) > tensor.size(1):
                # Transpose to [batch, seq_len, dim]
                tensor = tensor.permute(1, 0, 2)

        # Remove CLS token (first token) and reshape
        # tensor shape: [batch, seq_len, dim]
        result = tensor[:, 1:, :]  # Remove CLS token

        # Check if result is empty after removing CLS token
        if result.numel() == 0:
            print(
                f"Warning: Empty tensor after removing CLS token. Original shape: {tensor.shape}"
            )
            return None

        # Get dimensions
        batch_size = result.size(0)
        dim = result.size(2)

        # Reshape to [batch, height, width, dim]
        result = result.reshape(batch_size, height, width, dim)

        # Permute to [batch, dim, height, width]
        result = result.permute(0, 3, 1, 2)

        return result

    def generate_cam(
        self, input_image: torch.Tensor, target_embedding: torch.Tensor = None
    ) -> np.ndarray:
        """
        Generate CAM for CLIP

        Overrides parent method to handle CLIP's architecture
        """
        # Set model to train mode to enable gradients
        self.model.train()

        # Ensure input requires grad
        input_image = input_image.to(self.device)
        if not input_image.requires_grad:
            input_image.requires_grad = True

        # Forward through visual encoder
        output = self.full_model.encode_image(input_image)
        if target_embedding is not None:
            target_embedding = target_embedding.to(self.device)
            loss = F.cosine_similarity(output, target_embedding, dim=1).mean()
        else:
            loss = output.norm()

        # Clear previous gradients
        self.model.zero_grad()
        if input_image.grad is not None:
            input_image.grad.zero_()

        # Backward pass
        loss.backward()

        # Set back to eval mode
        self.model.eval()

        # Get activations and gradients (both are [batch, seq_len, dim])
        gradients = self.gradients
        activations = self.activations

        if gradients is None or activations is None:
            print("Warning: Gradients or activations are None")
            return np.zeros((224, 224), dtype=np.float32)

        # Print debug info
        print(
            f"Debug: gradients shape: {gradients.shape}, activations shape: {activations.shape}"
        )
        print(
            f"Debug: gradients device: {gradients.device}, activations device: {activations.device}"
        )
        print(
            f"Debug: gradients numel: {gradients.numel()}, activations numel: {activations.numel()}"
        )

        # MPS compatibility: move to CPU for processing
        if self.device == "mps":
            gradients = gradients.cpu()
            activations = activations.cpu()

        # Reshape transformer output to spatial format
        # Remove CLS token and reshape to [batch, channels, height, width]
        activations_reshaped = self.reshape_transform(activations)
        gradients_reshaped = self.reshape_transform(gradients)

        if activations_reshaped is None or gradients_reshaped is None:
            print("Warning: Failed to reshape activations or gradients")
            return np.zeros((224, 224), dtype=np.float32)

        # Convert to numpy for processing
        gradients_np = gradients_reshaped[0].cpu().detach().numpy()  # [C, H, W]
        activations_np = activations_reshaped[0].cpu().detach().numpy()  # [C, H, W]

        # Debug: check gradients and activations statistics
        print(
            f"Debug: gradients_np min={gradients_np.min():.6f}, max={gradients_np.max():.6f}, mean={gradients_np.mean():.6f}"
        )
        print(
            f"Debug: activations_np min={activations_np.min():.6f}, max={activations_np.max():.6f}, mean={activations_np.mean():.6f}"
        )

        # Compute weights by averaging gradients over spatial dimensions
        weights = np.mean(gradients_np, axis=(1, 2))  # [C]
        print(
            f"Debug: weights min={weights.min():.6f}, max={weights.max():.6f}, mean={weights.mean():.6f}"
        )

        # Weighted combination of activations
        cam = np.sum(
            weights[:, np.newaxis, np.newaxis] * activations_np, axis=0
        )  # [H, W]

        # Apply ReLU
        cam = np.maximum(cam, 0)
        print(f"Debug: cam before normalize min={cam.min():.6f}, max={cam.max():.6f}")

        # Check if cam is valid
        if cam.size == 0 or not np.isfinite(cam).all():
            print(
                f"Warning: Invalid CAM generated. cam.shape: {cam.shape}, cam.size: {cam.size}"
            )
            return np.zeros((224, 224), dtype=np.float32)

        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            print(f"Warning: CAM has no variance (min={cam_min}, max={cam_max})")
            cam = np.zeros_like(cam)

        # Ensure cam is in the right format for cv2.resize
        cam = np.float32(cam)

        # Check dimensions before resize
        if len(cam.shape) != 2:
            print(f"Warning: CAM has wrong dimensions: {cam.shape}")
            return np.zeros((224, 224), dtype=np.float32)

        # Resize to input image size
        try:
            cam = cv2.resize(cam, (224, 224))
        except Exception as e:
            print(f"Warning: cv2.resize failed: {e}")
            print(f"cam shape: {cam.shape}, dtype: {cam.dtype}")
            return np.zeros((224, 224), dtype=np.float32)

        return cam


def main():
    """Demo of Grad-CAM visualization"""
    print("=" * 50)
    print("Grad-CAM Visualization")
    print("=" * 50)

    config = Config()

    # Check for processed images
    from data.data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()

    processed_dir = config.PROCESSED_DATA_DIR
    image_paths = preprocessor.collect_image_paths(processed_dir, max_images=10)

    if len(image_paths) < 2:
        print("Need at least 2 images for pairwise comparison.")
        return

    print(f"\nFound {len(image_paths)} images")

    # Load CLIP model
    try:
        import clip

        print("\nLoading CLIP model...")
        clip_model, _ = clip.load("ViT-B/32", device=config.DEVICE)

        # Initialize CLIP embedder
        from clip_embedder import CLIPEmbedder

        embedder = CLIPEmbedder()

        # Initialize Grad-CAM
        grad_cam = CLIPGradCAM(clip_model, device=config.DEVICE)

        # Generate pairwise CAM
        print("\nGenerating Grad-CAM for image pair...")
        query_path = image_paths[0]
        ref_path = image_paths[1]

        save_dir = os.path.join(config.RESULTS_DIR, "gradcam_demo")

        grad_cam.generate_pairwise_cam(
            query_path, ref_path, embedder, save_dir=save_dir
        )

        print("\n✅ Grad-CAM visualization complete!")
        print(f"Results saved to: {save_dir}")

        # Clean up
        grad_cam.remove_hooks()

    except Exception as e:
        print(f"Error: {e}")
        print(
            "Make sure CLIP is installed: pip install git+https://github.com/openai/CLIP.git"
        )


if __name__ == "__main__":
    main()
