"""
Attention Visualization Module
Visualizes self-attention patterns from vision transformers (DINOv2)
Team Member: Hao-Cheng Chang
"""

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Optional, Dict
import cv2

from config import Config


class AttentionVisualizer:
    """
    Visualize attention patterns from vision transformers
    """

    def __init__(self, config: Config = None):
        """Initialize attention visualizer"""
        self.config = config or Config()

    def visualize_attention_map(
        self,
        attention_map: np.ndarray,
        image_path: str,
        save_path: Optional[str] = None,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Overlay attention map on image

        Args:
            attention_map: Attention weights [H, W]
            image_path: Path to original image
            save_path: Path to save visualization
            alpha: Transparency of overlay
            colormap: Matplotlib colormap (default: jet)

        Returns:
            Overlayed image
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        # Resize attention map to match image
        attn_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)

        # Normalize
        attn_min, attn_max = attn_resized.min(), attn_resized.max()
        print(f"Visualizing attention: min={attn_min:.4f}, max={attn_max:.4f}")

        attn_resized = (attn_resized - attn_min) / (attn_max - attn_min + 1e-8)

        # Apply colormap
        cmap = cm.get_cmap(colormap)
        attn_colored = cmap(attn_resized)[:, :, :3]
        attn_colored = (attn_colored * 255).astype(np.uint8)

        # Overlay
        overlayed = cv2.addWeighted(image_np, 1 - alpha, attn_colored, alpha, 0)

        # Save
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray(overlayed).save(save_path)

        return overlayed

    def visualize_multihead_attention(
        self,
        attention_weights: np.ndarray,
        image_path: str,
        save_dir: str,
        num_heads: int = 8,
    ):
        """
        Visualize attention from multiple heads

        Args:
            attention_weights: Attention weights [num_heads, N, N]
            image_path: Path to original image
            save_dir: Directory to save visualizations
            num_heads: Number of attention heads to visualize
        """
        os.makedirs(save_dir, exist_ok=True)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Get CLS token attention for each head
        num_heads_total = attention_weights.shape[0]
        heads_to_vis = min(num_heads, num_heads_total)

        fig, axes = plt.subplots(2, heads_to_vis // 2, figsize=(15, 6))
        axes = axes.flatten() if heads_to_vis > 1 else [axes]

        for i in range(heads_to_vis):
            # Get attention from CLS token to patches
            cls_attn = attention_weights[i, 0, 1:]  # Skip CLS token

            # Reshape to spatial
            num_patches = int(np.sqrt(len(cls_attn)))
            attn_map = cls_attn.reshape(num_patches, num_patches)

            # Visualize
            attn_vis = self.visualize_attention_map(
                attn_map, image_path, alpha=0.5, colormap="hot"
            )

            axes[i].imshow(attn_vis)
            axes[i].set_title(f"Head {i + 1}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "multihead_attention.jpg"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

    def visualize_attention_flow(
        self,
        attention_map: np.ndarray,
        query_image_path: str,
        reference_image_path: str,
        save_path: str,
    ):
        """
        Visualize attention flow between query and reference

        Args:
            attention_map: Attention weights from query to reference
            query_image_path: Path to query image
            reference_image_path: Path to reference image
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Query image
        query_img = Image.open(query_image_path).convert("RGB")
        axes[0].imshow(query_img)
        axes[0].set_title("Query Image")
        axes[0].axis("off")

        # Attention map
        axes[1].imshow(attention_map, cmap="hot", interpolation="nearest")
        axes[1].set_title("Attention Map")
        axes[1].axis("off")

        # Reference image with attention overlay
        ref_vis = self.visualize_attention_map(
            attention_map, reference_image_path, alpha=0.5
        )
        axes[2].imshow(ref_vis)
        axes[2].set_title("Reference with Attention")
        axes[2].axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

    def compare_layer_attention(
        self,
        layer_attention_maps: Dict[int, np.ndarray],
        image_path: str,
        save_path: str,
    ):
        """
        Compare attention patterns across layers

        Args:
            layer_attention_maps: Dictionary mapping layer index to attention map
            image_path: Path to image
            save_path: Path to save comparison
        """
        num_layers = len(layer_attention_maps)
        fig, axes = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))

        # Original image
        image = Image.open(image_path).convert("RGB")
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Attention from each layer
        for idx, (layer_idx, attn_map) in enumerate(
            sorted(layer_attention_maps.items()), 1
        ):
            attn_vis = self.visualize_attention_map(
                attn_map, image_path, alpha=0.6, colormap="jet"
            )
            axes[idx].imshow(attn_vis)
            axes[idx].set_title(f"Layer {layer_idx}")
            axes[idx].axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

    def generate_attention_rollout(
        self,
        attention_weights_list: List[np.ndarray],
        image_path: str,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate attention rollout across layers

        Args:
            attention_weights_list: List of attention weights from each layer
            image_path: Path to image
            save_path: Path to save visualization

        Returns:
            Rolled out attention map
        """
        # Start with identity
        rollout = np.eye(attention_weights_list[0].shape[-1])

        # Multiply attention matrices
        for attn in attention_weights_list:
            # Average over heads
            attn_avg = attn.mean(axis=0)

            # Add residual connection
            attn_avg = attn_avg + np.eye(attn_avg.shape[0])

            # Normalize
            attn_avg = attn_avg / attn_avg.sum(axis=-1, keepdims=True)

            # Multiply
            rollout = np.matmul(attn_avg, rollout)

        # Get CLS token attention to patches
        cls_attn = rollout[0, 1:]

        # Reshape to spatial
        num_patches = int(np.sqrt(len(cls_attn)))
        attn_map = cls_attn.reshape(num_patches, num_patches)

        # Visualize
        if save_path:
            self.visualize_attention_map(attn_map, image_path, save_path=save_path)

        return attn_map

    def visualize_pairwise_attention(
        self,
        query_path: str,
        reference_paths: List[str],
        embedder,
        save_dir: str,
        top_k: int = 5,
    ):
        """
        Visualize attention for top-k similar images

        Args:
            query_path: Path to query image
            reference_paths: List of reference image paths
            embedder: DINOv2 embedder instance
            save_dir: Directory to save visualizations
            top_k: Number of top results to visualize
        """
        os.makedirs(save_dir, exist_ok=True)

        # Extract attention for query
        query_attn_data = embedder.extract_attention_maps(query_path)
        query_attn = query_attn_data["attention_map"]

        # Create visualization grid
        num_refs = min(top_k, len(reference_paths))
        fig, axes = plt.subplots(2, num_refs + 1, figsize=(4 * (num_refs + 1), 8))

        # Query images
        query_img = Image.open(query_path).convert("RGB")
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title("Query\nOriginal")
        axes[0, 0].axis("off")

        query_attn_vis = self.visualize_attention_map(query_attn, query_path)
        axes[1, 0].imshow(query_attn_vis)
        axes[1, 0].set_title("Query\nAttention")
        axes[1, 0].axis("off")

        # Reference images
        for i, ref_path in enumerate(reference_paths[:num_refs], 1):
            # Original
            ref_img = Image.open(ref_path).convert("RGB")
            axes[0, i].imshow(ref_img)
            axes[0, i].set_title(f"Match {i}\nOriginal")
            axes[0, i].axis("off")

            # Attention
            ref_attn_data = embedder.extract_attention_maps(ref_path)
            ref_attn = ref_attn_data["attention_map"]
            ref_attn_vis = self.visualize_attention_map(ref_attn, ref_path)
            axes[1, i].imshow(ref_attn_vis)
            axes[1, i].set_title(f"Match {i}\nAttention")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "pairwise_attention.jpg"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

        print(f"Pairwise attention visualization saved to {save_dir}")


def main():
    """Demo of attention visualization"""
    print("=" * 50)
    print("Attention Visualization")
    print("=" * 50)

    config = Config()

    # Check for processed images
    from data.data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()

    processed_dir = config.PROCESSED_DATA_DIR
    image_paths = preprocessor.collect_image_paths(processed_dir, max_images=5)

    if len(image_paths) < 2:
        print("Need at least 2 images for visualization.")
        return

    print(f"\nFound {len(image_paths)} images")

    try:
        # Load DINOv2 embedder
        from embeddings.dinov2_embedder import DINOv2Embedder

        print("\nLoading DINOv2 model...")
        embedder = DINOv2Embedder()

        # Initialize visualizer
        visualizer = AttentionVisualizer()

        # Extract and visualize attention for first image
        print("\nExtracting attention maps...")
        query_path = image_paths[0]
        attn_data = embedder.extract_attention_maps(query_path)

        save_dir = os.path.join(config.RESULTS_DIR, "attention_demo")
        os.makedirs(save_dir, exist_ok=True)

        # Visualize single image attention
        print("Visualizing attention map...")
        attn_map = attn_data["attention_map"]
        visualizer.visualize_attention_map(
            attn_map,
            query_path,
            save_path=os.path.join(save_dir, "single_attention.jpg"),
        )

        # Visualize attention across layers
        if len(image_paths) >= 1:
            print("Extracting multi-layer attention...")
            layer_indices = [2, 5, 8, 11]  # Sample layers
            layer_attns = {}

            for layer_idx in layer_indices:
                attn_data_layer = embedder.extract_attention_maps(
                    query_path, layer_idx=layer_idx
                )
                layer_attns[layer_idx] = attn_data_layer["attention_map"]

            visualizer.compare_layer_attention(
                layer_attns,
                query_path,
                save_path=os.path.join(save_dir, "layer_comparison.jpg"),
            )

        # Visualize pairwise attention
        if len(image_paths) >= 5:
            print("Generating pairwise attention visualization...")
            visualizer.visualize_pairwise_attention(
                query_path, image_paths[1:5], embedder, save_dir, top_k=4
            )

        print(f"\n✅ Attention visualization complete!")
        print(f"Results saved to: {save_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
