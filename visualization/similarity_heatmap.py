"""
Similarity Heatmap Visualization
Shows which regions of query and result images are similar
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os


class SimilarityHeatmapGenerator:
    """Generate heatmaps showing similar regions between images"""
    
    def __init__(self, model_name='dinov2_vitb14', device='mps', model=None):
        """
        Initialize with DINOv2 for patch-level features
        
        Args:
            model_name: DINOv2 model variant
            device: Device to use
            model: Optional pre-loaded model to reuse
        """
        self.device = device
        self.model_name = model_name
        self.model = model
        self.preprocess = None
        
        # If model is provided, ensure we initialize preprocess
        if self.model is not None:
            self._init_preprocess()
            
    def _init_preprocess(self):
        """Initialize preprocessing transforms"""
        from torchvision import transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self):
        """Load DINOv2 model for feature extraction"""
        if self.model is not None:
            # Ensure preprocess is initialized even if model was passed
            if self.preprocess is None:
                self._init_preprocess()
            return
        
        import torch
        
        # Load DINOv2
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self._init_preprocess()
    
    def extract_patch_features(self, image_path):
        """
        Extract patch-level features from image
        
        Args:
            image_path: Path to image
            
        Returns:
            patch_features: [num_patches, feature_dim] features
            original_image: PIL Image
        """
        self.load_model()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        if self.device == "mps":
            image_input = image_input.float()
        
        # Extract features
        with torch.no_grad():
            # Get patch tokens (exclude CLS token)
            features = self.model.forward_features(image_input)
            
            if isinstance(features, dict):
                patch_features = features.get('x_norm_patchtokens', features.get('x', None))
                if patch_features is None:
                    # Fallback: use full output and remove CLS
                    output = self.model(image_input)
                    patch_features = output[:, 1:, :]  # Remove CLS token
            else:
                # Assume format [B, N, D] where N includes CLS
                patch_features = features[:, 1:, :]  # Remove CLS token
        
        # Move to CPU
        patch_features = patch_features.cpu().numpy()[0]  # [num_patches, dim]
        
        return patch_features, image
    
    def compute_similarity_map(self, query_features, result_features):
        """
        Compute spatial similarity map between query and result
        
        Args:
            query_features: [num_patches, dim] query patch features
            result_features: [num_patches, dim] result patch features
            
        Returns:
            similarity_map: [H, W] spatial similarity map
        """
        # Normalize features
        query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
        result_norm = result_features / (np.linalg.norm(result_features, axis=1, keepdims=True) + 1e-8)
        
        # Compute patch-to-patch similarities
        # For each query patch, find max similarity with any result patch
        similarities = np.dot(query_norm, result_norm.T)  # [num_patches_q, num_patches_r]
        
        # Take max similarity for each query patch
        max_similarities = similarities.max(axis=1)  # [num_patches_q]
        # max_similarities = similarities.mean(axis=1)  # [num_patches_q]
        
        # Reshape to spatial grid (14x14 for ViT)
        num_patches = len(max_similarities)
        grid_size = int(np.sqrt(num_patches))
        similarity_map = max_similarities.reshape(grid_size, grid_size)
        
        return similarity_map
    
    def create_heatmap_overlay(self, image, similarity_map, alpha=0.5):
        """
        Create heatmap overlay on image
        
        Args:
            image: PIL Image
            similarity_map: [H, W] similarity scores
            alpha: Overlay transparency
            
        Returns:
            overlaid_image: PIL Image with heatmap overlay
        """
        # Resize image to match heatmap
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        
        # Normalize similarity map to 0-1
        sim_min, sim_max = similarity_map.min(), similarity_map.max()
        if sim_max > sim_min:
            similarity_map = (similarity_map - sim_min) / (sim_max - sim_min)
        
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(similarity_map, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Create colormap (red = high similarity, blue = low similarity)
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlaid = cv2.addWeighted(image_np, 1 - alpha, heatmap_colored, alpha, 0)
        
        return Image.fromarray(overlaid)
    
    def create_enhanced_visualization(self, query_path, result_paths, similarities,
                                     save_path, k=10):
        """
        Create visualization with original images + heatmap overlays
        
        Args:
            query_path: Path to query image
            result_paths: List of result image paths
            similarities: Similarity scores
            save_path: Where to save visualization
            k: Number of results to show
        """
        print(f"    Generating similarity heatmaps...")
        
        # Extract query features once
        query_features, query_image = self.extract_patch_features(query_path)
        
        # Create figure with proper aspect ratio
        # Width: accommodate k results + query horizontally
        # Height: 2 rows (original + heatmap)
        fig_width = (k + 1) * 2.5  # 2.5 inches per result column
        fig_height = 6  # 2 rows of images
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
        
        from matplotlib.gridspec import GridSpec
        
        # Grid: 3 rows × (k+1) columns
        # Row 0: Headers
        # Row 1: Original images
        # Row 2: Heatmap overlays
        gs = GridSpec(3, k + 1, 
                     figure=fig,
                     left=0.05, right=0.92,  # Reserve right 8% for colorbar
                     top=0.92, bottom=0.08,
                     hspace=0.3, wspace=0.15)
        
        # Query column (column 0)
        # Header
        ax_query_header = fig.add_subplot(gs[0, 0])
        ax_query_header.text(0.5, 0.5, 'Query', 
                           ha='center', va='center', fontsize=12, fontweight='bold')
        ax_query_header.axis('off')
        
        # Query image (row 1)
        ax_query = fig.add_subplot(gs[1, 0])
        ax_query.imshow(query_image)
        ax_query.set_title('Reference', fontsize=9)
        ax_query.axis('off')
        
        # Empty space in query column row 2
        ax_query_empty = fig.add_subplot(gs[2, 0])
        ax_query_empty.axis('off')
        
        # Normalize query path for comparison
        query_path_norm = os.path.normpath(os.path.abspath(query_path))
        
        # Process each result (in columns)
        for idx in range(min(k, len(result_paths))):
            result_path = result_paths[idx]
            result_path_norm = os.path.normpath(os.path.abspath(result_path))
            
            col = idx + 1  # Column index (skip query column 0)
            
            # Header for this result
            ax_header = fig.add_subplot(gs[0, col])
            ax_header.text(0.5, 0.5, f'Result #{idx+1}\nSim: {similarities[idx]:.3f}',
                          ha='center', va='center', fontsize=10, fontweight='bold')
            ax_header.axis('off')
            
            # Check if this is the query image itself
            if query_path_norm == result_path_norm:
                # Same image - use identity map (all patches match perfectly)
                result_image = query_image
                num_patches = len(query_features)
                grid_size = int(np.sqrt(num_patches))
                similarity_map = np.ones((grid_size, grid_size))  # All 1.0 = all red
            else:
                # Different image - compute actual similarities
                result_features, result_image = self.extract_patch_features(result_path)
                similarity_map = self.compute_similarity_map(query_features, result_features)
            
            # Create heatmap overlay
            result_heatmap = self.create_heatmap_overlay(result_image, similarity_map, alpha=0.6)
            
            # Row 1: Original result image
            ax_orig = fig.add_subplot(gs[1, col])
            ax_orig.imshow(result_image)
            ax_orig.set_title('Original', fontsize=9)
            ax_orig.axis('off')
            
            # Row 2: Heatmap overlay
            ax_heat = fig.add_subplot(gs[2, col])
            ax_heat.imshow(result_heatmap)
            ax_heat.set_title('Similar Regions', fontsize=9)
            ax_heat.axis('off')
        
        # Add colorbar aligned with the 2 image rows (rows 1-2)
        # Position: [left, bottom, width, height]
        # Should span from bottom of row 1 to bottom of row 2
        cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.84])
        
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=0, vmax=1)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
            cax=cbar_ax,
            label='Similarity'
        )
        cb.ax.tick_params(labelsize=9)
        
        plt.suptitle('Face Similarity Search with Region Highlighting', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Save without bbox_inches='tight' to preserve our manual layout
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"    ✅ Saved: {os.path.basename(save_path)}")


def visualize_top_k_results_with_heatmaps(query_path, similar_paths, similarities, 
                                          save_path, k=10, device='mps'):
    """
    Create enhanced visualization with similarity heatmaps
    
    Args:
        query_path: Path to query image
        similar_paths: List of similar image paths
        similarities: Similarity scores
        save_path: Where to save
        k: Number of results
        device: Device to use
    """
    generator = SimilarityHeatmapGenerator(device=device)
    generator.create_enhanced_visualization(
        query_path, similar_paths, similarities, save_path, k=k
    )


if __name__ == "__main__":
    # Test
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python similarity_heatmap.py <query_image> <result_image>")
        sys.exit(1)
    
    query_path = sys.argv[1]
    result_path = sys.argv[2]
    
    generator = SimilarityHeatmapGenerator()
    query_feat, query_img = generator.extract_patch_features(query_path)
    result_feat, result_img = generator.extract_patch_features(result_path)
    
    sim_map = generator.compute_similarity_map(query_feat, result_feat)
    heatmap = generator.create_heatmap_overlay(result_img, sim_map)
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(query_img)
    axes[0].set_title('Query')
    axes[0].axis('off')
    
    axes[1].imshow(result_img)
    axes[1].set_title('Result')
    axes[1].axis('off')
    
    axes[2].imshow(heatmap)
    axes[2].set_title('Similar Regions')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()