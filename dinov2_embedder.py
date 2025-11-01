"""
DINOv2 Embeddings Module
Generates face embeddings using DINOv2 model with attention visualization
Team Member: Hao-Cheng Chang
"""
import os
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import h5py
from torchvision import transforms

from config import Config


class DINOv2Embedder:
    """Generate embeddings using DINOv2 model with attention extraction"""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize DINOv2 model
        
        Args:
            model_name: DINOv2 model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14)
            device: Device to run model on (cuda/cpu)
        """
        self.config = Config()
        self.model_name = model_name or self.config.DINOV2_MODEL_NAME
        self.device = device or self.config.DEVICE
        
        print(f"Loading DINOv2 model: {self.model_name}")
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.embed_dim
        print(f"DINOv2 model loaded. Embedding dimension: {self.embedding_dim}")
        
        # Setup preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def embed_image(self, image_path: str, return_cls_only: bool = True) -> np.ndarray:
        """
        Generate embedding for a single image
        
        Args:
            image_path: Path to image file
            return_cls_only: If True, return only CLS token embedding
            
        Returns:
            Embedding vector (normalized)
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if return_cls_only:
                    # Get CLS token embedding
                    embedding = self.model(image_input)
                else:
                    # Get all patch embeddings
                    embedding = self.model.forward_features(image_input)['x_norm_patchtokens']
                
                # Normalize
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            return embedding.cpu().numpy().flatten() if return_cls_only else embedding.cpu().numpy()
        except Exception as e:
            print(f"Error embedding image {image_path}: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_batch(self, images: torch.Tensor, return_cls_only: bool = True) -> np.ndarray:
        """
        Generate embeddings for a batch of images
        
        Args:
            images: Batch of preprocessed images [B, C, H, W]
            return_cls_only: If True, return only CLS token embeddings
            
        Returns:
            Embeddings [B, D] or [B, N, D] where N is number of patches
        """
        with torch.no_grad():
            images = images.to(self.device)
            
            if return_cls_only:
                embeddings = self.model(images)
            else:
                embeddings = self.model.forward_features(images)['x_norm_patchtokens']
            
            # Normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()
    
    def embed_dataset(self, image_paths: List[str], 
                     batch_size: int = None,
                     save_path: str = None) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for entire dataset
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            save_path: Path to save embeddings (optional)
            
        Returns:
            embeddings: Array of embeddings [N, D]
            valid_paths: List of valid image paths
        """
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        embeddings_list = []
        valid_paths = []
        
        print(f"Generating DINOv2 embeddings for {len(image_paths)} images...")
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="DINOv2 embedding"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_valid_paths = []
            
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_input = self.preprocess(image)
                    batch_images.append(image_input)
                    batch_valid_paths.append(img_path)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            if len(batch_images) > 0:
                batch_tensor = torch.stack(batch_images).to(self.device)
                batch_embeddings = self.embed_batch(batch_tensor)
                embeddings_list.append(batch_embeddings)
                valid_paths.extend(batch_valid_paths)
        
        embeddings = np.vstack(embeddings_list)
        
        print(f"Generated {len(embeddings)} embeddings with dimension {self.embedding_dim}")
        
        # Save embeddings
        if save_path:
            self.save_embeddings(embeddings, valid_paths, save_path)
        
        return embeddings, valid_paths
    
    def extract_attention_maps(self, image_path: str, 
                              layer_idx: int = -1) -> Dict[str, np.ndarray]:
        """
        Extract attention maps from specified layer
        
        Args:
            image_path: Path to image
            layer_idx: Layer index to extract attention from (-1 for last layer)
            
        Returns:
            Dictionary with attention maps and patch embeddings
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        attention_maps = {}
        
        def hook_fn_forward_attn(module, input, output):
            """Hook to capture attention weights"""
            attention_maps['attn'] = output
        
        # Register hook on attention layer
        if layer_idx == -1:
            layer_idx = len(self.model.blocks) - 1
        
        handle = self.model.blocks[layer_idx].attn.attn_drop.register_forward_hook(
            hook_fn_forward_attn
        )
        
        # Forward pass
        with torch.no_grad():
            features = self.model.forward_features(image_input)
        
        handle.remove()
        
        # Process attention maps
        attn = attention_maps['attn'].cpu().numpy()  # [B, num_heads, N, N]
        
        # Average over heads
        attn_avg = attn.mean(axis=1)[0]  # [N, N]
        
        # Get CLS token attention to all patches
        cls_attn = attn_avg[0, 1:]  # [num_patches]
        
        # Reshape to spatial grid
        num_patches = int(np.sqrt(len(cls_attn)))
        cls_attn_map = cls_attn.reshape(num_patches, num_patches)
        
        return {
            'attention_map': cls_attn_map,
            'full_attention': attn_avg,
            'patch_embeddings': features['x_norm_patchtokens'].cpu().numpy(),
            'cls_embedding': features['x_norm_clstoken'].cpu().numpy()
        }
    
    def get_layer_embeddings(self, image_path: str, 
                            layer_indices: List[int] = None) -> Dict[int, np.ndarray]:
        """
        Extract embeddings from intermediate layers
        
        Args:
            image_path: Path to image
            layer_indices: List of layer indices to extract
            
        Returns:
            Dictionary mapping layer index to embedding
        """
        if layer_indices is None:
            # Default: extract from multiple layers
            num_layers = len(self.model.blocks)
            layer_indices = [num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
        
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        layer_embeddings = {}
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                # Extract CLS token from output
                cls_token = output[:, 0, :]
                layer_embeddings[layer_idx] = cls_token.detach()
            return hook
        
        # Register hooks
        handles = []
        for idx in layer_indices:
            if idx < len(self.model.blocks):
                handle = self.model.blocks[idx].register_forward_hook(hook_fn(idx))
                handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(image_input)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Normalize and convert to numpy
        result = {}
        for idx, embedding in layer_embeddings.items():
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            result[idx] = embedding.cpu().numpy().flatten()
        
        return result
    
    def save_embeddings(self, embeddings: np.ndarray, 
                       image_paths: List[str], 
                       save_path: str):
        """Save embeddings to HDF5 file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"Saving embeddings to {save_path}...")
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('embeddings', data=embeddings)
            f.create_dataset('image_paths', 
                           data=np.array(image_paths, dtype=h5py.string_dtype()))
            f.attrs['model_name'] = self.model_name
            f.attrs['embedding_dim'] = self.embedding_dim
            f.attrs['num_images'] = len(image_paths)
        
        print(f"✅ Embeddings saved successfully!")
    
    @staticmethod
    def load_embeddings(load_path: str) -> Tuple[np.ndarray, List[str], Dict]:
        """Load embeddings from HDF5 file"""
        print(f"Loading embeddings from {load_path}...")
        with h5py.File(load_path, 'r') as f:
            embeddings = f['embeddings'][:]
            image_paths = [path.decode('utf-8') for path in f['image_paths'][:]]
            metadata = dict(f.attrs)
        
        print(f"Loaded {len(embeddings)} embeddings")
        return embeddings, image_paths, metadata


def main():
    """Demo of DINOv2 embedding generation"""
    print("=" * 50)
    print("DINOv2 Embedding Generation")
    print("=" * 50)
    
    config = Config()
    
    # Check if processed images exist
    from data.data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    
    processed_dir = config.PROCESSED_DATA_DIR
    if not os.path.exists(processed_dir):
        print(f"Processed data directory not found: {processed_dir}")
        print("Please run data_preprocessing.py first.")
        return
    
    image_paths = preprocessor.collect_image_paths(processed_dir, max_images=100)
    
    if len(image_paths) == 0:
        print("No images found in processed directory.")
        return
    
    # Initialize DINOv2 embedder
    embedder = DINOv2Embedder()
    
    # Generate embeddings
    save_path = os.path.join(config.EMBEDDINGS_DIR, 'dinov2_embeddings.h5')
    embeddings, valid_paths = embedder.embed_dataset(
        image_paths,
        batch_size=32,
        save_path=save_path
    )
    
    # Test attention extraction
    if len(valid_paths) > 0:
        print("\nExtracting attention maps for sample image...")
        attention_data = embedder.extract_attention_maps(valid_paths[0])
        print(f"Attention map shape: {attention_data['attention_map'].shape}")
        
        # Test layer embeddings
        layer_embs = embedder.get_layer_embeddings(valid_paths[0])
        print(f"Layer embeddings: {list(layer_embs.keys())}")
    
    print("\n✅ DINOv2 embedding generation complete!")


if __name__ == "__main__":
    main()
