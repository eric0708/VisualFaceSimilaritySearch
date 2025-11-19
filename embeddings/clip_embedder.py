"""
CLIP Embeddings Module
Generates face embeddings using CLIP model
Team Member: Yen-Shuo Su
"""
import os
import torch
import clip
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
from tqdm import tqdm
import h5py

from config import Config


class CLIPEmbedder:
    """Generate embeddings using CLIP model"""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize CLIP model
        
        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14)
            device: Device to run model on (mps/cuda/cpu, auto-detected if None)
        """
        self.config = Config()
        self.model_name = model_name or self.config.CLIP_MODEL_NAME
        self.device = device or self.config.DEVICE
        
        print(f"Loading CLIP model: {self.model_name}")
        print(f"Device: {self.device}")
        
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()
        
        # Handle MPS-specific optimizations
        if self.device == "mps":
            # MPS works better with float32
            self.model = self.model.float()
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
            if self.device == "mps":
                dummy_input = dummy_input.float()
            dummy_output = self.model.encode_image(dummy_input)
            self.embedding_dim = dummy_output.shape[1]
        
        print(f"CLIP model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized embedding vector
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error embedding image {image_path}: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Generate embeddings for a batch of images
        
        Args:
            images: Batch of preprocessed images [B, C, H, W]
            
        Returns:
            Normalized embeddings [B, D]
        """
        with torch.no_grad():
            images = images.to(self.device)
            if self.device == "mps":
                images = images.float()
            
            embeddings = self.model.encode_image(images)
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
        
        print(f"Generating CLIP embeddings for {len(image_paths)} images...")
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP embedding"):
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
    
    def save_embeddings(self, embeddings: np.ndarray, 
                       image_paths: List[str], 
                       save_path: str):
        """
        Save embeddings to HDF5 file
        
        Args:
            embeddings: Array of embeddings
            image_paths: List of image paths
            save_path: Output file path
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"Saving embeddings to {save_path}...")
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('embeddings', data=embeddings)
            f.create_dataset('image_paths', 
                           data=np.array(image_paths, dtype=h5py.string_dtype()))
            f.attrs['model_name'] = self.model_name
            f.attrs['embedding_dim'] = self.embedding_dim
            f.attrs['num_images'] = len(image_paths)
        
        print("✅ Embeddings saved successfully!")
    
    @staticmethod
    def load_embeddings(load_path: str) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Load embeddings from HDF5 file
        
        Args:
            load_path: Path to HDF5 file
            
        Returns:
            embeddings: Array of embeddings
            image_paths: List of image paths
            metadata: Dictionary of metadata
        """
        print(f"Loading embeddings from {load_path}...")
        with h5py.File(load_path, 'r') as f:
            embeddings = f['embeddings'][:]
            image_paths = [path.decode('utf-8') for path in f['image_paths'][:]]
            metadata = dict(f.attrs)
        
        print(f"Loaded {len(embeddings)} embeddings")
        return embeddings, image_paths, metadata
    
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
            layer_indices = [3, 6, 9, 11]  # Early, mid, late layers
        
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        layer_embeddings = {}
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                layer_embeddings[layer_idx] = output.detach()
            return hook
        
        # Register hooks
        handles = []
        for idx in layer_indices:
            if idx < len(self.model.visual.transformer.resblocks):
                handle = self.model.visual.transformer.resblocks[idx].register_forward_hook(
                    hook_fn(idx)
                )
                handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model.encode_image(image_input)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Normalize and convert to numpy
        result = {}
        for idx, embedding in layer_embeddings.items():
            # Take CLS token embedding
            cls_embedding = embedding[:, 0, :]
            cls_embedding = cls_embedding / cls_embedding.norm(dim=-1, keepdim=True)
            result[idx] = cls_embedding.cpu().numpy().flatten()
        
        return result


def main():
    """Demo of CLIP embedding generation"""
    print("=" * 50)
    print("CLIP Embedding Generation")
    print("=" * 50)
    
    config = Config()
    
    # Check if processed images exist
    from data_preprocessing import DataPreprocessor
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
    
    # Initialize CLIP embedder
    embedder = CLIPEmbedder()
    
    # Generate embeddings
    save_path = os.path.join(config.EMBEDDINGS_DIR, 'clip_embeddings.h5')
    embeddings, valid_paths = embedder.embed_dataset(
        image_paths,
        batch_size=32,
        save_path=save_path
    )
    
    # Test loading
    loaded_embeddings, loaded_paths, metadata = CLIPEmbedder.load_embeddings(save_path)
    print(f"\nMetadata: {metadata}")
    
    # Test single image embedding
    if len(valid_paths) > 0:
        test_embedding = embedder.embed_image(valid_paths[0])
        print(f"\nSingle image embedding shape: {test_embedding.shape}")
        
        # Test layer embeddings
        layer_embs = embedder.get_layer_embeddings(valid_paths[0])
        print(f"Layer embeddings: {list(layer_embs.keys())}")
    
    print("\n✅ CLIP embedding generation complete!")


if __name__ == "__main__":
    main()