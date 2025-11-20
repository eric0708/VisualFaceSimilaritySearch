"""
DINOv2 Embeddings Module
Generates face embeddings using DINOv2 model with attention visualization
Team Member: Hao-Cheng Chang
"""
import os
# Enable MPS fallback for unsupported operations (needed for DINOv2 on Apple Silicon)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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
            device: Device to run model on (mps/cuda/cpu, auto-detected if None)
        """
        self.config = Config()
        self.model_name = model_name or self.config.DINOV2_MODEL_NAME
        self.device = device or self.config.DEVICE
        
        print(f"Loading DINOv2 model: {self.model_name}")
        print(f"Device: {self.device}")
        
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Handle MPS-specific optimizations
        if self.device == "mps":
            self.model = self.model.float()
        
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
            if self.device == "mps":
                images = images.float()
            
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
    
    def extract_attention_maps(self, image_path: str, layer_idx: int = -1) -> Dict[str, np.ndarray]:
        """
        Extract attention maps from specified layer of DINOv2.

        Returns:
            {
                'attention_map': [H, W] CLS→patch attention (normalized),
                'full_attention': [N, N] averaged full attention matrix,
                'patch_embeddings': [1, num_patches, D],
                'cls_embedding': [1, D],
            }
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        if self.device == "mps":
            image_input = image_input.float()

        attention_maps: Dict[str, torch.Tensor] = {}

        # --- Hook that inspects inputs/outputs for a 4D attention tensor ---
        def hook_fn_forward_attn(module, inputs, output):
            # Search inputs and output for a tensor that looks like attention weights: [B, heads, N, N]
            candidates = []

            def collect_tensors(obj):
                if isinstance(obj, torch.Tensor):
                    candidates.append(obj.detach().cpu())
                elif isinstance(obj, (list, tuple)):
                    for o in obj:
                        collect_tensors(o)
                elif isinstance(obj, dict):
                    for o in obj.values():
                        collect_tensors(o)

            collect_tensors(inputs)
            collect_tensors(output)

            # pick the first 4D tensor with square last two dims
            for t in candidates:
                if t.ndim == 4:
                    b, h, n1, n2 = t.shape
                    if n1 == n2:
                        attention_maps['attn'] = t
                        return
            
            # If not found, try to compute it manually if we have the input and the module has qkv
            if 'attn' not in attention_maps and hasattr(module, 'qkv') and hasattr(module, 'num_heads') and hasattr(module, 'scale'):
                try:
                    x = inputs[0]
                    B, N, C = x.shape
                    qkv = module.qkv(x)
                    # Reshape: [B, N, 3, num_heads, head_dim]
                    qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
                    # Permute: [3, B, num_heads, N, head_dim]
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    
                    attn = (q @ k.transpose(-2, -1)) * module.scale
                    attn = attn.softmax(dim=-1)
                    attention_maps['attn'] = attn.detach().cpu()
                    return
                except Exception:
                    pass

        # choose layer
        if layer_idx == -1:
            layer_idx = len(self.model.blocks) - 1

        blk = self.model.blocks[layer_idx]

        # Try several plausible targets for registering the hook
        handle = None
        tried_targets = []
        try:
            # Prefer registering on a Dropout inside attention if available (keeps pre-dropout attn)
            if hasattr(blk, 'attn') and hasattr(blk.attn, 'attn_drop') and isinstance(blk.attn.attn_drop, torch.nn.Module):
                tried_targets.append('blk.attn.attn_drop')
                handle = blk.attn.attn_drop.register_forward_hook(hook_fn_forward_attn)
            # fall back to registering on the attention module itself
            elif hasattr(blk, 'attn') and isinstance(blk.attn, torch.nn.Module):
                tried_targets.append('blk.attn')
                handle = blk.attn.register_forward_hook(hook_fn_forward_attn)
            else:
                # try to find a submodule containing 'attn' in its name
                for name, sub in blk.named_modules():
                    if 'attn' in name:
                        tried_targets.append(f'blk.{name}')
                        handle = sub.register_forward_hook(hook_fn_forward_attn)
                        break

            if handle is None:
                raise RuntimeError(f"No attention submodule found in block (tried: {tried_targets})")
        except Exception as e:
            print(f"Warning: Could not register hook on attention module: {e}")
            # fallback synthetic map
            num_patches = 14
            return {
                "attention_map": np.ones((num_patches, num_patches), dtype=np.float32) * 0.5,
                "full_attention": np.eye(num_patches * num_patches + 1, dtype=np.float32),
                "patch_embeddings": np.zeros((1, 196, self.embedding_dim), dtype=np.float32),
                "cls_embedding": np.zeros((1, self.embedding_dim), dtype=np.float32),
            }

        # forward pass
        with torch.no_grad():
            features = self.model.forward_features(image_input)

        handle.remove()  # remove hook

        # --- Process attention ---
        if "attn" not in attention_maps:
            print("Warning: attention hook did not fire; using synthetic map")
            num_patches = 14
            attn_avg_np = np.eye(num_patches * num_patches + 1, dtype=np.float32)
            cls_attn_map = np.ones((num_patches, num_patches), dtype=np.float32) * 0.5
        else:
            attn = attention_maps["attn"]          # [B, heads, N, N]
            if isinstance(attn, torch.Tensor):
                attn = attn.cpu()
            attn_np = attn.numpy()

            # average over heads, take first batch
            attn_avg = attn_np.mean(axis=1)[0]     # [N, N]
            attn_avg_np = attn_avg

            # CLS → patches (skip token 0)
            cls_attn = attn_avg[0, 1:]            # [num_patches]
            num_tokens = cls_attn.shape[0]
            side = int(round(np.sqrt(num_tokens)))

            # robust reshape (in case tokens are not perfect square)
            if side * side != num_tokens:
                padded = np.zeros(side * side, dtype=cls_attn.dtype)
                padded[:min(num_tokens, side * side)] = cls_attn[:side * side]
                cls_attn_map = padded.reshape(side, side)
            else:
                cls_attn_map = cls_attn.reshape(side, side)

            # normalize to [0, 1]
            cls_attn_map = cls_attn_map - cls_attn_map.min()
            denom = cls_attn_map.max() - cls_attn_map.min() + 1e-8
            cls_attn_map = cls_attn_map / denom

        # --- Embeddings from forward_features ---
        if isinstance(features, dict):
            patch_embeddings = features.get("x_norm_patchtokens", features.get("x", None))
            cls_embedding = features.get("x_norm_clstoken", None)

            if isinstance(patch_embeddings, torch.Tensor):
                patch_embeddings = patch_embeddings.cpu().numpy()
            else:
                patch_embeddings = np.zeros((1, 196, self.embedding_dim), dtype=np.float32)

            if isinstance(cls_embedding, torch.Tensor):
                cls_embedding = cls_embedding.cpu().numpy()
            else:
                cls_embedding = np.zeros((1, self.embedding_dim), dtype=np.float32)
        else:
            patch_embeddings = np.zeros((1, 196, self.embedding_dim), dtype=np.float32)
            cls_embedding = np.zeros((1, self.embedding_dim), dtype=np.float32)

        return {
            "attention_map": cls_attn_map,
            "full_attention": attn_avg_np,
            "patch_embeddings": patch_embeddings,
            "cls_embedding": cls_embedding,
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