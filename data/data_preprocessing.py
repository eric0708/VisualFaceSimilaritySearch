"""
Data Collection & Preprocessing Module
Handles downloading, loading, and preprocessing face datasets
"""
import os
import requests
import tarfile
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import Config


class FaceDataset(Dataset):
    """Custom Dataset for face images"""
    
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, 224, 224), img_path


class DataPreprocessor:
    """Handles data collection and preprocessing"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.config.create_directories()
        
        # Optimize number of workers based on CPU count
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # Use min(cpu_count, 8) to avoid overwhelming the system
        self.optimal_workers = min(cpu_count, 8)
        print(f"Using {self.optimal_workers} workers for data loading (CPU count: {cpu_count})")
        
    def download_lfw(self) -> str:
        """Download LFW dataset"""
        print("Downloading LFW dataset...")
        output_path = os.path.join(self.config.RAW_DATA_DIR, 'lfw.tgz')
        
        if os.path.exists(output_path):
            print("LFW dataset already downloaded.")
            return output_path
            
        try:
            response = requests.get(self.config.LFW_URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc='LFW'
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            print("LFW dataset downloaded successfully!")
            return output_path
        except Exception as e:
            print(f"Error downloading LFW: {e}")
            return None
    
    def extract_lfw(self, archive_path: str) -> str:
        """Extract LFW dataset"""
        extract_path = os.path.join(self.config.RAW_DATA_DIR, 'lfw')
        
        if os.path.exists(extract_path):
            print("LFW dataset already extracted.")
            return extract_path
            
        print("Extracting LFW dataset...")
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=self.config.RAW_DATA_DIR)
            print("LFW dataset extracted successfully!")
            return extract_path
        except Exception as e:
            print(f"Error extracting LFW: {e}")
            return None
    
    def collect_image_paths(self, dataset_dir: str, max_images: Optional[int] = None) -> List[str]:
        """Collect all image paths from dataset directory"""
        print(f"Collecting image paths from {dataset_dir}...")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = []
        
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
                    if max_images and len(image_paths) >= max_images:
                        break
            if max_images and len(image_paths) >= max_images:
                break
        
        print(f"Found {len(image_paths)} images.")
        return image_paths
    
    def get_transform(self, for_model: str = 'clip') -> transforms.Compose:
        """Get appropriate image transformations"""
        if for_model == 'clip':
            return transforms.Compose([
                transforms.Resize(self.config.TARGET_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        elif for_model == 'dinov2':
            return transforms.Compose([
                transforms.Resize(self.config.TARGET_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.config.TARGET_IMAGE_SIZE),
                transforms.ToTensor()
            ])
    
    def preprocess_and_save(self, image_paths: List[str], output_dir: str, 
                           target_size: Tuple[int, int] = (224, 224)):
        """Preprocess images and save to output directory"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Preprocessing {len(image_paths)} images...")
        
        for img_path in tqdm(image_paths, desc="Preprocessing"):
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                
                # Resize
                img = img.resize(target_size, Image.LANCZOS)
                
                # Save with same relative structure
                relative_path = os.path.relpath(img_path, self.config.RAW_DATA_DIR)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img.save(output_path)
                
            except Exception as e:
                print(f"Error preprocessing {img_path}: {e}")
        
        print("Preprocessing complete!")
    
    def create_dataloader(self, image_paths: List[str], 
                         transform=None, batch_size: int = None) -> DataLoader:
        """Create DataLoader for batch processing"""
        if transform is None:
            transform = self.get_transform()
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
            
        dataset = FaceDataset(image_paths, transform=transform)
        
        # Pin memory if GPU is available for faster data transfer
        pin_memory = self.config.DEVICE in ['cuda', 'mps']
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.optimal_workers,
            pin_memory=pin_memory,
            persistent_workers=True if self.optimal_workers > 0 else False,
            prefetch_factor=2 if self.optimal_workers > 0 else None
        )
        return dataloader
    
    def detect_and_align_faces(self, image_paths: List[str], output_dir: str):
        """Detect and align faces using MTCNN (optional enhancement)"""
        try:
            from facenet_pytorch import MTCNN
            
            os.makedirs(output_dir, exist_ok=True)
            mtcnn = MTCNN(
                image_size=224, 
                margin=0,
                device=self.config.DEVICE
            )
            
            print("Detecting and aligning faces...")
            for img_path in tqdm(image_paths, desc="Face detection"):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_aligned = mtcnn(img)
                    
                    if img_aligned is not None:
                        relative_path = os.path.relpath(img_path, self.config.RAW_DATA_DIR)
                        output_path = os.path.join(output_dir, relative_path)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # Convert tensor to PIL and save
                        img_pil = transforms.ToPILImage()(img_aligned)
                        img_pil.save(output_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    
            print("Face detection and alignment complete!")
        except ImportError:
            print("facenet-pytorch not installed. Skipping face detection.")


def main():
    """Demo of data preprocessing pipeline"""
    print("=" * 50)
    print("Face Similarity Search - Data Preprocessing")
    print("=" * 50)
    
    preprocessor = DataPreprocessor()
    
    # Download and extract LFW
    # archive_path = preprocessor.download_lfw()
    # if archive_path:
    #     dataset_path = preprocessor.extract_lfw(archive_path)
    
    # For demo, use a sample dataset or your own images
    # Place images in data/raw/sample_faces/
    sample_dir = os.path.join(Config.RAW_DATA_DIR, 'sample_faces')
    
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
        print(f"\nPlease place face images in: {sample_dir}")
        print("Then run this script again.")
        return
    
    # Collect image paths
    image_paths = preprocessor.collect_image_paths(sample_dir, max_images=100)
    
    if len(image_paths) == 0:
        print("No images found. Please add images to the sample_faces directory.")
        return
    
    # Preprocess images
    preprocessor.preprocess_and_save(
        image_paths,
        Config.PROCESSED_DATA_DIR,
        target_size=Config.TARGET_IMAGE_SIZE
    )
    
    # Create dataloader for testing
    processed_paths = preprocessor.collect_image_paths(Config.PROCESSED_DATA_DIR)
    dataloader = preprocessor.create_dataloader(processed_paths, batch_size=8)
    
    print(f"\nDataLoader created with {len(dataloader)} batches")
    print("Sample batch shape:", next(iter(dataloader))[0].shape)
    
    print("\n✅ Data preprocessing complete!")
    print(f"Processed images saved to: {Config.PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()