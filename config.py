"""
Configuration file for Face Similarity Search Project
"""
import os

class Config:
    """Base configuration"""
    
    # Project paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, 'embeddings')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    FAISS_INDEX_DIR = os.path.join(PROJECT_ROOT, 'indexing', 'indices')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    
    # Dataset settings
    CELEBA_URL = "https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8"
    LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    TARGET_IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Model settings
    CLIP_MODEL_NAME = "ViT-B/32"  # Options: ViT-B/32, ViT-B/16, ViT-L/14
    DINOV2_MODEL_NAME = "dinov2_vitb14"  # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14
    
    # FAISS settings
    FAISS_INDEX_TYPE = "IVF"  # Options: Flat, IVF, HNSW
    FAISS_NLIST = 100  # Number of clusters for IVF
    FAISS_NPROBE = 10  # Number of clusters to visit during search
    TOP_K = 10  # Number of similar images to retrieve
    
    # Visualization settings
    HEATMAP_ALPHA = 0.4
    HEATMAP_COLORMAP = 'jet'
    
    # Performance settings
    USE_GPU = True
    DEVICE = "cuda" if USE_GPU else "cpu"
    
    # Target metrics
    TARGET_TOP10_ACCURACY = 0.95
    TARGET_QUERY_LATENCY = 2.0  # seconds
    TARGET_EXPLANATION_SCORE = 4.0  # out of 5
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.EMBEDDINGS_DIR,
            cls.MODELS_DIR,
            cls.FAISS_INDEX_DIR,
            cls.RESULTS_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("All directories created successfully!")

if __name__ == "__main__":
    Config.create_directories()
