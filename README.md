# Visual Face Similarity Search with Explainable Deep Embeddings

**Team #39**: Hao-Cheng Chang, Yu Chu Tsai, Jin-Lian Ho, Yen-Shuo Su  
Georgia Institute of Technology

## 📋 Project Overview

This project implements an interactive face similarity search system with explainability features. Users can upload a face image and retrieve similar faces from a large-scale dataset, with visualizations showing which facial features the AI considers important for similarity.

### Key Features

- **Multi-Model Embeddings**: Support for both CLIP and DINOv2 models
- **Efficient Search**: FAISS-based similarity search with multiple index types
- **Visual Explanations**: Grad-CAM and attention visualization
- **Multi-Scale Exploration**: Layer-wise feature analysis
- **Scalable**: Handles datasets with millions of images

## 🏗️ Project Structure

```
face_similarity_project/
├── config.py                      # Configuration settings
├── main.py                        # Main pipeline execution script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/                          # Data handling
│   ├── raw/                       # Raw images (user provided)
│   ├── processed/                 # Preprocessed images
│   └── data_preprocessing.py      # Data preprocessing module
│
├── embeddings/                    # Embedding generation
│   ├── clip_embedder.py          # CLIP embeddings (Yen-Shuo)
│   └── dinov2_embedder.py        # DINOv2 embeddings (Hao-Cheng)
│
├── indexing/                      # FAISS indexing
│   ├── indices/                   # Saved FAISS indices
│   └── faiss_indexer.py          # FAISS index builder (Jin-Lian)
│
├── visualization/                 # Explainability visualizations
│   ├── gradcam.py                # Grad-CAM implementation (Yen-Shuo)
│   └── attention_viz.py          # Attention visualization (Hao-Cheng)
│
├── utils/                         # Utility functions
│   └── helpers.py                # Helper functions
│
├── results/                       # Output visualizations and reports
└── models/                        # Model checkpoints (auto-downloaded)
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works)
- 8GB+ RAM (16GB+ recommended for large datasets)

### Step 1: Clone or Download the Project

```bash
cd face_similarity_project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# For GPU support with FAISS (optional but recommended):
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Step 4: Install CLIP (if not installed via requirements.txt)

```bash
pip install git+https://github.com/openai/CLIP.git
```

## 📊 Dataset Preparation

### Option 1: Use Your Own Images

1. Create the raw data directory:
```bash
mkdir -p data/raw/sample_faces
```

2. Place your face images in `data/raw/sample_faces/`:
   - Supported formats: JPG, JPEG, PNG, BMP
   - Recommended: At least 100 images for meaningful results
   - Images can be in subdirectories (will be recursively found)

### Option 2: Download Public Datasets

**LFW (Labeled Faces in the Wild)**
```bash
python -c "from data.data_preprocessing import DataPreprocessor; dp = DataPreprocessor(); dp.download_lfw(); dp.extract_lfw('data/raw/lfw.tgz')"
```

**CelebA** (manual download required):
- Visit: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Download and extract to `data/raw/celeba/`

## 🎯 Usage

### Quick Start - Run Complete Pipeline

Process everything with one command:

```bash
python main.py --step all --max-images 1000
```

This will:
1. Preprocess images
2. Generate CLIP embeddings
3. Generate DINOv2 embeddings
4. Build FAISS index
5. Create Grad-CAM visualizations
6. Create attention visualizations
7. Run demo similarity search

### Run Individual Steps

#### Step 1: Data Preprocessing

```bash
python main.py --step preprocess --max-images 1000
```

**What it does:**
- Collects images from `data/raw/`
- Resizes to 224x224
- Saves to `data/processed/`

**Alternative (direct script):**
```bash
python data/data_preprocessing.py
```

#### Step 2: Generate CLIP Embeddings

```bash
python main.py --step embed_clip --clip-model ViT-B/32 --batch-size 32
```

**What it does:**
- Loads CLIP model
- Generates embeddings for all processed images
- Saves to `embeddings/clip_embeddings.h5`

**Available CLIP models:**
- `ViT-B/32` (fastest, default)
- `ViT-B/16` (balanced)
- `ViT-L/14` (best quality, slower)

**Alternative (direct script):**
```bash
python embeddings/clip_embedder.py
```

**Output:**
- File: `embeddings/clip_embeddings.h5`
- Contains: Embeddings (N x D), image paths, metadata

#### Step 3: Generate DINOv2 Embeddings

```bash
python main.py --step embed_dinov2 --dinov2-model dinov2_vitb14 --batch-size 32
```

**What it does:**
- Loads DINOv2 model
- Generates embeddings with attention maps
- Saves to `embeddings/dinov2_embeddings.h5`

**Available DINOv2 models:**
- `dinov2_vits14` (small, fast)
- `dinov2_vitb14` (medium, default)
- `dinov2_vitl14` (large, best quality)

**Alternative (direct script):**
```bash
python embeddings/dinov2_embedder.py
```

**Output:**
- File: `embeddings/dinov2_embeddings.h5`
- Contains: Embeddings, attention maps

#### Step 4: Build FAISS Index

```bash
python main.py --step index --index-type IVF
```

**What it does:**
- Loads CLIP embeddings
- Builds FAISS index for fast similarity search
- Benchmarks search performance
- Saves index to `indexing/indices/`

**Available index types:**
- `Flat`: Exact search (slow but accurate)
- `IVF`: Inverted file index (fast, default)
- `HNSW`: Hierarchical graph (fastest, approximate)

**Alternative (direct script):**
```bash
python indexing/faiss_indexer.py
```

**Output:**
- Files: `indexing/indices/clip_ivf_index.*`
- Benchmark results printed to console

#### Step 5: Generate Grad-CAM Visualizations

```bash
python main.py --step gradcam
```

**What it does:**
- Loads CLIP model
- Generates Grad-CAM heatmaps showing important facial regions
- Creates pairwise similarity visualizations
- Saves to `results/gradcam_results/`

**Alternative (direct script):**
```bash
python visualization/gradcam.py
```

**Output:**
- Directory: `results/gradcam_results/`
- Files: `query_cam.jpg`, `comparison.jpg`

#### Step 6: Generate Attention Visualizations

```bash
python main.py --step attention
```

**What it does:**
- Loads DINOv2 model
- Extracts self-attention patterns
- Visualizes which image regions the model focuses on
- Saves to `results/attention_results/`

**Alternative (direct script):**
```bash
python visualization/attention_viz.py
```

**Output:**
- Directory: `results/attention_results/`
- Files: `single_attention.jpg`, `layer_comparison.jpg`, `pairwise_attention.jpg`

#### Step 7: Demo Similarity Search

```bash
python main.py --step demo
```

**What it does:**
- Loads FAISS index
- Performs similarity search for sample query
- Visualizes top-10 results
- Saves to `results/demo_search_results.jpg`

**Output:**
- Visualization showing query and top-10 similar faces
- Console output with similarity scores

## 🔧 Advanced Usage

### Custom Query Search

Create a custom search script:

```python
from indexing.faiss_indexer import FAISSIndexer
from embeddings.clip_embedder import CLIPEmbedder
from config import Config

config = Config()

# Load index
indexer = FAISSIndexer.load_index(
    config.FAISS_INDEX_DIR, 
    "clip_ivf_index"
)

# Load embedder
embedder = CLIPEmbedder()

# Query with your image
query_path = "path/to/your/query/image.jpg"
query_embedding = embedder.embed_image(query_path)

# Search
similar_paths, similarities = indexer.search(query_embedding, k=20)

# Print results
for i, (path, sim) in enumerate(zip(similar_paths, similarities), 1):
    print(f"{i}. {path}: {sim:.4f}")
```

### Multi-Layer Feature Exploration

```python
from embeddings.clip_embedder import CLIPEmbedder

embedder = CLIPEmbedder()

# Get embeddings from different layers
layer_embeddings = embedder.get_layer_embeddings(
    "image.jpg",
    layer_indices=[3, 6, 9, 11]  # Early, mid, late layers
)

# Each layer captures different features:
# - Early layers: edges, textures, colors
# - Mid layers: facial components (eyes, nose)
# - Late layers: identity, high-level features
```

### Batch Processing

```python
from data.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
image_paths = preprocessor.collect_image_paths("data/raw/")

# Create dataloader for efficient batch processing
dataloader = preprocessor.create_dataloader(
    image_paths,
    batch_size=64
)

# Process in batches
for batch_images, batch_paths in dataloader:
    # Your processing code here
    pass
```

## 📈 Performance Benchmarks

### Expected Performance (on 1M images)

| Index Type | Build Time | Query Latency | Accuracy |
|------------|------------|---------------|----------|
| Flat       | 10s        | 500ms         | 100%     |
| IVF        | 30s        | 50ms          | ~99%     |
| HNSW       | 120s       | 2ms           | ~98%     |

### Hardware Requirements

| Dataset Size | RAM Required | GPU VRAM | Recommended GPU |
|--------------|--------------|----------|-----------------|
| 10K images   | 4GB          | 2GB      | Any CUDA GPU    |
| 100K images  | 8GB          | 4GB      | GTX 1080+       |
| 1M images    | 16GB         | 8GB      | RTX 2080+       |

## 🎨 Output Examples

### 1. Similarity Search Results
- **Location**: `results/demo_search_results.jpg`
- **Content**: Grid showing query image and top-10 similar faces with similarity scores

### 2. Grad-CAM Visualizations
- **Location**: `results/gradcam_results/`
- **Content**: Heatmaps highlighting facial regions important for similarity

### 3. Attention Maps
- **Location**: `results/attention_results/`
- **Content**: Visualizations showing where the model focuses attention

### 4. Layer Comparisons
- **Location**: `results/attention_results/layer_comparison.jpg`
- **Content**: Side-by-side comparison of features from different layers

## 🧪 Testing and Validation

### Run Unit Tests

Test individual components:

```bash
# Test data preprocessing
python data/data_preprocessing.py

# Test CLIP embeddings
python embeddings/clip_embedder.py

# Test DINOv2 embeddings
python embeddings/dinov2_embedder.py

# Test FAISS indexing
python indexing/faiss_indexer.py

# Test visualizations
python visualization/gradcam.py
python visualization/attention_viz.py

# Test utilities
python utils/helpers.py
```

### Validate Results

Check that outputs exist and are reasonable:

```bash
# Check embeddings were created
ls -lh embeddings/*.h5

# Check index was built
ls -lh indexing/indices/

# Check visualizations were generated
ls -R results/
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'clip'`
```bash
pip install git+https://github.com/openai/CLIP.git
```

**Issue**: CUDA out of memory
```bash
# Reduce batch size
python main.py --step embed_clip --batch-size 16
```

**Issue**: No images found
```bash
# Make sure images are in correct location
ls data/raw/sample_faces/
```

**Issue**: FAISS index error
```bash
# Try a different index type
python main.py --step index --index-type Flat
```

**Issue**: DINOv2 model download fails
```bash
# Clear torch hub cache
rm -rf ~/.cache/torch/hub/facebookresearch_dinov2_main/
```

### Getting Help

1. Check this README thoroughly
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify dataset is in correct location
5. Check hardware requirements

## 📚 Technical Details

### Models Used

**CLIP (Contrastive Language-Image Pre-training)**
- Vision Transformer architecture
- Trained on 400M image-text pairs
- Embeddings: 512D (ViT-B/32), 768D (ViT-L/14)

**DINOv2 (Self-Supervised Vision Transformer)**
- Vision Transformer trained with self-distillation
- Better for fine-grained visual features
- Embeddings: 384D (vits14), 768D (vitb14), 1024D (vitl14)

### Explainability Methods

**Grad-CAM (Gradient-weighted Class Activation Mapping)**
- Computes gradients of output w.r.t. feature maps
- Shows which regions contribute to similarity
- Works with any CNN or ViT architecture

**Attention Visualization**
- Extracts self-attention patterns from transformers
- Shows which patches the model attends to
- Reveals hierarchical feature learning

## 📖 References

1. FaceNet: [Schroff et al., CVPR 2015]
2. ArcFace: [Deng et al., CVPR 2019]
3. CLIP: [Radford et al., ICML 2021]
4. DINOv2: [Oquab et al., arXiv 2023]
5. Grad-CAM: [Selvaraju et al., ICCV 2017]
6. FAISS: [Johnson et al., IEEE Trans 2019]

## 👥 Team Contributions

- **Yen-Shuo Su**: CLIP embeddings, Grad-CAM implementation
- **Hao-Cheng Chang**: DINOv2 embeddings, attention visualization
- **Jin-Lian Ho**: FAISS indexing, frontend (future), user study
- **Yu Chu Tsai**: Backend API (future), integration

## 📄 License

This project is for educational purposes as part of Georgia Tech coursework.

## 🙏 Acknowledgments

- Anthropic's Claude for code assistance
- Facebook AI Research for DINOv2 and FAISS
- OpenAI for CLIP
- Georgia Tech CS 6460 course staff

---

For questions or issues, please contact the team members at their Georgia Tech emails.
