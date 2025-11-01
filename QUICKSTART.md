# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 2. Add Your Images (1 minute)

```bash
mkdir -p data/raw/sample_faces
# Copy your face images to data/raw/sample_faces/
```

### 3. Run Pipeline (2 minutes for small dataset)

```bash
# Process up to 100 images
python main.py --step all --max-images 100
```

### 4. View Results

Check these locations:
- `embeddings/` - Saved embeddings
- `indexing/indices/` - FAISS search index
- `results/` - All visualizations

## What Each Step Does

### Step 1: Preprocessing
- Resizes images to 224x224
- Normalizes formats
- **Time**: ~5 sec for 100 images

### Step 2: CLIP Embeddings
- Generates 512D feature vectors
- **Time**: ~30 sec for 100 images (GPU)

### Step 3: DINOv2 Embeddings
- Generates 768D feature vectors with attention
- **Time**: ~45 sec for 100 images (GPU)

### Step 4: FAISS Index
- Builds search index
- **Time**: ~5 sec for 100 images

### Step 5-6: Visualizations
- Creates explainability heatmaps
- **Time**: ~10 sec per image pair

## Quick Commands

### Just want to search similar faces?

```bash
# 1. Preprocess
python main.py --step preprocess

# 2. Create embeddings
python main.py --step embed_clip

# 3. Build index
python main.py --step index

# 4. Search!
python main.py --step demo
```

### Just want explanations?

```bash
# After embeddings are created:
python main.py --step gradcam
python main.py --step attention
```

## Testing with Sample Images

If you don't have face images yet:

```bash
# Download sample dataset (LFW)
python -c "from data.data_preprocessing import DataPreprocessor; dp = DataPreprocessor(); dp.download_lfw()"
```

## Minimal Working Example

Create a file `quick_test.py`:

```python
from embeddings.clip_embedder import CLIPEmbedder
from indexing.faiss_indexer import FAISSIndexer

# Generate embeddings
embedder = CLIPEmbedder()
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
embeddings, paths = embedder.embed_dataset(image_paths)

# Build index
indexer = FAISSIndexer(embedding_dim=512, index_type="Flat")
indexer.build_index(embeddings, paths)

# Search
query_emb = embedder.embed_image("query.jpg")
results, similarities = indexer.search(query_emb, k=5)

print("Top 5 similar images:", results)
```

## Common First-Time Issues

**Import errors?**
```bash
pip install -r requirements.txt
```

**No images found?**
```bash
ls data/raw/sample_faces/  # Should show your images
```

**Out of memory?**
```bash
python main.py --step all --max-images 50 --batch-size 8
```

## Next Steps

After quick start works:
1. Read full README.md
2. Try different models (`--clip-model ViT-L/14`)
3. Experiment with index types (`--index-type HNSW`)
4. Explore layer embeddings
5. Scale to larger datasets

## Getting Results Fast

**Fastest setup (CPU, small dataset)**:
```bash
python main.py --step all --max-images 50 --batch-size 4
```

**Best quality (GPU, medium dataset)**:
```bash
python main.py --step all --max-images 1000 --clip-model ViT-L/14
```

**Large scale (GPU, big dataset)**:
```bash
python main.py --step all --max-images 10000 --index-type HNSW
```

---

**Need help?** Check README.md for detailed documentation.
