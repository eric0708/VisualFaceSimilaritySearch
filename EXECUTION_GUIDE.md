# Execution Guide

## Quick Reference for Running the Complete Project

---

## 📦 What You Have

A complete, production-ready face similarity search system with:
- 16 Python files implementing all 6 tasks
- Comprehensive documentation (README, Quickstart, Summary)
- Modular architecture for easy extension
- Complete visualization and explainability tools

---

## 🚀 Fastest Way to Get Started

### 1. Setup (5 minutes)

```bash
# Navigate to project
cd face_similarity_project

# Run automated setup
bash setup.sh

# OR manual setup:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 2. Add Images (2 minutes)

```bash
# Create directory
mkdir -p data/raw/sample_faces

# Add your face images (JPG, PNG, etc.)
# Or download sample dataset:
python -c "from data.data_preprocessing import DataPreprocessor; dp = DataPreprocessor(); dp.download_lfw()"
```

### 3. Run Everything (5-10 minutes for 100 images)

```bash
python main.py --step all --max-images 100
```

That's it! Results will be in the `results/` directory.

---

## 📋 What Each File Does

### Core Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `config.py` | Central configuration | Paths, settings, hyperparameters |
| `main.py` | Pipeline orchestration | Run all steps sequentially |
| `requirements.txt` | Dependencies | All required Python packages |

### Data Processing

| File | Task | Team Member |
|------|------|-------------|
| `data/data_preprocessing.py` | Image loading, resizing, normalization | All |

**Run standalone:**
```bash
python data/data_preprocessing.py
```

### Embeddings (Tasks 2-3)

| File | Task | Team Member |
|------|------|-------------|
| `embeddings/clip_embedder.py` | CLIP embeddings, layer extraction | Yen-Shuo Su |
| `embeddings/dinov2_embedder.py` | DINOv2 embeddings, attention | Hao-Cheng Chang |

**Run standalone:**
```bash
python embeddings/clip_embedder.py
python embeddings/dinov2_embedder.py
```

### Indexing (Task 4)

| File | Task | Team Member |
|------|------|-------------|
| `indexing/faiss_indexer.py` | FAISS index building, search | Jin-Lian Ho |

**Run standalone:**
```bash
python indexing/faiss_indexer.py
```

### Visualization (Tasks 5-6)

| File | Task | Team Member |
|------|------|-------------|
| `visualization/gradcam.py` | Grad-CAM heatmaps | Yen-Shuo Su |
| `visualization/attention_viz.py` | Attention maps | Hao-Cheng Chang |

**Run standalone:**
```bash
python visualization/gradcam.py
python visualization/attention_viz.py
```

### Utilities

| File | Purpose |
|------|---------|
| `utils/helpers.py` | Helper functions, metrics, visualization |
| `notebooks/demo_exploration.py` | Interactive demo script |

---

## 🎯 Running Specific Tasks

### Just Preprocessing
```bash
python main.py --step preprocess --max-images 1000
```

### Just CLIP Embeddings
```bash
python main.py --step embed_clip --clip-model ViT-B/32 --batch-size 32
```

### Just DINOv2 Embeddings
```bash
python main.py --step embed_dinov2 --dinov2-model dinov2_vitb14
```

### Just FAISS Index
```bash
python main.py --step index --index-type IVF
```

### Just Visualizations
```bash
python main.py --step gradcam
python main.py --step attention
```

### Just Demo Search
```bash
python main.py --step demo
```

---

## 🎮 Interactive Exploration

Run the demo notebook for hands-on exploration:

```bash
python notebooks/demo_exploration.py
```

This will:
1. Load and display sample images
2. Generate embeddings
3. Build search index
4. Perform similarity search
5. Extract multi-layer features
6. Generate Grad-CAM visualizations
7. Create attention maps
8. Benchmark performance

---

## 📊 Expected Outputs

### After Full Pipeline

```
face_similarity_project/
├── embeddings/
│   ├── clip_embeddings.h5       # CLIP embeddings
│   └── dinov2_embeddings.h5     # DINOv2 embeddings
│
├── indexing/indices/
│   ├── clip_ivf_index.index     # FAISS index
│   ├── clip_ivf_index_paths.pkl # Image paths
│   └── clip_ivf_index_metadata.json
│
└── results/
    ├── gradcam_results/         # Grad-CAM visualizations
    ├── attention_results/       # Attention visualizations
    └── demo_search_results.jpg  # Top-10 results
```

---

## ⚙️ Customization Options

### Change Models

```bash
# Use better CLIP model
python main.py --step embed_clip --clip-model ViT-L/14

# Use larger DINOv2 model
python main.py --step embed_dinov2 --dinov2-model dinov2_vitl14
```

### Change Index Type

```bash
# Fastest search (approximate)
python main.py --step index --index-type HNSW

# Exact search (slower)
python main.py --step index --index-type Flat
```

### Adjust Performance

```bash
# For limited GPU memory
python main.py --step all --batch-size 8

# For more images
python main.py --step all --max-images 10000
```

---

## 🔍 Verify Everything Works

### Check 1: Files Created
```bash
ls -lh embeddings/*.h5
ls -lh indexing/indices/
ls -R results/
```

### Check 2: Load Embeddings
```python
from embeddings.clip_embedder import CLIPEmbedder
embeddings, paths, metadata = CLIPEmbedder.load_embeddings('embeddings/clip_embeddings.h5')
print(f"Loaded {len(embeddings)} embeddings")
```

### Check 3: Test Search
```python
from indexing.faiss_indexer import FAISSIndexer
from embeddings.clip_embedder import CLIPEmbedder

indexer = FAISSIndexer.load_index('indexing/indices', 'clip_ivf_index')
embedder = CLIPEmbedder()

query_emb = embedder.embed_image('path/to/image.jpg')
results, similarities = indexer.search(query_emb, k=5)
print(results)
```

---

## 🚨 Troubleshooting

### Problem: Import Errors
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### Problem: CUDA Out of Memory
```bash
# Solution: Reduce batch size
python main.py --step all --batch-size 8
```

### Problem: No Images Found
```bash
# Solution: Check directory
ls data/raw/sample_faces/
# Add images to this directory
```

### Problem: Models Not Downloading
```bash
# Solution: Clear cache and retry
rm -rf ~/.cache/torch/hub/
python main.py --step embed_clip
```

---

## 📈 Performance Tips

### For Speed
- Use smaller models: `ViT-B/32`, `dinov2_vits14`
- Use HNSW index: `--index-type HNSW`
- Increase batch size: `--batch-size 64` (if GPU allows)

### For Quality
- Use larger models: `ViT-L/14`, `dinov2_vitl14`
- Use Flat index: `--index-type Flat`
- Process more images: `--max-images 10000`

### For Memory
- Reduce batch size: `--batch-size 8`
- Use IVF index: `--index-type IVF`
- Process in chunks: Run steps separately

---

## 🎓 Learning Path

### Beginner
1. Run `python main.py --step all --max-images 50`
2. Check results in `results/`
3. Read `QUICKSTART.md`

### Intermediate
1. Run individual steps to understand pipeline
2. Modify `config.py` settings
3. Try different models and index types
4. Read `README.md`

### Advanced
1. Explore `notebooks/demo_exploration.py`
2. Extract layer-wise embeddings
3. Implement custom search logic
4. Read source code with docstrings
5. Read `PROJECT_SUMMARY.md`

---

## ✅ Success Checklist

- [ ] Setup completed without errors
- [ ] Images added to `data/raw/sample_faces/`
- [ ] Pipeline runs successfully
- [ ] Embeddings created (check `embeddings/` directory)
- [ ] Index built (check `indexing/indices/` directory)
- [ ] Visualizations generated (check `results/` directory)
- [ ] Search returns reasonable results
- [ ] All tests pass

---

## 📞 Need Help?

1. **Check documentation**: README.md, QUICKSTART.md, PROJECT_SUMMARY.md
2. **Read error messages**: They usually indicate the problem
3. **Check file paths**: Ensure images are in correct location
4. **Verify installation**: All dependencies in requirements.txt
5. **Review code**: Well-commented with docstrings

---

## 🎉 What's Next?

After everything works:
1. Experiment with your own images
2. Try different model architectures
3. Analyze layer-wise features
4. Compare explanation methods
5. Scale to larger datasets
6. Integrate into your application

---

**Enjoy exploring face similarity with explainable AI! 🚀**
