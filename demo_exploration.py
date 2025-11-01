# Face Similarity Search - Demo Notebook

"""
Interactive demo for exploring face similarity search
Run this in Jupyter notebook or as a Python script
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add project to path
sys.path.insert(0, os.path.abspath('.'))

from config import Config
from embeddings.clip_embedder import CLIPEmbedder
from embeddings.dinov2_embedder import DINOv2Embedder
from indexing.faiss_indexer import FAISSIndexer
from visualization.gradcam import CLIPGradCAM
from visualization.attention_viz import AttentionVisualizer
from utils.helpers import visualize_top_k_results

# Initialize config
config = Config()
config.create_directories()

print("=" * 70)
print("FACE SIMILARITY SEARCH - INTERACTIVE DEMO")
print("=" * 70)

# =============================================================================
# SECTION 1: Data Exploration
# =============================================================================
print("\n--- Section 1: Data Exploration ---\n")

from data.data_preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()

# Find images
image_paths = preprocessor.collect_image_paths(config.PROCESSED_DATA_DIR, max_images=100)
print(f"Found {len(image_paths)} processed images")

if len(image_paths) > 0:
    # Display sample images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, img_path in enumerate(image_paths[:10]):
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f'Image {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/sample_images.jpg', dpi=150, bbox_inches='tight')
    print("Sample images saved to results/sample_images.jpg")
else:
    print("No images found. Please add images to data/raw/ and run preprocessing.")
    sys.exit(1)

# =============================================================================
# SECTION 2: Generate Embeddings
# =============================================================================
print("\n--- Section 2: Generate Embeddings ---\n")

# Check if embeddings exist
clip_emb_path = os.path.join(config.EMBEDDINGS_DIR, 'clip_embeddings.h5')

if not os.path.exists(clip_emb_path):
    print("Generating CLIP embeddings...")
    clip_embedder = CLIPEmbedder()
    clip_embeddings, clip_paths = clip_embedder.embed_dataset(
        image_paths[:100], 
        save_path=clip_emb_path
    )
else:
    print("Loading existing CLIP embeddings...")
    clip_embeddings, clip_paths, metadata = CLIPEmbedder.load_embeddings(clip_emb_path)

print(f"Embeddings shape: {clip_embeddings.shape}")
print(f"Embedding dimension: {clip_embeddings.shape[1]}")

# =============================================================================
# SECTION 3: Build Search Index
# =============================================================================
print("\n--- Section 3: Build Search Index ---\n")

indexer = FAISSIndexer(embedding_dim=clip_embeddings.shape[1], index_type="Flat")
indexer.build_index(clip_embeddings, clip_paths)

print("Index built successfully!")
print(f"Total vectors in index: {indexer.index.ntotal}")

# =============================================================================
# SECTION 4: Perform Similarity Search
# =============================================================================
print("\n--- Section 4: Perform Similarity Search ---\n")

# Use first image as query
query_idx = 0
query_path = clip_paths[query_idx]
query_embedding = clip_embeddings[query_idx]

print(f"Query image: {os.path.basename(query_path)}")

# Search
similar_paths, similarities = indexer.search(query_embedding, k=10)

print("\nTop 10 similar images:")
for i, (path, sim) in enumerate(zip(similar_paths, similarities), 1):
    print(f"{i:2d}. {os.path.basename(path):40s} | Similarity: {sim:.4f}")

# Visualize results
viz_path = os.path.join(config.RESULTS_DIR, 'similarity_search_demo.jpg')
visualize_top_k_results(query_path, similar_paths, similarities, viz_path, k=10)
print(f"\nVisualization saved to {viz_path}")

# =============================================================================
# SECTION 5: Multi-Layer Feature Exploration
# =============================================================================
print("\n--- Section 5: Multi-Layer Feature Exploration ---\n")

clip_embedder = CLIPEmbedder()

# Get embeddings from different layers
layer_embeddings = clip_embedder.get_layer_embeddings(
    query_path,
    layer_indices=[3, 6, 9, 11]
)

print("Layer embeddings extracted:")
for layer_idx, emb in layer_embeddings.items():
    print(f"  Layer {layer_idx}: {emb.shape}")

# Search using different layers
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for idx, (layer_idx, layer_emb) in enumerate(layer_embeddings.items()):
    # Search using this layer's embedding
    layer_similar, layer_sims = indexer.search(layer_emb, k=5)
    
    # Visualize
    ax = axes[idx]
    ax.axis('off')
    ax.set_title(f'Layer {layer_idx} - Top 5 Results')
    
    # Create mini grid
    mini_fig = plt.figure(figsize=(10, 2))
    for i, (path, sim) in enumerate(zip(layer_similar[:5], layer_sims[:5]), 1):
        plt.subplot(1, 6, i+1)
        img = Image.open(path)
        plt.imshow(img)
        plt.title(f'#{i}\n{sim:.3f}')
        plt.axis('off')
    plt.tight_layout()
    mini_fig.savefig(f'results/layer_{layer_idx}_results.jpg', dpi=100, bbox_inches='tight')
    plt.close(mini_fig)

plt.close()
print("Layer-wise results saved to results/layer_*_results.jpg")

# =============================================================================
# SECTION 6: Explainability - Grad-CAM
# =============================================================================
print("\n--- Section 6: Explainability - Grad-CAM ---\n")

try:
    import clip
    
    print("Generating Grad-CAM visualizations...")
    clip_model, _ = clip.load("ViT-B/32", device=config.DEVICE)
    grad_cam = CLIPGradCAM(clip_model, device=config.DEVICE)
    
    # Generate for top result
    reference_path = similar_paths[1]  # Second result (first is query itself)
    
    gradcam_dir = os.path.join(config.RESULTS_DIR, 'gradcam_demo')
    results = grad_cam.generate_pairwise_cam(
        query_path, reference_path, clip_embedder, save_dir=gradcam_dir
    )
    
    grad_cam.remove_hooks()
    print(f"Grad-CAM results saved to {gradcam_dir}")
    
except Exception as e:
    print(f"Grad-CAM generation skipped: {e}")

# =============================================================================
# SECTION 7: Explainability - Attention Visualization
# =============================================================================
print("\n--- Section 7: Explainability - Attention Visualization ---\n")

try:
    print("Generating attention visualizations...")
    dinov2_embedder = DINOv2Embedder()
    visualizer = AttentionVisualizer()
    
    # Extract attention
    attn_data = dinov2_embedder.extract_attention_maps(query_path)
    
    # Visualize
    attn_dir = os.path.join(config.RESULTS_DIR, 'attention_demo')
    os.makedirs(attn_dir, exist_ok=True)
    
    visualizer.visualize_attention_map(
        attn_data['attention_map'],
        query_path,
        save_path=os.path.join(attn_dir, 'attention_map.jpg')
    )
    
    print(f"Attention results saved to {attn_dir}")
    
except Exception as e:
    print(f"Attention visualization skipped: {e}")

# =============================================================================
# SECTION 8: Performance Benchmark
# =============================================================================
print("\n--- Section 8: Performance Benchmark ---\n")

print("Benchmarking search performance...")
benchmark_results = indexer.benchmark(clip_embeddings, k=10, num_queries=50)

print("\nBenchmark Summary:")
print(f"  Average latency: {benchmark_results['avg_latency']*1000:.2f} ms")
print(f"  Queries per second: {benchmark_results['queries_per_second']:.2f}")
print(f"  Total queries: {benchmark_results['num_queries']}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("DEMO COMPLETE!")
print("=" * 70)
print("\nGenerated outputs:")
print("  - Sample images: results/sample_images.jpg")
print(f"  - Search results: {viz_path}")
print("  - Layer results: results/layer_*_results.jpg")
print("  - Grad-CAM: results/gradcam_demo/")
print("  - Attention: results/attention_demo/")
print("\nYou can now:")
print("  1. Try different query images")
print("  2. Experiment with different models")
print("  3. Explore layer-wise embeddings")
print("  4. Analyze explainability visualizations")
print("=" * 70)
