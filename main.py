"""
Main Execution Script (No-Index Version)
Runs face similarity search pipeline WITHOUT FAISS indexing
Uses direct numpy similarity computation instead
"""
import os
import sys

# Enable MPS fallback for unsupported operations (needed for DINOv2 on Apple Silicon)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config


def compute_similarities(query_embedding: np.ndarray, 
                        all_embeddings: np.ndarray, 
                        top_k: int = 10) -> tuple:
    """
    Compute cosine similarity between query and all embeddings
    
    Args:
        query_embedding: Query vector [D]
        all_embeddings: All embeddings [N, D]
        top_k: Number of top results to return
        
    Returns:
        top_indices: Indices of top-k similar embeddings
        top_similarities: Similarity scores
    """
    # Normalize query
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    
    # Compute cosine similarities (dot product of normalized vectors)
    similarities = np.dot(all_embeddings, query_norm)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_similarities = similarities[top_indices]
    
    return top_indices, top_similarities


def create_summary_grid(query_indices, image_paths, embeddings_normalized, 
                       compute_similarities_func, save_path):
    """
    Create a summary grid showing all sample searches
    
    Args:
        query_indices: List of query image indices
        image_paths: List of all image paths
        embeddings_normalized: Normalized embeddings array
        compute_similarities_func: Function to compute similarities
        save_path: Path to save the summary grid
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    import time
    
    num_samples = len(query_indices)
    top_k = 10  # Show top 10 for summary
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, top_k + 1, figsize=(30, 3.5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Track timing
    search_times = []
    
    for sample_idx, query_idx in enumerate(query_indices):
        query_path = image_paths[query_idx]
        query_embedding = embeddings_normalized[query_idx]
        
        # Time the similarity computation
        start_time = time.time()
        top_indices, top_similarities = compute_similarities_func(
            query_embedding, embeddings_normalized, top_k=top_k
        )
        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        search_times.append(search_time)
        
        # Plot query image
        ax = axes[sample_idx, 0]
        query_img = Image.open(query_path)
        ax.imshow(query_img)
        ax.set_title(f'Sample {sample_idx + 1}\nQuery\n({search_time:.1f}ms)', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Plot top-k results
        for result_idx, (top_idx, sim) in enumerate(zip(top_indices, top_similarities)):
            ax = axes[sample_idx, result_idx + 1]
            result_img = Image.open(image_paths[top_idx])
            ax.imshow(result_img)
            ax.set_title(f'#{result_idx + 1}\nSim: {sim:.3f}', fontsize=9)
            ax.axis('off')
    
    # Calculate average search time
    avg_time = np.mean(search_times)
    
    plt.suptitle(f'Face Similarity Search - Sample Results\nAverage Search Time: {avg_time:.1f}ms ({len(image_paths)} images)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Summary grid saved: {os.path.basename(save_path)}")
    print(f"  ⏱️  Average search time: {avg_time:.1f}ms per query")


def create_embedding_comparison(query_indices, embeddings_dict, 
                                compute_similarities_func, save_dir):
    """
    Create side-by-side comparison of CLIP vs DINOv2 results
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    
    print(f"  Generating comparison visualizations for {len(query_indices)} samples...")
    
    clip_data = embeddings_dict['clip']
    dinov2_data = embeddings_dict['dinov2']
    
    clip_emb, clip_paths, _ = clip_data
    dinov2_emb, dinov2_paths, _ = dinov2_data
    
    # Normalize
    clip_norm = clip_emb / (np.linalg.norm(clip_emb, axis=1, keepdims=True) + 1e-8)
    dinov2_norm = dinov2_emb / (np.linalg.norm(dinov2_emb, axis=1, keepdims=True) + 1e-8)
    
    # For each query, create comparison
    for sample_num, query_idx in enumerate(query_indices, 1):
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        
        query_path = clip_paths[query_idx]
        
        # CLIP results
        clip_query = clip_norm[query_idx]
        clip_top_idx, clip_sims = compute_similarities_func(clip_query, clip_norm, top_k=5)
        
        # DINOv2 results
        dinov2_query = dinov2_norm[query_idx]
        dinov2_top_idx, dinov2_sims = compute_similarities_func(dinov2_query, dinov2_norm, top_k=5)
        
        # Plot CLIP row
        axes[0, 0].imshow(Image.open(query_path))
        axes[0, 0].set_title('Query\n(CLIP)', fontweight='bold')
        axes[0, 0].axis('off')
        
        for i, (idx, sim) in enumerate(zip(clip_top_idx, clip_sims)):
            axes[0, i+1].imshow(Image.open(clip_paths[idx]))
            axes[0, i+1].set_title(f'CLIP #{i+1}\n{sim:.3f}')
            axes[0, i+1].axis('off')
        
        # Plot DINOv2 row
        axes[1, 0].imshow(Image.open(query_path))
        axes[1, 0].set_title('Query\n(DINOv2)', fontweight='bold')
        axes[1, 0].axis('off')
        
        for i, (idx, sim) in enumerate(zip(dinov2_top_idx, dinov2_sims)):
            axes[1, i+1].imshow(Image.open(dinov2_paths[idx]))
            axes[1, i+1].set_title(f'DINOv2 #{i+1}\n{sim:.3f}')
            axes[1, i+1].axis('off')
        
        plt.suptitle(f'Sample {sample_num}: CLIP vs DINOv2 Comparison', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        comp_path = os.path.join(save_dir, f'comparison_sample_{sample_num}.jpg')
        plt.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  ✅ Created {len(query_indices)} comparison visualizations")


def run_pipeline(args):
    """Run the complete pipeline without FAISS"""
    config = Config()
    config.create_directories()
    
    print("=" * 70)
    print("VISUAL FACE SIMILARITY SEARCH (NO-INDEX VERSION)")
    print("Direct numpy-based similarity search")
    print("=" * 70)
    
    # Step 1: Data Preprocessing
    if args.step in ['all', 'preprocess']:
        print("\n" + "="*70)
        print("STEP 1: DATA PREPROCESSING")
        print("="*70)
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Collect images
        if os.path.exists(config.RAW_DATA_DIR):
            image_paths = preprocessor.collect_image_paths(
                config.RAW_DATA_DIR, 
                max_images=args.max_images
            )
            
            if len(image_paths) > 0:
                # Preprocess
                preprocessor.preprocess_and_save(
                    image_paths,
                    config.PROCESSED_DATA_DIR,
                    target_size=config.TARGET_IMAGE_SIZE
                )
                print("✅ Data preprocessing complete!")
            else:
                print(f"⚠️  No images found in {config.RAW_DATA_DIR}")
                print("Please add face images and run again.")
                return
        else:
            print(f"⚠️  Raw data directory not found: {config.RAW_DATA_DIR}")
            print("Please create it and add face images.")
            return
    
    # Get processed image paths
    from data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    image_paths = preprocessor.collect_image_paths(config.PROCESSED_DATA_DIR)
    
    if len(image_paths) == 0:
        print("No processed images found. Run preprocessing first.")
        return
    
    # Step 2: Generate CLIP Embeddings
    if args.step in ['all', 'embed_clip']:
        print("\n" + "="*70)
        print("STEP 2: GENERATING CLIP EMBEDDINGS")
        print("="*70)
        from clip_embedder import CLIPEmbedder
        
        clip_embedder = CLIPEmbedder(model_name=args.clip_model)
        clip_save_path = os.path.join(config.EMBEDDINGS_DIR, 'clip_embeddings.h5')
        
        if os.path.exists(clip_save_path):
            print(f"Loading existing CLIP embeddings from {clip_save_path}...")
            clip_embeddings, clip_paths, _ = CLIPEmbedder.load_embeddings(clip_save_path)
            print(f"✅ Loaded {len(clip_embeddings)} CLIP embeddings")
        else:
            clip_embeddings, clip_paths = clip_embedder.embed_dataset(
                image_paths,
                batch_size=args.batch_size,
                save_path=clip_save_path
            )
            print("✅ CLIP embeddings complete!")
    
    # Step 3: Generate DINOv2 Embeddings
    if args.step in ['all', 'embed_dinov2']:
        print("\n" + "="*70)
        print("STEP 3: GENERATING DINOV2 EMBEDDINGS")
        print("="*70)
        from dinov2_embedder import DINOv2Embedder
        
        dinov2_embedder = DINOv2Embedder(model_name=args.dinov2_model)
        dinov2_save_path = os.path.join(config.EMBEDDINGS_DIR, 'dinov2_embeddings.h5')
        
        if os.path.exists(dinov2_save_path):
            print(f"Loading existing DINOv2 embeddings from {dinov2_save_path}...")
            dinov2_embeddings, dinov2_paths, _ = DINOv2Embedder.load_embeddings(dinov2_save_path)
            print(f"✅ Loaded {len(dinov2_embeddings)} DINOv2 embeddings")
        else:
            dinov2_embeddings, dinov2_paths = dinov2_embedder.embed_dataset(
                image_paths,
                batch_size=args.batch_size,
                save_path=dinov2_save_path
            )
            print("✅ DINOv2 embeddings complete!")
    
    # Step 4: FAISS Indexing Experiments
    if args.step in ['all', 'index']:
        print("\n" + "="*70)
        print("STEP 4: FAISS INDEXING EXPERIMENTS")
        print("Comparing latency of different search methods")
        print("="*70)
        
        from faiss_indexer import FAISSIndexer
        from clip_embedder import CLIPEmbedder
        import time
        import faiss
        
        # Load CLIP embeddings
        clip_emb_path = os.path.join(config.EMBEDDINGS_DIR, 'clip_embeddings.h5')
        if not os.path.exists(clip_emb_path):
            print(f"Embeddings not found at {clip_emb_path}")
            print("Please run with --step embed_clip first")
        else:
            embeddings, paths, _ = CLIPEmbedder.load_embeddings(clip_emb_path)
            print(f"Loaded {len(embeddings)} embeddings for benchmarking")
            
            # Ensure float32
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            
            # Normalize for cosine similarity
            print("Normalizing embeddings...")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            
            results = []
            
            # 1. Pure Numpy Search
            print("\n--- Benchmarking Pure Numpy Search ---")
            start_time = time.time()
            # Run 100 random queries
            num_queries = 100
            np.random.seed(42)
            query_indices = np.random.choice(len(embeddings), num_queries, replace=False)
            queries = embeddings[query_indices]
            
            for query in queries:
                # Dot product
                sims = np.dot(embeddings, query)
                # Top k
                top_k = np.argsort(sims)[-10:][::-1]
            
            numpy_time = time.time() - start_time
            numpy_avg = (numpy_time / num_queries) * 1000
            print(f"Numpy Average Latency: {numpy_avg:.4f} ms")
            results.append(("Numpy (Exact)", numpy_avg))
            
            # 2. FAISS Index Types
            index_types = ['Flat', 'IVF', 'HNSW']
            
            for idx_type in index_types:
                print(f"\n--- Benchmarking FAISS {idx_type} ---")
                try:
                    indexer = FAISSIndexer(embedding_dim=embeddings.shape[1], index_type=idx_type)
                    # Note: normalize=False because we already normalized above
                    indexer.build_index(embeddings, paths, normalize=False) 
                    
                    bench_res = indexer.benchmark(embeddings, k=10, num_queries=num_queries)
                    results.append((f"FAISS {idx_type}", bench_res['avg_latency'] * 1000))
                    
                except Exception as e:
                    print(f"Error benchmarking {idx_type}: {e}")
            
            # Print Summary Table
            print("\n" + "="*60)
            print(f"{'Method':<20} | {'Latency (ms)':<15} | {'Speedup vs Numpy':<15}")
            print("-" * 60)
            
            numpy_baseline = results[0][1]
            for name, latency in results:
                speedup = numpy_baseline / latency if latency > 0 else 0
                print(f"{name:<20} | {latency:<15.4f} | {speedup:<15.2f}x")
            print("="*60)
    
    # Step 5: Grad-CAM Visualization
    if args.step in ['all', 'gradcam']:
        print("\n" + "="*70)
        print("STEP 5: GENERATING GRAD-CAM VISUALIZATIONS")
        print("="*70)
        
        if len(image_paths) < 2:
            print("Need at least 2 images for Grad-CAM demo.")
        else:
            try:
                from gradcam import CLIPGradCAM
                from clip_embedder import CLIPEmbedder
                import clip
                
                # Load models
                clip_model, _ = clip.load(args.clip_model, device=config.DEVICE)
                embedder = CLIPEmbedder(model_name=args.clip_model)
                grad_cam = CLIPGradCAM(clip_model, device=config.DEVICE)
                
                # Load embeddings for similarity calculation
                clip_emb_path = os.path.join(config.EMBEDDINGS_DIR, 'clip_embeddings.h5')
                if not os.path.exists(clip_emb_path):
                    print("Generating temporary embeddings for selection...")
                    embeddings, valid_paths = embedder.embed_dataset(image_paths[:100], batch_size=32)
                else:
                    embeddings, valid_paths, _ = CLIPEmbedder.load_embeddings(clip_emb_path)
                
                # Normalize embeddings
                embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
                
                save_dir = os.path.join(config.RESULTS_DIR, 'gradcam_results')
                os.makedirs(save_dir, exist_ok=True)
                
                # Select 5 SIMILAR pairs
                print("\nGenerating 5 SIMILAR pairs (High Similarity)...")
                # Pick 5 random query indices
                np.random.seed(42)
                query_indices = np.random.choice(len(embeddings), 5, replace=False)
                
                for i, query_idx in enumerate(query_indices):
                    query_emb = embeddings[query_idx]
                    
                    # Compute similarities
                    sims = np.dot(embeddings, query_emb)
                    sims[query_idx] = -1  # Exclude self
                    
                    # Find most similar
                    best_idx = np.argmax(sims)
                    similarity = sims[best_idx]
                    
                    query_path = valid_paths[query_idx]
                    ref_path = valid_paths[best_idx]
                    
                    output_filename = f"gradcam_similar_{i+1}_sim_{similarity:.2f}.jpg"
                    grad_cam.generate_pairwise_cam(
                        query_path, ref_path, embedder, 
                        save_dir=save_dir, 
                        output_filename=output_filename
                    )
                    print(f"  Saved {output_filename} (Sim: {similarity:.3f})")

                # Select 5 DISSIMILAR pairs
                print("\nGenerating 5 DISSIMILAR pairs (Low Similarity)...")
                # Pick 5 new random query indices
                query_indices_dissim = np.random.choice(len(embeddings), 5, replace=False)
                
                for i, query_idx in enumerate(query_indices_dissim):
                    query_emb = embeddings[query_idx]
                    
                    # Compute similarities
                    sims = np.dot(embeddings, query_emb)
                    
                    # Find least similar
                    worst_idx = np.argmin(sims)
                    similarity = sims[worst_idx]
                    
                    query_path = valid_paths[query_idx]
                    ref_path = valid_paths[worst_idx]
                    
                    output_filename = f"gradcam_dissimilar_{i+1}_sim_{similarity:.2f}.jpg"
                    grad_cam.generate_pairwise_cam(
                        query_path, ref_path, embedder, 
                        save_dir=save_dir, 
                        output_filename=output_filename
                    )
                    print(f"  Saved {output_filename} (Sim: {similarity:.3f})")
                
                grad_cam.remove_hooks()
                print(f"✅ Grad-CAM complete! Results in {save_dir}")
            except Exception as e:
                print(f"⚠️  Grad-CAM skipped due to error: {e}")
                print("   This is a known MPS compatibility issue.")
                print("   You can still use all other features!")
                print("   To try Grad-CAM, you can:")
                print("   1. Run with CPU: Set DEVICE='cpu' in config.py")
                print("   2. Skip this step: Run without gradcam step")
    
    # Step 6: Attention Visualization  
    if args.step in ['all', 'attention']:
        print("\n" + "="*70)
        print("STEP 6: GENERATING ATTENTION VISUALIZATIONS")
        print("="*70)
        
        if len(image_paths) < 2:
            print("Need at least 2 images for attention visualization.")
        else:
            try:
                from attention_viz import AttentionVisualizer
                from dinov2_embedder import DINOv2Embedder
                import matplotlib.pyplot as plt
                import cv2
                from PIL import Image
                
                embedder = DINOv2Embedder(model_name=args.dinov2_model)
                visualizer = AttentionVisualizer()
                
                save_dir = os.path.join(config.RESULTS_DIR, 'attention_results')
                os.makedirs(save_dir, exist_ok=True)
                
                # Generate all-layer attention maps for 10 samples
                num_samples = min(10, len(image_paths))
                print(f"Generating all-layer attention maps for {num_samples} samples...")
                
                for i in range(num_samples):
                    img_path = image_paths[i]
                    original_img = Image.open(img_path).convert('RGB')
                    
                    # Create figure: 3 rows, 5 columns
                    # Col 0: Original (Middle)
                    # Cols 1-4: Layers 0-11 (3x4 grid)
                    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
                    
                    # Clear all axes first and turn off axis
                    for ax in axes.flatten():
                        ax.axis('off')
                    
                    # Plot Original Image at (1, 0) - Middle Left
                    axes[1, 0].imshow(original_img)
                    axes[1, 0].set_title("Original Image", fontsize=12, fontweight='bold')
                    
                    # Plot Layers 0-11 in columns 1-4
                    for layer_idx in range(12):
                        row = layer_idx // 4
                        col = (layer_idx % 4) + 1
                        
                        try:
                            res = embedder.extract_attention_maps(img_path, layer_idx=layer_idx)
                            attn_map = res['attention_map']
                            
                            # Normalize
                            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                            
                            # Resize
                            w, h = original_img.size
                            attn_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_CUBIC)
                            
                            # Apply colormap
                            heatmap = plt.cm.jet(attn_resized)[:, :, :3]
                            
                            # Overlay
                            overlay = np.array(original_img) / 255.0 * 0.5 + heatmap * 0.5
                            
                            axes[row, col].imshow(overlay)
                            axes[row, col].set_title(f"Layer {layer_idx}")
                            
                        except Exception as e:
                            print(f"Error layer {layer_idx} for image {i}: {e}")
                    
                    plt.tight_layout()
                    save_path = os.path.join(save_dir, f'all_layers_attention_sample_{i+1}.jpg')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved {os.path.basename(save_path)}")
                
                print(f"✅ Attention visualization complete! Results in {save_dir}")
            except Exception as e:
                print(f"⚠️  Attention visualization skipped due to error: {e}")
                print("   This may be a model compatibility issue.")
                print("   You can still use all other features!")
                print("   The similarity search and sample generation work perfectly.")
    
    # Step 7: Demo Search (No FAISS - Direct numpy)
    if args.step in ['all', 'demo']:
        print("\n" + "="*70)
        print("STEP 7: DEMO SIMILARITY SEARCH (5 SAMPLES)")
        print("Using both CLIP and DINOv2 embeddings")
        print("="*70)
        
        from clip_embedder import CLIPEmbedder
        from dinov2_embedder import DINOv2Embedder
        from helpers import visualize_top_k_results
        import random
        
        # Load both embeddings
        embeddings_dict = {}
        
        # Load CLIP
        clip_emb_path = os.path.join(config.EMBEDDINGS_DIR, 'clip_embeddings.h5')
        if os.path.exists(clip_emb_path):
            print("\nLoading CLIP embeddings...")
            embeddings_dict['clip'] = CLIPEmbedder.load_embeddings(clip_emb_path)
            print(f"  ✅ Loaded {len(embeddings_dict['clip'][0])} CLIP embeddings")
        else:
            print(f"⚠️  CLIP embeddings not found at {clip_emb_path}")
            print("   Run: python main_no_index.py --step embed_clip")
        
        # Load DINOv2
        dinov2_emb_path = os.path.join(config.EMBEDDINGS_DIR, 'dinov2_embeddings.h5')
        if os.path.exists(dinov2_emb_path):
            print("Loading DINOv2 embeddings...")
            embeddings_dict['dinov2'] = DINOv2Embedder.load_embeddings(dinov2_emb_path)
            print(f"  ✅ Loaded {len(embeddings_dict['dinov2'][0])} DINOv2 embeddings")
        else:
            print(f"⚠️  DINOv2 embeddings not found at {dinov2_emb_path}")
            print("   Run: python main_no_index.py --step embed_dinov2")
        
        if not embeddings_dict:
            print("\n❌ No embeddings available. Please run embedding steps first:")
            print("   python main_no_index.py --step embed_clip")
            print("   python main_no_index.py --step embed_dinov2")
            return
        
        # Use first available embeddings for determining samples
        emb_type = list(embeddings_dict.keys())[0]
        embeddings, emb_paths, metadata = embeddings_dict[emb_type]
        
        # Generate 10 sample searches
        num_samples = min(10, len(emb_paths))
        print(f"\n{'='*70}")
        print(f"Generating {num_samples} sample searches...")
        print(f"{'='*70}")
        
        # Select diverse query images (evenly spaced)
        total_images = len(emb_paths)
        query_indices = [int(i * total_images / num_samples) for i in range(num_samples)]
        
        # Process each embedding type
        for emb_name, (embeddings, paths, metadata) in embeddings_dict.items():
            print(f"\n{'='*70}")
            print(f"PROCESSING {emb_name.upper()} EMBEDDINGS")
            print(f"{'='*70}")
            
            # Normalize embeddings
            print("Normalizing embeddings...")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / (norms + 1e-8)
            
            # Create results directory for this embedding type
            samples_dir = os.path.join(config.RESULTS_DIR, 'sample_searches', emb_name)
            os.makedirs(samples_dir, exist_ok=True)
            print(f"Results will be saved to: {samples_dir}")
            
            # Process each sample
            for sample_num, query_idx in enumerate(query_indices, 1):
                query_path = paths[query_idx]
                query_embedding = embeddings_normalized[query_idx]
                
                print(f"\n  Sample {sample_num}/{num_samples}: {os.path.basename(query_path)}")
                
                # Time the similarity computation
                import time
                start_time = time.time()
                
                # Compute similarities
                top_indices, top_similarities = compute_similarities(
                    query_embedding,
                    embeddings_normalized,
                    top_k=10
                )
                
                search_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Get paths
                similar_paths = [paths[idx] for idx in top_indices]
                
                # Print timing and top 5 for brevity
                print(f"  ⏱️  Search completed in {search_time_ms:.1f}ms")
                print(f"  Top 5 matches:")
                for i, (path, sim) in enumerate(zip(similar_paths[:5], top_similarities[:5]), 1):
                    print(f"    {i}. {os.path.basename(path):40s} | Sim: {sim:.4f}")
                
                # Visualize this sample - Standard version
                sample_viz_path = os.path.join(samples_dir, f'sample_{sample_num}_results.jpg')
                visualize_top_k_results(query_path, similar_paths, top_similarities, 
                                       sample_viz_path, k=10, include_heatmaps=False)
                print(f"    ✅ Standard visualization saved")
                
                # Visualize this sample - Heatmap version (with region highlighting)
                sample_heatmap_path = os.path.join(samples_dir, f'sample_{sample_num}_heatmap.jpg')
                print(f"    🎨 Generating heatmap with region highlighting...")
                visualize_top_k_results(query_path, similar_paths, top_similarities, 
                                       sample_heatmap_path, k=10, include_heatmaps=True,
                                       device=config.DEVICE)
            
            # Create summary visualization
            print(f"\n  Creating summary grid for {emb_name.upper()}...")
            
            summary_path = os.path.join(samples_dir, 'all_samples_summary.jpg')
            create_summary_grid(query_indices, paths, embeddings_normalized, 
                              compute_similarities, summary_path)
            
            print(f"\n  ✅ {emb_name.upper()} complete!")
            print(f"     Directory: {samples_dir}")
            print(f"     Standard results: sample_1_results.jpg ... sample_{num_samples}_results.jpg")
            print(f"     Heatmap results: sample_1_heatmap.jpg ... sample_{num_samples}_heatmap.jpg")
            print(f"     Summary: all_samples_summary.jpg (shows all {num_samples} samples)")
        
        # Create comparison if both embeddings were loaded
        if len(embeddings_dict) == 2:
            print(f"\n{'='*70}")
            print("CREATING CLIP vs DINOv2 COMPARISON")
            print(f"{'='*70}")
            
            comparison_dir = os.path.join(config.RESULTS_DIR, 'sample_searches', 'comparison')
            os.makedirs(comparison_dir, exist_ok=True)
            
            create_embedding_comparison(
                query_indices,
                embeddings_dict,
                compute_similarities,
                comparison_dir
            )
            
            print(f"\n  ✅ Comparison complete!")
            print(f"     Directory: {comparison_dir}")
            print(f"     Files: comparison_sample_1.jpg ... comparison_sample_{num_samples}.jpg")
        
        print(f"\n{'='*70}")
        print("ALL SEARCHES COMPLETE!")
        print(f"{'='*70}")
        print(f"\nResults structure:")
        print(f"  results/sample_searches/")
        if 'clip' in embeddings_dict:
            print(f"  ├── clip/")
            print(f"  │   ├── sample_1_results.jpg      (standard)")
            print(f"  │   ├── sample_1_heatmap.jpg      (with similar regions)")
            print(f"  │   ├── sample_2_results.jpg")
            print(f"  │   ├── sample_2_heatmap.jpg")
            print(f"  │   ├── ...")
            print(f"  │   ├── sample_{num_samples}_results.jpg")
            print(f"  │   ├── sample_{num_samples}_heatmap.jpg")
            print(f"  │   └── all_samples_summary.jpg   ({num_samples} samples × 10 results)")
        if 'dinov2' in embeddings_dict:
            print(f"  ├── dinov2/")
            print(f"  │   ├── sample_1_results.jpg      (standard)")
            print(f"  │   ├── sample_1_heatmap.jpg      (with similar regions)")
            print(f"  │   ├── sample_2_results.jpg")
            print(f"  │   ├── sample_2_heatmap.jpg")
            print(f"  │   ├── ...")
            print(f"  │   ├── sample_{num_samples}_results.jpg")
            print(f"  │   ├── sample_{num_samples}_heatmap.jpg")
            print(f"  │   └── all_samples_summary.jpg   ({num_samples} samples × 10 results)")
        if len(embeddings_dict) == 2:
            print(f"  └── comparison/")
            print(f"      ├── comparison_sample_1.jpg")
            print(f"      ├── comparison_sample_2.jpg")
            print(f"      ├── ...")
            print(f"      └── comparison_sample_{num_samples}.jpg")
        print(f"\n🎉 Generated {num_samples} sample searches for each embedding type!")
        print(f"   Heatmap versions show highlighted similar regions!")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Visual Face Similarity Search Pipeline (No-Index Version)'
    )
    
    parser.add_argument(
        '--step',
        type=str,
        default='all',
        choices=['all', 'preprocess', 'embed_clip', 'embed_dinov2', 
                'gradcam', 'attention', 'demo', 'index'],
        help='Pipeline step to run (note: no index step in this version)'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        default=1000,
        help='Maximum number of images to process'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation'
    )
    
    parser.add_argument(
        '--clip-model',
        type=str,
        default='ViT-B/32',
        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
        help='CLIP model variant'
    )
    
    parser.add_argument(
        '--dinov2-model',
        type=str,
        default='dinov2_vitb14',
        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'],
        help='DINOv2 model variant'
    )
    
    args = parser.parse_args()
    
    run_pipeline(args)


if __name__ == "__main__":
    main()