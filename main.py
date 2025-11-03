"""
Main Execution Script (No-Index Version)
Runs face similarity search pipeline WITHOUT FAISS indexing
Uses direct numpy similarity computation instead
"""

import os
import sys

# Enable MPS fallback for unsupported operations (needed for DINOv2 on Apple Silicon)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse

import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config


def compute_similarities(
    query_embedding: np.ndarray, all_embeddings: np.ndarray, top_k: int = 10
) -> tuple:
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


def create_summary_grid(
    query_indices,
    image_paths,
    embeddings_normalized,
    compute_similarities_func,
    save_path,
):
    """
    Create a summary grid showing all sample searches

    Args:
        query_indices: List of query image indices
        image_paths: List of all image paths
        embeddings_normalized: Normalized embeddings array
        compute_similarities_func: Function to compute similarities
        save_path: Path to save the summary grid
    """
    import time

    import matplotlib.pyplot as plt
    from PIL import Image

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
        ax.set_title(
            f"Sample {sample_idx + 1}\nQuery\n({search_time:.1f}ms)",
            fontsize=10,
            fontweight="bold",
        )
        ax.axis("off")

        # Plot top-k results
        for result_idx, (top_idx, sim) in enumerate(zip(top_indices, top_similarities)):
            ax = axes[sample_idx, result_idx + 1]
            result_img = Image.open(image_paths[top_idx])
            ax.imshow(result_img)
            ax.set_title(f"#{result_idx + 1}\nSim: {sim:.3f}", fontsize=9)
            ax.axis("off")

    # Calculate average search time
    avg_time = np.mean(search_times)

    plt.suptitle(
        f"Face Similarity Search - Sample Results\nAverage Search Time: {avg_time:.1f}ms ({len(image_paths)} images)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✅ Summary grid saved: {os.path.basename(save_path)}")
    print(f"  ⏱️  Average search time: {avg_time:.1f}ms per query")


def create_embedding_comparison(
    query_indices, embeddings_dict, compute_similarities_func, save_dir
):
    """
    Create side-by-side comparison of CLIP vs DINOv2 results
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    print(f"  Generating comparison visualizations for {len(query_indices)} samples...")

    clip_data = embeddings_dict["clip"]
    dinov2_data = embeddings_dict["dinov2"]

    clip_emb, clip_paths, _ = clip_data
    dinov2_emb, dinov2_paths, _ = dinov2_data

    # Normalize
    clip_norm = clip_emb / (np.linalg.norm(clip_emb, axis=1, keepdims=True) + 1e-8)
    dinov2_norm = dinov2_emb / (
        np.linalg.norm(dinov2_emb, axis=1, keepdims=True) + 1e-8
    )

    # For each query, create comparison
    for sample_num, query_idx in enumerate(query_indices, 1):
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))

        query_path = clip_paths[query_idx]

        # CLIP results
        clip_query = clip_norm[query_idx]
        clip_top_idx, clip_sims = compute_similarities_func(
            clip_query, clip_norm, top_k=5
        )

        # DINOv2 results
        dinov2_query = dinov2_norm[query_idx]
        dinov2_top_idx, dinov2_sims = compute_similarities_func(
            dinov2_query, dinov2_norm, top_k=5
        )

        # Plot CLIP row
        axes[0, 0].imshow(Image.open(query_path))
        axes[0, 0].set_title("Query\n(CLIP)", fontweight="bold")
        axes[0, 0].axis("off")

        for i, (idx, sim) in enumerate(zip(clip_top_idx, clip_sims)):
            axes[0, i + 1].imshow(Image.open(clip_paths[idx]))
            axes[0, i + 1].set_title(f"CLIP #{i + 1}\n{sim:.3f}")
            axes[0, i + 1].axis("off")

        # Plot DINOv2 row
        axes[1, 0].imshow(Image.open(query_path))
        axes[1, 0].set_title("Query\n(DINOv2)", fontweight="bold")
        axes[1, 0].axis("off")

        for i, (idx, sim) in enumerate(zip(dinov2_top_idx, dinov2_sims)):
            axes[1, i + 1].imshow(Image.open(dinov2_paths[idx]))
            axes[1, i + 1].set_title(f"DINOv2 #{i + 1}\n{sim:.3f}")
            axes[1, i + 1].axis("off")

        plt.suptitle(
            f"Sample {sample_num}: CLIP vs DINOv2 Comparison",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        comp_path = os.path.join(save_dir, f"comparison_sample_{sample_num}.jpg")
        plt.savefig(comp_path, dpi=150, bbox_inches="tight")
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
    if args.step in ["all", "preprocess"]:
        print("\n" + "=" * 70)
        print("STEP 1: DATA PREPROCESSING")
        print("=" * 70)
        from data.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor()

        # Collect images
        if os.path.exists(config.RAW_DATA_DIR):
            image_paths = preprocessor.collect_image_paths(
                config.RAW_DATA_DIR, max_images=args.max_images
            )

            if len(image_paths) > 0:
                # Preprocess
                preprocessor.preprocess_and_save(
                    image_paths,
                    config.PROCESSED_DATA_DIR,
                    target_size=config.TARGET_IMAGE_SIZE,
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
    from data.data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    image_paths = preprocessor.collect_image_paths(config.PROCESSED_DATA_DIR)

    if len(image_paths) == 0:
        print("No processed images found. Run preprocessing first.")
        return

    # Step 2: Generate CLIP Embeddings
    if args.step in ["all", "embed_clip"]:
        print("\n" + "=" * 70)
        print("STEP 2: GENERATING CLIP EMBEDDINGS")
        print("=" * 70)
        from embeddings.clip_embedder import CLIPEmbedder

        clip_embedder = CLIPEmbedder(model_name=args.clip_model)
        clip_save_path = os.path.join(config.EMBEDDINGS_DIR, "clip_embeddings.h5")

        clip_embeddings, clip_paths = clip_embedder.embed_dataset(
            image_paths, batch_size=args.batch_size, save_path=clip_save_path
        )
        print("✅ CLIP embeddings complete!")

    # Step 3: Generate DINOv2 Embeddings
    if args.step in ["all", "embed_dinov2"]:
        print("\n" + "=" * 70)
        print("STEP 3: GENERATING DINOV2 EMBEDDINGS")
        print("=" * 70)
        from embeddings.dinov2_embedder import DINOv2Embedder

        dinov2_embedder = DINOv2Embedder(model_name=args.dinov2_model)
        dinov2_save_path = os.path.join(config.EMBEDDINGS_DIR, "dinov2_embeddings.h5")

        dinov2_embeddings, dinov2_paths = dinov2_embedder.embed_dataset(
            image_paths, batch_size=args.batch_size, save_path=dinov2_save_path
        )
        print("✅ DINOv2 embeddings complete!")

    # Step 4: SKIP FAISS - Use direct numpy search instead
    print("\n" + "=" * 70)
    print("STEP 4: SKIPPING FAISS INDEX")
    print("Using direct numpy similarity search instead")
    print("=" * 70)
    print("✅ No indexing needed - embeddings ready for search!")

    # Step 5: Grad-CAM Visualization
    if args.step in ["all", "gradcam"]:
        print("\n" + "=" * 70)
        print("STEP 5: GENERATING GRAD-CAM VISUALIZATIONS")
        print("=" * 70)

        if len(image_paths) < 2:
            print("Need at least 2 images for Grad-CAM demo.")
        else:
            try:
                import clip

                from embeddings.clip_embedder import CLIPEmbedder
                from visualization.gradcam import CLIPGradCAM

                # Load models
                clip_model, _ = clip.load(args.clip_model, device=config.DEVICE)
                embedder = CLIPEmbedder(model_name=args.clip_model)
                grad_cam = CLIPGradCAM(clip_model, device=config.DEVICE)

                # Generate for first pair
                save_dir = os.path.join(config.RESULTS_DIR, "gradcam_results")
                print(f"{image_paths[0]=}, {image_paths[-1]=}")
                grad_cam.generate_pairwise_cam(
                    image_paths[0], image_paths[1], embedder, save_dir=save_dir
                )

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
    if args.step in ["all", "attention"]:
        print("\n" + "=" * 70)
        print("STEP 6: GENERATING ATTENTION VISUALIZATIONS")
        print("=" * 70)

        if len(image_paths) < 2:
            print("Need at least 2 images for attention visualization.")
        else:
            try:
                from embeddings.dinov2_embedder import DINOv2Embedder
                from visualization.attention_viz import AttentionVisualizer

                embedder = DINOv2Embedder(model_name=args.dinov2_model)
                visualizer = AttentionVisualizer()

                save_dir = os.path.join(config.RESULTS_DIR, "attention_results")

                # Single image attention
                attn_data = embedder.extract_attention_maps(image_paths[0])
                visualizer.visualize_attention_map(
                    attn_data["attention_map"],
                    image_paths[0],
                    save_path=os.path.join(save_dir, "attention_demo.jpg"),
                )

                print(f"✅ Attention visualization complete! Results in {save_dir}")
            except Exception as e:
                print(f"⚠️  Attention visualization skipped due to error: {e}")
                print("   This may be a model compatibility issue.")
                print("   You can still use all other features!")
                print("   The similarity search and sample generation work perfectly.")

    # Step 7: Demo Search (No FAISS - Direct numpy)
    if args.step in ["all", "demo"]:
        print("\n" + "=" * 70)
        print("STEP 7: DEMO SIMILARITY SEARCH (5 SAMPLES)")
        print("Using both CLIP and DINOv2 embeddings")
        print("=" * 70)

        from embeddings.clip_embedder import CLIPEmbedder
        from embeddings.dinov2_embedder import DINOv2Embedder
        from utils.helpers import visualize_top_k_results

        # Load both embeddings
        embeddings_dict = {}

        # Load CLIP
        clip_emb_path = os.path.join(config.EMBEDDINGS_DIR, "clip_embeddings.h5")
        if os.path.exists(clip_emb_path):
            print("\nLoading CLIP embeddings...")
            embeddings_dict["clip"] = CLIPEmbedder.load_embeddings(clip_emb_path)
            print(f"  ✅ Loaded {len(embeddings_dict['clip'][0])} CLIP embeddings")
        else:
            print(f"⚠️  CLIP embeddings not found at {clip_emb_path}")
            print("   Run: python main_no_index.py --step embed_clip")

        # Load DINOv2
        dinov2_emb_path = os.path.join(config.EMBEDDINGS_DIR, "dinov2_embeddings.h5")
        if os.path.exists(dinov2_emb_path):
            print("Loading DINOv2 embeddings...")
            embeddings_dict["dinov2"] = DINOv2Embedder.load_embeddings(dinov2_emb_path)
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
        print(f"\n{'=' * 70}")
        print(f"Generating {num_samples} sample searches...")
        print(f"{'=' * 70}")

        # Select diverse query images (evenly spaced)
        total_images = len(emb_paths)
        query_indices = [
            int(i * total_images / num_samples) for i in range(num_samples)
        ]

        # Process each embedding type
        for emb_name, (embeddings, paths, metadata) in embeddings_dict.items():
            print(f"\n{'=' * 70}")
            print(f"PROCESSING {emb_name.upper()} EMBEDDINGS")
            print(f"{'=' * 70}")

            # Normalize embeddings
            print("Normalizing embeddings...")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / (norms + 1e-8)

            # Create results directory for this embedding type
            samples_dir = os.path.join(config.RESULTS_DIR, "sample_searches", emb_name)
            os.makedirs(samples_dir, exist_ok=True)
            print(f"Results will be saved to: {samples_dir}")

            # Process each sample
            for sample_num, query_idx in enumerate(query_indices, 1):
                query_path = paths[query_idx]
                query_embedding = embeddings_normalized[query_idx]

                print(
                    f"\n  Sample {sample_num}/{num_samples}: {os.path.basename(query_path)}"
                )

                # Time the similarity computation
                import time

                start_time = time.time()

                # Compute similarities
                top_indices, top_similarities = compute_similarities(
                    query_embedding, embeddings_normalized, top_k=10
                )

                search_time_ms = (
                    time.time() - start_time
                ) * 1000  # Convert to milliseconds

                # Get paths
                similar_paths = [paths[idx] for idx in top_indices]

                # Print timing and top 5 for brevity
                print(f"  ⏱️  Search completed in {search_time_ms:.1f}ms")
                print("  Top 5 matches:")
                for i, (path, sim) in enumerate(
                    zip(similar_paths[:5], top_similarities[:5]), 1
                ):
                    print(f"    {i}. {os.path.basename(path):40s} | Sim: {sim:.4f}")

                # Visualize this sample - Standard version
                sample_viz_path = os.path.join(
                    samples_dir, f"sample_{sample_num}_results.jpg"
                )
                visualize_top_k_results(
                    query_path,
                    similar_paths,
                    top_similarities,
                    sample_viz_path,
                    k=10,
                    include_heatmaps=False,
                )
                print("    ✅ Standard visualization saved")

                # Visualize this sample - Heatmap version (with region highlighting)
                sample_heatmap_path = os.path.join(
                    samples_dir, f"sample_{sample_num}_heatmap.jpg"
                )
                print("    🎨 Generating heatmap with region highlighting...")
                visualize_top_k_results(
                    query_path,
                    similar_paths,
                    top_similarities,
                    sample_heatmap_path,
                    k=10,
                    include_heatmaps=True,
                    device=config.DEVICE,
                )

            # Create summary visualization
            print(f"\n  Creating summary grid for {emb_name.upper()}...")

            summary_path = os.path.join(samples_dir, "all_samples_summary.jpg")
            create_summary_grid(
                query_indices,
                paths,
                embeddings_normalized,
                compute_similarities,
                summary_path,
            )

            print(f"\n  ✅ {emb_name.upper()} complete!")
            print(f"     Directory: {samples_dir}")
            print(
                f"     Standard results: sample_1_results.jpg ... sample_{num_samples}_results.jpg"
            )
            print(
                f"     Heatmap results: sample_1_heatmap.jpg ... sample_{num_samples}_heatmap.jpg"
            )
            print(
                f"     Summary: all_samples_summary.jpg (shows all {num_samples} samples)"
            )

        # Create comparison if both embeddings were loaded
        if len(embeddings_dict) == 2:
            print(f"\n{'=' * 70}")
            print("CREATING CLIP vs DINOv2 COMPARISON")
            print(f"{'=' * 70}")

            comparison_dir = os.path.join(
                config.RESULTS_DIR, "sample_searches", "comparison"
            )
            os.makedirs(comparison_dir, exist_ok=True)

            create_embedding_comparison(
                query_indices, embeddings_dict, compute_similarities, comparison_dir
            )

            print("\n  ✅ Comparison complete!")
            print(f"     Directory: {comparison_dir}")
            print(
                f"     Files: comparison_sample_1.jpg ... comparison_sample_{num_samples}.jpg"
            )

        print(f"\n{'=' * 70}")
        print("ALL SEARCHES COMPLETE!")
        print(f"{'=' * 70}")
        print("\nResults structure:")
        print("  results/sample_searches/")
        if "clip" in embeddings_dict:
            print("  ├── clip/")
            print("  │   ├── sample_1_results.jpg      (standard)")
            print("  │   ├── sample_1_heatmap.jpg      (with similar regions)")
            print("  │   ├── sample_2_results.jpg")
            print("  │   ├── sample_2_heatmap.jpg")
            print("  │   ├── ...")
            print(f"  │   ├── sample_{num_samples}_results.jpg")
            print(f"  │   ├── sample_{num_samples}_heatmap.jpg")
            print(
                f"  │   └── all_samples_summary.jpg   ({num_samples} samples × 10 results)"
            )
        if "dinov2" in embeddings_dict:
            print("  ├── dinov2/")
            print("  │   ├── sample_1_results.jpg      (standard)")
            print("  │   ├── sample_1_heatmap.jpg      (with similar regions)")
            print("  │   ├── sample_2_results.jpg")
            print("  │   ├── sample_2_heatmap.jpg")
            print("  │   ├── ...")
            print(f"  │   ├── sample_{num_samples}_results.jpg")
            print(f"  │   ├── sample_{num_samples}_heatmap.jpg")
            print(
                f"  │   └── all_samples_summary.jpg   ({num_samples} samples × 10 results)"
            )
        if len(embeddings_dict) == 2:
            print("  └── comparison/")
            print("      ├── comparison_sample_1.jpg")
            print("      ├── comparison_sample_2.jpg")
            print("      ├── ...")
            print(f"      └── comparison_sample_{num_samples}.jpg")
        print(f"\n🎉 Generated {num_samples} sample searches for each embedding type!")
        print("   Heatmap versions show highlighted similar regions!")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Visual Face Similarity Search Pipeline (No-Index Version)"
    )

    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=[
            "all",
            "preprocess",
            "embed_clip",
            "embed_dinov2",
            "gradcam",
            "attention",
            "demo",
        ],
        help="Pipeline step to run (note: no index step in this version)",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Maximum number of images to process",
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for embedding generation"
    )

    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP model variant",
    )

    parser.add_argument(
        "--dinov2-model",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
        help="DINOv2 model variant",
    )

    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
