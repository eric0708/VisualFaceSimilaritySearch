"""
Main Execution Script
Runs the complete face similarity search pipeline
"""
import os
import argparse
from config import Config


def run_pipeline(args):
    """Run the complete pipeline"""
    config = Config()
    config.create_directories()
    
    print("=" * 70)
    print("VISUAL FACE SIMILARITY SEARCH WITH EXPLAINABLE DEEP EMBEDDINGS")
    print("=" * 70)
    
    # Step 1: Data Preprocessing
    if args.step in ['all', 'preprocess']:
        print("\n" + "="*70)
        print("STEP 1: DATA PREPROCESSING")
        print("="*70)
        from data.data_preprocessing import DataPreprocessor
        
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
    from data.data_preprocessing import DataPreprocessor
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
        from embeddings.clip_embedder import CLIPEmbedder
        
        clip_embedder = CLIPEmbedder(model_name=args.clip_model)
        clip_save_path = os.path.join(config.EMBEDDINGS_DIR, 'clip_embeddings.h5')
        
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
        from embeddings.dinov2_embedder import DINOv2Embedder
        
        dinov2_embedder = DINOv2Embedder(model_name=args.dinov2_model)
        dinov2_save_path = os.path.join(config.EMBEDDINGS_DIR, 'dinov2_embeddings.h5')
        
        dinov2_embeddings, dinov2_paths = dinov2_embedder.embed_dataset(
            image_paths,
            batch_size=args.batch_size,
            save_path=dinov2_save_path
        )
        print("✅ DINOv2 embeddings complete!")
    
    # Step 4: Build FAISS Index
    if args.step in ['all', 'index']:
        print("\n" + "="*70)
        print("STEP 4: BUILDING FAISS INDEX")
        print("="*70)
        from indexing.faiss_indexer import FAISSIndexer
        from embeddings.clip_embedder import CLIPEmbedder
        
        # Load CLIP embeddings
        clip_emb_path = os.path.join(config.EMBEDDINGS_DIR, 'clip_embeddings.h5')
        if not os.path.exists(clip_emb_path):
            print("CLIP embeddings not found. Run embed_clip step first.")
            return
        
        embeddings, paths, metadata = CLIPEmbedder.load_embeddings(clip_emb_path)
        
        # Build index
        indexer = FAISSIndexer(
            embedding_dim=embeddings.shape[1],
            index_type=args.index_type
        )
        
        indexer.build_index(embeddings, paths)
        
        # Save index
        indexer.save_index(config.FAISS_INDEX_DIR, index_name=f"clip_{args.index_type.lower()}_index")
        
        # Benchmark
        indexer.benchmark(embeddings, k=10, num_queries=100)
        print("✅ FAISS index complete!")
    
    # Step 5: Grad-CAM Visualization
    if args.step in ['all', 'gradcam']:
        print("\n" + "="*70)
        print("STEP 5: GENERATING GRAD-CAM VISUALIZATIONS")
        print("="*70)
        
        if len(image_paths) < 2:
            print("Need at least 2 images for Grad-CAM demo.")
        else:
            from visualization.gradcam import CLIPGradCAM
            from embeddings.clip_embedder import CLIPEmbedder
            import clip
            
            # Load models
            clip_model, _ = clip.load(args.clip_model, device=config.DEVICE)
            embedder = CLIPEmbedder(model_name=args.clip_model)
            grad_cam = CLIPGradCAM(clip_model, device=config.DEVICE)
            
            # Generate for first pair
            save_dir = os.path.join(config.RESULTS_DIR, 'gradcam_results')
            grad_cam.generate_pairwise_cam(
                image_paths[0], image_paths[1], embedder, save_dir=save_dir
            )
            
            grad_cam.remove_hooks()
            print(f"✅ Grad-CAM complete! Results in {save_dir}")
    
    # Step 6: Attention Visualization
    if args.step in ['all', 'attention']:
        print("\n" + "="*70)
        print("STEP 6: GENERATING ATTENTION VISUALIZATIONS")
        print("="*70)
        
        if len(image_paths) < 2:
            print("Need at least 2 images for attention visualization.")
        else:
            from visualization.attention_viz import AttentionVisualizer
            from embeddings.dinov2_embedder import DINOv2Embedder
            
            embedder = DINOv2Embedder(model_name=args.dinov2_model)
            visualizer = AttentionVisualizer()
            
            save_dir = os.path.join(config.RESULTS_DIR, 'attention_results')
            
            # Single image attention
            attn_data = embedder.extract_attention_maps(image_paths[0])
            visualizer.visualize_attention_map(
                attn_data['attention_map'],
                image_paths[0],
                save_path=os.path.join(save_dir, 'attention_demo.jpg')
            )
            
            print(f"✅ Attention visualization complete! Results in {save_dir}")
    
    # Step 7: Demo Search
    if args.step in ['all', 'demo']:
        print("\n" + "="*70)
        print("STEP 7: DEMO SIMILARITY SEARCH")
        print("="*70)
        
        from indexing.faiss_indexer import FAISSIndexer
        from embeddings.clip_embedder import CLIPEmbedder
        from utils.helpers import visualize_top_k_results
        
        # Load index
        index_path = config.FAISS_INDEX_DIR
        index_name = f"clip_{args.index_type.lower()}_index"
        
        if not os.path.exists(os.path.join(index_path, f"{index_name}.index")):
            print("Index not found. Run index step first.")
            return
        
        indexer = FAISSIndexer.load_index(index_path, index_name)
        embedder = CLIPEmbedder(model_name=args.clip_model)
        
        # Search for first image
        query_path = image_paths[0]
        query_emb = embedder.embed_image(query_path)
        
        similar_paths, similarities = indexer.search(query_emb, k=10)
        
        print(f"\nQuery: {os.path.basename(query_path)}")
        print("\nTop 10 similar images:")
        for i, (path, sim) in enumerate(zip(similar_paths, similarities), 1):
            print(f"{i:2d}. {os.path.basename(path):40s} | Similarity: {sim:.4f}")
        
        # Visualize
        viz_path = os.path.join(config.RESULTS_DIR, 'demo_search_results.jpg')
        visualize_top_k_results(query_path, similar_paths, similarities, viz_path, k=10)
        
        print(f"✅ Demo search complete! Visualization saved to {viz_path}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Visual Face Similarity Search Pipeline'
    )
    
    parser.add_argument(
        '--step',
        type=str,
        default='all',
        choices=['all', 'preprocess', 'embed_clip', 'embed_dinov2', 
                'index', 'gradcam', 'attention', 'demo'],
        help='Pipeline step to run'
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
    
    parser.add_argument(
        '--index-type',
        type=str,
        default='IVF',
        choices=['Flat', 'IVF', 'HNSW'],
        help='FAISS index type'
    )
    
    args = parser.parse_args()
    
    run_pipeline(args)


if __name__ == "__main__":
    main()
