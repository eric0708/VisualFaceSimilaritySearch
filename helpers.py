"""
Utility Functions
Helper functions for the face similarity search project
"""
import os
import json
import time
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def ensure_dir(directory: str):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict, filepath: str):
    """Save dictionary as JSON"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def timer(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2 + 1e-8)


def visualize_top_k_results(query_path: str,
                           result_paths: List[str],
                           similarities: List[float],
                           save_path: str,
                           k: int = 10,
                           include_heatmaps: bool = False,
                           device: str = 'mps'):
    """
    Visualize top-k retrieval results
    
    Args:
        query_path: Path to query image
        result_paths: List of retrieved image paths
        similarities: List of similarity scores
        save_path: Path to save visualization
        k: Number of results to show
        include_heatmaps: Whether to include similarity region heatmaps
        device: Device for heatmap computation
    """
    if include_heatmaps:
        # Use enhanced visualization with heatmaps
        try:
            from similarity_heatmap import visualize_top_k_results_with_heatmaps
            visualize_top_k_results_with_heatmaps(
                query_path, result_paths, similarities, save_path, k=k, device=device
            )
            return
        except Exception as e:
            print(f"      Warning: Heatmap generation failed ({e}), using standard visualization")
            # Fall back to standard visualization
    
    # Standard visualization (original)
    k = min(k, len(result_paths))
    
    # Create figure
    cols = 5
    rows = (k + cols) // cols + 1  # +1 for query row
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()
    
    # Show query image
    query_img = Image.open(query_path)
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Hide remaining cells in first row
    for i in range(1, cols):
        axes[i].axis('off')
    
    # Show results
    for i in range(k):
        ax_idx = cols + i
        img = Image.open(result_paths[i])
        axes[ax_idx].imshow(img)
        axes[ax_idx].set_title(f'#{i+1}\nSim: {similarities[i]:.3f}', 
                              fontsize=10)
        axes[ax_idx].axis('off')
    
    # Hide remaining cells
    for i in range(cols + k, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Visualization saved to {save_path}")


def calculate_retrieval_metrics(retrieved_ids: List[str],
                                relevant_ids: List[str],
                                k: int = 10) -> Dict[str, float]:
    """
    Calculate retrieval evaluation metrics
    
    Args:
        retrieved_ids: List of retrieved image IDs
        relevant_ids: List of ground truth relevant image IDs
        k: Number of top results to consider
        
    Returns:
        Dictionary with metrics (precision, recall, MAP)
    """
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    # Precision@k
    precision = len(retrieved_set & relevant_set) / k if k > 0 else 0.0
    
    # Recall@k
    recall = len(retrieved_set & relevant_set) / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    # Average Precision
    ap = 0.0
    num_hits = 0
    for i, ret_id in enumerate(retrieved_ids[:k], 1):
        if ret_id in relevant_set:
            num_hits += 1
            ap += num_hits / i
    
    ap = ap / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    return {
        'precision@k': precision,
        'recall@k': recall,
        'average_precision': ap
    }


def create_summary_report(metrics: Dict[str, Any], save_path: str):
    """
    Create a summary report of results
    
    Args:
        metrics: Dictionary containing various metrics
        save_path: Path to save report
    """
    report = []
    report.append("=" * 60)
    report.append("FACE SIMILARITY SEARCH - SUMMARY REPORT")
    report.append("=" * 60)
    report.append("")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            report.append(f"{key:30s}: {value:.4f}")
        elif isinstance(value, int):
            report.append(f"{key:30s}: {value}")
        else:
            report.append(f"{key:30s}: {value}")
    
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    # Print to console
    print(report_text)
    
    # Save to file
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        f.write(report_text)


def batch_process(items: List[Any], 
                 process_func,
                 batch_size: int = 32,
                 desc: str = "Processing") -> List[Any]:
    """
    Process items in batches
    
    Args:
        items: List of items to process
        process_func: Function to apply to each batch
        batch_size: Size of each batch
        desc: Description for progress bar
        
    Returns:
        List of processed results
    """
    from tqdm import tqdm
    
    results = []
    for i in tqdm(range(0, len(items), batch_size), desc=desc):
        batch = items[i:i+batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    
    return results


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test similarity computation
    vec1 = np.random.randn(512)
    vec2 = np.random.randn(512)
    sim = compute_cosine_similarity(vec1, vec2)
    print(f"Cosine similarity: {sim:.4f}")
    
    # Test metrics calculation
    retrieved = ['img1', 'img2', 'img3', 'img4', 'img5']
    relevant = ['img1', 'img3', 'img6']
    metrics = calculate_retrieval_metrics(retrieved, relevant, k=5)
    print(f"Metrics: {metrics}")
    
    print("✅ Utilities test complete!")