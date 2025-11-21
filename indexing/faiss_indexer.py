"""
FAISS Indexing Module
Builds efficient similarity search indices using FAISS
Team Member: Jin-Lian Ho
"""

import os

# Set environment variables to prevent FAISS hangs on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict
import pickle
import json

from config import Config


class FAISSIndexer:
    """Build and manage FAISS indices for similarity search"""

    def __init__(
        self, embedding_dim: int, index_type: str = "Flat", config: Config = None
    ):
        """
        Initialize FAISS indexer

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index (Flat, IVF, HNSW)
            config: Configuration object
        """
        self.config = config or Config()
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.image_paths = []
        self.metadata = {}

        print(f"Initializing FAISS indexer with {index_type} index")

    def build_index(
        self, embeddings: np.ndarray, image_paths: List[str], normalize: bool = True
    ) -> faiss.Index:
        """
        Build FAISS index from embeddings

        Args:
            embeddings: Array of embeddings [N, D]
            image_paths: List of corresponding image paths
            normalize: Whether to normalize embeddings (recommended for cosine similarity)

        Returns:
            Built FAISS index
        """
        assert embeddings.shape[0] == len(image_paths), (
            "Mismatch between embeddings and paths"
        )

        # Ensure float32
        if embeddings.dtype != np.float32:
            print("Converting embeddings to float32...")
            embeddings = embeddings.astype(np.float32)

        self.image_paths = image_paths
        n_vectors = embeddings.shape[0]

        # Normalize embeddings for cosine similarity
        if normalize:
            print("Normalizing embeddings...")
            faiss.normalize_L2(embeddings)

        print(f"Building {self.index_type} index for {n_vectors} vectors...")
        start_time = time.time()

        if self.index_type == "Flat":
            # Exact search using L2 distance (after normalization, equivalent to cosine)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            print("Adding vectors to Flat index...")
            self.index.add(embeddings)

        elif self.index_type == "IVF":
            # Inverted file index for faster search
            nlist = min(self.config.FAISS_NLIST, n_vectors // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)

            # Train the index
            print(
                f"Training IVF index with {nlist} clusters (this may take a moment)..."
            )
            self.index.train(embeddings)
            print("Adding vectors to IVF index...")
            self.index.add(embeddings)
            self.index.nprobe = self.config.FAISS_NPROBE  # Number of clusters to visit

        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graph
            M = 32  # Number of connections per layer
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            self.index.hnsw.efConstruction = 40  # Controls index construction quality
            self.index.hnsw.efSearch = 32  # Controls search quality
            print("Building HNSW graph (this may take a while)...")
            self.index.add(embeddings)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        build_time = time.time() - start_time
        print(f"✅ Index built in {build_time:.2f} seconds")
        print(f"Index contains {self.index.ntotal} vectors")

        # Store metadata
        self.metadata = {
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "num_vectors": n_vectors,
            "build_time": build_time,
            "normalized": normalize,
        }

        return self.index

    def search(
        self, query_embedding: np.ndarray, k: int = 10, return_distances: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        """
        Search for k nearest neighbors

        Args:
            query_embedding: Query embedding vector [D] or [1, D]
            k: Number of nearest neighbors to retrieve
            return_distances: Whether to return distances

        Returns:
            image_paths: List of k most similar image paths
            distances: Array of distances/similarities [k]
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize query
        faiss.normalize_L2(query_embedding)

        # Search
        start_time = time.time()
        distances, indices = self.index.search(query_embedding, k)
        search_time = time.time() - start_time

        # Convert to similarities (1 - normalized L2 distance)
        similarities = 1 - distances[0] / 2

        # Get corresponding image paths
        result_paths = [self.image_paths[idx] for idx in indices[0]]

        if return_distances:
            return result_paths, similarities
        return result_paths

    def batch_search(
        self, query_embeddings: np.ndarray, k: int = 10
    ) -> Tuple[List[List[str]], np.ndarray]:
        """
        Search for multiple queries

        Args:
            query_embeddings: Query embeddings [N, D]
            k: Number of nearest neighbors per query

        Returns:
            all_paths: List of lists of image paths
            all_similarities: Array of similarities [N, k]
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        # Normalize queries
        faiss.normalize_L2(query_embeddings)

        # Search
        distances, indices = self.index.search(query_embeddings, k)

        # Convert to similarities
        similarities = 1 - distances / 2

        # Get paths for all queries
        all_paths = []
        for query_indices in indices:
            paths = [self.image_paths[idx] for idx in query_indices]
            all_paths.append(paths)

        return all_paths, similarities

    def save_index(self, save_dir: str, index_name: str = "faiss_index"):
        """
        Save FAISS index and metadata

        Args:
            save_dir: Directory to save index
            index_name: Name prefix for saved files
        """
        os.makedirs(save_dir, exist_ok=True)

        index_path = os.path.join(save_dir, f"{index_name}.index")
        paths_path = os.path.join(save_dir, f"{index_name}_paths.pkl")
        metadata_path = os.path.join(save_dir, f"{index_name}_metadata.json")

        print(f"Saving index to {save_dir}...")

        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save image paths
        with open(paths_path, "wb") as f:
            pickle.dump(self.image_paths, f)

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✅ Index saved successfully!")
        print(f"  - Index: {index_path}")
        print(f"  - Paths: {paths_path}")
        print(f"  - Metadata: {metadata_path}")

    @classmethod
    def load_index(
        cls, load_dir: str, index_name: str = "faiss_index"
    ) -> "FAISSIndexer":
        """
        Load FAISS index from disk

        Args:
            load_dir: Directory containing saved index
            index_name: Name prefix of saved files

        Returns:
            Loaded FAISSIndexer instance
        """
        index_path = os.path.join(load_dir, f"{index_name}.index")
        paths_path = os.path.join(load_dir, f"{index_name}_paths.pkl")
        metadata_path = os.path.join(load_dir, f"{index_name}_metadata.json")

        print(f"Loading index from {load_dir}...")

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Create indexer instance
        indexer = cls(
            embedding_dim=metadata["embedding_dim"], index_type=metadata["index_type"]
        )

        # Load FAISS index
        indexer.index = faiss.read_index(index_path)

        # Load image paths
        with open(paths_path, "rb") as f:
            indexer.image_paths = pickle.load(f)

        indexer.metadata = metadata

        print(f"✅ Index loaded successfully!")
        print(f"  - Type: {metadata['index_type']}")
        print(f"  - Vectors: {metadata['num_vectors']}")
        print(f"  - Dimension: {metadata['embedding_dim']}")

        return indexer

    def benchmark(
        self, query_embeddings: np.ndarray, k: int = 10, num_queries: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark search performance

        Args:
            query_embeddings: Embeddings to use as queries
            k: Number of neighbors to retrieve
            num_queries: Number of queries to test

        Returns:
            Dictionary with benchmark results
        """
        print(f"Benchmarking with {num_queries} queries...")

        # Sample random queries
        num_queries = min(num_queries, len(query_embeddings))
        query_indices = np.random.choice(
            len(query_embeddings), num_queries, replace=False
        )
        test_queries = query_embeddings[query_indices]

        # Warm up
        _ = self.search(test_queries[0], k=k)

        # Benchmark
        start_time = time.time()
        for query in test_queries:
            _ = self.search(query, k=k)
        total_time = time.time() - start_time

        avg_latency = total_time / num_queries
        qps = num_queries / total_time

        results = {
            "num_queries": num_queries,
            "total_time": total_time,
            "avg_latency": avg_latency,
            "queries_per_second": qps,
            "k": k,
        }

        print(f"\nBenchmark Results:")
        print(f"  - Average latency: {avg_latency * 1000:.2f} ms")
        print(f"  - Queries per second: {qps:.2f}")

        return results


def compare_index_types(embeddings: np.ndarray, image_paths: List[str]):
    """
    Compare different FAISS index types

    Args:
        embeddings: Array of embeddings
        image_paths: List of image paths
    """
    print("=" * 50)
    print("Comparing FAISS Index Types")
    print("=" * 50)

    index_types = ["Flat", "IVF", "HNSW"]
    results = {}

    for idx_type in index_types:
        print(f"\n{'=' * 20} {idx_type} {'=' * 20}")

        indexer = FAISSIndexer(embedding_dim=embeddings.shape[1], index_type=idx_type)

        # Build index
        indexer.build_index(embeddings, image_paths)

        # Benchmark
        benchmark_results = indexer.benchmark(embeddings, k=10, num_queries=100)
        results[idx_type] = benchmark_results

    # Print comparison
    print("\n" + "=" * 50)
    print("Comparison Summary")
    print("=" * 50)
    for idx_type, res in results.items():
        print(
            f"{idx_type:10s}: {res['avg_latency'] * 1000:6.2f} ms/query, "
            f"{res['queries_per_second']:6.2f} QPS"
        )


def main():
    """Demo of FAISS indexing"""
    print("=" * 50)
    print("FAISS Index Building")
    print("=" * 50)

    config = Config()

    # Load embeddings (assuming they exist from previous steps)
    embeddings_path = os.path.join(config.EMBEDDINGS_DIR, "clip_embeddings.h5")

    if not os.path.exists(embeddings_path):
        print(f"Embeddings not found: {embeddings_path}")
        print("Please run clip_embedder.py first.")
        return

    # Load embeddings
    from embeddings.clip_embedder import CLIPEmbedder

    embeddings, image_paths, metadata = CLIPEmbedder.load_embeddings(embeddings_path)

    print(f"\nLoaded {len(embeddings)} embeddings")

    # Build index
    indexer = FAISSIndexer(
        embedding_dim=embeddings.shape[1],
        index_type="IVF",  # Change to "Flat" or "HNSW" to try different types
    )

    indexer.build_index(embeddings, image_paths)

    # Save index
    save_dir = config.FAISS_INDEX_DIR
    indexer.save_index(save_dir, index_name="clip_ivf_index")

    # Test search
    print("\nTesting similarity search...")
    query_idx = 0
    query_embedding = embeddings[query_idx]

    similar_paths, similarities = indexer.search(query_embedding, k=10)

    print(f"\nQuery image: {image_paths[query_idx]}")
    print("\nTop 10 similar images:")
    for i, (path, sim) in enumerate(zip(similar_paths, similarities), 1):
        print(f"{i}. {os.path.basename(path)}: {sim:.4f}")

    # Benchmark
    benchmark_results = indexer.benchmark(embeddings, k=10, num_queries=100)

    # Compare index types (optional, comment out if too slow)
    # compare_index_types(embeddings[:1000], image_paths[:1000])

    print("\n✅ FAISS indexing complete!")


if __name__ == "__main__":
    main()
