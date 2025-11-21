import logging
import os
import sys
import uuid

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config  # noqa: E402
from data.data_preprocessing import DataPreprocessor  # noqa: E402
from embeddings.clip_embedder import CLIPEmbedder  # noqa: E402
from embeddings.dinov2_embedder import DINOv2Embedder  # noqa: E402
from visualization.attention_viz import AttentionVisualizer  # noqa: E402
from visualization.gradcam import CLIPGradCAM  # noqa: E402
from visualization.similarity_heatmap import SimilarityHeatmapGenerator  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VizService:
    _instance = None

    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        self.heatmap_generators = {}  # model_name -> generator
        self.search_embedders = {}  # model_name -> embedder
        self.attention_visualizer = None
        self.cached_embeddings = {}  # model_name -> (embeddings, paths, metadata)
        # Cache for image paths to ensure consistency across requests
        # This replaces the paths from the H5 file which might be stale (absolute paths from another machine)
        self.cached_image_paths = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_image_paths(self):
        """Helper to load image paths from filesystem"""
        logger.info("Refreshing image paths from filesystem...")
        preprocessor = DataPreprocessor(self.config)
        if os.path.exists(self.config.PROCESSED_DATA_DIR):
            paths = preprocessor.collect_image_paths(self.config.PROCESSED_DATA_DIR)
            logger.info(
                f"Loaded {len(paths)} image paths from {self.config.PROCESSED_DATA_DIR}"
            )
            return paths
        else:
            logger.error(
                f"Processed data dir not found: {self.config.PROCESSED_DATA_DIR}"
            )
            return []

    def load_model(self, model_name):
        """Explicitly load model and embeddings"""
        logger.info(f"Loading model resources for: {model_name}")

        if model_name == "gradcam":
            # Grad-CAM relies on CLIP model and embeddings
            return self.load_model("clip")

        if model_name == "attention":
            # Attention relies on DINOv2 model
            return self.load_model("dinov2")

        # 0. Refresh image paths from filesystem to ensure validity
        if self.cached_image_paths is None:
            self.cached_image_paths = self._load_image_paths()

        # 1. Initialize Search Embedder (loads model weights)
        self.get_embedder(model_name)

        # 2. Load Embeddings File (into memory if not cached)
        # Check if already cached
        if model_name in self.cached_embeddings:
            logger.info(f"Using cached embeddings for {model_name}")
        else:
            emb_dir = self.config.EMBEDDINGS_DIR
            emb_file = (
                "clip_embeddings.h5" if model_name == "clip" else "dinov2_embeddings.h5"
            )
            emb_path = os.path.join(emb_dir, emb_file)

            if not os.path.exists(emb_path):
                logger.warning(f"Embeddings file missing: {emb_path}")
            else:
                # Trigger load and cache
                if model_name == "clip":
                    self.cached_embeddings[model_name] = CLIPEmbedder.load_embeddings(
                        emb_path
                    )
                else:
                    self.cached_embeddings[model_name] = DINOv2Embedder.load_embeddings(
                        emb_path
                    )

        # 3. Initialize Heatmap Generator
        # Map for heatmap generator
        gen_model_name = "dinov2_vitb14"
        if model_name == "clip":
            # Fallback or specific mapping
            gen_model_name = "dinov2_vitb14"

        self.get_generator(gen_model_name)

        return True

    def get_generator(self, model_name="dinov2_vitb14"):
        """Get or create a SimilarityHeatmapGenerator for the specified model"""
        # Map frontend model names to backend model names if necessary
        if model_name == "dinov2":
            model_name = "dinov2_vitb14"
        elif model_name == "clip":
            model_name = "dinov2_vitb14"

        if model_name not in self.heatmap_generators:
            logger.info(f"Initializing heatmap generator for {model_name}")

            # Try to reuse model from search embedder if available
            existing_model = None
            if "dinov2" in model_name:
                # Ensure DINOv2 embedder is loaded to reuse its model
                embedder = self.get_embedder("dinov2")
                existing_model = embedder.model

            self.heatmap_generators[model_name] = SimilarityHeatmapGenerator(
                model_name=model_name, device=self.device, model=existing_model
            )

        return self.heatmap_generators[model_name]

    def get_embedder(self, model_type):
        """Get or create an embedder for search"""
        if model_type not in self.search_embedders:
            logger.info(f"Initializing embedder for {model_type}")
            if model_type == "clip":
                self.search_embedders[model_type] = CLIPEmbedder(device=self.device)
            elif model_type == "dinov2":
                self.search_embedders[model_type] = DINOv2Embedder(device=self.device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        return self.search_embedders[model_type]

    def get_attention_visualizer(self):
        if self.attention_visualizer is None:
            self.attention_visualizer = AttentionVisualizer(self.config)
        return self.attention_visualizer

    def compute_similarity_matrix(self, query_path, result_path, model_name="dinov2"):
        """
        Compute the raw patch-to-patch similarity matrix.

        Returns:
            matrix: Flattened list of similarity scores [N_query * N_result]
            grid_size: Tuple (h, w) of the patch grid
            dimensions: Dict with 'query_patches', 'result_patches' counts
        """
        if model_name == "attention":
            # For Attention model, we want to return the actual attention map for the specified layer
            # The result_path contains the filename layer_X.jpg
            try:
                filename = os.path.basename(result_path)
                # Format is layer_{idx}.jpg
                if "layer_" in filename:
                    layer_part = filename.split("layer_")[1]
                    layer_idx = int(layer_part.split(".")[0])
                else:
                    # Default to last layer if can't parse
                    layer_idx = -1

                embedder = self.get_embedder("dinov2")
                res = embedder.extract_attention_maps(query_path, layer_idx=layer_idx)
                attn_map = res["attention_map"]  # [H, W] numpy array

                flat_attn = attn_map.flatten().tolist()
                grid_size = int(np.sqrt(len(flat_attn)))

                # Broadcast to create a fake "matrix" where every row is the same Attention Map
                # This ensures that "max" reduction in frontend gives back the Attention Map
                matrix = [flat_attn for _ in range(len(flat_attn))]

                return {
                    "matrix": matrix,
                    "shape": [len(flat_attn), len(flat_attn)],
                    "grid_size": grid_size,
                }
            except Exception as e:
                logger.error(f"Error computing attention matrix: {e}")
                # Fallback to zeros
                dummy_size = 16 * 16
                return {
                    "matrix": [0.0] * dummy_size,
                    "shape": [dummy_size, dummy_size],
                    "grid_size": 16,
                }

        if model_name == "gradcam":
            # Special handling for Grad-CAM
            # Use CLIP embedder to get the model and process images
            embedder = self.get_embedder("clip")

            # Initialize Grad-CAM with the CLIP model from embedder
            grad_cam = CLIPGradCAM(embedder.model, device=self.device)

            try:
                # 1. Get query embedding (target for CAM)
                query_emb_np = embedder.embed_image(query_path)
                query_emb_tensor = (
                    torch.from_numpy(query_emb_np).unsqueeze(0).to(self.device)
                )

                # 2. Prepare result image tensor (input for CAM)
                result_img = Image.open(result_path).convert("RGB")
                result_tensor = embedder.preprocess(result_img).unsqueeze(
                    0
                )  # [1, 3, 224, 224]

                # 3. Generate CAM on Result image, targeting Query embedding
                cam = grad_cam.generate_cam(
                    result_tensor, target_embedding=query_emb_tensor
                )

                flat_cam = cam.flatten().tolist()
                grid_size = int(np.sqrt(len(flat_cam)))

                if grid_size * grid_size != len(flat_cam):
                    logger.warning(f"CAM shape {cam.shape} is not square")

                # Broadcast to create a fake "matrix" where every row is the same CAM
                # This ensures that "max" reduction in frontend gives back the CAM
                matrix = [flat_cam for _ in range(len(flat_cam))]

                return {
                    "matrix": matrix,
                    "shape": [len(flat_cam), len(flat_cam)],
                    "grid_size": grid_size,
                }
            finally:
                grad_cam.remove_hooks()

        generator = self.get_generator(model_name)

        try:
            # Extract features
            # These return [num_patches, dim]
            query_features, _ = generator.extract_patch_features(query_path)
            result_features, _ = generator.extract_patch_features(result_path)

            # Normalize
            query_norm = query_features / (
                np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8
            )
            result_norm = result_features / (
                np.linalg.norm(result_features, axis=1, keepdims=True) + 1e-8
            )

            # Compute matrix: [num_patches_q, num_patches_r]
            similarity_matrix = np.dot(query_norm, result_norm.T)

            # Determine grid size
            # Assuming square grid
            num_patches = len(query_features)
            grid_size = int(np.sqrt(num_patches))

            return {
                "matrix": similarity_matrix.tolist(),  # JSON serializable
                "shape": [len(query_features), len(result_features)],
                "grid_size": grid_size,
            }
        except Exception as e:
            logger.error(f"Error computing similarity matrix for {model_name}: {e}")
            # Fallback to zeros
            dummy_size = 16 * 16
            return {
                "matrix": [0.0] * dummy_size,
                "shape": [dummy_size, dummy_size],
                "grid_size": 16,
            }

    def preprocess_uploaded_image(self, image_path, output_path=None):
        """
        Preprocess uploaded image:
        1. Try to detect and align face using MTCNN
        2. If failed or no face, center crop and resize
        """
        try:
            # Try MTCNN first for face detection
            try:
                from facenet_pytorch import MTCNN

                mtcnn = MTCNN(image_size=224, margin=0, device=self.device)

                img = Image.open(image_path).convert("RGB")
                img_aligned = mtcnn(img)

                if img_aligned is not None:
                    logger.info("Face detected and aligned successfully")
                    # Convert tensor to PIL
                    img_processed = transforms.ToPILImage()(img_aligned)
                else:
                    logger.warning("No face detected by MTCNN, falling back to resize")
                    img_processed = img.resize((224, 224), Image.LANCZOS)
            except ImportError:
                logger.warning("facenet-pytorch not found, skipping face detection")
                img = Image.open(image_path).convert("RGB")
                img_processed = img.resize((224, 224), Image.LANCZOS)
            except Exception as e:
                logger.error(f"MTCNN error: {e}")
                img = Image.open(image_path).convert("RGB")
                img_processed = img.resize((224, 224), Image.LANCZOS)

            # Save processed image
            if output_path:
                img_processed.save(output_path)
                return output_path
            else:
                return img_processed

        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            # Fallback to simple resize if everything fails
            img = Image.open(image_path).convert("RGB")
            img = img.resize((224, 224), Image.LANCZOS)
            if output_path:
                img.save(output_path)
                return output_path
            return img

    def search_similar(self, query_path, model_name="dinov2", top_k=5):
        """
        Search for similar images using global embeddings.
        Excludes results with similarity > 0.99 to avoid returning the query image itself.
        """
        if model_name == "gradcam":
            model_name = "clip"

        embedder = self.get_embedder(model_name)

        # Embed query
        query_embedding = embedder.embed_image(query_path)

        # Load all embeddings
        if model_name not in self.cached_embeddings:
            logger.info(f"Embeddings for {model_name} not cached, loading now...")
            self.load_model(model_name)

        if model_name in self.cached_embeddings:
            all_embeddings, _, _ = self.cached_embeddings[model_name]
        else:
            logger.error(f"Failed to load embeddings for {model_name}")
            return []

        # Lazy initialization of cached_image_paths if needed
        if self.cached_image_paths is None:
            self.cached_image_paths = self._load_image_paths()

        image_paths = self.cached_image_paths

        if len(image_paths) == 0:
            logger.error("No image paths found! Check processed data directory.")
            return []

        if len(image_paths) != len(all_embeddings):
            logger.warning(
                f"Count mismatch: {len(image_paths)} files vs {len(all_embeddings)} embeddings. "
                "Using minimum length."
            )
            min_len = min(len(image_paths), len(all_embeddings))
            all_embeddings = all_embeddings[:min_len]

        # Compute similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        all_norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        all_normalized = all_embeddings / (all_norms + 1e-8)
        sims = np.dot(all_normalized, query_norm)

        top_indices = np.argsort(sims)[::-1]

        results = []
        count = 0
        for idx in top_indices:
            if count >= top_k:
                break

            sim_score = float(sims[idx])

            if sim_score > 0.9999:
                continue

            if idx < len(image_paths):
                results.append({"path": image_paths[idx], "score": sim_score})
                count += 1
            else:
                logger.warning(
                    f"Index {idx} out of bounds for image_paths list (len={len(image_paths)})"
                )

        return results

    def generate_layer_attention_results(self, query_path):
        """
        Generate attention maps for all layers of the DINOv2 model.
        Returns results in a format compatible with frontend gallery.
        """
        embedder = self.get_embedder("dinov2")
        visualizer = self.get_attention_visualizer()

        # Directory to save temporary attention maps
        # We use a unique ID for this batch to avoid collisions if multiple users (though local demo)
        session_id = str(uuid.uuid4())[:8]
        temp_dir = os.path.join(project_root, "data", "temp_attention", session_id)
        os.makedirs(temp_dir, exist_ok=True)

        results = []

        # Layers 0 to 11
        for layer_idx in range(12):
            try:
                # Extract attention map
                res = embedder.extract_attention_maps(query_path, layer_idx=layer_idx)
                attn_map = res["attention_map"]

                # Save visualization
                filename = f"layer_{layer_idx}.jpg"
                save_path = os.path.join(temp_dir, filename)

                # Use visualizer to save overlay
                visualizer.visualize_attention_map(
                    attn_map, query_path, save_path=save_path, alpha=0.5, colormap="jet"
                )

                # Create result entry
                # score = layer_idx (to display "Sim: 0.000" -> "Layer 0")
                results.append({"path": save_path, "score": float(layer_idx)})

            except Exception as e:
                logger.error(f"Error generating attention for layer {layer_idx}: {e}")

        return results
