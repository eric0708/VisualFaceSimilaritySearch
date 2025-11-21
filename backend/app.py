import sys
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
from pathlib import Path
from typing import Optional
import logging

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.services.viz_service import VizService
from config import Config

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service instance
viz_service = VizService.get_instance()
config = Config()

# Mount processed images directory to serve images
app.mount("/data", StaticFiles(directory="data"), name="data")
# Also mount project root if needed (be careful in production)
app.mount("/images", StaticFiles(directory=project_root), name="images")


@app.get("/models")
async def get_models():
    return {
        "models": [
            {"id": "dinov2", "name": "DINOv2 (ViT-B/14)"},
            {"id": "clip", "name": "CLIP (ViT-B/32)"}
        ]
    }

@app.get("/samples")
async def get_samples():
    """Return list of sample images from processed directory"""
    processed_dir = Path(config.PROCESSED_DATA_DIR)
    if not processed_dir.exists():
        return {"images": []}
    
    # Get first 20 images
    images = []
    count = 0
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        for img_path in processed_dir.rglob(ext):
            if count >= 20: break
            # Return path relative to project root so we can serve it via /images
            rel_path = str(img_path.relative_to(project_root))
            images.append(rel_path)
            count += 1
        if count >= 20: break
            
    return {"images": images}

@app.post("/load_model")
async def load_model(model: str = Form(...)):
    """Explicitly load a model"""
    try:
        viz_service.load_model(model)
        return {"status": "loaded", "model": model}
    except Exception as e:
        logging.error(f"Load model error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(
    image: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    model: str = Form("dinov2")
):
    """
    Search for similar images.
    Accepts either an uploaded file or an existing image path.
    """
    temp_path = None
    
    try:
        if image:
            # Save uploaded file
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            # Use processed filename
            temp_path = temp_dir / f"raw_{image.filename}"
            with temp_path.open("wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            
            # Preprocess the uploaded image
            processed_filename = f"processed_{image.filename}"
            processed_path = temp_dir / processed_filename
            
            try:
                # Preprocess (Face detect -> Crop -> Resize)
                processed_path_str = viz_service.preprocess_uploaded_image(
                    str(temp_path), 
                    output_path=str(processed_path)
                )
                query_path = processed_path_str
            except Exception as e:
                logging.error(f"Preprocessing failed: {e}")
                # Fallback to raw image if preprocessing fails really badly
                query_path = str(temp_path)
                
        elif image_path:
            # Use existing path (ensure it's absolute)
            if not os.path.isabs(image_path):
                query_path = os.path.join(project_root, image_path)
            else:
                query_path = image_path
        else:
            raise HTTPException(status_code=400, detail="No image provided")
            
        results = viz_service.search_similar(query_path, model_name=model, top_k=6)
        
        # Fix paths for frontend
        cleaned_results = []
        for res in results:
            path_obj = Path(res["path"])
            try:
                rel_path = str(path_obj.relative_to(project_root))
            except ValueError:
                rel_path = str(path_obj) 
            
            cleaned_results.append({
                "path": rel_path,
                "score": res["score"]
            })
            
        # Determine return path (relative to project root for frontend serving)
        try:
            if os.path.isabs(query_path):
                return_query_path = str(Path(query_path).relative_to(project_root))
            else:
                return_query_path = query_path
        except ValueError:
            # If relative conversion fails (e.g. temp dir outside project), try best effort
            # But here temp_uploads is relative to cwd which is project root
            return_query_path = query_path

        return {
            "results": cleaned_results,
            "query_path": return_query_path
        }
        
    except Exception as e:
        logging.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze(
    query_path: str = Form(...),
    result_path: str = Form(...),
    model: str = Form("dinov2")
):
    """
    Compute similarity matrix between query and result.
    """
    try:
        # Ensure absolute paths
        if not os.path.isabs(query_path):
            q_path = os.path.join(project_root, query_path)
        else:
            q_path = query_path
            
        if not os.path.isabs(result_path):
            r_path = os.path.join(project_root, result_path)
        else:
            r_path = result_path
            
        data = viz_service.compute_similarity_matrix(q_path, r_path, model_name=model)
        return data
        
    except Exception as e:
        logging.error(f"Analyze error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
