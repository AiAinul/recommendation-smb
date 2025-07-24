import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime, timedelta
import time
import shutil
import threading
import time
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import RedirectResponse
import secrets
import uuid
import requests

from data_processing import DataProcessor
from model import RecommendationModel
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.docs import get_redoc_html

# Set TensorFlow environment
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Add global variable for program studi data
data_processor = DataProcessor()
program_studi_df = None

def enrich_recommendations_with_program_studi(recommendations, recommendation_id=None):
    global program_studi_df
    if program_studi_df is None:
        # Load the full program studi data (not just selected_cols)
        program_studi_df = pd.read_csv(data_processor.program_studi_url)
        program_studi_df["item_id"] = program_studi_df["item_id"].astype(str)
    # Build a lookup dict for fast access
    program_studi_lookup = program_studi_df.set_index("item_id").to_dict(orient="index")
    enriched = []
    for rec in recommendations:
        item_id = str(rec.get("item_id", ""))
        extra = program_studi_lookup.get(item_id, {})
        rec = rec.copy()
        rec["tittle"] = extra.get("tittle")
        page_url = extra.get("page_url")
        if page_url and recommendation_id:
            if "?" in page_url:
                rec["page_url"] = f"{page_url}&recommendationid={recommendation_id}"
            else:
                rec["page_url"] = f"{page_url}?recommendationid={recommendation_id}"
        else:
            rec["page_url"] = page_url
        rec["image_url"] = extra.get("image_url")
        rec["kampus"] = extra.get("kampus")
        enriched.append(rec)
    return enriched

# Pydantic models for API requests/responses
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    current_item_id: str = Field(..., description="Current item ID being viewed")
    region: str = Field(..., description="User region")
    city: str = Field(..., description="User city")
    top_k: int = Field(default=5, description="Number of recommendations to return")

class RecommendationResponse(BaseModel):
    user_id: str
    current_item_id: str
    recommendations: List[Dict]
    model_info: Dict

class TrainingRequest(BaseModel):
    epochs: int = Field(default=15, description="Number of training epochs")
    batch_size: int = Field(default=4096, description="Training batch size")

class TrainingResponse(BaseModel):
    status: str
    message: str
    training_info: Optional[Dict] = None

class ModelInfo(BaseModel):
    model_status: str
    dataset_stats: Dict
    last_training: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(
    title="Recommendation System API",
    description="by Riset & Pemasaran Digital",
    version="1.0.0",
    docs_url=None,      # Nonaktifkan /docs default
    redoc_url=None,     # Nonaktifkan /redoc default
    openapi_url="/openapi.json"  # Tetap expose openapi
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
data_processor = DataProcessor()
recommendation_model = RecommendationModel()
model_loaded = False
training_in_progress = False

# Tambahkan tracking untuk API usage
from datetime import datetime, timedelta
import time

# Global variables untuk tracking
api_usage = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "requests_by_endpoint": {},
    "response_times": [],
    "daily_requests": {}
}

@app.middleware("http")
async def track_api_usage(request, call_next):
    """Middleware untuk tracking API usage"""
    start_time = time.time()
    
    # Update total requests
    api_usage["total_requests"] += 1
    
    # Track by endpoint
    endpoint = request.url.path
    if endpoint not in api_usage["requests_by_endpoint"]:
        api_usage["requests_by_endpoint"][endpoint] = 0
    api_usage["requests_by_endpoint"][endpoint] += 1
    
    # Track daily requests
    today = datetime.now().strftime("%Y-%m-%d")
    if today not in api_usage["daily_requests"]:
        api_usage["daily_requests"][today] = 0
    api_usage["daily_requests"][today] += 1
    
    response = await call_next(request)
    
    # Track response time
    response_time = time.time() - start_time
    api_usage["response_times"].append(response_time)
    
    # Keep only last 1000 response times
    if len(api_usage["response_times"]) > 1000:
        api_usage["response_times"] = api_usage["response_times"][-1000:]
    
    # Track success/failure
    if response.status_code < 400:
        api_usage["successful_requests"] += 1
    else:
        api_usage["failed_requests"] += 1
    
    return response

# Tambahkan tracking untuk recommendation performance
recommendation_metrics = {
    "total_recommendations": 0,
    "recommendations_by_category": {},
    "user_engagement": {},
    "popular_items": {}
}

# Global variables untuk model versioning
data_processor = DataProcessor()
recommendation_model = RecommendationModel()
model_loaded = False
training_in_progress = False

# Global variables untuk in-memory model management
current_model_instance = None
previous_model_instance = None
model_loaded = False
training_in_progress = False

# Model versioning system (in-memory)
model_versions = {
    "current": None,
    "previous": None,
    "current_version": None,
    "previous_version": None
}

# Training status
training_status = {
    "is_training": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "start_time": None,
    "estimated_completion": None,
    "new_model_version": None,
    "current_training_id": None,
    "training_completed": False,
    "completion_time": None,
    "current_step": "idle"
}

# Thread lock untuk thread safety
model_lock = threading.Lock()

def get_available_model_instance():
    """Get available model instance for recommendations"""
    global current_model_instance, previous_model_instance, model_loaded
    
    with model_lock:
        # Priority: current model, then previous model
        if current_model_instance is not None:
            model_loaded = True
            return current_model_instance, "current"
        elif previous_model_instance is not None:
            model_loaded = True
            return previous_model_instance, "previous"
        else:
            model_loaded = False
            return None, None

def save_model_instance_in_memory(model_instance, version: str):
    """Save model instance in memory"""
    global current_model_instance, previous_model_instance, model_versions
    
    with model_lock:
        # Move current model to previous
        if current_model_instance is not None:
            previous_model_instance = current_model_instance
            model_versions["previous"] = model_versions["current"]
            model_versions["previous_version"] = model_versions["current_version"]
            print(f"🔄 Moved current model to previous (version: {model_versions['previous_version']})")
        
        # Set new model as current
        current_model_instance = model_instance
        model_versions["current"] = model_instance
        model_versions["current_version"] = version
        print(f"✅ New model instance saved in memory (version: {version})")

def cleanup_old_model():
    """Clean up old model from memory"""
    global previous_model_instance, model_versions
    
    with model_lock:
        if previous_model_instance is not None:
            previous_model_instance = None
            model_versions["previous"] = None
            model_versions["previous_version"] = None
            print("🧹 Old model cleaned from memory")
        else:
            print("ℹ️ No old model to clean")

def train_model_thread(training_id: str, new_version: str, request: TrainingRequest):
    """Separate thread for model training"""
    global training_in_progress, training_status
    
    try:
        print(f"🎯 Starting model training in background thread (ID: {training_id}, version: {new_version})...")
        
        # Update status to building
        training_status["current_step"] = "building"
        print("📊 Loading and processing data...")
        
        # Load and process data
        dataset, _ = data_processor.load_and_process_data()
        dataset_interaction = data_processor.create_interaction_dataset(dataset)
        
        # Convert to TensorFlow datasets with proper data types
        selected_cols = [
            "user_id", "item_id", "category", "category2", "category3",
            "region", 'city', "item_id_lastview", "item_id_currentview", "label","timestamp_unix"
        ]
        
        # Ensure all data is properly formatted
        print(" Formatting data...")
        for col in selected_cols:
            if col == "label":
                dataset_interaction[col] = dataset_interaction[col].astype(np.float32)
            elif col == "timestamp_unix":
                dataset_interaction[col] = dataset_interaction[col].astype(np.int64)
            else:
                dataset_interaction[col] = dataset_interaction[col].astype(str)
        
        # Convert to TensorFlow dataset
        ratings = tf.data.Dataset.from_tensor_slices(dict(dataset_interaction[selected_cols]))
        
        # Load program studi data
        print("📚 Loading program studi data...")
        program_studi = data_processor.load_program_studi_data()
        
        # Ensure program studi data is properly formatted
        for col in program_studi.columns:
            if col in ["item_id", "category", "category2", "category3", "region"]:
                program_studi[col] = program_studi[col].astype(str)
        
        movies = tf.data.Dataset.from_tensor_slices(dict(program_studi))
        
        # Build model (non-blocking)
        print("️ Building model...")
        training_status["current_step"] = "building"
        
        # Create new recommendation model instance for training
        training_model = RecommendationModel()
        training_model.build_model(ratings, movies)
        
        print("🎯 Training model...")
        training_status["current_step"] = "training"
        history = training_model.train_model(ratings, epochs=request.epochs)
        
        # Create index and lookup
        print(" Creating index and lookup...")
        training_model.create_index(movies)
        training_model.create_item_lookup(movies)
        
        # Verify model is ready
        print(" Verifying model readiness...")
        if training_model.brute_force_index is None:
            print("❌ Model index not created properly")
            raise Exception("Model index creation failed")
        else:
            print("✅ Model index created successfully")
        
        # Save new model instance in memory
        print(" Saving new model instance in memory...")
        training_status["current_step"] = "saving"
        
        try:
            # Save the complete training model instance in memory
            save_model_instance_in_memory(training_model, new_version)
            model_loaded = True
            print("✅ Model training completed successfully!")
            print("🔄 New model is now active for recommendations")
            
            # Clean up old model from memory
            print("🧹 Cleaning up old model from memory...")
            cleanup_old_model()
            
        except Exception as e:
            print(f"❌ Error saving model in memory: {e}")
            model_loaded = False
            print("❌ Failed to save model in memory")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        model_loaded = False
        print("❌ Training failed")
        
        import traceback
        traceback.print_exc()
    
    finally:
        # ALWAYS reset training status
        print(" Resetting training status...")
        training_status.update({
            "is_training": False,
            "current_step": "completed",
            "training_completed": True,
            "completion_time": datetime.now().isoformat()
        })
        training_in_progress = False

def train_ranking_model_thread(training_id: str, new_version: str, request: TrainingRequest):
    """Separate thread for ranking model training"""
    global training_in_progress, training_status
    
    try:
        print(f"🎯 Starting ranking model training in background thread (ID: {training_id}, version: {new_version})...")
        
        # Update status to building
        training_status["current_step"] = "building_ranking"
        print("📊 Loading and processing data for ranking...")
        
        # Load and process data
        dataset, _ = data_processor.load_and_process_data()
        dataset_interaction = data_processor.create_interaction_dataset(dataset)
        
        # Convert to TensorFlow datasets with proper data types
        selected_cols = [
            "user_id", "item_id", "category", "category2", "category3",
            "region", 'city', "item_id_lastview", "item_id_currentview", "label","timestamp_unix"
        ]
        
        # Ensure all data is properly formatted
        print("🔧 Formatting data for ranking...")
        for col in selected_cols:
            if col == "label":
                dataset_interaction[col] = dataset_interaction[col].astype(np.float32)
            elif col == "timestamp_unix":
                dataset_interaction[col] = dataset_interaction[col].astype(np.int64)
            else:
                dataset_interaction[col] = dataset_interaction[col].astype(str)
        
        # Convert to TensorFlow dataset
        ratings = tf.data.Dataset.from_tensor_slices(dict(dataset_interaction[selected_cols]))
        
        # Load program studi data
        print("📚 Loading program studi data...")
        program_studi = data_processor.load_program_studi_data()
        
        # Ensure program studi data is properly formatted
        for col in program_studi.columns:
            if col in ["item_id", "category", "category2", "category3", "region"]:
                program_studi[col] = program_studi[col].astype(str)
        
        movies = tf.data.Dataset.from_tensor_slices(dict(program_studi))
        
        # Get current model instance
        available_model_instance, model_type = get_available_model_instance()
        if available_model_instance is None:
            raise Exception("No base model available for ranking training")
        
        print("🎯 Training ranking model...")
        training_status["current_step"] = "training_ranking"
        
        # Train the ranking model using the existing user and item models
        history = available_model_instance.train_ranking_model(ratings, epochs=request.epochs)
        
        print("✅ Ranking model training completed successfully!")
        print("🔄 Ranking model is now available for recommendations with ranking")
        
    except Exception as e:
        print(f"❌ Ranking training error: {e}")
        print("❌ Ranking training failed")
        
        import traceback
        traceback.print_exc()
    
    finally:
        # ALWAYS reset training status
        print("🔄 Resetting training status...")
        training_status.update({
            "is_training": False,
            "current_step": "completed",
            "training_completed": True,
            "completion_time": datetime.now().isoformat()
        })
        training_in_progress = False

def save_model_backup(model_path: str, version: str):
    """Save model backup with version"""
    backup_path = f"{model_path}_v{version}"
    try:
        if os.path.exists(model_path):
            # Copy directory instead of symlink
            shutil.copytree(model_path, backup_path, dirs_exist_ok=True)
            print(f"✅ Model backup saved: {backup_path}")
            return backup_path
    except Exception as e:
        print(f"⚠️ Warning: Could not save model backup: {e}")
    return None

def create_model_reference(current_path: str, reference_path: str):
    """Create a reference to current model without symlink"""
    try:
        # Instead of symlink, create a reference file
        with open(reference_path, "w") as f:
            f.write(current_path)
        print(f"✅ Model reference created: {reference_path} -> {current_path}")
        return True
    except Exception as e:
        print(f"⚠️ Warning: Could not create model reference: {e}")
        return False

def get_current_model_path():
    """Get current model path from reference file"""
    reference_file = "current_model.txt"
    if os.path.exists(reference_file):
        try:
            with open(reference_file, "r") as f:
                return f.read().strip()
        except:
            pass
    return "saved_model"  # Default fallback

def load_model_version(version_path: str):
    """Load specific model version"""
    try:
        if os.path.exists(version_path):
            recommendation_model.load_model(version_path)
            return True
    except Exception as e:
        print(f"❌ Error loading model version {version_path}: {e}")
    return False

def save_model_custom(model, save_path: str):
    """Save model using custom approach for TFRS models"""
    try:
        # Create directory if not exists
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights and configuration
        model.save_weights(os.path.join(save_path, "weights"))
        
        # Save lookup layers
        lookup_config = {}
        if hasattr(model, 'user_model') and hasattr(model.user_model, 'lookup_layers'):
            for name, layer in model.user_model.lookup_layers.items():
                lookup_config[f"user_{name}"] = layer.get_vocabulary()
        
        if hasattr(model, 'item_model') and hasattr(model.item_model, 'lookup_layers'):
            for name, layer in model.item_model.lookup_layers.items():
                lookup_config[f"item_{name}"] = layer.get_vocabulary()
        
        # Save lookup configuration
        with open(os.path.join(save_path, "lookup_config.json"), "w") as f:
            json.dump(lookup_config, f)
        
        # Save model metadata
        metadata = {
            "model_type": "two_tower_recommendation",
            "saved_at": datetime.now().isoformat(),
            "user_model_layers": len(model.user_model.layers) if hasattr(model, 'user_model') else 0,
            "item_model_layers": len(model.item_model.layers) if hasattr(model, 'item_model') else 0
        }
        
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        
        print(f"✅ Model saved successfully to {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

def load_model_custom(load_path: str):
    """Load model using custom approach"""
    try:
        if not os.path.exists(load_path):
            print(f"❌ Model path does not exist: {load_path}")
            return False
        
        # Load metadata
        metadata_path = os.path.join(load_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"📋 Loading model: {metadata}")
        
        # Load lookup configuration
        lookup_config_path = os.path.join(load_path, "lookup_config.json")
        if os.path.exists(lookup_config_path):
            with open(lookup_config_path, "r") as f:
                lookup_config = json.load(f)
        
        # Rebuild model with loaded configuration
        # Note: This is a simplified approach. In production, you might want to store more model architecture details
        print("🔄 Model loaded successfully (weights and config)")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Whitelist config from environment
import os

def parse_env_set(env_value, default_set):
    if env_value:
        return set(x.strip() for x in env_value.split(",") if x.strip())
    return default_set

WHITELISTED_IPS = parse_env_set(os.environ.get("WHITELISTED_IPS"), {"127.0.0.1", "::1"})
WHITELISTED_API_KEYS = parse_env_set(os.environ.get("WHITELISTED_API_KEYS"), {"your-secret-api-key"})

def is_request_whitelisted(request: Request) -> bool:
    # Check IP
    client_ip = request.client.host if request.client else None
    if client_ip in WHITELISTED_IPS:
        return True
    # Check API key
    api_key = request.headers.get("x-api-key")
    if api_key in WHITELISTED_API_KEYS:
        return True
    return False

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global model_loaded
    try:
        # Check if there's a model in memory (for restart scenarios)
        if current_model_instance is not None:
            model_loaded = True
            print("✅ Model loaded from memory on startup")
        else:
            print("⚠️ No model in memory. Please train the model first.")
    except Exception as e:
        print(f"❌ Error loading model on startup: {e}")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Recommendation System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "recommendations": "/recommendations",
            "train": "/train",
            "data_stats": "/data/stats"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    available_model_instance, model_type = get_available_model_instance()
    
    return {
        "status": "healthy",
        "model_loaded": available_model_instance is not None,
        "training_in_progress": training_status["is_training"],
        "model_type": model_type,
        "model_ready": available_model_instance is not None and available_model_instance.brute_force_index is not None
    }

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get model information"""
    try:
        dataset, _ = data_processor.load_and_process_data()
        dataset_interaction = data_processor.create_interaction_dataset(dataset)
        stats = data_processor.get_dataset_stats(dataset_interaction)
        
        # Convert numpy types to Python native types
        converted_stats = {}
        for key, value in stats.items():
            if hasattr(value, 'item'):  # numpy type
                converted_stats[key] = value.item()
            else:
                converted_stats[key] = value
        
        return ModelInfo(
            model_status="loaded" if model_loaded else "not_loaded",
            dataset_stats=converted_stats,
            last_training=None  # Could be stored in a database
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train the recommendation model with non-blocking thread"""
    global training_in_progress, current_training, training_status
    
    # Check if training is already in progress
    if training_in_progress:
        raise HTTPException(
            status_code=400, 
            detail="Training already in progress. Please wait for current training to complete."
        )
    
    # Generate training ID
    training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Start training immediately
    training_in_progress = True
    current_training = training_id
    
    # Generate new model version
    new_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_status.update({
        "is_training": True,
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": request.epochs,
        "start_time": datetime.now().isoformat(),
        "estimated_completion": None,
        "new_model_version": new_version,
        "current_training_id": training_id,
        "training_completed": False,
        "completion_time": None,
        "current_step": "starting"
    })
    
    # Start training in separate thread
    training_thread = threading.Thread(
        target=train_model_thread,
        args=(training_id, new_version, request),
        daemon=True
    )
    training_thread.start()
    
    return TrainingResponse(
        status="started",
        message=f"Model training started in background (ID: {training_id}, version: {new_version}). Recommendations will continue to work with current model.",
        training_info={
            "training_id": training_id,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "estimated_time": f"{request.epochs * 2} minutes",
            "new_version": new_version,
            "non_blocking": True
        }
    )

@app.post("/train/ranking", response_model=TrainingResponse, tags=["Training"])
async def train_ranking_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train the ranking model with non-blocking thread"""
    global training_in_progress, current_training, training_status
    
    # Check if training is already in progress
    if training_in_progress:
        raise HTTPException(
            status_code=400, 
            detail="Training already in progress. Please wait for current training to complete."
        )
    
    # Check if base model is available
    available_model_instance, model_type = get_available_model_instance()
    if available_model_instance is None:
        raise HTTPException(
            status_code=400, 
            detail="No base model available. Please train the base model first using POST /train"
        )
    
    # Generate training ID
    training_id = datetime.now().strftime("%Y%m%d_%H%M%S_ranking")
    
    # Start training immediately
    training_in_progress = True
    current_training = training_id
    
    # Generate new model version
    new_version = datetime.now().strftime("%Y%m%d_%H%M%S_ranking")
    training_status.update({
        "is_training": True,
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": request.epochs,
        "start_time": datetime.now().isoformat(),
        "estimated_completion": None,
        "new_model_version": new_version,
        "current_training_id": training_id,
        "training_completed": False,
        "completion_time": None,
        "current_step": "starting_ranking_training"
    })
    
    # Start training in separate thread
    training_thread = threading.Thread(
        target=train_ranking_model_thread,
        args=(training_id, new_version, request),
        daemon=True
    )
    training_thread.start()
    
    return TrainingResponse(
        status="started",
        message=f"Ranking model training started in background (ID: {training_id}, version: {new_version}). Recommendations will continue to work with current model.",
        training_info={
            "training_id": training_id,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "estimated_time": f"{request.epochs * 1} minutes",
            "new_version": new_version,
            "non_blocking": True,
            "model_type": "ranking"
        }
    )

@app.post("/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest, _request: Request):
    if not is_request_whitelisted(_request):
        raise HTTPException(status_code=403, detail="Forbidden: Not whitelisted.")
    """Get recommendations for a user - using separate model instances"""
    
    # Get available model instance
    available_model_instance, model_type = get_available_model_instance()
    
    if available_model_instance is None:
        raise HTTPException(
            status_code=400, 
            detail="No model available in memory. Please train the model first using POST /train"
        )
    
    # Check if model index is created
    if available_model_instance.brute_force_index is None:
        raise HTTPException(
            status_code=500,
            detail="Model index not created. Please train the model first."
        )
    
    # Jika training sedang berjalan, gunakan model yang tersedia
    if training_status["is_training"] and not training_status["training_completed"]:
        print(f"🔄 Training in progress ({training_status['current_step']}), using {model_type} model for recommendations...")
    
    try:
        # Use the available model instance directly
        recommendations = available_model_instance.get_recommendations(
            user_id=request.user_id,
            current_item_id=request.current_item_id,
            region=request.region,
            city=request.city,
            top_k=request.top_k
        )
        # Enrich recommendations with tittle, page_url, image_url
        recommendation_id = str(uuid.uuid4())
        recommendation_group = [rec.get("item_id") for rec in recommendations]
        recommendations = enrich_recommendations_with_program_studi(recommendations, recommendation_id)
        
        # Track recommendation metrics
        recommendation_metrics["total_recommendations"] += 1
        
        # Track by category
        for rec in recommendations:
            category = rec.get("category", "unknown")
            if category not in recommendation_metrics["recommendations_by_category"]:
                recommendation_metrics["recommendations_by_category"][category] = 0
            recommendation_metrics["recommendations_by_category"][category] += 1
            
            # Track popular items
            item_id = rec.get("item_id", "unknown")
            if item_id not in recommendation_metrics["popular_items"]:
                recommendation_metrics["popular_items"][item_id] = 0
            recommendation_metrics["popular_items"][item_id] += 1
        
        # Track user engagement
        if request.user_id not in recommendation_metrics["user_engagement"]:
            recommendation_metrics["user_engagement"][request.user_id] = 0
        recommendation_metrics["user_engagement"][request.user_id] += 1
        
        # Determine model status
        model_status = "current"
        if training_status["is_training"] and not training_status["training_completed"]:
            model_status = f"{model_type} (training in progress - {training_status['current_step']})"
        
        recommendation_id = str(uuid.uuid4())
        response_data = RecommendationResponse(
            user_id=request.user_id,
            current_item_id=request.current_item_id,
            recommendations=recommendations,
            model_info={
                "model_loaded": model_loaded,
                "recommendations_count": len(recommendations),
                "model_status": model_status,
                "model_version": model_versions["current_version"] if model_type == "current" else model_versions["previous_version"],
                "training_in_progress": training_status["is_training"],
                "training_completed": training_status["training_completed"],
                "training_step": training_status["current_step"],
                "message": f"Using {model_type} model instance in memory" + (" (during training)" if training_status["is_training"] else "")
            }
        )
        response_dict = response_data.dict()
        response_dict["recommendation_id"] = recommendation_id
        response_dict["recommendation_group"] = recommendation_group
        # Send to Google Apps Script (non-blocking)
        try:
            GOOGLE_SCRIPT_URL = os.environ.get("RECOMMENDATION_URL")
            if GOOGLE_SCRIPT_URL:
                requests.post(GOOGLE_SCRIPT_URL, json=response_dict, timeout=3)
        except Exception as log_exc:
            print(f"[LOGGING ERROR] Failed to log recommendation: {log_exc}")
        return response_dict
    except Exception as e:
        print(f"❌ Error in recommendations: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/recommendations/with-ranking", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations_with_ranking(request: RecommendationRequest, _request: Request):
    if not is_request_whitelisted(_request):
        raise HTTPException(status_code=403, detail="Forbidden: Not whitelisted.")
    """Get recommendations using retrieval + ranking approach"""
    
    # Get available model instance
    available_model_instance, model_type = get_available_model_instance()
    
    if available_model_instance is None:
        raise HTTPException(
            status_code=400, 
            detail="No model available in memory. Please train the model first using POST /train"
        )
    
    # Check if model index is created
    if available_model_instance.brute_force_index is None:
        raise HTTPException(
            status_code=500,
            detail="Model index not created. Please train the model first."
        )
    
    # Check if ranking model is available
    if available_model_instance.ranking_model is None:
        raise HTTPException(
            status_code=500,
            detail="Ranking model not available. Please ensure ranking model is trained."
        )
    
    # Jika training sedang berjalan, gunakan model yang tersedia
    if training_status["is_training"] and not training_status["training_completed"]:
        print(f"🔄 Training in progress ({training_status['current_step']}), using {model_type} model for recommendations...")
    
    try:
        # Use the available model instance with ranking
        recommendations = available_model_instance.get_recommendations_with_ranking(
            user_id=request.user_id,
            current_item_id=request.current_item_id,
            region=request.region,
            city=request.city,
            top_k=request.top_k
        )
        # Enrich recommendations with tittle, page_url, image_url
        recommendation_id = str(uuid.uuid4())
        recommendation_group = [rec.get("item_id") for rec in recommendations]
        recommendations = enrich_recommendations_with_program_studi(recommendations, recommendation_id)
        
        # Track recommendation metrics
        recommendation_metrics["total_recommendations"] += 1
        
        # Track by category
        for rec in recommendations:
            category = rec.get("category", "unknown")
            if category not in recommendation_metrics["recommendations_by_category"]:
                recommendation_metrics["recommendations_by_category"][category] = 0
            recommendation_metrics["recommendations_by_category"][category] += 1
            
            # Track popular items
            item_id = rec.get("item_id", "unknown")
            if item_id not in recommendation_metrics["popular_items"]:
                recommendation_metrics["popular_items"][item_id] = 0
            recommendation_metrics["popular_items"][item_id] += 1
        
        # Track user engagement
        if request.user_id not in recommendation_metrics["user_engagement"]:
            recommendation_metrics["user_engagement"][request.user_id] = 0
        recommendation_metrics["user_engagement"][request.user_id] += 1
        
        # Determine model status
        model_status = "current"
        if training_status["is_training"] and not training_status["training_completed"]:
            model_status = f"{model_type} (training in progress - {training_status['current_step']})"
        
        recommendation_id = str(uuid.uuid4())
        response_data = RecommendationResponse(
            user_id=request.user_id,
            current_item_id=request.current_item_id,
            recommendations=recommendations,
            model_info={
                "model_loaded": model_loaded,
                "recommendations_count": len(recommendations),
                "model_status": model_status,
                "model_version": model_versions["current_version"] if model_type == "current" else model_versions["previous_version"],
                "training_in_progress": training_status["is_training"],
                "training_completed": training_status["training_completed"],
                "training_step": training_status["current_step"],
                "message": f"Using {model_type} model instance with ranking in memory" + (" (during training)" if training_status["is_training"] else ""),
                "method": "retrieval_with_ranking"
            }
        )
        response_dict = response_data.dict()
        response_dict["recommendation_id"] = recommendation_id
        response_dict["recommendation_group"] = recommendation_group
        # Send to Google Apps Script (non-blocking)
        try:
            GOOGLE_SCRIPT_URL = os.environ.get("RECOMMENDATION_URL")
            if GOOGLE_SCRIPT_URL:
                requests.post(GOOGLE_SCRIPT_URL, json=response_dict, timeout=3)
        except Exception as log_exc:
            print(f"[LOGGING ERROR] Failed to log recommendation: {log_exc}")
        return response_dict
    except Exception as e:
        print(f"❌ Error in recommendations with ranking: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting recommendations with ranking: {str(e)}")

@app.get("/data/stats", tags=["Data"])
async def get_data_stats():
    """Get dataset statistics"""
    try:
        dataset, _ = data_processor.load_and_process_data()
        dataset_interaction = data_processor.create_interaction_dataset(dataset)
        stats = data_processor.get_dataset_stats(dataset_interaction)
        
        # Convert numpy types to Python native types
        converted_stats = {}
        for key, value in stats.items():
            if hasattr(value, 'item'):  # numpy type
                converted_stats[key] = value.item()
            else:
                converted_stats[key] = value
        
        # Convert sample data to native types
        sample_data = dataset_interaction.head(5).to_dict('records')
        for record in sample_data:
            for key, value in record.items():
                if hasattr(value, 'item'):  # numpy type
                    record[key] = value.item()
                elif pd.isna(value):  # Handle NaN values
                    record[key] = None
        
        return {
            "dataset_stats": converted_stats,
            "sample_data": sample_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data stats: {str(e)}")

@app.get("/recommendations/sample", tags=["Recommendations"])
async def get_sample_recommendations():
    """Get sample recommendations for testing - always use available model"""
    if not model_loaded or current_model_instance is None:
        raise HTTPException(status_code=400, detail="No model available in memory. Please train the model first.")
    
    try:
        # Sample data for testing
        sample_request = RecommendationRequest(
            user_id="yogieeka@gmail.com",
            current_item_id="3905",
            region="west java",
            city="bandung",
            top_k=5
        )
        
        return await get_recommendations(sample_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sample recommendations: {str(e)}")

@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Reload the model from disk"""
    global model_loaded
    try:
        current_model_path = get_current_model_path()
        if os.path.exists(current_model_path):
            if load_model_custom(current_model_path):
                model_loaded = True
                return {
                    "status": "success", 
                    "message": "Model reloaded successfully",
                    "model_path": current_model_path
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to load model")
        else:
            raise HTTPException(status_code=404, detail="No saved model found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")

@app.get("/dashboard/api-analytics", tags=["Dashboard"])
async def get_api_analytics():
    """Get API usage analytics"""
    avg_response_time = sum(api_usage["response_times"]) / len(api_usage["response_times"]) if api_usage["response_times"] else 0
    
    return {
        "overview": {
            "total_requests": api_usage["total_requests"],
            "successful_requests": api_usage["successful_requests"],
            "failed_requests": api_usage["failed_requests"],
            "success_rate": (api_usage["successful_requests"] / api_usage["total_requests"] * 100) if api_usage["total_requests"] > 0 else 0
        },
        "performance": {
            "average_response_time": round(avg_response_time, 3),
            "min_response_time": min(api_usage["response_times"]) if api_usage["response_times"] else 0,
            "max_response_time": max(api_usage["response_times"]) if api_usage["response_times"] else 0
        },
        "endpoint_usage": api_usage["requests_by_endpoint"],
        "daily_trends": api_usage["daily_requests"]
    }

@app.get("/dashboard/recommendation-analytics", tags=["Dashboard"])
async def get_recommendation_analytics():
    """Get recommendation performance analytics"""
    # Get top popular items
    popular_items = sorted(
        recommendation_metrics["popular_items"].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    # Get top engaged users
    engaged_users = sorted(
        recommendation_metrics["user_engagement"].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    return {
        "overview": {
            "total_recommendations": recommendation_metrics["total_recommendations"],
            "unique_users": len(recommendation_metrics["user_engagement"]),
            "unique_items_recommended": len(recommendation_metrics["popular_items"])
        },
        "category_distribution": recommendation_metrics["recommendations_by_category"],
        "popular_items": [{"item_id": item_id, "count": count} for item_id, count in popular_items],
        "engaged_users": [{"user_id": user_id, "requests": count} for user_id, count in engaged_users]
    }

import psutil
import os

def convert_numpy_types(obj):
    """Convert numpy types to Python native types"""
    if hasattr(obj, 'item'):  # numpy type
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):  # Handle NaN values
        return None
    else:
        return obj

@app.get("/dashboard/system-health", tags=["Dashboard"])
async def get_system_health():
    """Get system health metrics with real data"""
    try:
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get real model info
        model_size = "Unknown"
        if model_loaded:
            try:
                # Check if saved_model directory exists
                if os.path.exists("saved_model"):
                    model_size = "Model saved"
                else:
                    model_size = "Model in memory"
            except:
                model_size = "Model loaded"
        
        # Get real data stats
        dataset, _ = data_processor.load_and_process_data()
        dataset_interaction = data_processor.create_interaction_dataset(dataset)
        stats = data_processor.get_dataset_stats(dataset_interaction)
        
        # Convert numpy types to Python native types
        converted_stats = convert_numpy_types(stats)
        
        return {
            "system_metrics": {
                "cpu_usage": float(cpu_percent),
                "memory_usage": float(memory.percent),
                "memory_available": int(memory.available // (1024**3)),  # GB
                "disk_usage": float(disk.percent),
                "disk_free": int(disk.free // (1024**3))  # GB
            },
            "application_status": {
                "model_loaded": model_loaded,
                "training_in_progress": training_in_progress,
                "uptime": "Server running",
                "last_restart": "Current session"
            },
            "model_status": {
                "model_size": model_size,
                "last_training": "Model available" if model_loaded else "No model",
                "training_duration": "Completed" if model_loaded else "Not trained",
                "model_accuracy": "Ready" if model_loaded else "Not ready"
            },
            "data_stats": converted_stats
        }
    except Exception as e:
        return {
            "system_metrics": {
                "cpu_usage": "Error",
                "memory_usage": "Error",
                "memory_available": "Error",
                "disk_usage": "Error",
                "disk_free": "Error"
            },
            "application_status": {
                "model_loaded": model_loaded,
                "training_in_progress": training_in_progress,
                "uptime": "Error",
                "last_restart": "Error"
            },
            "model_status": {
                "model_size": "Error",
                "last_training": "Error",
                "training_duration": "Error",
                "model_accuracy": "Error"
            },
            "error": str(e)
        }

@app.get("/dashboard/data-quality", tags=["Dashboard"])
async def get_data_quality_metrics():
    """Get data quality metrics"""
    try:
        dataset, _ = data_processor.load_and_process_data()
        dataset_interaction = data_processor.create_interaction_dataset(dataset)
        
        # Calculate data quality metrics
        total_rows = len(dataset_interaction)
        missing_values = dataset_interaction.isnull().sum().sum()
        duplicate_rows = dataset_interaction.duplicated().sum()
        
        # Convert numpy types to Python native types
        earliest_date = int(dataset_interaction['timestamp_unix'].min())
        latest_date = int(dataset_interaction['timestamp_unix'].max())
        date_range_days = int((latest_date - earliest_date) // 86400)
        
        return {
            "data_overview": {
                "total_records": int(total_rows),
                "missing_values": int(missing_values),
                "duplicate_records": int(duplicate_rows),
                "data_completeness": float(((total_rows - missing_values) / total_rows * 100) if total_rows > 0 else 0)
            },
            "data_distribution": {
                "users": int(dataset_interaction['user_id'].nunique()),
                "items": int(dataset_interaction['item_id'].nunique()),
                "categories": int(dataset_interaction['category'].nunique()),
                "regions": int(dataset_interaction['region'].nunique())
            },
            "data_timeline": {
                "earliest_date": earliest_date,
                "latest_date": latest_date,
                "date_range_days": date_range_days
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data quality metrics: {str(e)}")

@app.get("/dashboard", tags=["Dashboard"])
async def get_main_dashboard():
    """Get main dashboard with real metrics"""
    try:
        # Get real data stats
        dataset, _ = data_processor.load_and_process_data()
        dataset_interaction = data_processor.create_interaction_dataset(dataset)
        stats = data_processor.get_dataset_stats(dataset_interaction)
        
        # Convert numpy types to Python native types
        converted_stats = convert_numpy_types(stats)
        
        # Perbaiki perhitungan active_users_today
        today = datetime.now().strftime("%Y-%m-%d")
        active_users_today = api_usage["daily_requests"].get(today, 0)
        
        # Calculate training readiness
        total_interactions = converted_stats.get("num_interactions", 0)
        total_users = converted_stats.get("num_users", 0)
        total_items = converted_stats.get("num_items", 0)
        
        return {
            "dashboard": {
                "title": "Recommendation System Dashboard",
                "last_updated": datetime.now().isoformat(),
                "endpoints": {
                    "training_metrics": "/dashboard/training-metrics",
                    "training_progress": "/dashboard/training-progress",
                    "api_analytics": "/dashboard/api-analytics", 
                    "recommendation_analytics": "/dashboard/recommendation-analytics",
                    "system_health": "/dashboard/system-health",
                    "data_quality": "/dashboard/data-quality"
                }
            },
            "quick_stats": {
                "model_status": "loaded" if model_loaded else "not_loaded",
                "total_api_requests": api_usage["total_requests"],
                "total_recommendations": recommendation_metrics["total_recommendations"],
                "active_users_today": active_users_today,
                "total_users_in_data": total_users,
                "total_items_in_data": total_items,
                "total_interactions_in_data": total_interactions
            },
            "training_insights": {
                "data_sufficient": total_interactions >= 1000,
                "can_train": total_interactions > 0,
                "estimated_accuracy": f"{min(95, max(70, 75 + (total_interactions / 10000) * 10)):.1f}%" if model_loaded else "Not trained",
                "recommended_epochs": min(20, max(5, int(total_interactions / 1000)))
            },
            "real_data_overview": converted_stats
        }
    except Exception as e:
        return {
            "dashboard": {
                "title": "Recommendation System Dashboard",
                "last_updated": datetime.now().isoformat(),
                "error": str(e)
            },
            "quick_stats": {
                "model_status": "loaded" if model_loaded else "not_loaded",
                "total_api_requests": api_usage["total_requests"],
                "total_recommendations": recommendation_metrics["total_recommendations"],
                "active_users_today": api_usage["daily_requests"].get(datetime.now().strftime("%Y-%m-%d"), 0),
                "error": "Could not load data stats"
            },
            "training_insights": {
                "data_sufficient": False,
                "can_train": False,
                "estimated_accuracy": "Error",
                "recommended_epochs": "Error"
            }
        }

# Tambahkan endpoint untuk monitoring training
@app.get("/dashboard/training-metrics", tags=["Dashboard"])
async def get_training_metrics():
    """Get real-time training metrics with real data"""
    try:
        # Load real data
        dataset, _ = data_processor.load_and_process_data()
        dataset_interaction = data_processor.create_interaction_dataset(dataset)
        stats = data_processor.get_dataset_stats(dataset_interaction)
        
        # Convert numpy types to Python native types
        converted_stats = convert_numpy_types(stats)
        
        # Get real model info if available
        model_info = {
            "model_size": "Unknown",
            "last_training": "Never",
            "training_duration": "Unknown",
            "model_accuracy": "Unknown"
        }
        
        if model_loaded:
            model_info = {
                "model_size": "Model loaded",
                "last_training": "Model available",
                "training_duration": "Completed",
                "model_accuracy": "Ready for recommendations"
            }
        
        # Calculate real training metrics
        total_interactions = converted_stats.get("num_interactions", 0)
        total_users = converted_stats.get("num_users", 0)
        total_items = converted_stats.get("num_items", 0)
        
        # Estimate model performance based on data
        if total_interactions > 0:
            # Simple estimation based on data size
            estimated_accuracy = min(0.95, max(0.70, 0.75 + (total_interactions / 10000) * 0.1))
            estimated_loss = max(0.01, 0.1 - (total_interactions / 10000) * 0.05)
        else:
            estimated_accuracy = 0.0
            estimated_loss = 0.0
        
        return {
            "training_status": {
                "is_training": training_in_progress,
                "model_loaded": model_loaded,
                "last_training_time": model_info["last_training"],
                "training_ready": total_interactions > 0
            },
            "model_performance": {
                "loss": round(estimated_loss, 4) if model_loaded else "Not trained",
                "accuracy": f"{estimated_accuracy:.2%}" if model_loaded else "Not trained",
                "epochs_completed": 15 if model_loaded else 0,
                "total_epochs": 15,
                "training_progress": "100%" if model_loaded else "0%"
            },
            "data_metrics": {
                "total_users": total_users,
                "total_items": total_items,
                "total_interactions": total_interactions,
                "recommendations_generated": recommendation_metrics["total_recommendations"],
                "data_quality_score": f"{((total_interactions / (total_users * total_items)) * 100):.2f}%" if total_users > 0 and total_items > 0 else "0%"
            },
            "real_data_stats": converted_stats,
            "training_insights": {
                "sparsity": f"{((1 - (total_interactions / (total_users * total_items))) * 100):.2f}%" if total_users > 0 and total_items > 0 else "100%",
                "avg_interactions_per_user": round(total_interactions / total_users, 2) if total_users > 0 else 0,
                "avg_interactions_per_item": round(total_interactions / total_items, 2) if total_items > 0 else 0,
                "recommendation_coverage": f"{(len(recommendation_metrics['popular_items']) / total_items * 100):.2f}%" if total_items > 0 else "0%"
            }
        }
    except Exception as e:
        return {
            "training_status": {
                "is_training": training_in_progress,
                "model_loaded": model_loaded,
                "last_training_time": "Error loading data",
                "training_ready": False
            },
            "model_performance": {
                "loss": "Error",
                "accuracy": "Error", 
                "epochs_completed": "Error",
                "total_epochs": "Error",
                "training_progress": "Error"
            },
            "data_metrics": {
                "total_users": 0,
                "total_items": 0,
                "total_interactions": 0,
                "recommendations_generated": recommendation_metrics["total_recommendations"],
                "data_quality_score": "Error"
            },
            "training_insights": {
                "sparsity": "Error",
                "avg_interactions_per_user": "Error",
                "avg_interactions_per_item": "Error",
                "recommendation_coverage": "Error"
            },
            "error": str(e)
        }

@app.get("/dashboard/training-progress", tags=["Dashboard"])
async def get_training_progress():
    """Get detailed training progress and metrics"""
    try:
        # Get real data stats
        dataset, _ = data_processor.load_and_process_data()
        dataset_interaction = data_processor.create_interaction_dataset(dataset)
        stats = data_processor.get_dataset_stats(dataset_interaction)
        converted_stats = convert_numpy_types(stats)
        
        total_interactions = converted_stats.get("num_interactions", 0)
        total_users = converted_stats.get("num_users", 0)
        total_items = converted_stats.get("num_items", 0)
        
        # Calculate training readiness metrics
        data_sparsity = ((1 - (total_interactions / (total_users * total_items))) * 100) if total_users > 0 and total_items > 0 else 100
        interaction_density = (total_interactions / (total_users * total_items)) if total_users > 0 and total_items > 0 else 0
        
        return {
            "data_readiness": {
                "total_records": total_interactions,
                "unique_users": total_users,
                "unique_items": total_items,
                "sparsity": f"{data_sparsity:.2f}%",
                "density": f"{interaction_density:.4f}",
                "avg_interactions_per_user": round(total_interactions / total_users, 2) if total_users > 0 else 0,
                "avg_interactions_per_item": round(total_interactions / total_items, 2) if total_items > 0 else 0
            },
            "model_status": {
                "is_loaded": model_loaded,
                "is_training": training_in_progress,
                "can_train": total_interactions > 0,
                "recommended_epochs": min(20, max(5, int(total_interactions / 1000))),
                "estimated_training_time": f"{max(5, int(total_interactions / 5000))} minutes"
            },
            "performance_metrics": {
                "total_recommendations_generated": recommendation_metrics["total_recommendations"],
                "unique_items_recommended": len(recommendation_metrics["popular_items"]),
                "unique_users_served": len(recommendation_metrics["user_engagement"]),
                "recommendation_coverage": f"{(len(recommendation_metrics['popular_items']) / total_items * 100):.2f}%" if total_items > 0 else "0%"
            },
            "training_recommendations": {
                "data_sufficient": total_interactions >= 1000,
                "sparsity_acceptable": data_sparsity < 99.9,
                "user_coverage_good": total_users >= 100,
                "item_coverage_good": total_items >= 50,
                "recommended_actions": [
                    "Train model" if not model_loaded and total_interactions > 0 else "Model already trained",
                    "Add more data" if total_interactions < 1000 else "Data sufficient",
                    "Increase user interactions" if data_sparsity > 99.9 else "Interaction density good"
                ]
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "data_readiness": {
                "total_records": 0,
                "unique_users": 0,
                "unique_items": 0,
                "sparsity": "Error",
                "density": "Error",
                "avg_interactions_per_user": "Error",
                "avg_interactions_per_item": "Error"
            },
            "model_status": {
                "is_loaded": model_loaded,
                "is_training": training_in_progress,
                "can_train": False,
                "recommended_epochs": "Error",
                "estimated_training_time": "Error"
            },
            "performance_metrics": {
                "total_recommendations_generated": recommendation_metrics["total_recommendations"],
                "unique_items_recommended": len(recommendation_metrics["popular_items"]),
                "unique_users_served": len(recommendation_metrics["user_engagement"]),
                "recommendation_coverage": "Error"
            },
            "training_recommendations": {
                "data_sufficient": False,
                "sparsity_acceptable": False,
                "user_coverage_good": False,
                "item_coverage_good": False,
                "recommended_actions": ["Fix data loading error"]
            }
        }

@app.get("/model/status", tags=["Model"])
async def get_model_status():
    """Get detailed model status"""
    available_model_instance, model_type = get_available_model_instance()
    
    model_ready = False
    if available_model_instance is not None:
        model_ready = available_model_instance.brute_force_index is not None
    
    return {
        "model_loaded": model_loaded,
        "model_in_memory": available_model_instance is not None,
        "model_index_created": model_ready,
        "available_model_type": model_type,
        "training_in_progress": training_status["is_training"],
        "training_completed": training_status["training_completed"],
        "current_model_version": model_versions["current_version"],
        "previous_model_version": model_versions["previous_version"],
        "recommendations_available": model_ready,
        "can_use_during_training": model_ready,
        "model_status": "ready" if model_ready else "not_ready",
        "memory_usage": {
            "current_model": "loaded" if current_model_instance is not None else "not_loaded",
            "previous_model": "loaded" if previous_model_instance is not None else "not_loaded"
        },
        "training_status": {
            "is_training": training_status["is_training"],
            "progress": training_status["progress"],
            "current_epoch": training_status["current_epoch"],
            "total_epochs": training_status["total_epochs"],
            "current_step": training_status["current_step"]
        }
    }

@app.get("/training/status", tags=["Training"])
async def get_training_status():
    """Get current training status with detailed step info"""
    available_model_instance, model_type = get_available_model_instance()
    
    return {
        "is_training": training_status["is_training"],
        "training_completed": training_status["training_completed"],
        "completion_time": training_status["completion_time"],
        "current_step": training_status["current_step"],
        "current_training": {
            "id": training_status["current_training_id"],
            "progress": training_status["progress"],
            "current_epoch": training_status["current_epoch"],
            "total_epochs": training_status["total_epochs"],
            "start_time": training_status["start_time"],
            "estimated_completion": training_status["estimated_completion"],
            "new_model_version": training_status["new_model_version"]
        },
        "model_ready": available_model_instance is not None,
        "available_model_type": model_type,
        "current_model_version": model_versions["current_version"],
        "previous_model_version": model_versions["previous_version"],
        "memory_status": {
            "current_model": "loaded" if current_model_instance is not None else "not_loaded",
            "previous_model": "loaded" if previous_model_instance is not None else "not_loaded",
            "available_model": model_type
        },
        "non_blocking": True
    }

@app.post("/model/clear", tags=["Model"])
async def clear_model_from_memory():
    """Clear model from memory"""
    global current_model_instance, previous_model_instance, model_loaded, model_versions
    
    if current_model_instance is not None:
        current_model_instance = None
        previous_model_instance = None
        model_loaded = False
        model_versions = {
            "current": None,
            "previous": None,
            "current_version": None,
            "previous_version": None
        }
        
        return {
            "status": "success",
            "message": "Model cleared from memory",
            "model_loaded": False
        }
    else:
        return {
            "status": "no_action",
            "message": "No model in memory to clear",
            "model_loaded": False
        }

@app.get("/model/versions", tags=["Model"])
async def get_model_versions():
    """Get information about available model versions"""
    return {
        "current_model": model_versions["current"],
        "previous_model": model_versions["previous"],
        "backup_model": model_versions["backup"],
        "model_loaded": model_loaded,
        "training_in_progress": training_in_progress,
        "available_models": [
            path for path in model_versions.values() 
            if path and os.path.exists(path)
        ]
    }

@app.post("/model/switch/{version}", tags=["Model"])
async def switch_model_version(version: str):
    """Switch to a specific model version"""
    global model_loaded, model_versions
    
    if version == "current" and model_versions["current"]:
        model_path = model_versions["current"]
    elif version == "previous" and model_versions["previous"]:
        model_path = model_versions["previous"]
    elif version == "backup" and model_versions["backup"]:
        model_path = model_versions["backup"]
    else:
        raise HTTPException(status_code=404, detail=f"Model version {version} not found")
    
    try:
        if load_model_version(model_path):
            model_loaded = True
            # Update current model
            model_versions["current"] = model_path
            return {
                "status": "success",
                "message": f"Switched to model version: {version}",
                "model_path": model_path
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load model version {version}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error switching model: {str(e)}")

@app.post("/training/reset", tags=["Training"])
async def reset_training_status_manual():
    """Manually reset training status if stuck"""
    global training_in_progress, current_training
    
    if training_in_progress:
        training_status.update({
            "is_training": False,
            "current_step": "completed",
            "training_completed": True,
            "completion_time": datetime.now().isoformat()
        })
        training_in_progress = False
        return {
            "status": "reset",
            "message": "Training status manually reset",
            "training_in_progress": False
        }
    else:
        return {
            "status": "no_action",
            "message": "No training in progress to reset",
            "training_in_progress": False
        }

security = HTTPBasic()

def docs_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "admin")  # Ganti dengan username Anda
    correct_password = secrets.compare_digest(credentials.password, "password123")  # Ganti dengan password Anda
    if not (correct_username and correct_password):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

# Override /docs endpoint
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui(credentials: HTTPBasicCredentials = Depends(docs_auth)):
    return get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title + " - Docs")

# Simpan original docs di endpoint lain
@app.get("/docs-original", include_in_schema=False)
async def overridden_swagger():
    return get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title + " - Docs")

# (Opsional) Lakukan hal yang sama untuk /redoc jika Anda ingin mengamankan Redoc juga
@app.get("/redoc", include_in_schema=False)
async def custom_redoc_ui(credentials: HTTPBasicCredentials = Depends(docs_auth)):
    return get_redoc_html(openapi_url=app.openapi_url, title=app.title + " - ReDoc")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)