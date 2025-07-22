import os
import sys
import argparse
import uvicorn
import numpy as np

def run_fastapi_server(host="0.0.0.0", port=8000, reload=False):
    """Run the FastAPI server"""
    print(f"🌐 Starting FastAPI server on {host}:{port}")
    
    try:
        uvicorn.run(
            "fastapi_app_fixed:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ FastAPI server failed: {e}")

def train_model_directly():
    """Train the model directly without TFX pipeline"""
    print("🎯 Training model directly...")
    
    try:
        from data_processing import DataProcessor
        from model import RecommendationModel
        import tensorflow as tf
        
        # Set TensorFlow environment
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        
        # Initialize components
        data_processor = DataProcessor()
        recommendation_model = RecommendationModel()
        
        # Load and process data
        print("📊 Loading and processing data...")
        dataset, _ = data_processor.load_and_process_data()
        dataset_interaction = data_processor.create_interaction_dataset(dataset)
        
        # Convert to TensorFlow datasets with proper data types
        selected_cols = [
            "user_id", "item_id", "category", "category2", "category3",
            "region", 'city', "item_id_lastview", "item_id_currentview", "label","timestamp_unix"
        ]
        
        # Ensure all data is properly formatted
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
        program_studi = data_processor.load_program_studi_data()
        
        # Ensure program studi data is properly formatted
        for col in program_studi.columns:
            if col in ["item_id", "category", "category2", "category3", "region"]:
                program_studi[col] = program_studi[col].astype(str)
        
        movies = tf.data.Dataset.from_tensor_slices(dict(program_studi))
        
        # Build and train model
        print("️ Building model...")
        recommendation_model.build_model(ratings, movies)
        
        print("🎯 Training model...")
        recommendation_model.train_model(ratings, epochs=15)
        
        # Create index and lookup
        print(" Creating index and lookup...")
        recommendation_model.create_index(movies)
        recommendation_model.create_item_lookup(movies)
        
        # Save model (try to save, but don't fail if it doesn't work)
        print(" Saving model...")
        try:
            recommendation_model.save_model("saved_model")
            print("✅ Model saved successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not save model: {e}")
            print(" Model is ready for recommendations but not saved.")
        
        print("✅ Model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Recommendation System with FastAPI")
    parser.add_argument(
        "--mode",
        choices=["direct", "server", "full"],
        default="server",
        help="Mode to run: direct (training), server (FastAPI only), full (training + server)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for FastAPI server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for FastAPI server"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for FastAPI server"
    )
    
    args = parser.parse_args()
    
    print("🎯 Recommendation System with FastAPI")
    print("=" * 50)
    
    if args.mode == "direct":
        # Train model directly
        success = train_model_directly()
        if success:
            print("🎉 Model training completed!")
        else:
            print("💥 Model training failed!")
            sys.exit(1)
    
    elif args.mode == "server":
        # Run FastAPI server only
        run_fastapi_server(args.host, args.port, args.reload)
    
    elif args.mode == "full":
        # Run complete pipeline and server
        print("🔄 Running complete pipeline...")
        
        # First, train the model
        print("1️⃣ Training model...")
        success = train_model_directly()
        
        if success:
            print("✅ Model training completed!")
            
            # Then start the server
            print("2️⃣ Starting FastAPI server...")
            run_fastapi_server(args.host, args.port, args.reload)
        else:
            print("💥 Model training failed! Cannot start server.")
            sys.exit(1)

if __name__ == "__main__":
    main()
