# Code Structure Documentation

## 📁 **File Organization**

### 1. **`data_processing.py`** - Data Processing Layer
**Responsibilities:**
- Loading and processing raw data
- Data filtering and validation
- Dataset creation for different training types
- TensorFlow ↔ Pandas conversion utilities
- Recommendation feedback integration

**Key Methods:**
```python
# Data Loading
- load_and_process_data()
- load_recommendation_dataset()
- load_program_studi_data()

# Dataset Creation
- create_interaction_dataset(min_items=2)
- create_ranking_dataset()  # users with >5 items
- create_retrieval_dataset()  # users with >=3 items
- create_enhanced_training_dataset()

# Data Conversion Utilities
- convert_tf_dataset_to_pandas()
- convert_pandas_to_tf_dataset()

# Model Support
- create_timestamp_normalization_layer()
```

### 2. **`model.py`** - Model Layer
**Responsibilities:**
- Model architecture definition
- Training logic
- Recommendation generation
- Model evaluation

**Key Classes:**
```python
# Model Architectures
- UserModel
- ItemModel
- MyTwoTowerModel (Retrieval)
- RankingModel

# Main Model Class
- RecommendationModel
  - build_model()
  - train_model()
  - train_ranking_model()
  - train_ranking_model_with_feedback()
  - get_recommendations()
  - evaluate_recommendation_accuracy()
```

### 3. **`fastapi_app_fixed.py`** - API Layer
**Responsibilities:**
- HTTP endpoints
- Request/response handling
- Background training orchestration
- Model management

**Key Endpoints:**
```python
# Training Endpoints
- POST /train (Retrieval model)
- POST /train/ranking (Ranking model with feedback)
- POST /train/full (Both models)

# Recommendation Endpoints
- POST /recommendations
- POST /recommendations/with-ranking

# Monitoring Endpoints
- GET /model/info
- GET /dashboard/*
- GET /training/status
```

## 🔄 **Data Flow with Recommendation Feedback**

### 1. **Training Flow (`/train/ranking`)**
```
1. Load Base Data
   ↓
2. Create Ranking Dataset (users >5 items)
   ↓
3. Load Recommendation Feedback
   ↓
4. Convert TF Dataset → Pandas (data_processing.py)
   ↓
5. Create Enhanced Dataset (data_processing.py)
   ↓
6. Convert Pandas → TF Dataset (data_processing.py)
   ↓
7. Train Ranking Model (model.py)
```

### 2. **Recommendation Generation Flow**
```
1. User Request
   ↓
2. Get Available Model (API)
   ↓
3. Generate Recommendations (model.py)
   ↓
4. Enrich with Program Studi Data (API)
   ↓
5. Return Response
```

## 🏗️ **Separation of Concerns**

### **Data Processing Layer (`data_processing.py`)**
- ✅ **Data Loading**: CSV, URL, environment variables
- ✅ **Data Filtering**: User interaction counts, data quality
- ✅ **Data Conversion**: TensorFlow ↔ Pandas utilities
- ✅ **Dataset Creation**: Ranking, retrieval, enhanced datasets
- ✅ **Data Validation**: Type checking, format validation

### **Model Layer (`model.py`)**
- ✅ **Model Architecture**: User/Item models, ranking models
- ✅ **Training Logic**: Standard and feedback-enhanced training
- ✅ **Recommendation Logic**: Retrieval and ranking algorithms
- ✅ **Model Evaluation**: Accuracy metrics, feedback evaluation
- ✅ **Model Management**: Save/load, index creation

### **API Layer (`fastapi_app_fixed.py`)**
- ✅ **HTTP Endpoints**: RESTful API design
- ✅ **Request Handling**: Validation, authentication
- ✅ **Background Processing**: Non-blocking training
- ✅ **Model Orchestration**: Model switching, versioning
- ✅ **Monitoring**: Metrics, health checks, dashboards

## 🔧 **Key Improvements**

### 1. **Proper Data Conversion**
```python
# Centralized in data_processing.py
def convert_tf_dataset_to_pandas(self, tf_dataset):
    # Handles TensorFlow → Pandas conversion
    # Proper error handling and data type management

def convert_pandas_to_tf_dataset(self, df):
    # Handles Pandas → TensorFlow conversion
    # Ensures proper data types for TensorFlow
```

### 2. **Clean Model Training**
```python
# Clean separation of concerns
def train_ranking_model_with_feedback(self, ranking_dataset, feedback_dataset):
    # Uses data_processor for conversions
    # Focuses only on training logic
```

### 3. **Simplified API Logic**
```python
# Clean orchestration
def train_ranking_model_thread():
    # Uses data_processor methods
    # Focuses on API orchestration
```

## 📊 **Recommendation Feedback Integration**

### **Data Structure**
```python
# Recommendation Feedback Format
{
    'user_id': 'user@email.com',
    'current_item_id': '3325',
    'recommendation_group': '[3327, 3705, 3311, 3702, 3319]',
    'match': False,
    'rank': -1,
    'timestamp': '2024-01-01 10:00:00'
}
```

### **Processing Steps**
1. **Load Feedback**: Parse CSV from URL
2. **Parse Recommendations**: Convert string to list
3. **Create Training Examples**: Positive examples from recommendations
4. **Enhance Base Dataset**: Combine with interaction data
5. **Train Model**: Use enhanced dataset for better accuracy

### **Benefits**
- ✅ **Improved Accuracy**: Model learns from actual user feedback
- ✅ **Context Awareness**: Uses current item context
- ✅ **Ranking Quality**: Learns from recommendation positions
- ✅ **Fallback Safety**: Graceful degradation if feedback unavailable

## 🚀 **Usage Examples**

### **Training with Feedback**
```bash
# Train ranking model with recommendation feedback
POST /train/ranking
{
    "epochs": 15,
    "batch_size": 4096
}
```

### **Getting Recommendations**
```bash
# Standard recommendations
POST /recommendations
{
    "user_id": "user@email.com",
    "current_item_id": "3325",
    "region": "west java",
    "city": "bandung",
    "top_k": 5
}

# Recommendations with ranking
POST /recommendations/with-ranking
{
    "user_id": "user@email.com",
    "current_item_id": "3325",
    "region": "west java",
    "city": "bandung",
    "top_k": 5
}
```

## 🔍 **Monitoring and Debugging**

### **Training Status**
```bash
GET /training/status
GET /model/status
GET /dashboard/training-metrics
```

### **Data Quality**
```bash
GET /data/stats
GET /dashboard/data-quality
GET /model/evaluate-accuracy
```

## 🧹 **Code Cleanup Summary**

### **Removed Unused Methods:**
- ❌ `create_ranking_dataset_2d()` - Not used in main system
- ❌ `create_retrieval_dataset_2d()` - Not used in main system  
- ❌ `train_ranking_model_2d()` - Not used in main system
- ❌ `create_ranking_dataset_with_feedback()` - Redundant with enhanced approach

### **Benefits of Cleanup:**
- ✅ **Reduced Complexity**: Fewer unused methods
- ✅ **Better Maintainability**: Cleaner codebase
- ✅ **Focused Functionality**: Only essential methods remain
- ✅ **Consistent Approach**: Standard training methods throughout

This structure ensures clean separation of concerns, maintainable code, and proper error handling throughout the recommendation system. 