import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple

# Set TensorFlow environment
os.environ['TF_USE_LEGACY_KERAS'] = '1'

class DataProcessor:
    def __init__(self):
        self.item_view_url = os.environ.get('ITEM_VIEW_URL')
        self.item_purchase_url = os.environ.get('ITEM_PURCHASE_URL')
        self.program_studi_url = os.environ.get('PROGRAM_STUDI_URL')
        self.recommendation_dataset_url = os.environ.get('RECOMMENDATION_DATASET_URL')
        self.recommendation_feedback_dataset_url = os.environ.get('RECOMMENDATION_DATASET_URL')
        # Optional: error handling jika env tidak di-set
        if not self.item_view_url or not self.item_purchase_url or not self.program_studi_url:
            raise ValueError("One or more data URLs are not set in environment variables.")
        
    def load_and_process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process item view and purchase data"""
        
        # Load item view data
        item_view = pd.read_csv(self.item_view_url)
        item_view['timestamp'] = pd.to_datetime(item_view['timestamp'])
        item_view_selected = item_view.sort_values('timestamp')
        item_view_deduplication = item_view_selected.drop_duplicates(
            subset=[
                'user_id', 'event_name', 'item_id', 'item_category',
                'item_category2', 'item_category3'
            ],
            keep='last'
        )
        item_view_selected = item_view_deduplication[[
            'user_id', 'event_name', 'item_id',
            'item_category', 'item_category2', 'item_category3',
            'event_country', 'event_region', 'region', 'city', 'timestamp'
        ]]

        # Load item purchase data
        item_purchase = pd.read_csv(self.item_purchase_url)
        item_purchase['timestamp'] = pd.to_datetime(item_purchase['timestamp'])
        item_purchase = item_purchase.sort_values('timestamp')
        item_purchase_deduplication = item_purchase.drop_duplicates(
            subset=[
                'user_id', 'event_name', 'item_id', 'item_category',
                'item_category2', 'item_category3', 'sdp2', 'transaction_id'
            ],
            keep='last'
        )
        item_purchase_selected = item_purchase_deduplication[[
            'user_id', 'event_name', 'item_id',
            'item_category', 'item_category2', 'item_category3',
            'event_country', 'event_region', 'region', 'city', 'timestamp'
        ]]

        # Combine datasets
        dataset = pd.concat([item_view_selected, item_purchase_selected], ignore_index=True)
        dataset['label'] = np.where(
            dataset.event_name.isin(['purchase']),
            1.0,
            0.2,
        )
        dataset['item_id'] = dataset['item_id'].astype(str).str.strip()
        
        return dataset, item_view_selected

    def find_last_different_item(self, group):
        """Find the last different item for each user interaction"""
        item_ids = group["item_id"].tolist()
        last_views = []

        for i in range(len(item_ids)):
            found = False
            for j in range(i - 1, -1, -1):
                if item_ids[j] != item_ids[i]:
                    last_views.append(item_ids[j])
                    found = True
                    break
            if not found:
                last_views.append(item_ids[i])

        group["item_id_lastview"] = last_views
        return group

    def create_interaction_dataset(self, dataset: pd.DataFrame, min_items: int = 2) -> pd.DataFrame:
        """Create the final interaction dataset with all features"""
        
        # Filter out users who interact with fewer than min_items unique items
        user_item_counts = dataset.groupby('user_id')['item_id'].nunique()
        users_with_sufficient_items = user_item_counts[user_item_counts >= min_items].index
        dataset_filtered = dataset[dataset['user_id'].isin(users_with_sufficient_items)]
        
        print(f"📊 Data filtering: Removed {len(dataset) - len(dataset_filtered)} interactions from users with less than {min_items} items")
        print(f"📊 Remaining: {len(dataset_filtered)} interactions from {len(users_with_sufficient_items)} users")
        
        dataset_interaction = dataset_filtered.sort_values(by=["user_id", "timestamp"])
        dataset_interaction = (
            dataset_interaction
            .groupby("user_id", group_keys=False)
            .apply(self.find_last_different_item)
        )
        dataset_interaction["item_id_currentview"] = dataset_interaction["item_id"]

        # Convert timestamp to unix
        dataset_interaction['timestamp'] = pd.to_datetime(dataset_interaction['timestamp'], errors='coerce')
        dataset_interaction['timestamp_unix'] = dataset_interaction['timestamp'].astype(np.int64) // 10**9
        
        # Select and format columns
        dataset_interaction = dataset_interaction[[
            'user_id', 'event_name', 'item_id',
            'item_category', 'item_category2', 'item_category3',
            'event_country', 'event_region', 'region', 'city', 
            'item_id_lastview','item_id_currentview', 'label', 'timestamp_unix'
        ]]

        # Convert to string types
        string_columns = [
            "user_id", "item_id", "item_category", "item_category2", 
            "item_category3", "event_country", "event_region", 
            "item_id_lastview", "item_id_currentview"
        ]
        
        for col in string_columns:
            dataset_interaction[col] = dataset_interaction[col].astype(str)
            
        dataset_interaction["label"] = dataset_interaction["label"].astype(np.float32)
        dataset_interaction["region"] = dataset_interaction["region"].astype(str)
        dataset_interaction["city"] = dataset_interaction["city"].astype(str)
        dataset_interaction["timestamp_unix"] = dataset_interaction["timestamp_unix"].astype(np.int64)

        # Rename columns for consistency
        dataset_interaction = dataset_interaction.rename(columns={
            'item_category': 'category',
            'item_category2': 'category2', 
            'item_category3': 'category3',
            'region': 'region',
            'city': 'city'
        })

        return dataset_interaction

    def create_ranking_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Create dataset for ranking training - users with more than 5 items
        """
        print("🔄 Creating ranking dataset (users with >5 items)...")
        return self.create_interaction_dataset(dataset, min_items=3)  # >5 means >=6

    def create_retrieval_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Create dataset for retrieval - users with minimum 3 items
        """
        print("🔄 Creating retrieval dataset (users with >=3 items)...")
        return self.create_interaction_dataset(dataset, min_items=3)

    def load_program_studi_data(self) -> pd.DataFrame:
        """Load program studi data"""
        program_studi = pd.read_csv(self.program_studi_url)
        program_studi["item_id"] = program_studi["item_id"].astype(str)
        program_studi["category"] = program_studi["item_category"].astype(str)
        program_studi["category2"] = program_studi["item_category2"].astype(str)
        program_studi["category3"] = program_studi["item_category3"].astype(str)
        
        selected_cols = ["item_id", "category", "category2", "category3"]
        return program_studi[selected_cols]

    def get_dataset_stats(self, dataset: pd.DataFrame) -> Dict:
        """Get dataset statistics"""
        return {
            'num_users': dataset['user_id'].nunique(),
            'num_items': dataset['item_id'].nunique(),
            'num_interactions': len(dataset),
            'num_purchases': dataset['label'].sum()
        }

    def create_timestamp_normalization_layer(self, ratings_dataset):
        """Create and adapt timestamp normalization layer"""
        timestamp_normalization_layer = tf.keras.layers.Normalization(axis=None)
        
        # Convert timestamp_unix to float32 for proper normalization
        def extract_timestamp(x):
            return tf.cast(x['timestamp_unix'], tf.float32)
        
        # Adapt the normalization layer with timestamp data
        timestamp_normalization_layer.adapt(
            ratings_dataset.map(extract_timestamp)
        )
        
        print(f"✅ Timestamp normalization layer created and adapted")
        return timestamp_normalization_layer

    def load_recommendation_dataset(self) -> pd.DataFrame:
        """Load and process recommendation feedback dataset"""
        if not self.recommendation_dataset_url:
            print("⚠️ RECOMMENDATION_DATASET_URL not set, skipping recommendation dataset")
            return pd.DataFrame()
            
        try:
            print("📊 Loading recommendation feedback dataset...")
            recommendation_df = pd.read_csv(self.recommendation_dataset_url)
            
            # Store original count for accurate calculation
            original_count = len(recommendation_df)
            print(f"📊 Original dataset: {original_count} rows")
            
            # Convert current_item_id to string for consistent comparison
            recommendation_df['current_item_id'] = recommendation_df['current_item_id'].astype(str)
            
            # Remove duplicates based on the specified columns, keeping the last occurrence
            recommendation_df = recommendation_df.drop_duplicates(
                subset=['user_id', 'current_item_id', 'recommendation_group'],
                keep='last'
            )
            
            final_count = len(recommendation_df)
            removed_count = original_count - final_count
            
            print(f"📊 After deduplication: {final_count} rows")
            print(f"📊 Removed {removed_count} duplicate rows")
            
            # Parse recommendation groups and create training examples
            training_examples = []
            
            for _, row in recommendation_df.iterrows():
                user_id = row['user_id']
                current_item_id = str(row['current_item_id'])
                recommendation_group = row['recommendation_group']
                timestamp = pd.to_datetime(row['timestamp'])
                
                # Parse recommendation group (convert string representation to list)
                if isinstance(recommendation_group, str):
                    try:
                        # Remove brackets and quotes, split by comma
                        rec_items = recommendation_group.strip('[]').replace('"', '').split(',')
                        rec_items = [item.strip() for item in rec_items if item.strip()]
                    except:
                        rec_items = []
                else:
                    rec_items = []
                
                # Create positive examples (items that were recommended and potentially clicked)
                for i, rec_item in enumerate(rec_items):
                    if rec_item and rec_item != current_item_id:
                        training_examples.append({
                            'user_id': user_id,
                            'item_id': rec_item,
                            'current_item_id': current_item_id,
                            'label': 1.0,  # Positive example
                            'rank': i + 1,  # Position in recommendation list
                            'timestamp': timestamp,
                            'source': 'recommendation_feedback'
                        })
                
                # Create negative examples (items not in recommendation list)
                # We'll add some negative sampling later
                
            recommendation_training_df = pd.DataFrame(training_examples)
            
            if len(recommendation_training_df) > 0:
                print(f"✅ Loaded {len(recommendation_training_df)} recommendation training examples")
                print(f"📊 Positive examples: {len(recommendation_training_df[recommendation_training_df['label'] == 1.0])}")
            else:
                print("⚠️ No recommendation training examples found")
                
            return recommendation_training_df
            
        except Exception as e:
            print(f"❌ Error loading recommendation dataset: {e}")
            return pd.DataFrame()

    def create_enhanced_training_dataset(self, base_dataset: pd.DataFrame, recommendation_dataset: pd.DataFrame = None) -> pd.DataFrame:
        """Create enhanced training dataset by combining base data with recommendation feedback"""
        
        enhanced_dataset = base_dataset.copy()
        
        if recommendation_dataset is not None and len(recommendation_dataset) > 0:
            print("🔄 Enhancing training dataset with recommendation feedback...")
            
            # Add recommendation feedback data
            # We need to match the column structure of the base dataset
            if 'source' not in enhanced_dataset.columns:
                enhanced_dataset['source'] = 'base_data'
            
            # Prepare recommendation data to match base dataset structure
            if len(recommendation_dataset) > 0:
                # Get program studi data for category information
                program_studi = self.load_program_studi_data()
                program_studi_dict = program_studi.set_index('item_id').to_dict('index')
                
                enhanced_rec_data = []
                for _, row in recommendation_dataset.iterrows():
                    item_id = str(row['item_id'])
                    current_item_id = str(row['current_item_id'])
                    
                    # Get category information
                    item_info = program_studi_dict.get(item_id, {})
                    current_item_info = program_studi_dict.get(current_item_id, {})
                    
                    # Ensure all required columns exist with proper data types
                    enhanced_rec_data.append({
                        'user_id': str(row['user_id']),
                        'item_id': str(item_id),
                        'current_item_id': str(current_item_id),
                        'label': float(row['label']),
                        'rank': int(row.get('rank', 0)),
                        'timestamp': row['timestamp'],
                        'source': 'recommendation_feedback',
                        # Add category information
                        'category': str(item_info.get('category', 'unknown')),
                        'category2': str(item_info.get('category2', 'unknown')),
                        'category3': str(item_info.get('category3', 'unknown')),
                        'current_category': str(current_item_info.get('category', 'unknown')),
                        'current_category2': str(current_item_info.get('category2', 'unknown')),
                        'current_category3': str(current_item_info.get('category3', 'unknown')),
                        # Add dummy values for required fields
                        'event_name': 'recommendation_feedback',
                        'item_category': str(item_info.get('category', 'unknown')),
                        'item_category2': str(item_info.get('category2', 'unknown')),
                        'item_category3': str(item_info.get('category3', 'unknown')),
                        'event_country': 'ID',
                        'event_region': 'unknown',
                        'region': 'unknown',
                        'city': 'unknown',
                        'item_id_lastview': str(current_item_id),
                        'item_id_currentview': str(current_item_id),
                        'timestamp_unix': int(row['timestamp'].timestamp()) if hasattr(row['timestamp'], 'timestamp') else 0
                    })
                
                rec_df = pd.DataFrame(enhanced_rec_data)
                
                # Ensure all columns in rec_df match the base dataset
                for col in enhanced_dataset.columns:
                    if col not in rec_df.columns:
                        if col == 'label':
                            rec_df[col] = 1.0
                        elif col == 'timestamp_unix':
                            rec_df[col] = 0
                        else:
                            rec_df[col] = 'unknown'
                
                # Combine datasets
                enhanced_dataset = pd.concat([enhanced_dataset, rec_df], ignore_index=True)
                
                print(f"✅ Enhanced dataset: {len(enhanced_dataset)} total examples")
                print(f"📊 Base data: {len(base_dataset)} examples")
                print(f"📊 Recommendation feedback: {len(rec_df)} examples")
                
        return enhanced_dataset

    def convert_tf_dataset_to_pandas(self, tf_dataset, batch_size=1000):
        """Convert TensorFlow dataset to pandas DataFrame"""
        print("🔄 Converting TensorFlow dataset to pandas DataFrame...")
        
        try:
            ratings_data = []
            for batch in tf_dataset.batch(batch_size):
                # Convert batch to dictionary with proper data types
                batch_data = {}
                for key, value in batch.items():
                    if hasattr(value, 'numpy'):
                        # Convert to list of strings for string columns, keep numeric as is
                        if key in ['user_id', 'item_id', 'category', 'category2', 'category3', 
                                 'region', 'city', 'item_id_lastview', 'item_id_currentview']:
                            batch_data[key] = [str(x) for x in value.numpy()]
                        else:
                            batch_data[key] = value.numpy().tolist()
                
                # Create DataFrame from this batch
                batch_df = pd.DataFrame(batch_data)
                ratings_data.append(batch_df)
            
            # Combine all batches
            if ratings_data:
                ratings_df = pd.concat(ratings_data, ignore_index=True)
                print(f"✅ Converted TensorFlow dataset to DataFrame: {len(ratings_df)} rows")
                return ratings_df
            else:
                print("⚠️ No data in TensorFlow dataset")
                return None
                
        except Exception as e:
            print(f"❌ Error converting TensorFlow dataset: {e}")
            return None

    def convert_pandas_to_tf_dataset(self, df, selected_cols=None):
        """Convert pandas DataFrame to TensorFlow dataset with proper data types"""
        if selected_cols is None:
            selected_cols = [
                "user_id", "item_id", "category", "category2", "category3",
                "region", 'city', "item_id_lastview", "item_id_currentview", "label", "timestamp_unix"
            ]
        
        print("🔄 Converting pandas DataFrame to TensorFlow dataset...")
        
        try:
            # Ensure all data is properly formatted
            for col in selected_cols:
                if col in df.columns:
                    if col == "label":
                        df[col] = df[col].astype(np.float32)
                    elif col == "timestamp_unix":
                        df[col] = df[col].astype(np.int64)
                    else:
                        df[col] = df[col].astype(str)
            
            # Convert to TensorFlow dataset
            tf_dataset = tf.data.Dataset.from_tensor_slices(dict(df[selected_cols]))
            print(f"✅ Converted DataFrame to TensorFlow dataset: {len(df)} rows")
            return tf_dataset
            
        except Exception as e:
            print(f"❌ Error converting DataFrame to TensorFlow dataset: {e}")
            return None 