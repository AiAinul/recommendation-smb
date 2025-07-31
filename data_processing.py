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

    def create_interaction_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Create the final interaction dataset with all features"""
        
        # Filter out users who only interact with 1 item_id
        user_item_counts = dataset.groupby('user_id')['item_id'].nunique()
        users_with_multiple_items = user_item_counts[user_item_counts > 1].index
        dataset_filtered = dataset[dataset['user_id'].isin(users_with_multiple_items)]
        
        print(f"📊 Data filtering: Removed {len(dataset) - len(dataset_filtered)} interactions from users with only 1 item")
        print(f"📊 Remaining: {len(dataset_filtered)} interactions from {len(users_with_multiple_items)} users")
        
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

    def create_ranking_dataset(self, dataset: pd.DataFrame) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create 2D tensor datasets for TensorFlow Ranking
        Returns: (user_features_dataset, item_features_dataset)
        """
        print("🔄 Creating ranking datasets with 2D tensors...")
        
        # Group by user_id to create lists of interactions
        user_groups = dataset.groupby('user_id')
        
        user_features_list = []
        item_features_list = []
        
        for user_id, group in user_groups:
            if len(group) < 2:  # Skip users with less than 2 interactions
                continue
                
            # Sort by timestamp
            group = group.sort_values('timestamp_unix')
            
            # Create user features (same for all items in the group)
            user_features = {
                'user_id': [user_id] * len(group),
                'region': [group['region'].iloc[0]] * len(group),
                'city': [group['city'].iloc[0]] * len(group),
                'item_id_currentview': [group['item_id'].iloc[-1]] * len(group),  # Current item
                'timestamp_unix': [group['timestamp_unix'].iloc[-1]] * len(group),  # Current timestamp
                'item_id_lastview': [group['item_id'].iloc[-2] if len(group) > 1 else group['item_id'].iloc[-1]] * len(group)
            }
            
            # Create item features (different for each item)
            item_features = {
                'item_id': group['item_id'].tolist(),
                'category': group['category'].tolist(),
                'category2': group['category2'].tolist(),
                'category3': group['category3'].tolist(),
                'label': group['label'].tolist()
            }
            
            user_features_list.append(user_features)
            item_features_list.append(item_features)
        
        # Convert to TensorFlow datasets
        user_features_dataset = tf.data.Dataset.from_tensor_slices(user_features_list)
        item_features_dataset = tf.data.Dataset.from_tensor_slices(item_features_list)
        
        print(f"✅ Created ranking datasets: {len(user_features_list)} user groups")
        return user_features_dataset, item_features_dataset

    def create_ranking_dataset_simple(self, dataset: pd.DataFrame) -> tf.data.Dataset:
        """
        Create a 2D tensor dataset for ranking with proper tensor shapes
        Returns a dataset where each element is a group of items for a user
        """
        print("🔄 Creating 2D ranking dataset...")
        
        # Group by user_id
        user_groups = dataset.groupby('user_id')
        
        ranking_data = []
        
        for user_id, group in user_groups:
            if len(group) < 2:  # Skip users with less than 2 interactions
                continue
                
            # Sort by timestamp
            group = group.sort_values('timestamp_unix')
            
            # Create a single record with all items for this user
            user_record = {
                'user_id': user_id,
                'region': group['region'].iloc[0],
                'city': group['city'].iloc[0],
                'item_id_currentview': group['item_id'].iloc[-1],
                'timestamp_unix': group['timestamp_unix'].iloc[-1],
                'item_id_lastview': group['item_id'].iloc[-2] if len(group) > 1 else group['item_id'].iloc[-1],
                'item_ids': group['item_id'].tolist(),
                'categories': group['category'].tolist(),
                'categories2': group['category2'].tolist(),
                'categories3': group['category3'].tolist(),
                'labels': group['label'].tolist()
            }
            
            ranking_data.append(user_record)
        
        # Convert to TensorFlow dataset
        ranking_dataset = tf.data.Dataset.from_tensor_slices(ranking_data)
        
        print(f"✅ Created 2D ranking dataset: {len(ranking_data)} user groups")
        return ranking_dataset

    def create_ranking_dataset_2d(self, dataset: pd.DataFrame) -> tf.data.Dataset:
        """
        Create proper 2D tensor dataset for TensorFlow Ranking
        Returns dataset with 2D tensors for labels and predictions
        """
        print("🔄 Creating proper 2D ranking dataset...")
        
        # Group by user_id
        user_groups = dataset.groupby('user_id')
        
        ranking_data = []
        
        for user_id, group in user_groups:
            if len(group) < 2:  # Skip users with less than 2 interactions
                continue
                
            # Sort by timestamp
            group = group.sort_values('timestamp_unix')
            
            # Create 2D tensors for labels and item features
            num_items = len(group)
            
            # Create user features (same for all items)
            user_features = {
                'user_id': [user_id] * num_items,
                'region': [group['region'].iloc[0]] * num_items,
                'city': [group['city'].iloc[0]] * num_items,
                'item_id_currentview': [group['item_id'].iloc[-1]] * num_items,
                'timestamp_unix': [group['timestamp_unix'].iloc[-1]] * num_items,
                'item_id_lastview': [group['item_id'].iloc[-2] if len(group) > 1 else group['item_id'].iloc[-1]] * num_items
            }
            
            # Create item features (different for each item)
            item_features = {
                'item_id': group['item_id'].tolist(),
                'category': group['category'].tolist(),
                'category2': group['category2'].tolist(),
                'category3': group['category3'].tolist(),
                'label': group['label'].tolist()
            }
            
            # Create 2D tensor record
            record = {
                'user_features': user_features,
                'item_features': item_features,
                'labels_2d': tf.reshape(tf.constant(group['label'].tolist()), [1, -1]),  # [1, num_items]
                'num_items': num_items
            }
            
            ranking_data.append(record)
        
        # Convert to TensorFlow dataset
        ranking_dataset = tf.data.Dataset.from_tensor_slices(ranking_data)
        
        print(f"✅ Created proper 2D ranking dataset: {len(ranking_data)} user groups")
        return ranking_dataset 