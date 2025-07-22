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
            1,
            0,
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
        
        dataset_interaction = dataset.sort_values(by=["user_id", "timestamp"])
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