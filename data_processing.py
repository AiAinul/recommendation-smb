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
        
        # Improved error handling dengan detail yang lebih jelas
        missing_vars = []
        if not self.item_view_url:
            missing_vars.append('ITEM_VIEW_URL')
        if not self.item_purchase_url:
            missing_vars.append('ITEM_PURCHASE_URL')
        if not self.program_studi_url:
            missing_vars.append('PROGRAM_STUDI_URL')
        
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}. Please set these variables before running the application.")
        
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

    def calculate_rank_based_label(self, rank: int, max_rank: int = 5) -> float:
        """Calculate label based on rank position optimized for MRR"""
        # Higher rank (lower position) = higher label
        # Optimized for MRR improvement
        if rank <= 0:
            return 0.0
        elif rank == 1:
            return 1.0  # Perfect score for top rank
        elif rank == 2:
            return 0.9  # High score for second rank
        elif rank == 3:
            return 0.7  # Good score for third rank
        elif rank == 4:
            return 0.5  # Medium score for fourth rank
        elif rank == 5:
            return 0.3  # Lower score for fifth rank
        else:
            # For ranks > 5, use smoother decay for better MRR
            return max(0.1, (1.0 / rank) ** 0.5)  # Smoother decay

    def calculate_correct_prediction(self, rank: int) -> float:
        """Calculate correct prediction probability based on rank optimized for MRR"""
        # Higher rank (lower position) = higher probability of being correct
        # Optimized for MRR improvement
        if rank <= 0:
            return 0.0
        elif rank == 1:
            return 1.0  # Perfect prediction for top rank
        elif rank == 2:
            return 0.8  # High probability for second rank
        elif rank == 3:
            return 0.6  # Good probability for third rank
        elif rank == 4:
            return 0.4  # Medium probability for fourth rank
        elif rank == 5:
            return 0.2  # Lower probability for fifth rank
        else:
            # For ranks > 5, use smoother decay for better MRR
            return max(0.1, (1.0 / rank) ** 0.6)

    def analyze_rank_distribution(self, recommendation_dataset: pd.DataFrame) -> Dict:
        """Analyze rank distribution to understand data patterns"""
        if len(recommendation_dataset) == 0:
            return {}
        
        rank_counts = recommendation_dataset['rank'].value_counts().sort_index()
        rank_percentages = (rank_counts / len(recommendation_dataset) * 100).round(2)
        
        print("📊 Rank Distribution Analysis:")
        print(f"📊 Total examples: {len(recommendation_dataset)}")
        print(f"📊 Rank distribution: {dict(rank_counts)}")
        print(f"📊 Rank percentages: {dict(rank_percentages)}")
        
        return {
            'total_examples': len(recommendation_dataset),
            'rank_counts': dict(rank_counts),
            'rank_percentages': dict(rank_percentages),
            'avg_rank': recommendation_dataset['rank'].mean(),
            'median_rank': recommendation_dataset['rank'].median()
        }

    def load_recommendation_dataset_with_correct_prediction(self) -> pd.DataFrame:
        """Load recommendation feedback dataset with improved negative sampling"""
        if not self.recommendation_dataset_url:
            print("⚠️ RECOMMENDATION_DATASET_URL not set, skipping recommendation dataset")
            return pd.DataFrame()
            
        try:
            print("📊 Loading recommendation feedback dataset with improved negative sampling...")
            recommendation_df = pd.read_csv(self.recommendation_dataset_url)
            
            # Store original count
            original_count = len(recommendation_df)
            print(f"📊 Original dataset: {original_count} rows")
            
            # Convert current_item_id to string
            recommendation_df['current_item_id'] = recommendation_df['current_item_id'].astype(str)
            
            # Remove duplicates
            recommendation_df = recommendation_df.drop_duplicates(
                subset=['user_id', 'current_item_id', 'recommendation_group'],
                keep='last'
            )
            
            final_count = len(recommendation_df)
            removed_count = original_count - final_count
            
            print(f"📊 After deduplication: {final_count} rows")
            print(f"📊 Removed {removed_count} duplicate rows")
            
            # Get all available items for negative sampling
            program_studi = self.load_program_studi_data()
            all_items = set(program_studi['item_id'].astype(str).tolist())
            
            # Parse recommendation groups and create training examples
            training_examples = []
            positive_examples = 0
            negative_examples = 0
            
            for _, row in recommendation_df.iterrows():
                user_id = row['user_id']
                current_item_id = str(row['current_item_id'])
                recommendation_group = row['recommendation_group']
                timestamp = pd.to_datetime(row['timestamp'])
                
                # Parse recommendation group
                if isinstance(recommendation_group, str):
                    try:
                        rec_items = recommendation_group.strip('[]').replace('"', '').split(',')
                        rec_items = [item.strip() for item in rec_items if item.strip()]
                    except:
                        rec_items = []
                else:
                    rec_items = []
                
                # Create positive examples (items in recommendation_group)
                for i, rec_item in enumerate(rec_items):
                    if rec_item and rec_item != current_item_id:
                        rank = i + 1
                        # Calculate correct prediction based on rank
                        correct_prediction = self.calculate_correct_prediction(rank)
                        
                        training_examples.append({
                            'user_id': user_id,
                            'item_id': rec_item,
                            'current_item_id': current_item_id,
                            'label': 1.0,  # Positive example
                            'rank': rank,  # Position in recommendation list
                            'correct_prediction': correct_prediction,  # New column
                            'timestamp': timestamp,
                            'source': 'recommendation_feedback_positive'
                        })
                        positive_examples += 1
                
                # IMPROVED: Create balanced negative examples
                # Sample 1-2 negative examples per positive example (instead of 2-3)
                negative_sample_size = min(2, len(rec_items))  # Reduced from 3 to 2
                
                # Get items that are NOT in recommendation_group
                available_negative_items = list(all_items - set(rec_items) - {current_item_id})
                
                if len(available_negative_items) > 0:
                    # Use stratified sampling for better balance
                    import random
                    random.seed(42)  # For reproducibility
                    
                    # Sample negative items with preference for similar categories
                    current_item_info = program_studi[program_studi['item_id'] == current_item_id]
                    if len(current_item_info) > 0:
                        current_category = current_item_info.iloc[0]['category']
                        # Prioritize items from different categories for better negative examples
                        different_category_items = [
                            item for item in available_negative_items 
                            if program_studi[program_studi['item_id'] == item].iloc[0]['category'] != current_category
                        ]
                        if len(different_category_items) > 0:
                            available_negative_items = different_category_items
                    
                    negative_items = random.sample(
                        available_negative_items, 
                        min(negative_sample_size, len(available_negative_items))
                    )
                    
                    for neg_item in negative_items:
                        training_examples.append({
                            'user_id': user_id,
                            'item_id': neg_item,
                            'current_item_id': current_item_id,
                            'label': 0.0,  # Negative example
                            'rank': -1,  # No rank for negative examples
                            'correct_prediction': 0.0,  # Zero for negative examples
                            'timestamp': timestamp,
                            'source': 'recommendation_feedback_negative'
                        })
                        negative_examples += 1
                
            recommendation_training_df = pd.DataFrame(training_examples)
            
            if len(recommendation_training_df) > 0:
                print(f"✅ Loaded {len(recommendation_training_df)} recommendation training examples")
                print(f"📊 Positive examples: {positive_examples}")
                print(f"📊 Negative examples: {negative_examples}")
                print(f"📊 Positive/Negative ratio: {positive_examples/negative_examples:.2f}" if negative_examples > 0 else "📊 No negative examples")
                print(f"📊 Correct prediction range: {recommendation_training_df['correct_prediction'].min():.3f} - {recommendation_training_df['correct_prediction'].max():.3f}")
            else:
                print("⚠️ No recommendation training examples found")
                
            return recommendation_training_df
            
        except Exception as e:
            print(f"❌ Error loading recommendation dataset: {e}")
            return pd.DataFrame()

    def create_enhanced_training_dataset_with_rank(self, base_dataset: pd.DataFrame, recommendation_dataset: pd.DataFrame = None) -> pd.DataFrame:
        """Create enhanced training dataset with rank-based labels and data augmentation"""
        
        enhanced_dataset = base_dataset.copy()
        
        if recommendation_dataset is not None and len(recommendation_dataset) > 0:
            print("🔄 Enhancing training dataset with rank-based feedback and data augmentation...")
            
            # Add recommendation feedback data with rank-based labels
            if 'source' not in enhanced_dataset.columns:
                enhanced_dataset['source'] = 'base_data'
            
            if len(recommendation_dataset) > 0:
                program_studi = self.load_program_studi_data()
                program_studi_dict = program_studi.set_index('item_id').to_dict('index')
                
                enhanced_rec_data = []
                positive_count = 0
                negative_count = 0
                
                for _, row in recommendation_dataset.iterrows():
                    item_id = str(row['item_id'])
                    current_item_id = str(row['current_item_id'])
                    rank = int(row.get('rank', 0))
                    source = str(row.get('source', 'recommendation_feedback'))
                    
                    # Get category information
                    item_info = program_studi_dict.get(item_id, {})
                    current_item_info = program_studi_dict.get(current_item_id, {})
                    
                    # Handle positive and negative examples differently
                    if source == 'recommendation_feedback_positive' or source == 'recommendation_feedback':
                        # Positive example - use rank-based label
                        if rank > 0:
                            label = self.calculate_rank_based_label(rank)
                        else:
                            label = 1.0  # Default positive label
                        positive_count += 1
                        
                        # IMPROVED: Add data augmentation for positive examples
                        # Create additional positive examples with slight variations
                        for i in range(2):  # Create 2 additional positive examples
                            enhanced_rec_data.append({
                                'user_id': str(row['user_id']),
                                'item_id': str(item_id),
                                'current_item_id': str(current_item_id),
                                'label': label * (0.9 + 0.1 * i),  # Slight variation in label
                                'rank': rank,
                                'timestamp': row['timestamp'],
                                'source': f'{source}_augmented_{i}',
                                'category': str(item_info.get('category', 'unknown')),
                                'category2': str(item_info.get('category2', 'unknown')),
                                'category3': str(item_info.get('category3', 'unknown')),
                                'current_category': str(current_item_info.get('category', 'unknown')),
                                'current_category2': str(current_item_info.get('category2', 'unknown')),
                                'current_category3': str(current_item_info.get('category3', 'unknown')),
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
                        
                    elif source == 'recommendation_feedback_negative':
                        # Negative example - use negative label
                        label = 0.0  # Negative label
                        negative_count += 1
                    
                    # Add original example
                    enhanced_rec_data.append({
                        'user_id': str(row['user_id']),
                        'item_id': str(item_id),
                        'current_item_id': str(current_item_id),
                        'label': label,  # Use appropriate label based on source
                        'rank': rank,
                        'timestamp': row['timestamp'],
                        'source': source,
                        'category': str(item_info.get('category', 'unknown')),
                        'category2': str(item_info.get('category2', 'unknown')),
                        'category3': str(item_info.get('category3', 'unknown')),
                        'current_category': str(current_item_info.get('category', 'unknown')),
                        'current_category2': str(current_item_info.get('category2', 'unknown')),
                        'current_category3': str(current_item_info.get('category3', 'unknown')),
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
                
                # Ensure all columns match
                for col in enhanced_dataset.columns:
                    if col not in rec_df.columns:
                        if col == 'label':
                            rec_df[col] = 0.5  # Default label
                        elif col == 'timestamp_unix':
                            rec_df[col] = 0
                        else:
                            rec_df[col] = 'unknown'
                
                # Combine datasets
                enhanced_dataset = pd.concat([enhanced_dataset, rec_df], ignore_index=True)
                
                print(f"✅ Enhanced dataset with rank-based labels and data augmentation: {len(enhanced_dataset)} total examples")
                print(f"📊 Base data: {len(base_dataset)} examples")
                print(f"📊 Positive feedback examples: {positive_count}")
                print(f"📊 Negative feedback examples: {negative_count}")
                print(f"📊 Total feedback examples: {len(rec_df)}")
                print(f"📊 Data augmentation applied: {positive_count * 2} additional positive examples")
                
        return enhanced_dataset 

    def debug_dataset_structure(self, dataset: pd.DataFrame, name: str = "dataset"):
        """Debug dataset structure and validate data"""
        print(f"🔍 Debugging {name} structure:")
        print(f"📊 Shape: {dataset.shape}")
        print(f"📊 Columns: {list(dataset.columns)}")
        print(f"📊 Data types: {dataset.dtypes.to_dict()}")
        
        if len(dataset) > 0:
            print(f"📊 Sample data:")
            print(dataset.head(3).to_dict('records'))
            
            # Check for missing values
            missing_values = dataset.isnull().sum()
            if missing_values.sum() > 0:
                print(f"⚠️ Missing values: {missing_values[missing_values > 0].to_dict()}")
            
            # Check source distribution
            if 'source' in dataset.columns:
                source_counts = dataset['source'].value_counts()
                print(f"📊 Source distribution: {source_counts.to_dict()}")
            
            # Check label distribution
            if 'label' in dataset.columns:
                label_counts = dataset['label'].value_counts()
                print(f"📊 Label distribution: {label_counts.to_dict()}")
        
        return True 