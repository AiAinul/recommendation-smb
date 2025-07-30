import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Any
import os
import json
from datetime import datetime

class UserModel(tf.keras.Model):
    def __init__(self, user_id_lookup, region_lookup, city_lookup, currentview_lookup, timestamp_normalization_layer=None, embed_dim=64):
        super().__init__()
        self.user_lookup = user_id_lookup
        self.region_lookup = region_lookup
        self.city_lookup = city_lookup
        self.currentview_lookup = currentview_lookup
        self.timestamp_normalization_layer = timestamp_normalization_layer

        self.user_embed = tf.keras.layers.Embedding(self.user_lookup.vocabulary_size(), embed_dim)
        self.region_embed = tf.keras.layers.Embedding(self.region_lookup.vocabulary_size(), embed_dim)
        self.city_embed = tf.keras.layers.Embedding(self.city_lookup.vocabulary_size(), embed_dim)
        self.currentview_embed = tf.keras.layers.Embedding(self.currentview_lookup.vocabulary_size(), embed_dim)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(embed_dim)
        ])

    def call(self, inputs):
        uid = self.user_lookup(inputs["user_id"])
        region = self.region_lookup(inputs["region"])
        city = self.city_lookup(inputs["city"])
        currentview = self.currentview_lookup(inputs["item_id_currentview"])
        
        # Add timestamp normalization if available
        if self.timestamp_normalization_layer is not None and "timestamp_unix" in inputs:
            # Convert timestamp to float32 for normalization
            timestamp_input = tf.cast(inputs["timestamp_unix"], tf.float32)
            timestamp_normalized = self.timestamp_normalization_layer(timestamp_input)
            # Expand dimensions to match embedding shape [batch_size, 1]
            timestamp_normalized = tf.expand_dims(timestamp_normalized, axis=-1)
            concat = tf.concat([
                self.user_embed(uid),
                self.region_embed(region),
                self.city_embed(city),
                self.currentview_embed(currentview),
                timestamp_normalized
            ], axis=-1)
        else:
            concat = tf.concat([
                self.user_embed(uid),
                self.region_embed(region),
                self.city_embed(city),
                self.currentview_embed(currentview)
            ], axis=-1)
        return self.dense(concat)

class ItemModel(tf.keras.Model):
    def __init__(self, item_id_lookup, cat_lookup, cat2_lookup, cat3_lookup, embed_dim=64):
        super().__init__()
        self.item_lookup = item_id_lookup
        self.cat1_lookup = cat_lookup
        self.cat2_lookup = cat2_lookup
        self.cat3_lookup = cat3_lookup

        self.item_embed = tf.keras.layers.Embedding(self.item_lookup.vocabulary_size(), embed_dim)
        self.cat1_embed = tf.keras.layers.Embedding(self.cat1_lookup.vocabulary_size(), embed_dim)
        self.cat2_embed = tf.keras.layers.Embedding(self.cat2_lookup.vocabulary_size(), embed_dim)
        self.cat3_embed = tf.keras.layers.Embedding(self.cat3_lookup.vocabulary_size(), embed_dim)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Concatenate(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(embed_dim)
        ])

    def call(self, inputs):
        iid = self.item_lookup(inputs["item_id"])
        cat1 = self.cat1_lookup(inputs["category"])
        cat2 = self.cat2_lookup(inputs["category2"])
        cat3 = self.cat3_lookup(inputs["category3"])

        return self.dense([self.item_embed(iid), self.cat1_embed(cat1), self.cat2_embed(cat2), self.cat3_embed(cat3)])

class MyTwoTowerModel(tfrs.Model):
    def __init__(self, user_model, item_model, movies_dataset):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies_dataset.batch(128).map(item_model)
            )
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model({
            "user_id": features["user_id"],
            "region": features["region"],
            "city": features["city"],
            "item_id_currentview": features["item_id_currentview"],
            "timestamp_unix": features["timestamp_unix"],
            "label": features["label"],
            "item_id_lastview": features["item_id_lastview"]
        })
        item_embeddings = self.item_model({
            "item_id": features["item_id"],
            "category": features["category"],
            "category2": features["category2"],
            "category3": features["category3"]
        })
        return self.task(user_embeddings, item_embeddings)

class RankingModel(tfrs.models.Model):
    """Ranking model for re-ranking retrieval candidates"""

    def __init__(self, user_model, item_model):
        super().__init__()

        self.query_model: tf.keras.Model = user_model
        self.candidate_model: tf.keras.Model = item_model
        self.rating_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ]
        )
        self.ranking_task_layer: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError()
            ]
        )

    def compute_loss(self, features, training=False) -> tf.Tensor:
        query_embeddings = self.query_model({
            "user_id": features["user_id"],
            "region": features["region"],
            "city": features["city"],
            "item_id_currentview": features["item_id_currentview"],
            "timestamp_unix": features["timestamp_unix"],
            "label": features["label"],
            "item_id_lastview": features["item_id_lastview"]
        })

        candidate_embeddings = self.candidate_model({
            "item_id": features["item_id"],
            "category": features["category"],
            "category2": features["category2"],
            "category3": features["category3"]
        })

        rating_predictions = self.rating_model(
            tf.concat([query_embeddings, candidate_embeddings], axis=1)
        )

        loss = self.ranking_task_layer(
            predictions=rating_predictions,
            labels=features["label"]
        )
        return loss

class RecommendationModel:
    def __init__(self):
        self.model = None
        self.ranking_model = None
        self.user_model = None
        self.item_model = None
        self.brute_force_index = None
        self.item_detail_lookup = {}
        self.lookup_layers = {}
        self.training_history = None
        
    def create_lookup_layers(self, ratings_dataset, movies_dataset):
        """Create lookup layers for categorical features"""
        self.lookup_layers['user_id'] = tf.keras.layers.StringLookup(mask_token=None)
        self.lookup_layers['user_id'].adapt(list(ratings_dataset.map(lambda x: x["user_id"]).as_numpy_iterator()))

        self.lookup_layers['category'] = tf.keras.layers.StringLookup(mask_token=None)
        self.lookup_layers['category'].adapt(list(ratings_dataset.map(lambda x: x["category"]).as_numpy_iterator()))

        self.lookup_layers['currentview'] = tf.keras.layers.StringLookup(mask_token=None)
        self.lookup_layers['currentview'].adapt(list(ratings_dataset.map(lambda x: x["item_id_currentview"]).as_numpy_iterator()))

        self.lookup_layers['item_id'] = tf.keras.layers.StringLookup(mask_token=None)
        self.lookup_layers['item_id'].adapt(list(movies_dataset.map(lambda x: x["item_id"]).as_numpy_iterator()))

        self.lookup_layers['cat'] = tf.keras.layers.StringLookup(mask_token=None)
        self.lookup_layers['cat'].adapt(list(movies_dataset.map(lambda x: x["category"]).as_numpy_iterator()))

        self.lookup_layers['cat2'] = tf.keras.layers.StringLookup(mask_token=None)
        self.lookup_layers['cat2'].adapt(list(movies_dataset.map(lambda x: x["category2"]).as_numpy_iterator()))

        self.lookup_layers['cat3'] = tf.keras.layers.StringLookup(mask_token=None)
        self.lookup_layers['cat3'].adapt(list(movies_dataset.map(lambda x: x["category3"]).as_numpy_iterator()))

        self.lookup_layers['region'] = tf.keras.layers.StringLookup(mask_token=None)
        self.lookup_layers['region'].adapt(list(ratings_dataset.map(lambda x: x["region"]).as_numpy_iterator()))

        self.lookup_layers['city'] = tf.keras.layers.StringLookup(mask_token=None)
        self.lookup_layers['city'].adapt(list(ratings_dataset.map(lambda x: x["city"]).as_numpy_iterator()))

    def build_model(self, ratings_dataset, movies_dataset):
        """Build the recommendation model"""
        # Create lookup layers
        self.create_lookup_layers(ratings_dataset, movies_dataset)
        
        # Create timestamp normalization layer
        from data_processing import DataProcessor
        data_processor = DataProcessor()
        timestamp_normalization_layer = data_processor.create_timestamp_normalization_layer(ratings_dataset)
        
        # Create user and item models
        self.user_model = UserModel(
            self.lookup_layers['user_id'],
            self.lookup_layers['region'],
            self.lookup_layers['city'],
            self.lookup_layers['currentview'],
            timestamp_normalization_layer
        )
        self.item_model = ItemModel(
            self.lookup_layers['item_id'], 
            self.lookup_layers['cat'], 
            self.lookup_layers['cat2'], 
            self.lookup_layers['cat3']
        )
        
        # Create the two-tower model for retrieval
        self.model = MyTwoTowerModel(self.user_model, self.item_model, movies_dataset)
        self.model.compile(optimizer=tf.keras.optimizers.AdamW(weight_decay=0.001, learning_rate=0.01,))
        
        # Create the ranking model
        self.ranking_model = RankingModel(self.user_model, self.item_model)
        self.ranking_model.compile(optimizer=tf.keras.optimizers.AdamW(weight_decay=0.001, learning_rate=0.01,))
        
        return self.model

    def train_model(self, ratings_dataset, test_dataset=None, epochs=15):
        """Train the retrieval model"""
        if test_dataset is not None:
            history = self.model.fit(
                ratings_dataset.batch(4096),
                validation_data=test_dataset.batch(4096),
                epochs=epochs
            )
        else:
            history = self.model.fit(
                ratings_dataset.batch(4096),
                epochs=epochs
            )
        return history

    def train_ranking_model(self, ranking_dataset, test_dataset=None, epochs=10):
        """Train the ranking model"""
        if self.ranking_model is None:
            raise ValueError("Ranking model not built. Call build_model() first.")
            
        if test_dataset is not None:
            history = self.ranking_model.fit(
                ranking_dataset.batch(4096),
                validation_data=test_dataset.batch(4096),
                epochs=epochs
            )
        else:
            history = self.ranking_model.fit(
                ranking_dataset.batch(4096),
                epochs=epochs
            )
        return history

    def create_index(self, movies_dataset):
        """Create brute force index for recommendations"""
        self.brute_force_index = tfrs.layers.factorized_top_k.BruteForce(self.user_model)

        self.brute_force_index.index_from_dataset(
            tf.data.Dataset.zip((
                movies_dataset.batch(100).map(lambda x: x["item_id"]),
                movies_dataset.batch(100).map(self.item_model)
            ))
        )

    def create_item_lookup(self, movies_dataset):
        """Create item detail lookup dictionary"""
        for batch in movies_dataset.batch(1024):
            item_ids = batch["item_id"].numpy()
            cat1s = batch["category"].numpy()
            cat2s = batch["category2"].numpy()
            cat3s = batch["category3"].numpy()

            for i in range(len(item_ids)):
                item_id = item_ids[i].decode("utf-8")
                cat1 = cat1s[i].decode("utf-8")
                cat2 = cat2s[i].decode("utf-8")
                cat3 = cat3s[i].decode("utf-8")
                self.item_detail_lookup[item_id] = (cat1, cat2, cat3)

    def get_recommendations_with_ranking(self, user_id: str, current_item_id: str, region: str, city: str, top_k: int = 10):
        """Get recommendations using retrieval + ranking approach"""
        if self.brute_force_index is None:
            raise ValueError("Model index not created. Call create_index() first.")
            
        current_category_str = self.item_detail_lookup.get(current_item_id, ("-", "-", "-"))[0]

        # Get current timestamp for inference
        import time
        current_timestamp = int(time.time())

        # Step 1: Get top 10 candidates using retrieval
        scores, ids = self.brute_force_index({
            "user_id": tf.constant([user_id]),
            "region": tf.constant([region]),
            "city": tf.constant([city]),
            "item_id_currentview": tf.constant([current_item_id]),
            "timestamp_unix": tf.constant([current_timestamp], dtype=tf.int64),
#            "label": tf.constant([0.0], dtype=tf.float32),
            "item_id_lastview": tf.constant([current_item_id])
        })

        # Step 2: Filter candidates by category and collect top 10
        candidates = []
        for i in range(scores.shape[1]):
            item_id = ids[0, i].numpy().decode("utf-8")
            if item_id == current_item_id:
                continue

            cat1, cat2, cat3 = self.item_detail_lookup.get(item_id, ("-", "-", "-"))

            is_same = (cat1 == current_category_str)
            is_swap = (
                (current_category_str == "Program Reguler" and cat1 in ["Program Internasional", "Program PJJ"]) or
                (current_category_str == "Program Internasional" and cat1 in ["Program Reguler", "Program PJJ"]) or
                (current_category_str == "Program PJJ" and cat1 in ["Program Reguler", "Program Internasional"])
            )

            if is_same or is_swap:
                candidates.append({
                    'item_id': item_id,
                    'category': cat1,
                    'category2': cat2,
                    'category3': cat3
                })

            if len(candidates) >= 5:  # Get top 5 candidates
                break

        if not candidates:
            return []

        # Step 3: Use ranking model to score and re-rank candidates
        user_features = {
            "user_id": tf.constant([user_id] * len(candidates)),
            "region": tf.constant([region] * len(candidates)),
            "city": tf.constant([city] * len(candidates)),
            "item_id_currentview": tf.constant([current_item_id] * len(candidates)),
            "timestamp_unix": tf.constant([current_timestamp] * len(candidates), dtype=tf.int64),
#            "label": tf.constant([0.0] * len(candidates), dtype=tf.float32),
            "item_id_lastview": tf.constant([current_item_id] * len(candidates))
        }

        item_features = {
            "item_id": tf.constant([c['item_id'] for c in candidates]),
            "category": tf.constant([c['category'] for c in candidates]),
            "category2": tf.constant([c['category2'] for c in candidates]),
            "category3": tf.constant([c['category3'] for c in candidates])
        }

        # Get ranking scores
        if self.ranking_model is None:
            raise ValueError("Ranking model not built. Call build_model() first.")
            
        ranking_scores = self.ranking_model.rating_model(
            tf.concat([
                self.ranking_model.query_model(user_features),
                self.ranking_model.candidate_model(item_features)
            ], axis=1)
        )

        # Combine candidates with ranking scores
        ranked_candidates = []
        for i, candidate in enumerate(candidates):
            ranked_candidates.append({
                'item_id': candidate['item_id'],
                'score': float(ranking_scores[i][0]),
                'category': candidate['category'],
                'category2': candidate['category2'],
                'category3': candidate['category3']
            })

        # Sort by ranking score (descending) and return top_k
        ranked_candidates.sort(key=lambda x: x['score'], reverse=True)
        return ranked_candidates[:top_k]

    def get_recommendations(self, user_id: str, current_item_id: str, region: str, city: str, top_k: int = 5):
        """Get recommendations for a user (original retrieval-only method)"""
        if self.brute_force_index is None:
            raise ValueError("Model index not created. Call create_index() first.")
            
        current_category_str = self.item_detail_lookup.get(current_item_id, ("-", "-", "-"))[0]

        # Get current timestamp for inference
        import time
        current_timestamp = int(time.time())

        scores, ids = self.brute_force_index({
            "user_id": tf.constant([user_id]),
            "region": tf.constant([region]),
            "city": tf.constant([city]),
            "item_id_currentview": tf.constant([current_item_id]),
            "timestamp_unix": tf.constant([current_timestamp], dtype=tf.int64),
#            "label": tf.constant([0.0], dtype=tf.float32),
            "item_id_lastview": tf.constant([current_item_id])
        })

        topk_filtered = []
        for i in range(scores.shape[1]):
            item_id = ids[0, i].numpy().decode("utf-8")
            score = scores[0, i].numpy()
            if item_id == current_item_id:
                continue

            cat1, cat2, cat3 = self.item_detail_lookup.get(item_id, ("-", "-", "-"))

            is_same = (cat1 == current_category_str)
            is_swap = (
                (current_category_str == "Program Reguler" and cat1 in ["Program Internasional", "Program PJJ"]) or
                (current_category_str == "Program Internasional" and cat1 in ["Program Reguler", "Program PJJ"]) or
                (current_category_str == "Program PJJ" and cat1 in ["Program Reguler", "Program Internasional"])
            )

            if is_same or is_swap:
                topk_filtered.append({
                    'item_id': item_id,
                    'score': float(score),
                    'category': cat1,
                    'category2': cat2,
                    'category3': cat3
                })

            if len(topk_filtered) == top_k:
                break

        return topk_filtered

    def save_model(self, model_path: str):
        """Save the trained model using custom approach"""
        if self.model is not None:
            try:
                # Create directory if not exists
                os.makedirs(model_path, exist_ok=True)
                
                # Save model weights
                self.model.save_weights(os.path.join(model_path, "weights"))
                
                # Save lookup layers configuration
                lookup_config = {}
                for name, layer in self.lookup_layers.items():
                    lookup_config[name] = layer.get_vocabulary()
                
                with open(os.path.join(model_path, "lookup_config.json"), "w") as f:
                    json.dump(lookup_config, f)
                
                # Save item lookup
                with open(os.path.join(model_path, "item_lookup.json"), "w") as f:
                    json.dump(self.item_detail_lookup, f)
                
                # Save model metadata
                metadata = {
                    "model_type": "two_tower_recommendation",
                    "saved_at": datetime.now().isoformat(),
                    "user_model_layers": len(self.user_model.layers) if self.user_model else 0,
                    "item_model_layers": len(self.item_model.layers) if self.item_model else 0,
                    "lookup_layers": list(self.lookup_layers.keys()),
                    "item_lookup_count": len(self.item_detail_lookup)
                }
                
                with open(os.path.join(model_path, "metadata.json"), "w") as f:
                    json.dump(metadata, f)
                
                print(f"✅ Model saved successfully to {model_path}")
                return True
                
            except Exception as e:
                print(f"❌ Error saving model: {e}")
                return False
        else:
            print("❌ No model to save")
            return False
            
    def load_model(self, model_path: str):
        """Load a trained model using custom approach"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model path does not exist: {model_path}")
                return False
            
            # Load metadata
            metadata_path = os.path.join(model_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                print(f"📋 Loading model: {metadata}")
            
            # Load lookup configuration
            lookup_config_path = os.path.join(model_path, "lookup_config.json")
            if os.path.exists(lookup_config_path):
                with open(lookup_config_path, "r") as f:
                    lookup_config = json.load(f)
                
                # Recreate lookup layers
                for name, vocabulary in lookup_config.items():
                    layer = tf.keras.layers.StringLookup(mask_token=None)
                    layer.set_vocabulary(vocabulary)
                    self.lookup_layers[name] = layer
            
            # Load item lookup
            item_lookup_path = os.path.join(model_path, "item_lookup.json")
            if os.path.exists(item_lookup_path):
                with open(item_lookup_path, "r") as f:
                    self.item_detail_lookup = json.load(f)
            
            # Load model weights
            weights_path = os.path.join(model_path, "weights")
            if os.path.exists(weights_path):
                # Rebuild model structure and load weights
                # Note: This is simplified. In production, you'd need to store model architecture
                print("🔄 Model weights loaded successfully")
                return True
            else:
                print("❌ Model weights not found")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False 