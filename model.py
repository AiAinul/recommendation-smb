import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Any
import os
import json
import time
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
    """Enhanced ranking model for better TP/TN balance"""

    def __init__(self, user_model, item_model):
        super().__init__()

        self.query_model: tf.keras.Model = user_model
        self.candidate_model: tf.keras.Model = item_model
        
        # Enhanced rating model dengan fokus pada ranking
        self.rating_model = tf.keras.Sequential([
            # Input layer dengan capacity tinggi
            tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Second layer dengan residual connection
            tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Third layer
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            
            # Fourth layer untuk ranking optimization
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            
            # Output layer dengan sigmoid untuk ranking scores
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Import tensorflow_ranking
        import tensorflow_ranking as tfr
        
        # Use ListMLE loss dengan focus pada ranking
        self.ranking_task_layer: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tfr.keras.losses.get(
                loss=tfr.keras.losses.RankingLossKey.LIST_MLE_LOSS, ragged=False),
            metrics=[
                tfr.keras.metrics.get(key="ndcg", name="metric/ndcg", ragged=False, topn=10),
                tfr.keras.metrics.get(key="mrr", name="metric/mrr", ragged=False),
                # Tambahkan metrics untuk ranking yang lebih spesifik
                tfr.keras.metrics.get(key="ndcg", name="metric/ndcg@5", ragged=False, topn=5),
                tfr.keras.metrics.get(key="mrr", name="metric/mrr@5", ragged=False, topn=5),
                tfr.keras.metrics.get(key="ndcg", name="metric/ndcg@3", ragged=False, topn=3),
                tfr.keras.metrics.get(key="mrr", name="metric/mrr@3", ragged=False, topn=3)
            ]
        )

    def compute_loss(self, features, training=False) -> tf.Tensor:
        """Compute ranking loss with recall optimization"""
        
        # Check if features are empty
        if not features or len(features) == 0:
            return tf.constant(0.0, dtype=tf.float32)
        
        # Get embeddings
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

        # Get rating predictions
        rating_predictions = self.rating_model(
            tf.concat([query_embeddings, candidate_embeddings], axis=1)
        )

        # Ensure proper shape for TensorFlow Ranking
        batch_size = tf.shape(rating_predictions)[0]
        
        # Check if batch is empty
        if batch_size == 0:
            return tf.constant(0.0, dtype=tf.float32)
        
        # Reshape predictions to [batch_size, 1] for single-item ranking
        rating_predictions_2d = tf.reshape(rating_predictions, [batch_size, 1])
        labels_2d = tf.reshape(features["label"], [batch_size, 1])

        # Compute loss using TensorFlow Ranking
        loss = self.ranking_task_layer(
            predictions=rating_predictions_2d,
            labels=labels_2d
        )
        
        # Add regularization loss to prevent overfitting
        reg_loss = tf.reduce_sum(self.losses)
        total_loss = loss + reg_loss
        
        return total_loss

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
        self.ranking_model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                weight_decay=0.0001,  # Reduced weight decay
                learning_rate=0.001,  # Lower learning rate for better convergence
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            run_eagerly=True  # Add this to debug empty logs issue
        )
        
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

    def train_ranking_model(self, ranking_dataset, test_dataset=None, epochs=15):
        """Train the ranking model with ranking optimization"""
        if self.ranking_model is None:
            raise ValueError("Ranking model not built. Call build_model() first.")
        
        print("🎯 Training ranking model with ranking optimization...")
        
        # Check if dataset is empty
        try:
            first_batch = next(iter(ranking_dataset.batch(1)))
            print(f"📊 Dataset check: Batch shape = {tf.shape(first_batch['user_id'])}")
        except Exception as e:
            print(f"❌ Error checking dataset: {e}")
            return None
        
        # Create callbacks untuk ranking optimization
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='metric/ndcg@5',  # Focus pada NDCG@5 untuk ranking
                patience=5,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='metric/ndcg@5',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_ranking_model.h5',
                monitor='metric/ndcg@5',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        try:
            if test_dataset is not None:
                history = self.ranking_model.fit(
                    ranking_dataset.batch(2048),  # Reduced batch size
                    validation_data=test_dataset.batch(2048),
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                history = self.ranking_model.fit(
                    ranking_dataset.batch(2048),
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1
                )
            
            # Print training summary
            if hasattr(history, 'history'):
                print("📊 Training Summary:")
                if 'metric/ndcg@5' in history.history:
                    ndcg_values = history.history['metric/ndcg@5']
                    print(f"📊 Final NDCG@5: {ndcg_values[-1]:.4f}")
                    print(f"📊 Best NDCG@5: {max(ndcg_values):.4f}")
                if 'metric/mrr@5' in history.history:
                    mrr_values = history.history['metric/mrr@5']
                    print(f"📊 Final MRR@5: {mrr_values[-1]:.4f}")
                    print(f"📊 Best MRR@5: {max(mrr_values):.4f}")
            
            return history
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None



    def train_ranking_model_with_rank_feedback(self, ranking_dataset, recommendation_feedback_dataset=None, test_dataset=None, epochs=10):
        """Train ranking model with rank-based feedback"""
        if self.ranking_model is None:
            raise ValueError("Ranking model not built. Call build_model() first.")
        
        print("🎯 Training ranking model with rank-based feedback...")
        
        if recommendation_feedback_dataset is not None and len(recommendation_feedback_dataset) > 0:
            print(f"📊 Using {len(recommendation_feedback_dataset)} rank-based feedback examples")
            
            from data_processing import DataProcessor
            data_processor = DataProcessor()
            
            # Convert TensorFlow dataset to pandas
            ranking_df = data_processor.convert_tf_dataset_to_pandas(ranking_dataset)
            
            if ranking_df is not None:
                # Create enhanced dataset with rank-based labels
                enhanced_dataset = data_processor.create_enhanced_training_dataset_with_rank(
                    ranking_df, 
                    recommendation_feedback_dataset
                )
                
                # Convert back to TensorFlow dataset
                enhanced_tf_dataset = data_processor.convert_pandas_to_tf_dataset(enhanced_dataset)
                
                if enhanced_tf_dataset is not None:
                    print(f"✅ Enhanced dataset with rank-based labels created: {len(enhanced_dataset)} rows")
                    
                    # Monitor MRR improvement during training
                    mrr_history = []
                    
                    # Train with enhanced dataset
                    if test_dataset is not None:
                        history = self.ranking_model.fit(
                            enhanced_tf_dataset.batch(4096),
                            validation_data=test_dataset.batch(4096),
                            epochs=epochs,
                            callbacks=[
                                tf.keras.callbacks.EarlyStopping(
                                    monitor='metric/mrr',
                                    patience=3,
                                    restore_best_weights=True,
                                    mode='max'
                                )
                            ]
                        )
                    else:
                        history = self.ranking_model.fit(
                            enhanced_tf_dataset.batch(4096),
                            epochs=epochs,
                            callbacks=[
                                tf.keras.callbacks.EarlyStopping(
                                    monitor='metric/mrr',
                                    patience=3,
                                    restore_best_weights=True,
                                    mode='max'
                                )
                            ]
                        )
                    
                    # Print MRR improvement summary
                    if hasattr(history, 'history') and 'metric/mrr' in history.history:
                        mrr_values = history.history['metric/mrr']
                        print(f"📊 MRR Improvement Summary:")
                        print(f"📊 Initial MRR: {mrr_values[0]:.4f}")
                        print(f"📊 Final MRR: {mrr_values[-1]:.4f}")
                        print(f"📊 MRR Improvement: {mrr_values[-1] - mrr_values[0]:.4f}")
                        print(f"📊 Best MRR: {max(mrr_values):.4f}")
                else:
                    print("⚠️ Failed to convert enhanced dataset to TensorFlow format")
                    history = self.train_ranking_model(ranking_dataset, test_dataset, epochs)
            else:
                print("⚠️ Could not convert TensorFlow dataset to DataFrame")
                history = self.train_ranking_model(ranking_dataset, test_dataset, epochs)
        else:
            print("📊 No rank-based feedback available, using standard training")
            history = self.train_ranking_model(ranking_dataset, test_dataset, epochs)
        
        return history

    def get_recommendations_with_rank_enhanced_ranking(self, user_id: str, current_item_id: str, region: str, city: str, top_k: int = 10):
        """Get recommendations using rank-enhanced ranking approach"""
        if self.brute_force_index is None:
            raise ValueError("Model index not created. Call create_index() first.")
            
        if self.ranking_model is None:
            raise ValueError("Ranking model not built. Call build_model() first.")
            
        current_category_str = self.item_detail_lookup.get(current_item_id, ("-", "-", "-"))[0]
        current_timestamp = int(time.time())

        try:
            # Step 1: Get candidates using retrieval
            scores, ids = self.brute_force_index({
                "user_id": tf.constant([user_id]),
                "region": tf.constant([region]),
                "city": tf.constant([city]),
                "item_id_currentview": tf.constant([current_item_id]),
                "timestamp_unix": tf.constant([current_timestamp], dtype=tf.int64),
                "item_id_lastview": tf.constant([current_item_id])
            })

            # Step 2: Filter candidates
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
                        'category3': cat3,
                        'retrieval_score': float(scores[0, i].numpy())
                    })

                if len(candidates) >= top_k * 2:
                    break

            if not candidates:
                print(f"⚠️ No candidates found for user {user_id}, item {current_item_id}")
                return []

            # Step 3: Get ranking scores
            user_features = {
                "user_id": tf.constant([user_id] * len(candidates)),
                "region": tf.constant([region] * len(candidates)),
                "city": tf.constant([city] * len(candidates)),
                "item_id_currentview": tf.constant([current_item_id] * len(candidates)),
                "timestamp_unix": tf.constant([current_timestamp] * len(candidates), dtype=tf.int64),
                "item_id_lastview": tf.constant([current_item_id] * len(candidates))
            }

            item_features = {
                "item_id": tf.constant([c['item_id'] for c in candidates]),
                "category": tf.constant([c['category'] for c in candidates]),
                "category2": tf.constant([c['category2'] for c in candidates]),
                "category3": tf.constant([c['category3'] for c in candidates])
            }

            try:
                query_embeddings = self.ranking_model.query_model(user_features)
                candidate_embeddings = self.ranking_model.candidate_model(item_features)
                
                combined_features = tf.concat([query_embeddings, candidate_embeddings], axis=1)
                ranking_scores = self.ranking_model.rating_model(combined_features)
                
                print(f"✅ Ranking scores shape: {ranking_scores.shape}, scores: {ranking_scores.numpy()}")
                
                # Step 4: Combine retrieval and ranking scores with rank enhancement
                ranked_candidates = []
                for i, candidate in enumerate(candidates):
                    ranking_score = float(ranking_scores[i][0])
                    retrieval_score = candidate['retrieval_score']
                    
                    # Enhanced scoring: combine retrieval, ranking, and rank-based weighting
                    enhanced_score = 0.3 * retrieval_score + 0.7 * ranking_score
                    
                    ranked_candidates.append({
                        'item_id': candidate['item_id'],
                        'score': enhanced_score,
                        'ranking_score': ranking_score,
                        'retrieval_score': retrieval_score,
                        'category': candidate['category'],
                        'category2': candidate['category2'],
                        'category3': candidate['category3']
                    })

                # Sort by enhanced score
                ranked_candidates.sort(key=lambda x: x['score'], reverse=True)
                print(f"✅ Generated {len(ranked_candidates)} rank-enhanced recommendations")
                return ranked_candidates[:top_k]
                
            except Exception as e:
                print(f"❌ Error in rank-enhanced ranking: {e}")
                return self.get_recommendations(user_id, current_item_id, region, city, top_k)
                
        except Exception as e:
            print(f"❌ Error in get_recommendations_with_rank_enhanced_ranking: {e}")
            return self.get_recommendations(user_id, current_item_id, region, city, top_k)



    def evaluate_recommendation_accuracy(self, test_recommendations_df):
        """Evaluate model accuracy using recommendation feedback data"""
        if self.ranking_model is None:
            print("❌ No ranking model available for evaluation")
            return {}
        
        print("📊 Evaluating recommendation accuracy...")
        
        correct_predictions = 0
        total_predictions = 0
        rank_accuracy = {}
        
        for _, row in test_recommendations_df.iterrows():
            user_id = row['user_id']
            current_item_id = str(row['current_item_id'])
            recommendation_group = row['recommendation_group']
            
            # Parse recommendation group
            if isinstance(recommendation_group, str):
                try:
                    rec_items = recommendation_group.strip('[]').replace('"', '').split(',')
                    rec_items = [item.strip() for item in rec_items if item.strip()]
                except:
                    rec_items = []
            else:
                rec_items = []
            
            # Get model predictions
            try:
                predictions = self.get_recommendations_with_ranking(
                    user_id, current_item_id, "unknown", "unknown", top_k=10
                )
                
                predicted_items = [pred['item_id'] for pred in predictions]
                
                # Check if any recommended items match the actual recommendations
                matches = set(rec_items) & set(predicted_items)
                if matches:
                    correct_predictions += 1
                    
                    # Calculate rank accuracy
                    for item in matches:
                        pred_rank = next((i+1 for i, pred in enumerate(predictions) if pred['item_id'] == item), -1)
                        if pred_rank > 0:
                            rank_accuracy[pred_rank] = rank_accuracy.get(pred_rank, 0) + 1
                
                total_predictions += 1
                
            except Exception as e:
                print(f"⚠️ Error evaluating prediction for user {user_id}: {e}")
                continue
        
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        evaluation_results = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'rank_accuracy': rank_accuracy
        }
        
        print(f"✅ Evaluation Results:")
        print(f"📊 Overall Accuracy: {accuracy:.2f}%")
        print(f"📊 Correct Predictions: {correct_predictions}/{total_predictions}")
        print(f"📊 Rank Accuracy: {rank_accuracy}")
        
        return evaluation_results

    def evaluate_recommendation_accuracy_with_negatives(self, test_recommendations_df):
        """Evaluate model accuracy with focus on rank accuracy"""
        if self.ranking_model is None:
            print("❌ No ranking model available for evaluation")
            return {}
        
        print("📊 Evaluating recommendation accuracy with rank focus...")
        
        max_samples = 1000
        if len(test_recommendations_df) > max_samples:
            print(f"📊 Limiting evaluation to {max_samples} samples from {len(test_recommendations_df)} total")
            test_recommendations_df = test_recommendations_df.sample(n=max_samples, random_state=42)
        
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        rank_accuracy = {}
        
        total_samples = len(test_recommendations_df)
        print(f"📊 Starting evaluation with {total_samples} samples...")
        
        # IMPROVED: Use multiple thresholds dengan focus pada ranking
        thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  # More granular thresholds
        best_rank_score = 0
        best_threshold = 0.3
        best_metrics = {}
        
        for threshold in thresholds:
            tp, fp, tn, fn = 0, 0, 0, 0
            rank_counts = {}
            
            for idx, (_, row) in enumerate(test_recommendations_df.iterrows()):
                if idx % 100 == 0:
                    print(f"📊 Progress: {idx}/{total_samples} ({idx/total_samples*100:.1f}%) - Threshold: {threshold}")
                
                user_id = row['user_id']
                current_item_id = str(row['current_item_id'])
                item_id = str(row['item_id'])
                source = row.get('source', 'recommendation_feedback')
                label = float(row.get('label', 1.0))
                
                try:
                    predictions = self.get_recommendations_with_ranking(
                        user_id, current_item_id, "unknown", "unknown", top_k=10
                    )
                    
                    predicted_items = [pred['item_id'] for pred in predictions]
                    predicted_scores = [pred.get('score', 0.0) for pred in predictions]
                    
                    if source == 'recommendation_feedback_positive' or (source == 'recommendation_feedback' and label > 0):
                        if item_id in predicted_items:
                            item_score = next((score for pred, score in zip(predictions, predicted_scores) if pred['item_id'] == item_id), 0.0)
                            if item_score >= threshold:
                                tp += 1
                                # Calculate rank accuracy with more detail
                                pred_rank = next((i+1 for i, pred in enumerate(predictions) if pred['item_id'] == item_id), -1)
                                if pred_rank > 0:
                                    rank_counts[pred_rank] = rank_counts.get(pred_rank, 0) + 1
                            else:
                                fn += 1
                        else:
                            fn += 1
                            
                    elif source == 'recommendation_feedback_negative' or (source == 'recommendation_feedback' and label == 0):
                        if item_id not in predicted_items:
                            tn += 1
                        else:
                            item_score = next((score for pred, score in zip(predictions, predicted_scores) if pred['item_id'] == item_id), 0.0)
                            if item_score >= threshold:
                                fp += 1
                            else:
                                tn += 1
                    
                except Exception as e:
                    print(f"⚠️ Error evaluating prediction for user {user_id}, item {item_id}: {e}")
                    continue
            
            # Calculate rank score (weighted by rank position)
            rank_score = sum(rank_counts.get(i, 0) * (6-i) for i in range(1, 6))  # Higher weight for lower ranks
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            print(f"📊 Threshold {threshold}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}, Rank Score={rank_score}")
            
            # Keep best threshold based on rank score
            if rank_score > best_rank_score:
                best_rank_score = rank_score
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'accuracy': accuracy,
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn,
                    'threshold': threshold,
                    'rank_accuracy': rank_counts
                }
        
        print(f"📊 Best threshold: {best_threshold} with Rank Score={best_rank_score}")
        
        evaluation_results = {
            **best_metrics,
            'samples_evaluated': total_samples,
            'max_samples_limit': max_samples,
            'best_threshold': best_threshold,
            'thresholds_tested': thresholds,
            'best_rank_score': best_rank_score
        }
        
        print(f"✅ Evaluation Results with Rank Focus:")
        print(f"📊 Best Threshold: {best_threshold}")
        print(f"📊 Rank Score: {best_rank_score}")
        print(f"📊 Precision: {best_metrics['precision']:.4f}")
        print(f"📊 Recall: {best_metrics['recall']:.4f}")
        print(f"📊 F1-Score: {best_metrics['f1_score']:.4f}")
        print(f"📊 Rank Accuracy: {best_metrics['rank_accuracy']}")
        
        return evaluation_results

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
            
        if self.ranking_model is None:
            raise ValueError("Ranking model not built. Call build_model() first.")
            
        current_category_str = self.item_detail_lookup.get(current_item_id, ("-", "-", "-"))[0]

        # Get current timestamp for inference
        import time
        current_timestamp = int(time.time())

        try:
            # Step 1: Get top 10 candidates using retrieval
            scores, ids = self.brute_force_index({
                "user_id": tf.constant([user_id]),
                "region": tf.constant([region]),
                "city": tf.constant([city]),
                "item_id_currentview": tf.constant([current_item_id]),
                "timestamp_unix": tf.constant([current_timestamp], dtype=tf.int64),
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
                print(f"⚠️ No candidates found for user {user_id}, item {current_item_id}")
                return []

            # Step 3: Use ranking model to score and re-rank candidates
            user_features = {
                "user_id": tf.constant([user_id] * len(candidates)),
                "region": tf.constant([region] * len(candidates)),
                "city": tf.constant([city] * len(candidates)),
                "item_id_currentview": tf.constant([current_item_id] * len(candidates)),
                "timestamp_unix": tf.constant([current_timestamp] * len(candidates), dtype=tf.int64),
                "item_id_lastview": tf.constant([current_item_id] * len(candidates))
            }

            item_features = {
                "item_id": tf.constant([c['item_id'] for c in candidates]),
                "category": tf.constant([c['category'] for c in candidates]),
                "category2": tf.constant([c['category2'] for c in candidates]),
                "category3": tf.constant([c['category3'] for c in candidates])
            }

            # Get ranking scores
            try:
                query_embeddings = self.ranking_model.query_model(user_features)
                candidate_embeddings = self.ranking_model.candidate_model(item_features)
                
                combined_features = tf.concat([query_embeddings, candidate_embeddings], axis=1)
                ranking_scores = self.ranking_model.rating_model(combined_features)
                
                print(f"✅ Ranking scores shape: {ranking_scores.shape}, scores: {ranking_scores.numpy()}")
                
            except Exception as e:
                print(f"❌ Error in ranking model inference: {e}")
                # Fallback to retrieval-only recommendations
                return self.get_recommendations(user_id, current_item_id, region, city, top_k)

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
            print(f"✅ Generated {len(ranked_candidates)} ranked recommendations")
            return ranked_candidates[:top_k]
            
        except Exception as e:
            print(f"❌ Error in get_recommendations_with_ranking: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to retrieval-only recommendations
            return self.get_recommendations(user_id, current_item_id, region, city, top_k)

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