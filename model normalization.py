import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Any

class UserModel(tf.keras.Model):
    def __init__(self, user_id_lookup, embed_dim=64):
        super().__init__()
        self.user_lookup = user_id_lookup
        self.user_embed = tf.keras.layers.Embedding(self.user_lookup.vocabulary_size(), embed_dim)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(embed_dim)
        ])

    def call(self, inputs):
        uid = self.user_lookup(inputs["user_id"])
        return self.dense(self.user_embed(uid))

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
            "city": features["city"]
        })
        item_embeddings = self.item_model({
            "item_id": features["item_id"],
            "category": features["category"],
            "category2": features["category2"],
            "category3": features["category3"]
        })
        return self.task(user_embeddings, item_embeddings)

class RecommendationModel:
    def __init__(self):
        self.model = None
        self.user_model = None
        self.item_model = None
        self.brute_force_index = None
        self.item_detail_lookup = {}
        self.lookup_layers = {}
        
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
        self.lookup_layers['region'].adapt(list(movies_dataset.map(lambda x: x["region"]).as_numpy_iterator()))

    def build_model(self, ratings_dataset, movies_dataset):
        """Build the recommendation model"""
        # Create lookup layers
        self.create_lookup_layers(ratings_dataset, movies_dataset)
        
        # Create user and item models
        self.user_model = UserModel(self.lookup_layers['user_id'])
        self.item_model = ItemModel(
            self.lookup_layers['item_id'], 
            self.lookup_layers['cat'], 
            self.lookup_layers['cat2'], 
            self.lookup_layers['cat3']
        )
        
        # Create the two-tower model
        self.model = MyTwoTowerModel(self.user_model, self.item_model, movies_dataset)
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01))
        
        return self.model

    def train_model(self, ratings_dataset, test_dataset=None, epochs=15):
        """Train the recommendation model"""
        if test_dataset is not None:
            self.model.fit(
                ratings_dataset.batch(4096),
                validation_data=test_dataset.batch(4096),
                epochs=epochs
            )
        else:
            self.model.fit(
                ratings_dataset.batch(4096),
                epochs=epochs
            )

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

    def get_recommendations(self, user_id: str, current_item_id: str, region: str, city: str, top_k: int = 5):
        """Get recommendations for a user"""
        if self.brute_force_index is None:
            raise ValueError("Model index not created. Call create_index() first.")
            
        current_category_str = self.item_detail_lookup.get(current_item_id, ("-", "-", "-"))[0]

        scores, ids = self.brute_force_index({
            "user_id": tf.constant([user_id]),
            "region": tf.constant([region]),
            "city": tf.constant([city]),
            "category": tf.constant([current_category_str]),
            "item_id_currentview": tf.constant([current_item_id])
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
                (current_category_str == "Program Reguler" and cat1 == "Program Internasional") or
                (current_category_str == "Program Internasional" and cat1 == "Program Reguler")
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
        """Save the trained model"""
        if self.model is not None:
            self.model.save(model_path)
            
    def load_model(self, model_path: str):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        # Recreate user and item models from the loaded model
        self.user_model = self.model.user_model
        self.item_model = self.model.item_model 