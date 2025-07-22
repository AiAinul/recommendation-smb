# Sistem Rekomendasi dengan Ranking

Sistem ini mengimplementasikan **hybrid recommendation approach** yang menggabungkan:
1. **Retrieval Model** - untuk mengambil top 10 kandidat
2. **Ranking Model** - untuk mengurutkan kandidat berdasarkan skor ranking

## Cara Kerja

### 1. Retrieval + Ranking Flow
```
User Request → Retrieval Model → Top 10 Candidates → Ranking Model → Top 5 Recommendations
```

### 2. Endpoint Baru

#### A. Training Ranking Model
```bash
POST /train/ranking
```
Body:
```json
{
  "epochs": 10,
  "batch_size": 4096
}
```

#### B. Get Recommendations dengan Ranking
```bash
POST /recommendations/with-ranking
```
Body:
```json
{
  "user_id": "yogieeka@gmail.com",
  "current_item_id": "3905",
  "region": "west java",
  "city": "bandung",
  "top_k": 5
}
```

### 3. Perbedaan dengan Retrieval-Only

| Metode | Endpoint | Deskripsi |
|--------|----------|-----------|
| Retrieval Only | `/recommendations` | Menggunakan similarity score dari two-tower model |
| Retrieval + Ranking | `/recommendations/with-ranking` | Menggunakan ranking model untuk re-ranking |

## Implementasi Teknis

### 1. RankingModel Class
```python
class RankingModel(tfrs.models.Model):
    def __init__(self, user_model, item_model):
        # Neural network untuk scoring
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
```

### 2. Hybrid Recommendation Process
```python
def get_recommendations_with_ranking(self, user_id, current_item_id, region, city, top_k=5):
    # Step 1: Get top 10 candidates using retrieval
    scores, ids = self.brute_force_index({...})
    
    # Step 2: Filter by category and collect candidates
    candidates = filter_candidates(ids, scores)
    
    # Step 3: Use ranking model to score and re-rank
    ranking_scores = self.ranking_model.rating_model(
        tf.concat([user_embeddings, item_embeddings], axis=1)
    )
    
    # Step 4: Sort by ranking score and return top_k
    return sort_by_ranking_score(candidates, ranking_scores)[:top_k]
```

## Langkah Penggunaan

### 1. Train Base Model (Retrieval)
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 15, "batch_size": 4096}'
```

### 2. Train Ranking Model
```bash
curl -X POST "http://localhost:8000/train/ranking" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10, "batch_size": 4096}'
```

### 3. Get Recommendations dengan Ranking
```bash
curl -X POST "http://localhost:8000/recommendations/with-ranking" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "yogieeka@gmail.com",
    "current_item_id": "3905",
    "region": "west java",
    "city": "bandung",
    "top_k": 5
  }'
```

## Response Format

```json
{
  "user_id": "yogieeka@gmail.com",
  "current_item_id": "3905",
  "recommendations": [
    {
      "item_id": "1234",
      "score": 0.85,
      "category": "Program Reguler",
      "category2": "Teknik Informatika",
      "category3": "S1"
    }
  ],
  "model_info": {
    "method": "retrieval_with_ranking",
    "recommendations_count": 5,
    "model_status": "current"
  }
}
```

## Keuntungan Hybrid Approach

1. **Efisiensi**: Retrieval cepat untuk filtering, ranking untuk precision
2. **Akurasi**: Ranking model dapat mempelajari pola kompleks
3. **Scalability**: Retrieval mengurangi kandidat, ranking fokus pada subset kecil
4. **Flexibility**: Dapat menggunakan retrieval-only atau hybrid sesuai kebutuhan

## Monitoring

- **Training Progress**: `/training/status`
- **Model Info**: `/model/info`
- **API Analytics**: `/dashboard/api-analytics`
- **Recommendation Analytics**: `/dashboard/recommendation-analytics`

## Troubleshooting

### Error: "Ranking model not available"
- Pastikan ranking model sudah di-train dengan `/train/ranking`
- Cek status training dengan `/training/status`

### Error: "No base model available"
- Train base model terlebih dahulu dengan `/train`
- Cek model status dengan `/model/status`

### Performance Issues
- Ranking model memerlukan lebih banyak komputasi
- Gunakan retrieval-only untuk high-throughput scenarios
- Monitor response time dengan `/dashboard/api-analytics` 