# Credit Card Fraud Detection API

Real-time fraud detection system using XGBoost with SMOTE-balanced training data.

## 🎯 Model Performance
- **Algorithm**: XGBoost Classifier with SMOTE oversampling
- **Fraud Detection Rate (Recall)**: 87%
- **Precision**: 62%
- **False Positive Rate**: 0.08%
- **ROC-AUC Score**: 0.98

## 🚨 Problem Statement
Credit card fraud detection with extreme class imbalance (99.83% legitimate, 0.17% fraud). Traditional accuracy metrics are meaningless here—a model predicting "not fraud" for everything would get 99.83% accuracy but catch zero fraud.

## 🚀 Quick Start

### Run with Docker
```bash
docker build -t fraud-detection-api .
docker run -p 8080:8080 fraud-detection-api
```

Visit `http://localhost:8080/docs` for interactive API documentation.

### Run Locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8080
```

## 📡 API Usage

### `POST /predict`
Analyze a transaction for fraud.

**Request Example:**
```json
{
  "Time": 406,
  "V1": -2.3122,
  "V2": 1.9519,
  "V3": -1.6098,
  "V4": 3.9979,
  "V5": -0.5226,
  "V6": -1.4267,
  "V7": -2.5373,
  "V8": 1.3919,
  "V9": -1.3507,
  "V10": -0.2794,
  "V11": 0.0025,
  "V12": 0.6408,
  "V13": 0.8417,
  "V14": -0.5897,
  "V15": -0.2705,
  "V16": -0.1545,
  "V17": -1.1375,
  "V18": -0.0027,
  "V19": 0.4032,
  "V20": 0.2515,
  "V21": -0.0181,
  "V22": -0.1073,
  "V23": -0.2269,
  "V24": 0.1957,
  "V25": 0.8203,
  "V26": 0.5135,
  "V27": 0.0352,
  "V28": -0.0018,
  "Amount": 0.89
}
```

**Response:**
```json
{
  "prediction": "Legit",
  "fraud_probability_percent": 0.64
}
```

## 🛠️ Tech Stack
- **ML**: XGBoost, SMOTE (imbalanced-learn), scikit-learn
- **API**: FastAPI, Uvicorn
- **Deployment**: Docker
- **Dataset**: Kaggle Credit Card Fraud (284,807 transactions)

## 📊 Handling Class Imbalance

### The Challenge
- Original: 284,315 legitimate (99.83%) vs 492 fraud (0.17%)
- A naive "predict all legitimate" model gets 99.83% accuracy but 0% fraud detection

### Solution
1. **SMOTE Oversampling**: Created synthetic fraud examples to balance training data
2. **XGBoost scale_pos_weight**: Weighted fraud class higher during training
3. **Optimized for Recall**: Prioritized catching fraud over minimizing false alarms
4. **Proper Metrics**: Used Recall, Precision, F1, and ROC-AUC instead of accuracy

### Results
- 87% of fraud transactions caught
- Only 0.08% false positive rate (78 false alarms per 93,838 legit transactions)
- Business-ready tradeoff: Better to verify 78 transactions than miss 20 frauds

## 🎓 Key Technical Decisions

**Why XGBoost over Random Forest?**
- Better precision (62% vs 37%) with similar recall
- Handles imbalanced data better through scale_pos_weight
- Fewer false positives = less customer friction

**Why SMOTE?**
- Creates synthetic minority class examples
- Better than simple oversampling (avoids exact duplicates)
- Model learns fraud patterns rather than memorizing few examples

**Why Recall > Precision?**
- Missing fraud = unrecoverable financial loss
- False positive = temporary inconvenience, easily resolved
- 87% recall with 0.08% FP rate is production-viable

## 📁 Project Structure
## Folder Structure

```bash
.
├── app.py
├── cc.ipynb
├── creditcard.csv
├── Dockerfile.api
├── readme.md
├── requirements.txt
└── saved_model.pkl
```

## 🔮 Production Considerations

**Current Implementation:**
- Synchronous prediction (suitable for <1000 requests/min)
- In-memory model loading

**For High-Scale Production:**
- Add Redis caching for frequent patterns
- Implement async batch processing for bulk checks
- Deploy behind load balancer with auto-scaling
- Add model monitoring (data drift, performance degradation)
- Implement A/B testing for threshold optimization

## 📈 Model Metrics Explained

**Confusion Matrix:**

[[93760    78]   ← 93,760 legit correctly identified, 78 false alarms

[   20   129]]  ← 20 frauds missed, 129 frauds caught

**Recall (87%)**: Of 149 actual frauds, caught 129 (missed 20)

**Precision (62%)**: Of 207 fraud predictions, 129 were correct (78 false alarms)

**ROC-AUC (0.98)**: Excellent discrimination between fraud and legitimate

## 🚀 Future Enhancements
- [ ] MLflow experiment tracking
- [ ] SHAP values for explainability ("Why was this flagged?")
- [ ] Real-time dashboard with Streamlit
- [ ] A/B testing framework for threshold tuning
- [ ] Deploy to AWS/GCP with CI/CD pipeline

## 📚 Lessons Learned
1. **Accuracy is meaningless for imbalanced data** - Always check class distribution first
2. **SMOTE > simple oversampling** - Synthetic examples help model generalize
3. **Business context matters** - Fraud detection prioritizes recall over precision
4. **Threshold tuning is crucial** - Default 0.5 might not be optimal for your use case
5. **Docker simplifies deployment** - One command to run anywhere

---

**Dataset Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
