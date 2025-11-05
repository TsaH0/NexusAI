# Nexus: AI‑Powered Material Demand Forecasting System

Accurate material demand prediction is a cornerstone of efficient supply chain and infrastructure project execution. Traditional forecasting methods often struggle with real‑time variability, resulting in either stockouts that halt execution or excessive inventory that blocks capital and storage space.

**Nexus** is a neural‑network‑powered demand forecasting system designed for large‑scale infrastructure projects (like those executed by **POWERGRID**). Nexus intelligently analyzes multiple factors — including budget, tower type, location, season, material usage patterns, project progress, and more — to forecast future demand with high accuracy.

Nexus provides a **data‑driven solution** to transform procurement planning into a proactive, optimized, and agile operation.

---

## 🚀 Key Features

✅ Machine learning–driven demand forecasting using a **multi‑layer perceptron neural network**

✅ Synthetic dataset generation with **10,000+ realistic procurement samples**

✅ End‑to‑end pipeline: preprocessing, training, evaluation, and visualization

✅ Support for categorical encoding & numeric feature scaling

✅ Streamlit dashboard with:

* Demand forecasts per project, material, and month
* Confidence score visualization
* Procurement recommendations
* Trend charts and comparison graphs
* Options to regenerate data, retrain the model, and export results (CSV/Excel)

✅ Performance evaluated using **MSE and MAE**

✅ Exportable results for business decision‑making

---

## 🧠 Model Architecture

```
Input Layer → Dense (ReLU) → Dropout → Dense (ReLU) → Dropout → Output Layer (Regression)
```

### Techniques Used

* Feedforward neural network (MLP)
* ReLU activation functions
* Dropout for regularization
* Mean Squared Error (MSE) loss
* Adam optimizer

---

## 📊 Dataset

Since real procurement data is confidential, Nexus uses **realistically generated synthetic data** representing:

| Feature Category | Examples                                       |
| ---------------- | ---------------------------------------------- |
| Project Info     | Budget, Tower Type, Substation Type, Geography |
| Progress Metrics | Completion %, Lead Time                        |
| Temporal Factors | Month, Season                                  |
| Usage History    | Past Material Consumption                      |
| Financial Inputs | Tax Rate                                       |

> Over **10,000 samples** with realistic distributions and noise to simulate real‑world scenarios.

---

## 🖥️ Streamlit Dashboard

The interactive app provides:

* 📈 Forecast trends by project & material
* 📊 Confidence interval bands
* 📁 Export results (CSV/Excel)
* 🔁 Synthetic data regeneration
* 🧠 On‑demand retraining
* 📦 Suggested procurement plan

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/your-username/nexus-forecasting.git
cd nexus-forecasting

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the training module

```bash
python train.py
```

### Launch Streamlit UI

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
📂 nexus-forecasting
 ┣ app.py
 ┣ requirements.txt
 ┗ README.md
```

📂 nexus-forecasting
┣ 📁 data
┣ 📁 models
┣ 📁 app
┣ train.py
┣ app.py
┣ utils.py
┣ requirements.txt
┗ README.md

```

---
## ✅ Performance Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Confidence scores on Streamlit UI

---
## 🎯 Impact
With Nexus, procurement teams can:

- Reduce stockouts and avoid project delays
- Minimize surplus inventory & save warehouse cost
- Improve budget allocation efficiency
- Enable proactive & intelligent supply planning
- Strengthen resilience and agility of supply chain operations

---
## 🛠️ Future Enhancements
- Incorporate RNN/LSTM time‑series forecasting
- Integrate real enterprise ERP data
- Auto‑tuning with Bayesian optimization
- Deployment with FastAPI & Docker
- Real‑time demand updates & alerting

---
## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---
## 📜 License
MIT License

---
## 📧 Contact
For queries or contributions:
**Author:** Tejesh Sahoo
LinkedIn | GitHub | Email

---
**Nexus — Engineering Insight‑Driven Supply Chains**

```
