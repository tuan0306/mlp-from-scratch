# 🚢 Titanic Survival Predictor: From Scratch Deep Learning to Cloud Deployment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Math_Engine-013243?logo=numpy&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend_API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend_UI-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Deep_Learning-Built_From_Scratch-success)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI/CD-Automated-2088FF?logo=githubactions&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-Cloud_Hosted-0089D6?logo=microsoftazure&logoColor=white)

## 🎯 Project Overview
While it is easy to import a pre-built model from high-level libraries like `scikit-learn` or `TensorFlow`, this project takes the fundamental approach. I architected a **Multilayer Perceptron (Neural Network) completely from scratch** using only **Python** and **NumPy**. 

Moving beyond a simple Jupyter Notebook, this project bridges the gap between Data Science and Software Engineering. It is deployed as a **Production-ready Microservice**, fully containerized with **Docker**, and features an automated **CI/CD pipeline** hosted on a **Microsoft Azure Virtual Machine**.

🟢 **Live Web App:** [http://tuan-webs.koreacentral.cloudapp.azure.com/](http://tuan-webs.koreacentral.cloudapp.azure.com/)

🟢 **Live API Docs (Swagger):** [http://tuan-webs.koreacentral.cloudapp.azure.com:8000/docs/](http://tuan-webs.koreacentral.cloudapp.azure.com:8000/docs/)

---

## 🏗️ System Architecture (Microservices & MLOps)
This project strictly follows the **Separation of Concerns** principle and modern DevOps practices:

1. **The AI Engine (Backend):** A robust RESTful API built with **FastAPI**. It securely loads the custom NumPy neural network, performs real-time data preprocessing, and exposes a `/predict` endpoint.
2. **The Client (Frontend):** An interactive web application built with **Streamlit**. It acts as the presentation layer, packaging user inputs into JSON payloads for the backend.
3. **Container Orchestration:** Both services are isolated in independent **Docker Containers**, communicating via a custom Docker bridge network, orchestrated by `docker-compose`.
4. **Continuous Deployment (CI/CD):** **GitHub Actions** monitors the `main` branch. When pushing code, it automatically SSHs into the Azure VM, pulls the latest commits, rebuilds the Docker images, and restarts the containers with zero manual intervention.

---

## 🚀 Core Technical Highlights

### 🧠 AI & Machine Learning Engineering
* **100% Custom Neural Network:** Implemented a modular, Object-Oriented Deep Learning framework covering Forward Propagation, Backpropagation, and Gradient Descent.
* **Under-the-Hood Mathematics:** Manually coded activation functions (ReLU, Sigmoid) and their derivatives.
* **Numerical Stability:** Engineered secure Binary Cross-Entropy (BCE) loss functions using epsilon clipping to prevent exploding gradients and `log(0)` errors.
* **Advanced Feature Engineering:** Handled skewed distributions, manually encoded categorical variables, and extracted complex non-linear relationships.

### ⚙️ DevOps & Cloud Infrastructure
* **Automated Pipeline:** Wrote YAML workflows for GitHub Actions to achieve 100% hands-free deployment.
* **Port Mapping & Security:** Configured Docker internal networks (`EXPOSE`) to hide the backend API from public access while allowing the Streamlit frontend to route traffic through port `80`.
* **Server Administration:** Provisioned and secured an Ubuntu-based Azure VM with explicit SSH key authentication and Git credential management.

---

## 🧠 Mathematical Foundation
The network is optimized using a Custom Binary Cross-Entropy Loss function:

$$Loss = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

*To ensure stability, predictions ($\hat{y}$) are clipped to avoid iteration over 0-d arrays and infinite logarithmic penalties.*

---

## 📁 Repository Structure

```text
mlp-from-scratch/
├── .github/workflows/         
│   └── deploy.yml             # Automated CI/CD deployment pipeline
├── core/                      # Custom Deep Learning Framework (from scratch)
│   ├── activation.py          # ReLU, Sigmoid & activation derivatives
│   ├── layer.py               # Dense/Linear layer implementations
│   ├── loss.py                # Binary Cross-Entropy (BCE) Loss
│   ├── metrics.py             # Accuracy and performance evaluation metrics
│   ├── network.py             # Main Neural Network class (Forward/Backward passes)
│   └── utils.py               # Data processing & numerical utilities
├── data/                      # Raw and processed Titanic datasets
├── models/                    # Serialized weights (.pkl) & feature scalers
├── notebooks/                 # Exploratory Data Analysis & training experiments
├── .gitignore                 # Files excluded from version control
├── api.py                     # FastAPI Backend (Inference Service)
├── app.py                     # Streamlit Frontend (User Interface)
├── docker-compose.yml         # Multi-container orchestration
├── Dockerfile                 # Docker image blueprint for the environment
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── submission.csv             # Final predictions for Kaggle submission
```

---

## 🚀 How to Run

This project supports two execution modes: **Inference Mode** (using pre-trained weights) and **Development Mode** (for retraining/research).

### 📋 Prerequisites
* **Docker** & **Docker Compose** (Recommended)
* **Python 3.9+** (For manual installation)

---

### 📦 Option 1: Running with Docker (Recommended)
This is the fastest way to get both the Frontend and Backend up and running in a synchronized environment.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tuan0306/mlp-from-scratch.git
   cd mlp-from-scratch
   ```
2. **Launch the Services:**
   ```bash
   docker-compose up -d --build
   ```
3. **Access the Application:**
* **Interactive Web UI:** [http://localhost:8501](http://localhost:8501)
* **Inference API (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 🛠️ Option 2: Manual Installation (Development)
Use this method if you want to run the project locally without Docker.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Data Setup (Optional):**
* The `data/` directory is empty by default (to keep the repo lightweight).
* **To use the Web App/API:** No action is needed. The system will automatically use the pre-trained weights in the models/ directory
* **To retrain the model:** Download `train.csv` and `test.csv` from the [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic/data) and place them in the `data/raw` folder.

3. **Start the Backend API:**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

4. **Start the Frontend UI:**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Model Performance & Kaggle Results

The model was evaluated using the **Titanic - Machine Learning from Disaster** dataset on Kaggle. Since the architecture was built entirely from scratch (NumPy only), these results demonstrate the mathematical correctness of the backpropagation and optimization algorithms.

### 1. Model Configuration
To achieve the best balance between bias and variance, the following hyperparameters were selected:

| Parameter | Value |
| :--- | :--- |
| **Architecture** | Input (8) → Hidden (16, ReLU) → Hidden (8, ReLU) → Output (1, Sigmoid) |
| **Optimizer** | Stochastic Gradient Descent (SGD) with Momentum |
| **Loss Function** | Binary Cross-Entropy |
| **Learning Rate** | 0.05 (with decay) |
| **Epochs** | 1000 |

### 2. Training Metrics
The training process showed stable convergence, proving the efficacy of the custom-built gradient descent implementation.

* **Validation Accuracy (70/30/30 Split):** ~85.1%
* **Test Accuracy (70/30/30 Split):** ~79.85%
* **Test Precision:** 0.8
* **Test Precision:** ~0.63
* **Test F1-Score:** ~0.7

### 3. Kaggle Submission Results
The generated `submission.csv` was uploaded to the Kaggle Leaderboard:

* **Public Score:** **0.76794**
* **Status:** Successfully predicted survival for 418 passengers in the test set.

---

## 👨‍💻 Author
**Nguyễn Đình Tuấn**
* **Github Profile:** [https://github.com/tuan0306](https://github.com/tuan0306) 