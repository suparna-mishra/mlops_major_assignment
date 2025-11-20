# MLOps Major Assignment – End-to-End ML Pipeline

## Overview

This repository implements a complete **End-to-End MLOps workflow**. It includes

* Model development using scikit-learn
* CI/CD workflow via GitHub Actions
* Flask application for model inference
* Docker containerization
* Kubernetes deployment with 3 replicas

---

# Git Branching Strategy

* Git branching strategy with `main`, `dev`, and `docker_cicd`

### **1️⃣ main (Initial Setup)**

Contains basic project files:

* `README.md`
* `.gitignore`

### **2️⃣ dev (Model Training + CI/CD)**

Contains:

* `train.py`
* `test.py`
* `.github/workflows/ci.yml`
* Model artifacts (generated via CI)

### **3️⃣ docker_cicd (Flask App + Docker + K8s)**

Contains:

* `app.py`
* `templates/index.html`
* `Dockerfile`
* `deployment.yml`

---

# Repository Structure

```
├── train.py
├── test.py
├── app.py
├── templates/
│   └── index.html
├── deployment.yml
├── Dockerfile
├── .github/
│   └── workflows/ci.yml
└── README.md
```

---

# Model Development

This project uses the **Olivetti Faces dataset** and trains a **DecisionTreeClassifier**.

# CI/CD Pipeline (GitHub Actions)

The workflow file at `.github/workflows/ci.yml`:

* Checks out the repo
* Installs dependencies
* Runs `train.py`
* Runs `test.py`
* Prints accuracy

### Trigger:

Runs on every `push` to the `dev` branch.

---

# Flask Web Application

A simple Flask UI for uploading face images and predicting classes.

### Endpoints:

| Endpoint   | Method | Description        |
| ---------- | ------ | ------------------ |
| `/`        | GET    | Upload page        |
| `/predict` | POST   | Predict face class |

---

# Docker Containerization

### Docker Steps:

Build image:

```
docker build -t <dockerhub-user>/olivetti-app .
```

Run container:

```
docker run -p 5000:5000 <dockerhub-user>/olivetti-app
```

Push to Docker Hub:

```
docker push <dockerhub-user>/olivetti-app
```

---

# Kubernetes Deployment

The `deployment.yml` defines:

* 3 replicas
* LoadBalancer service

Apply the deployment:

```
kubectl apply -f deployment.yml
kubectl get pods
kubectl get svc
```

# Running Project Locally

Install dependencies:

```
pip install scikit-learn joblib flask pillow numpy
```

Train model:

```
python train.py
```

Test model:

```
python test.py
```

Run Flask app:

```
python app.py
```

Open in browser:

```
http://localhost:5000
```

---

# Final Notes

This project implements a complete MLOps workflow including ML training, CI/CD, containerization, and Kubernetes orchestration. All assignment requirements are fulfilled.

