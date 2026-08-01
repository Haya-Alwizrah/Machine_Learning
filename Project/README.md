# AI Job Recommendation System

An AI-powered job recommendation system that recommends suitable job titles based on a user's skills and identifies the key skills they should learn to improve their career opportunities.

The system applies **TF-IDF vectorization** to represent job skills and uses a **K-Nearest Neighbors (KNN)** recommendation model to retrieve the most relevant job titles. For each recommendation, it also performs a **Skill Gap Analysis** by comparing the user's skills with those commonly required in similar job postings.

## Features

- Recommend the **Top 10** matching job titles.
- Display a **similarity score** for each recommendation.
- Perform **Skill Gap Analysis** for every recommended job.
- Interactive web interface built with **Flask**.
- Fast inference using pre-trained and serialized models.

## Technologies

- Python
- Flask
- Scikit-learn
- Pandas
- TF-IDF Vectorizer
- K-Nearest Neighbors (KNN)
- HTML
- CSS
- JavaScript

## Project Structure

```text
Project/
│
├── phase1_EDA/
│   ├── filtering.py
│   └── job-recommendation.ipynb
│
├── phase2_Model/
│   ├── data/
│   ├── models/
│   ├── train_model.py
│   ├── model.py
│   └── model_experiments.ipynb
│
└── phase3_Interface/
    ├── static/
    ├── templates/
    └── app.py
```

## Run the Project

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```
