# AI Job Recommendation System

An AI-powered job recommendation system that recommends suitable job titles based on a user's skills and identifies the most important skills they should acquire to improve their career opportunities.

The project covers the complete machine learning workflow, starting from data exploration and preprocessing, followed by model development and evaluation, and finally deploying the recommendation system through an interactive Flask web application.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

The project begins with exploring the LinkedIn Jobs and Skills dataset to better understand the data distribution and quality. During this phase:

- Explored job titles and skill frequencies.
- Analyzed the distribution of skills across different jobs.
- Cleaned noisy and duplicated skill names.
- Applied skill normalization using regex, semantic similarity, and fuzzy matching.
- Produced a clean dataset for model training.

---

### 2. Model Development

After preprocessing, multiple recommendation approaches were implemented and compared, including:

- TF-IDF + Cosine Similarity
- TF-IDF + K-Nearest Neighbors (KNN)
- TF-IDF + Truncated SVD + MiniBatch K-Means

Each model was evaluated using:

- Precision@5
- Mean Reciprocal Rank (MRR)

The K-Nearest Neighbors (KNN) model achieved the best overall performance and was selected for deployment.

---

### 3. Web Application

The selected model was integrated into a Flask web application where users can:

- Enter their skills.
- Receive the Top 10 recommended job titles.
- View the similarity score for each recommendation.
- Discover missing skills (Skill Gap Analysis) for every recommended job.

---

## Dataset

The project uses the **LinkedIn Jobs and Skills** dataset, which contains job postings along with their associated skills.

After preprocessing, the dataset was cleaned, normalized, and transformed into a structured format suitable for machine learning.

---

## Technologies

- Python
- Flask
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- K-Nearest Neighbors (KNN)
- HTML
- CSS
- JavaScript

---

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

---

## Run the Project

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```
