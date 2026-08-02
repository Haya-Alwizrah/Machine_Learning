import os
import kagglehub
import ast
import joblib, json

import pandas as pd
from huggingface_hub import HfApi

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load dataset ---------------------------------------------------------

path = kagglehub.dataset_download("hayaalwizrah1/job-recommendation")
data = pd.read_csv(path + "/linkedin-jobs-and-skills.csv")
df = data.copy()
df = df.drop(columns=['job_title', 'company', 'job_location', 'first_seen', 'search_city', 'search_country', 'search_position', 'job_level', 'job_type', 'job_skills'])
df = df.rename(columns={"job_title_c": "job_title", "job_skills_c": "job_skills"})
df["job_skills"] = df["job_skills"].apply(ast.literal_eval)

DEMO = True
if DEMO:
    df = df.sample(n=500000, random_state=42)

# Cleaning -------------------------------------------------------------------

junk_single_chars = {
    'h','s','3','e','n','j','a','m','5','9','z', 'p','x','k',
    'i','*','2','f','+','$','6','g','>','q','t','b','d','o'
}

def clean_skills(skills):
    return [s for s in skills if s not in junk_single_chars]

df["job_skills"] = df["job_skills"].apply(clean_skills)

def format_skill(skill):
    if skill == "r": return "lang_r"
    if skill == "c": return "lang_c"
    return skill.replace(" ", "_")

df["skills_text"] = df["job_skills"].apply(
    lambda x: " ".join(format_skill(s) for s in x)
)

# Train ------------------------------------------------------------
tfidf = TfidfVectorizer(min_df=15)
tfidf_matrix = tfidf.fit_transform(df["skills_text"])

nn = NearestNeighbors(n_neighbors=10, metric="cosine")
nn.fit(tfidf_matrix)

# Save clean data and model --------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# data
df = df[["job_title", "job_skills"]]
df.to_parquet(
    os.path.join(DATA_DIR, "clean_jobs.parquet"),
    compression="snappy"
)

# tf-idf
joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf.pkl"), compress=9)
joblib.dump(tfidf_matrix, os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), compress=9)
joblib.dump(nn, os.path.join(MODELS_DIR, "nn.pkl"), compress=9)

# hugging face ----------------------
api = HfApi()
repo_id = "x-hayush/job-recommendation-model"

files = [
    (os.path.join(DATA_DIR, "clean_jobs.parquet"), "clean_jobs.parquet"),
    (os.path.join(MODELS_DIR, "tfidf.pkl"), "tfidf.pkl"),
    (os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "tfidf_matrix.pkl"),
    (os.path.join(MODELS_DIR, "nn.pkl"), "nn.pkl"),
]

print("Uploading files to Hugging Face...")

for local_path, remote_name in files:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_name,
        repo_id=repo_id,
        repo_type="model",
    )

print("Upload completed successfully!")