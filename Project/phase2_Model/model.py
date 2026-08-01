import os
import joblib
import pandas as pd
import ast
import json
from huggingface_hub import snapshot_download

# Load data and models -------------------

BASE_DIR = os.path.dirname(__file__)
HF_DIR = os.path.join(BASE_DIR, "hf_files")
# MODELS_DIR = os.path.join(BASE_DIR, "models")
# DATA_DIR = os.path.join(BASE_DIR, "data")

if not os.path.exists(os.path.join(HF_DIR, "KNN.pkl")):
    print("Downloading model files from Hugging Face...")
    snapshot_download(
        repo_id="x-hayush/job-recommendation-model",
        local_dir=HF_DIR,
        local_dir_use_symlinks=False
    )
    print("Download completed.")
else:
    print("Using cached model files.")
    
df = pd.read_csv(os.path.join(HF_DIR, "clean_jobs.csv"))
df["job_skills"] = df["job_skills"].apply(ast.literal_eval)

tfidf = joblib.load(os.path.join(HF_DIR, "tfidf.pkl"))
tfidf_matrix = joblib.load(os.path.join(HF_DIR, "tfidf_matrix.pkl"))
nn = joblib.load(os.path.join(HF_DIR, "KNN.pkl"))

with open(os.path.join(HF_DIR, "skills.json"), "r", encoding="utf-8") as f:
    skill_list = json.load(f)

# -----------------------------------------
def format_skill(skill):

    skill = skill.strip().lower()

    if skill == "r": return "lang_r"
    if skill == "c": return "lang_c"
    return skill.replace(" ", "_")

# -----------------------------------------

def prepare_query(user_skills):
    query = " ".join(format_skill(skill) for skill in user_skills)
    return tfidf.transform([query])

# -----------------------------------------

def recommend_jobs(user_skills, top_n=10, gap_top_n=5, n_similar=30):

    query_vec = prepare_query(user_skills)
    distances, indices = nn.kneighbors(query_vec, n_neighbors=top_n)
    job_indices = indices[0]

    batch_vectors = tfidf_matrix[job_indices]
    n_neighbors = min(n_similar + 1, len(df))
    _, batch_neighbors = nn.kneighbors(batch_vectors, n_neighbors=n_neighbors)

    user_skills_clean = {s.strip().lower() for s in user_skills}

    results = []
    for row, job_idx, dist in zip(range(len(job_indices)), job_indices, distances[0]):
        neighbors = batch_neighbors[row]
        neighbors = neighbors[neighbors != job_idx][:n_similar]

        similar_skills = (df.iloc[neighbors]["job_skills"].explode())
        top_skills = (similar_skills.value_counts().head(gap_top_n).index.tolist())

        missing = [skill for skill in top_skills if skill.lower() not in user_skills_clean]

        results.append({
            "job_title": df.iloc[job_idx]["job_title"],
            "similarity": round(1 - dist, 3),
            "missing_skills": missing
        })

    return results