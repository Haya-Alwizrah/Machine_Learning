import kagglehub
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# 1. Data Loading --------------------------------------------------------------------------
print("Loading data and model... Please wait.")
path = kagglehub.dataset_download("hayaalwizrah1/job-recommendation")
df = pd.read_csv(path + "/linkedin-jobs-and-skills.csv")
df = df.drop(columns=['job_title', 'company' ,'job_location', 'first_seen', 'search_city', 'search_country', 'search_position',	'job_level', 'job_type', 'job_skills'])
df = df.rename(columns={"job_title_c": "job_title", "job_skills_c": "job_skills"})
df['job_skills'] = df['job_skills'].apply(ast.literal_eval)

# 2. TF-IDF ----------------------------------------------------------------------
# 2.1: Remove junk single-character skills 
junk_single_chars = {'h', 's', '3', 'e', 'n', 'j', 'a', 'm', '5', '9', 'z', 'p', 'x', 'k',
                     'i', '*', '2', 'f', '+', '$', '6', 'g', '>', 'q', 't', 'b', 'd', 'o'}

def clean_skills(skills_list):
    return [s for s in skills_list if s not in junk_single_chars]

df['job_skills'] = df['job_skills'].apply(clean_skills)

# 2.2: Convert skills list to text 
def format_skill(skill):
    if skill == "r": return "lang_r"
    if skill == "c": return "lang_c"
    return skill.replace(" ", "_")

df["skills_text"] = df["job_skills"].apply(
    lambda skills: " ".join([format_skill(s) for s in skills])
)

# 2.3: Build TF-IDF 
tfidf = TfidfVectorizer(min_df=15)
tfidf_matrix = tfidf.fit_transform(df["skills_text"])

# 3: query ----------------------------------------------------------------------------
def prepare_query(user_skills):
    query = " ".join([format_skill(skill.strip().lower()) for skill in user_skills])
    return tfidf.transform([query])


# 4: Nearest Neighbors ------------------------------------------------------------------------
nn = NearestNeighbors(n_neighbors=10, metric='cosine')
nn.fit(tfidf_matrix)
print("Model is ready!")

# 5: Extract available skills for the dropdown -----------------------------
all_skills = sorted(list(set(skill.lower() for skills in df['job_skills'] for skill in skills)))

def get_recommendations(user_skills):
    """
    Takes a list of user skills and returns the top matching job titles.
    Parameters:
        user_skills (list of str): e.g. ["python", "machine learning", "nlp"]
        top_n (int): number of recommendations to return
    Returns:
        pandas.DataFrame with columns: job_title, similarity
    """
    
    query_vec = prepare_query(user_skills)
    distances, indices = nn.kneighbors(query_vec, n_neighbors=10)
    
    recommended_jobs = []
    job_skills_pool = set()
    
    for idx in indices[0]:
        job_title = df.iloc[idx]['job_title']
        recommended_jobs.append(job_title)
        
        for skill in df.iloc[idx]['job_skills']:
            job_skills_pool.add(skill.lower())
            
    user_skills_lower = [s.lower() for s in user_skills]
    missing_skills = [skill for skill in job_skills_pool if skill not in user_skills_lower]
    
    return {
        "jobs": recommended_jobs,
        "missing_skills": list(set(missing_skills))[:15]
    }