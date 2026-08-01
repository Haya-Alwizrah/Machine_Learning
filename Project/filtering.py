import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import torch.nn.functional as F
from rapidfuzz import process, fuzz
import os
import json
os.makedirs("json", exist_ok=True)

class Filtering:
    def __init__(self, df, column, similarity_threshold=0.80, fuzzy_threshold=95, coverage_target=0.80):
        self.df = df.copy()
        self.column = column
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.coverage_target = coverage_target

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

        self.values_counts = None
        self.common_values = None
        self.sem_mapping = {}
        self.fuz_mapping = {}
        self.fin_mapping = {}

        self.embedding_file = os.path.join("json", f"{column}_embedding_mapping.json")
        self.fuzzy_file = os.path.join("json", f"{column}_fuzzy_mapping.json")
        self.final_file = os.path.join("json", f"{column}_final_mapping.json")

    @staticmethod
    def normalize_regex(skill):
        skill = skill.lower().strip()
        skill = re.sub(r"[-/_]", " ", skill)
        skill = re.sub(r"\bskills?\b$", "", skill)
        skill = re.sub(r"\b(strong|excellent|good|basic|advanced|proven|solid)\b", "", skill)
        skill = re.sub(r"\b\d+\+?\s*(years?|yrs?)\b.*", "", skill)
        skill = re.sub(r"^(ability to|experience in|experience with|knowledge of)\s+", "", skill)
        skill = re.sub(r"\s+", " ", skill)
        skill = skill.strip()

        if len(skill.split()) > 4:
                return None

        return skill
    
    def clean(self):
        if self.column ==  "job_skills":
            self.df[f"{self.column}_clean"] = self.df[self.column].str.split(",").apply(
                    lambda values: [
                        v for v in
                        (self.normalize_regex(x) for x in values)
                        if v is not None and v != ""
                    ]
                )
        else:
            self.df[f"{self.column}_clean"] = (
                self.df[self.column]
                .str.replace(r"\s*\([^)]*\)", "", regex=True)
                .str.strip()
            )
        
    def count_values(self):
        all_values = self.df[f"{self.column}_clean"].explode().dropna()
        all_values = all_values[all_values != ""]

        self.values_counts = all_values.value_counts()
        print("Top 10 values")
        print(self.values_counts.head(10))

        print(f"Unique skills: {len(self.values_counts):,}")
        print(f"Skills appearing once: {(self.values_counts == 1).sum():,}")
        print(f"Skills appearing more than once: {(self.values_counts != 1).sum():,}")

    def choose_min_count(self) -> int:
        total = self.values_counts.sum()

        best_n = 1
        print("Coverage Table")
        for n in [2, 3, 5, 10, 20, 50, 57, 100]:

            values = self.values_counts[self.values_counts >= n].sum()
            coverage = values / total

            print(f"{n:<3} -> {values:<8} values, Coverage = {coverage:.2%}")

            if coverage >= self.coverage_target:
                best_n = n

        print(f"\nSelected min_count = {best_n}")

        return best_n

    def semantic_mapping(self):
        min_count = self.choose_min_count()
        self.common_values = self.values_counts[self.values_counts >= min_count].index.tolist()

        if self.model is None:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        embeddings = self.model.encode(
            self.common_values,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=256,
            device=self.device
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)

        self.sem_mapping = {}

        for i, value in enumerate(tqdm(self.common_values)):

            if value in self.sem_mapping:
                continue

            sims = torch.matmul(embeddings[i], embeddings.T)
            idx = torch.where(sims >= self.similarity_threshold)[0].cpu().numpy()

            similar = [self.common_values[j] for j in idx]

            canonical = max(similar, key=lambda x: self.values_counts[x])

            for s in similar:
                self.sem_mapping[s] = canonical

    def fuzzy_mapping(self):
        canonical = list(set(self.sem_mapping.values()))

        self.fuz_mapping = {}

        for c in tqdm(canonical):

            matches = process.extract(
                c,
                canonical,
                scorer=fuzz.ratio,
                limit=5
            )

            for match, score, _ in matches:

                if match == c:
                    continue

                if score >= self.fuzzy_threshold:
                    canonical = max([c, match], key=lambda x: self.values_counts[x])

                    self.fuz_mapping[c] = canonical
                    self.fuz_mapping[match] = canonical

    def final_mapping(self):
        self.fin_mapping = {
            k: self.fuz_mapping.get(v, v)
            for k, v in self.sem_mapping.items()
        }

    def run(self):
        print("start cleaning")
        self.clean()

        if os.path.exists(self.final_file):
            print("Loading existing mappings...")
            with open(self.final_file, "r", encoding="utf-8") as f:
                self.fin_mapping = json.load(f)            
        else:
            print("Mappings not found. Building them...")
            self.count_values()

            # Semantic Mapping
            if os.path.exists(self.embedding_file):
                print("Loading embedding mapping...")
                with open(self.embedding_file, "r", encoding="utf-8") as f:
                    self.sem_mapping = json.load(f)
            else:
                print("Building embedding mapping...")
                self.semantic_mapping()
                with open(self.embedding_file, "w", encoding="utf-8") as f:
                    json.dump(self.sem_mapping, f, ensure_ascii=False, indent=4)

            # Fuzzy Mapping
            if os.path.exists(self.fuzzy_file):
                print("Loading fuzzy mapping...")
                with open(self.fuzzy_file, "r", encoding="utf-8") as f:
                    self.fuz_mapping = json.load(f)
            else:
                print("Building fuzzy mapping...")
                self.fuzzy_mapping()
                with open(self.fuzzy_file, "w", encoding="utf-8") as f:
                    json.dump(self.fuz_mapping, f, ensure_ascii=False, indent=4)

            print("Building final mapping...")
            self.final_mapping()
            with open(self.final_file, "w", encoding="utf-8") as f:
                json.dump(self.fin_mapping, f, ensure_ascii=False, indent=4)

        if self.column ==  "job_skills":
            self.df[f"{self.column}_mapped"] = self.df[f"{self.column}_clean"].apply(
                lambda values: [
                    self.fin_mapping.get(value, value)
                    for value in values
                ]
            )
        else:
            self.df[f"{self.column}_mapped"] = self.df[f"{self.column}_clean"].apply(
                lambda value: self.fin_mapping.get(value, value)
            )

        print("Done")
        return self.df[f"{self.column}_mapped"]