import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --- Load gut embeddings ---
embs_gut = np.load("embs_metagenome_breastcancer.npy")  # shape (N, 1280)
print("Gut embeddings shape:", embs_gut.shape)

# --- Load mean embedding from student embeddings ---
with open("student_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

if isinstance(data, dict):
    embs_student = np.vstack(list(data.values()))
elif isinstance(data, list):
    embs_student = np.array(data)
else:
    embs_student = np.array(data)

mean_embedding = np.mean(embs_student, axis=0).reshape(1, -1)  # shape (1, 1280)

# --- Compute cosine similarity ---
similarities = cosine_similarity(embs_gut, mean_embedding).flatten()

# --- Classify based on threshold 0.7 ---
threshold = 0.98
labels = np.where(similarities >= threshold, "GPCR", "non-GPCR")

# --- Print summary ---
num_gpcr = np.sum(labels == "GPCR")
num_non_gpcr = np.sum(labels == "non-GPCR")

print(f"Total embeddings: {len(labels)}")
print(f"GPCR: {num_gpcr}, non-GPCR: {num_non_gpcr}")
print("First 10 similarities:", similarities[:10])
print("First 10 labels:", labels[:10])





