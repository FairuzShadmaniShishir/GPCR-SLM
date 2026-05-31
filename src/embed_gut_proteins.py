import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModel, EsmTokenizer, EsmModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.decomposition import PCA
from Bio import SeqIO

# ---------------------
# 1. Student Model Definition (same as training)
# ---------------------
class StudentModel(nn.Module):
    def __init__(self, base_model="distilbert-base-uncased", target_dim=1280):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.proj = nn.Linear(self.encoder.config.hidden_size, target_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.proj(cls)


# ---------------------
# 2. Load Trained Student Model
# ---------------------
def load_student_model(model_path="student_model.pt", base_model="distilbert-base-uncased", target_dim=1280):
    """Load the trained student model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = StudentModel(base_model=base_model, target_dim=target_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------
# 3. Generate Student Embeddings
# ---------------------
def generate_student_embeddings(sequences, model, tokenizer, batch_size=1, max_len=512):
    """Generate embeddings using the trained student model"""
    device = next(model.parameters()).device
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Generating student embeddings"):
            if hasattr(sequences, 'iloc'):
                batch_sequences = sequences.iloc[i:i + batch_size].tolist()
            else:
                batch_sequences = sequences[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(batch_sequences,
                               truncation=True,
                               padding="max_length",
                               max_length=max_len,
                               return_tensors="pt")

            # Move to device
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # Generate embeddings
            output = model(input_ids, attention_mask)
            embeddings.append(output)

    return torch.cat(embeddings, dim=0)


def main_test():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import random
    import time

    sequences = []
    for record in SeqIO.parse("/home/f087s426/FragGeneScan1.32/cleaned_breastCancer.faa", "fasta"):
        seq_str = str(record.seq)
        sequences.append(seq_str)

        # Load student model
    print("Loading student model...")
    student_model, student_tokenizer = load_student_model()

    # Time original embeddings
    print("Generating original embeddings...")
    start_time_orig = time.time()
    student_embs = generate_student_embeddings(sequences, student_model, student_tokenizer)
    end_time_orig = time.time()
    print(f"Time taken for original embeddings: {end_time_orig - start_time_orig:.3f} seconds")

    # Convert to numpy if tensor
    embs_np = student_embs.detach().cpu().numpy() if hasattr(student_embs, "detach") else student_embs
    np.save("embs_metagenome_breastcancer.npy", embs_np)



if __name__ == "__main__":
    main_test()
