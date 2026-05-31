import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
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
# 1. Enable Attention & Hidden States
# ---------------------
def visualize_transformer_interpretability(sequence, model, tokenizer, max_len=256):
    """
    Visualizes Transformer interpretability for a single sequence:
    - Attention map from last layer & selected head
    - Layer-wise representation similarity
    """
    device = next(model.parameters()).device

    # Re-enable attention outputs
    model.encoder.config.output_attentions = True
    model.encoder.config.output_hidden_states = True
    model.eval()

    # Tokenize input
    inputs = tokenizer(sequence, truncation=True, padding="max_length",
                       max_length=max_len, return_tensors="pt").to(device)

    # Forward pass with attention + hidden states
    with torch.no_grad():
        outputs = model.encoder(**inputs, output_attentions=True, output_hidden_states=True)

    attentions = outputs.attentions          # list of (num_layers) tensors [B, heads, L, L]
    hidden_states = outputs.hidden_states    # list of (num_layers+1) tensors [B, L, hidden_size]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tokens = [t.replace("##", "") for t in tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])]

    # ---------------------
    # 2. Attention Visualization
    # ---------------------
    layer = -1  # last layer
    head = 0    # first head
    attn = attentions[layer][0, head].detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(attn[:40, :40], cmap="viridis", xticklabels=tokens[:40], yticklabels=tokens[:40])
    plt.title(f"Attention Map (Layer {layer}, Head {head})")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("Attention Map.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------------
    # 3. Layer Representation Similarity
    # ---------------------
    # CLS representations from each layer
    cls_embs = [h[:, 0, :].detach().cpu().numpy().squeeze() for h in hidden_states]
    final_cls = cls_embs[-1]

    sims = [cosine_similarity(final_cls.reshape(1, -1), h.reshape(1, -1))[0, 0] for h in cls_embs]

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(sims)), sims, marker="o")
    plt.title("Layer Contribution to Final Embedding")
    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Similarity to Final Layer CLS")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("layer_contribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✅ Visualization complete.")
    print(f"Total Layers: {len(sims)} | Final similarity: {sims[-1]:.3f}")

# ---------------------
# 4. Example Usage
# ---------------------
if __name__ == "__main__":
    # Load your trained student model
    model_path = "student_model.pt"
    base_model = "distilbert-base-uncased"
    student_model, student_tokenizer = load_student_model(model_path, base_model)

    # Example: visualize one protein sequence
    test_seq = "VNGVVRNYWVEGERRREDKLEVKSMGSTHKLLSKLFFLSSAEVGGIPGVAVECVDIGRNTSGEGDLLGTN"  # replace with one of your protein sequences
    visualize_transformer_interpretability(test_seq, student_model, student_tokenizer)
