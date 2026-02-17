import pandas as pd
import numpy as np
import torch
import random
import pickle
import time
from tqdm import tqdm

from transformers import (
    T5Tokenizer, T5EncoderModel,
    XLNetTokenizer, XLNetModel,BertTokenizer, BertModel,BertForMaskedLM,AutoModelForMaskedLM,AutoTokenizer,AutoModel
)

# Optional import if using ESM
try:
    import esm
    HAS_ESM = True
except ImportError:
    HAS_ESM = False

# === Load and preprocess data ===
train_df = pd.read_csv('/home/f087s426/Downloads/data/GPCR/cv_9/train.txt', header=None, sep='\t')
test_df = pd.read_csv('/home/f087s426/Downloads/data/GPCR/cv_9/test.txt', header=None, sep='\t')
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
df = df.rename(columns={0: 'Class', 1: 'sequence'})
df['filtered_sequence'] = df['sequence'].apply(lambda x: ''.join([char for char in x if char.isalnum()]))

# === Fragment sequences (30 AA, 10x augmentation) ===
dff = []
for j in range(10):
    temp_df = df.copy()
    temp_df['fragmented_sequence'] = temp_df['filtered_sequence'].apply(
        lambda x: x[random.randint(0, len(x) - 30):random.randint(0, len(x) - 30) + 30] if len(x) > 30 else x
    )
    dff.append(temp_df)

merged_df = pd.concat(dff, ignore_index=True)
sequences = list(merged_df['fragmented_sequence'])

# === Define models to evaluate ===
#model_list = ['esm2', 'prott5', 'protxlnet']
model_list = ['protbert']
timing_results = {}

for model_name in model_list:
    print(f"\n=== Running model: {model_name.upper()} ===")
    start_time = time.time()

    if model_name == "esm2":
        if not HAS_ESM:
            print("Skipping ESM-2 because esm is not installed.")
            continue
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval().cuda()
        embedding_dim = 1280

        def get_embedding(seq):
            data = [("seq", seq.upper())]
            _, _, tokens = batch_converter(data)
            tokens = tokens.cuda()
            with torch.no_grad():
                results = model(tokens, repr_layers=[33], return_contacts=False)
            reps = results["representations"][33]
            return reps[0, 1:-1].mean(0).cpu().numpy()

    elif model_name == "protbert":
        tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
        model  = AutoModel.from_pretrained("Rostlab/prot_bert").eval().cuda()
        embedding_dim = 1024

        def get_embedding(seq):
            seq = ' '.join(list(seq.upper()))
            seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
            tokens = tokenizer(seq, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**{k: v.cuda() for k, v in tokens.items()})
            embeddings = outputs.last_hidden_state  # shape: [1, seq_len, hidden_dim]

            input_mask_expanded = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float().cuda()
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask  # [1, hidden_dim]
            return mean_embeddings.squeeze(0).cpu().numpy()



    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # === Generate embeddings ===
    embedding_matrices = []
    for seq in tqdm(sequences, desc=f"Embedding ({model_name})", unit="sequence"):
        try:
            embedding = get_embedding(seq)
            embedding_matrices.append(embedding)
        except Exception as e:
            print(f"Error with {model_name} on seq {seq[:10]}...: {e}")
            embedding_matrices.append(np.zeros(embedding_dim))

    # === Save embeddings ===
    print(embedding_matrices)
    out_file = f"GPCR_{model_name}_embedding.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump(embedding_matrices, f)

    elapsed_time = time.time() - start_time
    timing_results[model_name] = elapsed_time
    print(f">>> {model_name.upper()} completed in {elapsed_time:.2f} seconds.")

# === Summary of Timing ===
print("\n=== Summary: Embedding Generation Times ===")
for model_name, elapsed in timing_results.items():
    print(f"{model_name.upper()}: {elapsed:.2f} seconds")
