import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, EsmTokenizer, EsmModel
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import pickle

# ---------------------
# 1. Load Your Data
# ---------------------
df = pd.read_csv('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR.csv')
#df = df.rename(columns={0: 'Class', 1: 'sequence'})
sequences = df['fragmented_sequence']

# ---------------------
# 2. Generate ESM2 Embeddings
# ---------------------
def generate_esm2_embeddings(sequences, model_name="facebook/esm2_t33_650M_UR50D", batch_size=32):
    """
    Generate ESM2 embeddings for protein sequences
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ESM2 model and tokenizer
    print(f"Loading ESM2 model: {model_name}")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()
    
    embeddings = []
    
    print("Generating ESM2 embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i:i+batch_size].tolist()
            
            # Tokenize sequences
            tokens = tokenizer(batch_sequences, 
                             padding=True, 
                             truncation=True, 
                             max_length=1024,  # ESM2 max length
                             return_tensors="pt")
            
            # Move to device
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            # Get embeddings
            outputs = model(**tokens)
            
            # Use mean pooling over sequence length (excluding special tokens)
            attention_mask = tokens['attention_mask']
            last_hidden_states = outputs.last_hidden_state
            
            # Mean pooling
            masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
            summed_embeddings = masked_embeddings.sum(dim=1)
            seq_lengths = attention_mask.sum(dim=1, keepdim=True)
            mean_embeddings = summed_embeddings / seq_lengths
            
            embeddings.append(mean_embeddings.cpu())
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(embeddings, dim=0)
    print(f"Generated embeddings shape: {all_embeddings.shape}")
    
    return all_embeddings

# Generate ESM2 embeddings
print("Generating ESM2 embeddings for sequences...")
#esm2_embeddings = generate_esm2_embeddings(sequences)

# Optional: Save embeddings for future use
print("Saving embeddings...")
# with open('generated_esm2_embeddings.pkl', 'wb') as f:
#     pickle.dump(esm2_embeddings.numpy(), f)

# ---------------------
# 3. Dataset Class
# ---------------------
class SequenceEmbeddingDataset(Dataset):
    def __init__(self, sequences, embeddings, tokenizer, max_len=512):
        self.sequences = sequences
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences.iloc[idx] if hasattr(self.sequences, 'iloc') else self.sequences[idx]
        emb = self.embeddings[idx]
        
        inputs = self.tokenizer(seq, 
                               truncation=True, 
                               padding="max_length", 
                               max_length=self.max_len, 
                               return_tensors="pt")
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'target': torch.tensor(emb, dtype=torch.float)
        }

# ---------------------
# 4. Student Model
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
# 5. Cosine Similarity Loss
# ---------------------
def cosine_loss(student_emb, teacher_emb):
    s = F.normalize(student_emb, dim=-1)
    t = F.normalize(teacher_emb, dim=-1)
    return 1 - (s * t).sum(dim=-1).mean()

# ---------------------
# 6. Training Loop
# ---------------------
def train_student_model(sequences, embeddings, base_model="distilbert-base-uncased", 
                       epochs=10, batch_size=32, lr=2e-5):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = SequenceEmbeddingDataset(sequences, embeddings, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Get target dimension from embeddings
    target_dim = embeddings.shape[1]
    model = StudentModel(base_model=base_model, target_dim=target_dim).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            target = batch['target'].cuda()
            
            output = model(input_ids, attention_mask)
            loss = cosine_loss(output, target)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(dataloader):.4f}")
    
    torch.save(model.state_dict(), "student_model.pt")
    return model

# ---------------------
# 7. Train the Model                  
# ---------------------
print("Starting student model training...")
model = train_student_model(sequences, esm2_embeddings, epochs=10)

print("Training completed!")
print(f"Student model saved as 'student_model.pt'")
print(f"ESM2 embeddings saved as 'generated_esm2_embeddings.pkl'")
