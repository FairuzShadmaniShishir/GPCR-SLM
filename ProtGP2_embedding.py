import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar

# Load the dataset
df = pd.read_csv('metal_binding_data.csv')
df2 = df.dropna(axis=0)

final_sequence = df2[(df2['SEQUENCE'].str.len() > 50) & (df2['SEQUENCE'].str.len() < 1000)]['SEQUENCE']
final_sequence = list(final_sequence)
#
# Load model and tokenizer
# model = GPT2LMHeadModel.from_pretrained('nferruz/ProtGPT2')
# tokenizer = GPT2Tokenizer.from_pretrained('nferruz/ProtGPT2')
#
# # Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# # Function to extract sequence embeddings
# def get_protgpt2_embedding(sequence):
#     tokenizer.pad_token = tokenizer.eos_token
#     input_ids = tokenizer(sequence, return_tensors="pt", padding=True).input_ids.to(device)
#
#     # Run model inference
#     with torch.no_grad():
#         outputs = model(input_ids, output_hidden_states=True)
#
#     # Extract hidden states from the last layer
#     embeddings = outputs.hidden_states[-1].squeeze(0).cpu().numpy()
#     embeddings=embeddings.mean(axis=0)
#     #print(embeddings.shape)
#     return embeddings
#
# # Loop through the sequences and compute embeddings with tqdm for progress tracking
# embedding_matrices = []
# embedding=[]
# for seq in tqdm(final_sequence, desc="Processing sequences", unit="sequence"):
#     try:
#         embedding_matrix = get_protgpt2_embedding(seq)
#         embedding_matrices.append(embedding_matrix)
#     except Exception as e:
#         print(f"Error processing sequence: {seq}")
#         print(f"Error message: {e}")
#
# # Concatenate all embeddings
#         #embedding(embedding_matrices)
#
# # Print embedding details
# #print("Embedding shape:",len(embedding_matrices))
# #np.save('/home/f087s426/Research/Nanobody_Rudro/metallm_protgpt2_embedding.npz', embedding_matrices)
#
#
# # import numpy as np
# # l=np.load(('/home/f087s426/Research/Nanobody_Rudro/metallm_protgpt2_embedding.npy', embedding_matrix))
# # print(l.shape)
# import pickle
# # Save to a pickle file
# with open('/home/f087s426/Research/Nanobody_Rudro/metallm_protgpt2_embedding.pkl', 'wb') as file:
#     pickle.dump(embedding_matrices, file)

from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
import torch
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig

torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import torch.nn.functional as F
#sequence="1MEVVIVTGMSGAGK"
# load model and tokenizer

#bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained("hugohrban/progen2-base",trust_remote_code=True)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,   # Adjust if using another task
    r=8,                          # Rank (smaller = more efficient, but less expressive)
    lora_alpha=32,                # Scaling factor
    lora_dropout=0.1,             # Dropout to avoid overfitting
    target_modules=["query", "value"]  # Apply LoRA to attention layers
)
tokenizer = Tokenizer.from_pretrained("hugohrban/progen2-base")
tokenizer.no_padding()
#print(model.summary())
model.half()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def get_progen2_embedding(sequence):

    input_ids = torch.tensor(tokenizer.encode(sequence).ids).to(device)
    #
    #Run model inference
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    #
    #     # Extract hidden states from the last layer
    embeddings = outputs.hidden_states[-1].squeeze(0).cpu().numpy()
    embeddings = embeddings.mean(axis=0)
    #print(embeddings.shape)
    #print(embeddings)
    return embeddings


embedding_matrices=[]
embedding=[]
for seq in tqdm(final_sequence[:2], desc="Processing sequences", unit="sequence"):
    print(type(seq))
    embedding_matrix = get_progen2_embedding(seq)
    embedding_matrices.append(embedding_matrix)
# import pickle
# # #np.save('/home/f087s426/Research/Nanobody_Rudro/metallm_progen2_embedding.npy', embedding_matrices)
# with open('/home/f087s426/Research/MetaLLM_ACM_Rebuttal/metallm_progen2_embedding.pkl', 'wb') as file:
#     pickle.dump(embedding_matrices, file)


# import pickle
#
# file_path = 'metallm_progen2_embedding.pkl'  # Replace with the actual path to your pickle file
#
# with open(file_path, 'rb') as file:
#     loaded_data = pickle.load(file)
#     print(len(loaded_data))
#     print((loaded_data[0].shape))

