import numpy as np
import pickle
import os
import pandas as pd

# ==============================
# Step 1: Load embeddings
# ==============================
embs_path = "embs_metagenome_breastcancer.npy"
X_new = np.load(embs_path)
print("✅ Loaded embeddings:", X_new.shape)

# ==============================
# Step 2: Load all XGBoost models
# ==============================
model_dir = "ml_results_multi_embedding_saved_model/saved_models"   # folder containing 86 XGBoost .pkl models
models = {}

for file in os.listdir(model_dir):
    if file.endswith(".pkl"):
        model_path = os.path.join(model_dir, file)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model_name = file.replace(".pkl", "")
        models[model_name] = model

print(f"✅ Loaded {len(models)} models from {model_dir}")

# ==============================
# Step 3: Run predictions
# ==============================
results = {}

for model_name, model in models.items():
    try:
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_new)[:, 1]  # probability of positive class
            preds_binary = (preds >= 0.3).astype(int)  # convert to 0/1
        else:
            preds_binary = model.predict(X_new)
        results[model_name] = preds_binary
        print(f"✓ Predictions done for {model_name}")
    except Exception as e:
        print(f"✗ Error with {model_name}: {e}")

# ==============================
# Step 4: Save full prediction results
# ==============================
results_df = pd.DataFrame(results)
results_df.to_csv("predictions_metagenome_breastcancer.csv", index=False)
print("✅ Saved full predictions to predictions_metagenome_breastcancer.csv")

# ==============================
# Step 5: Extract top predictions per sample
# ==============================
top_pred = results_df.idxmax(axis=1)
top_score = results_df.max(axis=1)

summary_df = pd.DataFrame({
    "Sample_Index": np.arange(len(top_pred)),
    "Predicted_Class": top_pred,
    "Confidence": top_score
})

# ==============================
# Step 6: Summarize class predictions (positive/negative counts)
# ==============================
summary_counts = {}

for model_name in results_df.columns:
    positives = int(results_df[model_name].sum())
    negatives = int(len(results_df) - positives)
    summary_counts[model_name] = {"Positive": positives, "Negative": negatives}

# Convert to DataFrame
class_summary_df = pd.DataFrame.from_dict(summary_counts, orient="index")
class_summary_df.index.name = "Class_Name"

# Save summary
class_summary_df.to_csv("class_prediction_summary.csv")
print("✅ Saved class-level summary to class_prediction_summary.csv")
print(class_summary_df.head())


summary_df.to_csv("top_predictions_summary.csv", index=False)
print("✅ Saved top predictions to top_predictions_summary.csv")


from Bio import SeqIO

# ==============================
# Step 7: Extract FASTA IDs for predicted GPCRs
# ==============================

fasta_path = "/home/f087s426/FragGeneScan1.32/cleaned_breastCancer.faa"  # metagenome FASTA file
output_fasta = "predicted_gpcrs.fasta"
output_ids = "predicted_gpcr_ids.txt"

# Step 7.1: Identify samples predicted positive by any of the 86 models
# Create a boolean mask: True if any model predicted 1
positive_mask = results_df.any(axis=1)

# Get corresponding indices (these correspond to sequence order in FASTA)
positive_indices = np.where(positive_mask)[0]

print(f"✅ Found {len(positive_indices)} predicted GPCR sequences")

# Step 7.2: Extract sequences from FASTA by index
selected_records = []
for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
    if i in positive_indices:
        selected_records.append(record)

# Step 7.3: Write out selected sequences to FASTA
SeqIO.write(selected_records, output_fasta, "fasta")

# Step 7.4: Also save sequence IDs to a plain text file
with open(output_ids, "w") as f:
    for record in selected_records:
        f.write(record.id + "\n")

print(f"✅ Saved {len(selected_records)} predicted GPCR sequences to:")
print(f"   FASTA: {output_fasta}")
print(f"   IDs:   {output_ids}")



# ==============================
# Compare two text files of FASTA IDs
# ==============================

file1 = "predicted_gpcr_ids.txt"      # first list
file2 = "/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/gpcr_like_ids.txt"         # second list (e.g., HMMER/GPCRdb hits)

# Load IDs from both files
with open(file1) as f:
    ids1 = set(line.strip() for line in f if line.strip())

with open(file2) as f:
    ids2 = set(line.strip() for line in f if line.strip())

# Find overlaps and differences
common_ids = ids1.intersection(ids2)
unique_to_file1 = ids1 - ids2
unique_to_file2 = ids2 - ids1

# Print summary
print("✅ Comparison Summary")
print(f"Total IDs in {file1}: {len(ids1)}")
print(f"Total IDs in {file2}: {len(ids2)}")
print(f"Common IDs: {len(common_ids)}")
print(f"Unique to {file1}: {len(unique_to_file1)}")
print(f"Unique to {file2}: {len(unique_to_file2)}")

# Save results
with open("common_ids.txt", "w") as f:
    f.write("\n".join(sorted(common_ids)))

with open("unique_to_file1.txt", "w") as f:
    f.write("\n".join(sorted(unique_to_file1)))

with open("unique_to_file2.txt", "w") as f:
    f.write("\n".join(sorted(unique_to_file2)))

print("\n✅ Saved:")
print("  common_ids.txt")
print("  unique_to_file1.txt")
print("  unique_to_file2.txt")


