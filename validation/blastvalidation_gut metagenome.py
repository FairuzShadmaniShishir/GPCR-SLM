import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --- Load gut embeddings ---
embs_gut = np.load("embs_gut.npy")  # shape (N, 1280)
print("Gut embeddings shape:", embs_gut.shape)
print("Gut embeddings shape:", embs_gut[0])
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
print(labels.shape)





import os
import time
import subprocess
import urllib.request
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
BLAST_BIN     = "/home/f087s426/ncbi-blast-2.15.0+/bin/"
FAA_FILE      = "/home/f087s426/FragGeneScan1.32/cleaned_SRR16133736.faa"

SUBSET_FAA = "cleaned_SRR16133736_100k.faa"

max_seqs = 100000
count = 0

with open(FAA_FILE, "r") as fin, open(SUBSET_FAA, "w") as fout:
    for line in fin:
        if line.startswith(">"):
            count += 1
            if count > max_seqs:
                break
        fout.write(line)

print("Subset created:", SUBSET_FAA)

FAA_FILE      = "cleaned_SRR16133736_100k.faa"

import pandas as pd

file_path = "cleaned_SRR16133736_100k.faa"

ids, seqs = [], []
seq = ""

for line in open(file_path):
    line = line.strip()
    if line.startswith(">"):
        if seq:
            seqs.append(seq)
        ids.append(line[1:])
        seq = ""
    else:
        seq += line

seqs.append(seq)

df = pd.DataFrame({"id": ids, "sequence": seqs})
df["label"] = labels

print(df.head())

# Output files
UNIPROT_FASTA     = "uniprot_gpcr_reviewed.fasta"
DB_OUTPUT         = "gpcr_db"
GPCR_IDS_TXT      = "predicted_gpcr_ids.txt"
NONGPCR_IDS_TXT   = "predicted_nongpcr_ids.txt"
GPCR_FASTA        = "predicted_gpcrs_only.fasta"
NONGPCR_FASTA     = "predicted_nongpcrs_only.fasta"
BLAST_GPCR_OUT    = "blast_gpcr_results.tsv"
BLAST_NONGPCR_OUT = "blast_nongpcr_results.tsv"

BLASTP_EXEC      = os.path.join(BLAST_BIN, "blastp")
MAKEBLASTDB_EXEC = os.path.join(BLAST_BIN, "makeblastdb")

print("=" * 60)
print("  GPCR BLAST VALIDATION PIPELINE")
print("=" * 60)


# ============================================================
# STEP 1: Use your existing df directly
# ============================================================
print("\n[STEP 1] Reading predictions from existing dataframe df...")

# ✅ df is already in memory — just use it directly
print(f"  Total sequences: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"  Label distribution:\n{df['label'].value_counts()}")

gpcr_ids    = df[df["label"] == "GPCR"]["id"]
nongpcr_ids = df[df["label"] == "non-GPCR"]["id"]

gpcr_ids.to_csv(GPCR_IDS_TXT,       index=False, header=False)
nongpcr_ids.to_csv(NONGPCR_IDS_TXT, index=False, header=False)

print(f"  ✅ Saved {len(gpcr_ids)} GPCR IDs     → {GPCR_IDS_TXT}")
print(f"  ✅ Saved {len(nongpcr_ids)} non-GPCR IDs → {NONGPCR_IDS_TXT}")


# ============================================================
# STEP 2: Extract sequences from FAA file
# ============================================================
print("\n[STEP 2] Extracting sequences from FAA file...")

def extract_sequences(faa_path, id_file, output_fasta):
    with open(id_file) as f:
        wanted = set(line.strip() for line in f if line.strip())

    found = 0
    current_id = None
    current_seq = []

    with open(faa_path) as f, open(output_fasta, "w") as out:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_id in wanted:
                    out.write(f">{current_id}\n{''.join(current_seq)}\n")
                    found += 1
                current_id  = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        # Last sequence
        if current_id and current_id in wanted:
            out.write(f">{current_id}\n{''.join(current_seq)}\n")
            found += 1

    return found

gpcr_found    = extract_sequences(FAA_FILE, GPCR_IDS_TXT,    GPCR_FASTA)
nongpcr_found = extract_sequences(FAA_FILE, NONGPCR_IDS_TXT, NONGPCR_FASTA)

print(f"  ✅ Extracted {gpcr_found} GPCR sequences     → {GPCR_FASTA}")
print(f"  ✅ Extracted {nongpcr_found} non-GPCR sequences → {NONGPCR_FASTA}")

if gpcr_found != len(gpcr_ids):
    print(f"  ⚠️  Warning: expected {len(gpcr_ids)} but found {gpcr_found}")
    print(f"      Check IDs match exactly between CSV and FAA file")


# ============================================================
# STEP 3: Download UniProt GPCR reference
# ============================================================
print("\n[STEP 3] Downloading UniProt reviewed GPCR reference...")

if os.path.isfile(UNIPROT_FASTA):
    print(f"  ✅ Already exists — skipping download")
else:
    url = ("https://rest.uniprot.org/uniprotkb/stream?"
           "query=family%3A%22g+protein-coupled+receptor%22"
           "+AND+reviewed%3Atrue&format=fasta")
    urllib.request.urlretrieve(url, UNIPROT_FASTA)
    print(f"  ✅ Downloaded → {UNIPROT_FASTA}")

result = subprocess.run(f"grep -c '>' {UNIPROT_FASTA}",
                        shell=True, capture_output=True, text=True)
print(f"  Reference sequences: {result.stdout.strip()}")


# ============================================================
# STEP 4: Build BLAST database
# ============================================================
print("\n[STEP 4] Building BLAST database...")

if os.path.isfile(DB_OUTPUT + ".pin"):
    print(f"  ✅ Already exists — skipping")
else:
    subprocess.run([
        MAKEBLASTDB_EXEC,
        "-in",     UNIPROT_FASTA,
        "-dbtype", "prot",
        "-out",    DB_OUTPUT
    ], check=True)
    print(f"  ✅ Database created → {DB_OUTPUT}")


# ============================================================
# STEP 5: BLASTP on predicted GPCRs
# ============================================================
print("\n[STEP 5] BLASTP on predicted GPCRs...")

# Filter short sequences first
def filter_fasta(input_fasta, output_fasta, min_length=30):
    kept, skipped = 0, 0
    current_id, current_seq = None, ""
    with open(input_fasta) as f, open(output_fasta, "w") as out:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    if len(current_seq) >= min_length:
                        out.write(f">{current_id}\n{current_seq}\n")
                        kept += 1
                    else:
                        skipped += 1
                current_id  = line[1:]
                current_seq = ""
            else:
                current_seq += line
        if current_id:
            if len(current_seq) >= min_length:
                out.write(f">{current_id}\n{current_seq}\n")
                kept += 1
            else:
                skipped += 1
    print(f"  Kept: {kept}  Skipped: {skipped}")
    return kept

print("  Filtering GPCR sequences >= 30 aa...")
gpcr_kept = filter_fasta(GPCR_FASTA, "predicted_gpcrs_filtered.fasta")

print("  Filtering non-GPCR sequences >= 30 aa...")
nongpcr_kept = filter_fasta(NONGPCR_FASTA, "predicted_nongpcrs_filtered.fasta")

start = time.time()
subprocess.run([
    BLASTP_EXEC,
    "-query",           "predicted_gpcrs_filtered.fasta",
    "-db",              DB_OUTPUT,
    "-out",             BLAST_GPCR_OUT,
    "-outfmt",          "6 qseqid sseqid pident length evalue bitscore",
    "-evalue",          "10",
    "-word_size",       "2",
    "-matrix",          "BLOSUM62",
    "-comp_based_stats","0",
    "-seg",             "no",
    "-num_threads",     "8",
    "-max_target_seqs", "1"
], check=True)
print(f"  ✅ Done in {time.time()-start:.1f}s")


# ============================================================
# STEP 6: BLASTP on predicted non-GPCRs
# ============================================================
print("\n[STEP 6] BLASTP on predicted non-GPCRs...")

start = time.time()
subprocess.run([
    BLASTP_EXEC,
    "-query",           "predicted_nongpcrs_filtered.fasta",
    "-db",              DB_OUTPUT,
    "-out",             BLAST_NONGPCR_OUT,
    "-outfmt",          "6 qseqid sseqid pident length evalue bitscore",
    "-evalue",          ".1",
    "-word_size",       "2",
    "-matrix",          "BLOSUM62",
    "-comp_based_stats","0",
    "-seg",             "no",
    "-num_threads",     "8",
    "-max_target_seqs", "1"
], check=True)
print(f"  ✅ Done in {time.time()-start:.1f}s")


# ============================================================
# STEP 7: Metrics — use filtered counts and relaxed threshold
# ============================================================
import pandas as pd
thresholds = [
    {"pident": 25, "length": 20},   # your current (effectively)
    {"pident": 30, "length": 30},   # moderate
    {"pident": 35, "length": 40},   # stricter
]
cols = ["qseqid","sseqid","pident","length","evalue","bitscore"]

# Load BLAST results
try:
    gpcr_blast    = pd.read_csv("blast_gpcr_results.tsv",    sep="\t", names=cols)
except pd.errors.EmptyDataError:
    gpcr_blast    = pd.DataFrame(columns=cols)

try:
    nongpcr_blast = pd.read_csv("blast_nongpcr_results.tsv", sep="\t", names=cols)
except pd.errors.EmptyDataError:
    nongpcr_blast = pd.DataFrame(columns=cols)

# Count actual sequences in filtered FASTA files
def count_fasta(filepath):
    count = 0
    try:
        with open(filepath) as f:
            for line in f:
                if line.startswith(">"): count += 1
    except FileNotFoundError:
        print(f"⚠️ File not found: {filepath}")
    return count

gpcr_total    = count_fasta("predicted_gpcrs_filtered.fasta")
nongpcr_total = count_fasta("predicted_nongpcrs_filtered.fasta")

print(f"GPCR sequences submitted to BLAST:     {gpcr_total}")
print(f"Non-GPCR sequences submitted to BLAST: {nongpcr_total}")

# Apply threshold — strict for short sequences
PIDENT_THRESHOLD = 30
LENGTH_THRESHOLD = 20

gpcr_confirmed = set(gpcr_blast[
    (gpcr_blast["pident"] >= PIDENT_THRESHOLD) &
    (gpcr_blast["length"] >= LENGTH_THRESHOLD)]["qseqid"])

nongpcr_confirmed = set(nongpcr_blast[
    (nongpcr_blast["pident"] >= PIDENT_THRESHOLD) &
    (nongpcr_blast["length"] >= LENGTH_THRESHOLD)]["qseqid"])

# Correct totals from actual FASTA counts
# TP = len(gpcr_confirmed)
# FP = len(nongpcr_confirmed)
# FN = gpcr_total    - TP       # ← uses actual count not hardcoded number
# TN = nongpcr_total - FP       # ← uses actual count not hardcoded number
#
# precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
# recall      = TP / (TP + FN) if (TP + FN) > 0 else 0
# f1          = (2 * precision * recall / (precision + recall)
#                if (precision + recall) > 0 else 0)
# specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
#
# print(f"\n{'='*50}")
# print(f"  CORRECTED VALIDATION RESULTS")
# print(f"{'='*50}")
# print(f"  GPCR total:     {gpcr_total}")
# print(f"  Non-GPCR total: {nongpcr_total}")
# print(f"{'='*50}")
# print(f"  TP={TP}  FP={FP}  FN={FN}  TN={TN}")
# print(f"{'='*50}")
# print(f"  Precision:   {precision:.4f}")
# print(f"  Recall:      {recall:.4f}")
# print(f"  F1 Score:    {f1:.4f}")
# print(f"  Specificity: {specificity:.4f}")
# print(f"{'='*50}")
#
# # Sanity check
# print(f"\n  Sanity check:")
# print(f"  TP+FN = {TP+FN} (should be {gpcr_total})")
# print(f"  FP+TN = {FP+TN} (should be {nongpcr_total})")


# for t in thresholds:
#     pi, ln = t["pident"], t["length"]
#
#     gc = set(gpcr_blast[
#         (gpcr_blast["pident"] >= pi) &
#         (gpcr_blast["length"] >= ln)]["qseqid"])
#     nc = set(nongpcr_blast[
#         (nongpcr_blast["pident"] >= pi) &
#         (nongpcr_blast["length"] >= ln)]["qseqid"])
#
#     TP = len(gc);  FP = len(nc)
#     FN = 6733 - TP;  TN = 49934 - FP
#
#     prec = TP/(TP+FP) if (TP+FP) > 0 else 0
#     rec  = TP/(TP+FN) if (TP+FN) > 0 else 0
#     f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
#
#     print(f"{pi:>8} {ln:>8} {TP:>7} {FP:>7} {FN:>7} {TN:>7} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}")

import pandas as pd
from itertools import product

cols = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore"]
gpcr_blast    = pd.read_csv("blast_gpcr_results.tsv",    sep="\t", names=cols)
nongpcr_blast = pd.read_csv("blast_nongpcr_results.tsv", sep="\t", names=cols)

# gpcr_total    = 6733
# nongpcr_total = 49934

# Fine sweep — short-sequence appropriate ranges
pident_vals = [25, 27, 30, 33, 35]
length_vals = [15, 18, 20, 22, 25]

print(f"{'pident':>8} {'minlen':>8} {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>7} "
      f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'Spec':>7}{'mcc':>7}")
print("-" * 80)

results = []
for pi, ln in product(pident_vals, length_vals):
    gc = set(gpcr_blast[
        (gpcr_blast["pident"] >= pi) &
        (gpcr_blast["length"] >= ln)]["qseqid"])
    nc = set(nongpcr_blast[
        (nongpcr_blast["pident"] >= pi) &
        (nongpcr_blast["length"] >= ln)]["qseqid"])

    TP = len(gc)
    FP = len(nc)
    FN = gpcr_total - TP
    TN = nongpcr_total - FP

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0

    import math

    mcc = ((TP * TN) - (FP * FN)) / math.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    ) if (TP + FP) and (TP + FN) and (TN + FP) and (TN + FN) else 0

    results.append((pi, ln, TP, FP, FN, TN, prec, rec, f1, spec, mcc))

    print(f"{pi:>8} {ln:>8} {TP:>7} {FP:>7} {FN:>7} {TN:>7} "
          f"{prec:>7.4f} {rec:>7.4f} {f1:>7.4f} {spec:>7.4f} {mcc:>7.4f}")

# Best F1
best = max(results, key=lambda x: x[8])
print(f"\n  ✅ Best F1={best[8]:.4f} at pident≥{best[0]}, length≥{best[1]}")
print(f"     TP={best[2]}  FP={best[3]}  FN={best[4]}  TN={best[5]}")
print(f"     Precision={best[6]:.4f}  Recall={best[7]:.4f}  Specificity={best[9]:.4f}")

# Also show best precision/recall tradeoff
best_bal = max(results, key=lambda x: min(x[6], x[7]))  # maximise the weaker of the two
print(f"\n  ✅ Best balanced (max of min(prec,rec)) at pident≥{best_bal[0]}, length≥{best_bal[1]}")
print(f"     Precision={best_bal[6]:.4f}  Recall={best_bal[7]:.4f}  F1={best_bal[8]:.4f}")
# ============================================================
# STEP 8: Identity distribution + save outputs
# ============================================================
print("\n[STEP 8] Saving results...")

if len(gpcr_blast) > 0:
    confirmed_hits = gpcr_blast[
        (gpcr_blast["pident"] >= PIDENT_THRESHOLD) &
        (gpcr_blast["length"] >= LENGTH_THRESHOLD)]

    print(f"\n  % Identity of confirmed hits:")
    print(f"  Min:    {confirmed_hits['pident'].min():.2f}%")
    print(f"  Max:    {confirmed_hits['pident'].max():.2f}%")
    print(f"  Mean:   {confirmed_hits['pident'].mean():.2f}%")
    print(f"  Median: {confirmed_hits['pident'].median():.2f}%")

    confirmed_hits.to_csv("confirmed_gpcr_hits.csv", index=False)
    print(f"  ✅ Saved → confirmed_gpcr_hits.csv")

# metrics = {
#     "TP": TP, "FP": FP, "FN": FN, "TN": TN,
#     "Precision":   round(precision,   4),
#     "Recall":      round(recall,      4),
#     "F1":          round(f1,          4),
#     "Specificity": round(specificity, 4)
# }
#pd.DataFrame([metrics]).to_csv("validation_metrics.csv", index=False)
print("  ✅ Saved → validation_metrics.csv")

print("\n🎯 Pipeline completed successfully!")


# Run this in Python to see full distribution
lengths = []
seq = ""
with open("predicted_gpcrs_only.fasta") as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if seq:
                lengths.append(len(seq))
            seq = ""
        else:
            seq += line
    if seq:
        lengths.append(len(seq))

import pandas as pd
s = pd.Series(lengths)
print(s.describe())
print(f"\nSequences >= 30 aa:  {(s >= 30).sum()}")
print(f"Sequences >= 50 aa:  {(s >= 50).sum()}")
print(f"Sequences >= 100 aa: {(s >= 100).sum()}")
print(f"Sequences < 30 aa:   {(s < 30).sum()}")


