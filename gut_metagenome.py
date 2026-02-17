import csv
import pandas as pd

df = pd.read_csv('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR.csv')
with open('/home/f087s426/PycharmProjects/Protein_Family_Prediction/GPCR.csv') as f, open("gpcr_reference.fasta", "w") as out:
    reader = csv.DictReader(f)
    for row in reader:
        seq_id = row['Class'].strip()
        seq = row['sequence'].replace(" ", "").strip()
        out.write(f">{seq_id}\n{seq}\n")

import os
import subprocess
import time

# === Paths ===
blast_bin = "/home/f087s426/ncbi-blast-2.15.0+/bin/"
blastp_executable = os.path.join(blast_bin, "blastp")
makeblastdb_executable = os.path.join(blast_bin, "makeblastdb")

# === Input files ===
gpcr_fasta = "/home/f087s426/PycharmProjects/Protein_Family_Prediction/gpcr_reference.fasta"
gut_proteins = "/home/f087s426/FragGeneScan1.32/cleaned_breastCancer.faa"

# === Output files ===
db_output = "/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/gpcr_db"
blast_output = "/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/gpcr_hits.tsv"
filtered_ids = "/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/gpcr_like_ids.txt"
filtered_fasta = "/home/f087s426/ncbi-blast-2.15.0+/bin/GPCR_Fasta/breastcancer_gpcr_like.faa"

# === Step 1: Create BLAST database ===
print("Checking if GPCR BLAST database exists...")
if not os.path.isfile(db_output + ".pin"):
    print("Database not found — creating GPCR BLAST database...")
    makeblastdb_cmd = [
        makeblastdb_executable,
        "-in", gpcr_fasta,
        "-dbtype", "prot",
        "-out", db_output
    ]
    subprocess.run(makeblastdb_cmd, check=True)
    print("✅ GPCR BLAST database created successfully.")
else:
    print("✅ Database already exists — skipping creation.")

# === Step 2: Run BLASTP ===
blastp_cmd = [
    blastp_executable,
    "-query", gut_proteins,
    "-db", db_output,
    "-out", blast_output,
    "-outfmt", "6 qseqid sseqid pident length evalue bitscore",
    "-word_size", "2",
    "-seg", "no",
    "-comp_based_stats", "0",
    "-evalue", "10"
]


print("\nRunning BLASTP...")
start_time = time.time()
subprocess.run(blastp_cmd, check=True)
elapsed_time = time.time() - start_time
print(f"✅ BLASTP completed in {elapsed_time:.2f} seconds.")
print(f"📄 Raw results saved at: {blast_output}")

# === Step 3: Filter GPCR-like hits ===
print("\nFiltering hits with ≥30% identity and alignment length ≥60...")
awk_cmd = f"awk '$3 >= 30 && $4 >= 30 {{print $1}}' {blast_output} | sort | uniq > {filtered_ids}"
subprocess.run(awk_cmd, shell=True, check=True)
print(f"✅ Filtered sequence IDs saved in: {filtered_ids}")

# === Step 4: Extract sequences using seqkit (if installed) ===
print("\nExtracting GPCR-like sequences from the metagenome...")
seqkit_cmd = f"seqkit grep -f {filtered_ids} {gut_proteins} > {filtered_fasta}"
try:
    subprocess.run(seqkit_cmd, shell=True, check=True)
    print(f"✅ GPCR-like sequences saved to: {filtered_fasta}")
except subprocess.CalledProcessError:
    print("⚠️  seqkit not found or extraction failed — please install it with:")
    print("    conda install -c bioconda seqkit")

print("\n🎯 Pipeline completed successfully.")
