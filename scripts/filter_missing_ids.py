import os
import csv
import glob

REPO_ROOT = r"c:\Users\zeynel\Desktop\ciceklab\DeepLMI"
FASTA_DIR = os.path.join(REPO_ROOT, "custom_dataset")
MIRNA_FASTA = os.path.join(FASTA_DIR, "mirna.fasta")
LNCRNA_FASTA = os.path.join(FASTA_DIR, "lncrna.fasta")

POSITIVES_PATH = os.path.join(REPO_ROOT, "processed_data", "training_positives.csv")
SPLITS_DIR = os.path.join(REPO_ROOT, "processed_data", "splits")

def load_fasta_ids(fasta_path):
    ids = set()
    if not os.path.exists(fasta_path):
        return ids
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                ids.add(line.strip()[1:].split()[0])
    return ids

def filter_csv(file_path, valid_mirnas, valid_lncrnas):
    if not os.path.exists(file_path):
        return

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    original_count = len(rows)
    filtered_rows = []
    
    for row in rows:
        m_id = row["miRNA_id"]
        l_id = row["lncRNA_id"]
        # In validation, sometimes "hsa-miR-XXX" is used instead of "MIXXX" 
        # But we must check if we found the sequence for it
        if m_id in valid_mirnas and l_id in valid_lncrnas:
            filtered_rows.append(row)

    if len(filtered_rows) < original_count:
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_rows)
        print(f"Filtered {file_path}: {original_count} -> {len(filtered_rows)} records (dropped {original_count - len(filtered_rows)})")
    else:
        print(f"Kept all {original_count} records in {file_path}")

def main():
    valid_mirnas = load_fasta_ids(MIRNA_FASTA)
    valid_lncrnas = load_fasta_ids(LNCRNA_FASTA)
    
    print(f"Loaded {len(valid_mirnas)} valid miRNAs and {len(valid_lncrnas)} valid lncRNAs")

    filter_csv(POSITIVES_PATH, valid_mirnas, valid_lncrnas)

    for split_file in glob.glob(os.path.join(SPLITS_DIR, "*.csv")):
        filter_csv(split_file, valid_mirnas, valid_lncrnas)

if __name__ == "__main__":
    main()
