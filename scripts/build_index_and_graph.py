#!/usr/bin/env python3
"""
Step 1.4: Build index_value.csv and node_link.csv for the custom dataset.

Reads:
  - processed_data/training_positives.csv  (training positives from training_chunks)
  - processed_data/splits/train_negatives.csv  (training negatives from data_with_negatives)
  - processed_data/splits/valid_*.csv
  - processed_data/splits/test_*.csv
  - custom_dataset/mirna.fasta
  - custom_dataset/lncrna.fasta

Creates:
  - custom_dataset/index_value.csv     (index → rna_name mapping)
  - custom_dataset/node_link.csv       (all training positive interactions as index pairs)
  - custom_dataset/mirna_id.fasta      (just miRNA names, one per line, no > prefix)
  - processed_data/splits/train.csv    (combined positives + negatives for training, with index IDs)

Also creates indexed versions of all split files in processed_data/splits_indexed/
"""

import os
import csv
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSITIVES_PATH = os.path.join(REPO_ROOT, "processed_data", "training_positives.csv")
SPLITS_DIR = os.path.join(REPO_ROOT, "processed_data", "splits")
DATASET_DIR = os.path.join(REPO_ROOT, "custom_dataset")
INDEXED_DIR = os.path.join(REPO_ROOT, "processed_data", "splits_indexed")


def load_fasta_names(fasta_path):
    """Load RNA names from a FASTA file."""
    names = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                names.append(line[1:])
    return names


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(INDEXED_DIR, exist_ok=True)

    print("=" * 60)
    print("INDEX AND GRAPH CONSTRUCTION REPORT")
    print("=" * 60)

    # Step 1: Load all unique RNA names from FASTA files
    mirna_fasta = os.path.join(DATASET_DIR, "mirna.fasta")
    lncrna_fasta = os.path.join(DATASET_DIR, "lncrna.fasta")

    if not os.path.exists(mirna_fasta) or not os.path.exists(lncrna_fasta):
        print("ERROR: Run build_fasta_files.py first!")
        return

    lncrna_names = load_fasta_names(lncrna_fasta)
    mirna_names = load_fasta_names(mirna_fasta)

    print(f"  lncRNAs in FASTA: {len(lncrna_names)}")
    print(f"  miRNAs in FASTA:  {len(mirna_names)}")

    # Step 2: Build index mapping (lncRNAs first, then miRNAs — same as original DeepLMI)
    all_names = lncrna_names + mirna_names
    name_to_index = {name: idx for idx, name in enumerate(all_names)}

    # Write index_value.csv
    index_value_path = os.path.join(DATASET_DIR, "index_value.csv")
    with open(index_value_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "rna"])
        for idx, name in enumerate(all_names):
            writer.writerow([idx, name])
    print(f"  index_value.csv: {len(all_names)} entries")

    # Write mirna_id.fasta (just names, one per line, no > prefix)
    mirna_id_path = os.path.join(DATASET_DIR, "mirna_id.fasta")
    with open(mirna_id_path, "w", encoding="utf-8") as f:
        for name in mirna_names:
            f.write(f"{name}\n")
    print(f"  mirna_id.fasta: {len(mirna_names)} entries")

    # Step 3: Load training positives and build node_link.csv
    train_pairs_pos = []
    missing_ids = set()

    with open(POSITIVES_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mirna_id = row["miRNA_id"]
            lncrna_id = row["lncRNA_id"]

            if mirna_id not in name_to_index:
                missing_ids.add(("miRNA", mirna_id))
                continue
            if lncrna_id not in name_to_index:
                missing_ids.add(("lncRNA", lncrna_id))
                continue

            node1 = name_to_index[lncrna_id]  # lncRNA index = node1 (convention from original)
            node2 = name_to_index[mirna_id]    # miRNA index = node2
            train_pairs_pos.append((node1, node2))

    # Write node_link.csv (training positives only — used for graph construction)
    node_link_path = os.path.join(DATASET_DIR, "node_link.csv")
    with open(node_link_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node1", "node2"])
        for n1, n2 in train_pairs_pos:
            writer.writerow([n1, n2])
    print(f"  node_link.csv: {len(train_pairs_pos)} training positive edges")

    if missing_ids:
        print(f"\n  ⚠ WARNING: {len(missing_ids)} IDs in interactions but not in FASTA:")
        for rna_type, rna_id in list(missing_ids)[:10]:
            print(f"    {rna_type}: {rna_id}")

    # Step 4: Create combined training CSV (positives + negatives with indices)
    train_neg_path = os.path.join(SPLITS_DIR, "train_negatives.csv")
    train_combined_path = os.path.join(INDEXED_DIR, "train.csv")

    train_records = []

    # Add positives from training_chunks
    for n1, n2 in train_pairs_pos:
        train_records.append((n1, n2, 1))

    # Add negatives from data_with_negatives
    neg_count = 0
    neg_missing = 0
    if os.path.exists(train_neg_path):
        with open(train_neg_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mirna_id = row["miRNA_id"]
                lncrna_id = row["lncRNA_id"]
                label = int(row["label"])

                if mirna_id not in name_to_index or lncrna_id not in name_to_index:
                    neg_missing += 1
                    continue

                node1 = name_to_index[lncrna_id]
                node2 = name_to_index[mirna_id]
                train_records.append((node1, node2, label))
                neg_count += 1

    with open(train_combined_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node1", "node2", "label"])
        for row in train_records:
            writer.writerow(row)
    print(f"  train.csv (indexed): {len(train_pairs_pos)} pos + {neg_count} neg = {len(train_records)} total")
    if neg_missing:
        print(f"    ⚠ {neg_missing} training negatives skipped (missing FASTA)")

    # Step 5: Create indexed versions of all val/test split files
    for split_file in glob.glob(os.path.join(SPLITS_DIR, "*.csv")):
        basename = os.path.basename(split_file)
        if basename == "train_negatives.csv":
            continue  # Already handled above

        output_path = os.path.join(INDEXED_DIR, basename)
        records = []
        skipped = 0

        with open(split_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mirna_id = row["miRNA_id"]
                lncrna_id = row["lncRNA_id"]
                label = int(row["label"])

                if mirna_id not in name_to_index or lncrna_id not in name_to_index:
                    skipped += 1
                    continue

                node1 = name_to_index[lncrna_id]
                node2 = name_to_index[mirna_id]
                records.append((node1, node2, label))

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["node1", "node2", "label"])
            for row in records:
                writer.writerow(row)

        pos = sum(1 for r in records if r[2] == 1)
        neg = sum(1 for r in records if r[2] == 0)
        skip_msg = f", {skipped} skipped" if skipped else ""
        print(f"  {basename} (indexed): {pos} pos + {neg} neg{skip_msg}")

    print(f"\n  OK All indexed splits written to: {INDEXED_DIR}")
    print(f"  Custom dataset files in: {DATASET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
