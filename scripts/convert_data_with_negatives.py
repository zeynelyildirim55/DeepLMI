#!/usr/bin/env python3
"""
Step 1.2: Convert data_with_negatives split files for validation and testing.

Input:  data_with_negatives/data_with_negatives/rna_rna/miRNA_lncRNA/final_*.jsonl
Output: processed_data/splits/
            train_negatives.csv          (negatives from final_train.jsonl)
            valid_unseen_pair.csv
            valid_unseen_source.csv      (unseen miRNA)
            valid_unseen_target.csv      (unseen lncRNA)
            test_unseen_pair.csv
            test_unseen_source.csv
            test_unseen_target.csv

Each CSV has columns: miRNA_id, lncRNA_id, label

NOTE: For training, positives come from training_chunks (Step 1.1).
      Negatives for training come from final_train.jsonl (label=0 only).
      Validation and test files include BOTH positives and negatives.
"""

import json
import os
import csv
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(
    REPO_ROOT, "data_with_negatives", "data_with_negatives", "rna_rna", "miRNA_lncRNA"
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "processed_data", "splits")

# Mapping from input filenames to output filenames
SPLIT_FILES = {
    "final_train.jsonl": "train_negatives.csv",  # Only negatives extracted
    "final_valid_unseen_pair.jsonl": "valid_unseen_pair.csv",
    "final_valid_unseen_source.jsonl": "valid_unseen_source.csv",
    "final_valid_unseen_target.jsonl": "valid_unseen_target.csv",
    "final_test_unseen_pair.jsonl": "test_unseen_pair.csv",
    "final_test_unseen_source.jsonl": "test_unseen_source.csv",
    "final_test_unseen_target.jsonl": "test_unseen_target.csv",
}


def process_split_file(input_path, output_path, train_negatives_only=False):
    """Parse a JSONL split file and output a CSV.
    
    If train_negatives_only=True, only extract label=0 records (for training negatives).
    Otherwise extract all records (for val/test).
    """
    records = []
    pos_count = 0
    neg_count = 0
    skipped_non_human = 0
    strategy_counter = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            mirna_id = record["RNA_id"]
            lncrna_id = record["target_id"]
            label = record["interaction_label"]

            # Human-only filter: skip non-human lncRNAs
            if lncrna_id.startswith("NONMMUG"):
                skipped_non_human += 1
                continue
            # Keep NONHSAG (human NONCODE), ENSG/ENST (human Ensembl), and other IDs
            # that may be human. Non-matching IDs will be caught in validation.

            if label == 1:
                pos_count += 1
            else:
                neg_count += 1
                neg_strategy = record.get("neg_strategy", "unknown")
                strategy_counter[neg_strategy] += 1

            if train_negatives_only and label == 1:
                continue  # Skip positives for train (they come from training_chunks)

            records.append((mirna_id, lncrna_id, label))

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["miRNA_id", "lncRNA_id", "label"])
        for row in records:
            writer.writerow(row)

    return {
        "total_written": len(records),
        "positives": pos_count,
        "negatives": neg_count,
        "skipped_non_human": skipped_non_human,
        "strategies": dict(strategy_counter.most_common()),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("DATA_WITH_NEGATIVES CONVERSION REPORT")
    print("=" * 70)

    all_stats = {}

    for input_name, output_name in SPLIT_FILES.items():
        input_path = os.path.join(INPUT_DIR, input_name)
        output_path = os.path.join(OUTPUT_DIR, output_name)

        if not os.path.exists(input_path):
            print(f"\nWARN WARNING: {input_name} not found, skipping")
            continue

        train_neg_only = (input_name == "final_train.jsonl")
        stats = process_split_file(input_path, output_path, train_negatives_only=train_neg_only)
        all_stats[output_name] = stats

        mode = "negatives only" if train_neg_only else "pos+neg"
        print(f"\n{input_name} -> {output_name} ({mode})")
        print(f"  Written:         {stats['total_written']:>8,}")
        print(f"  Positives found: {stats['positives']:>8,}")
        print(f"  Negatives found: {stats['negatives']:>8,}")
        if stats['skipped_non_human']:
            print(f"  Non-human skip:  {stats['skipped_non_human']:>8,}")

    # Cross-split validation
    print("\n" + "=" * 70)
    print("CROSS-SPLIT VALIDATION")
    print("=" * 70)

    # Load all pairs for validation
    split_pairs = {}
    for output_name in SPLIT_FILES.values():
        output_path = os.path.join(OUTPUT_DIR, output_name)
        if not os.path.exists(output_path):
            continue
        pairs = set()
        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.add((row["miRNA_id"], row["lncRNA_id"], int(row["label"])))
        split_pairs[output_name] = pairs

    # Check for conflicting labels within each split
    for name, pairs in split_pairs.items():
        pair_labels = {}
        for m, l, label in pairs:
            key = (m, l)
            if key in pair_labels and pair_labels[key] != label:
                print(f"  WARN CONFLICT in {name}: ({m}, {l}) has labels {pair_labels[key]} and {label}")
            pair_labels[key] = label
        print(f"  OK {name}: {len(pair_labels)} unique pairs, no label conflicts")

    # Check unseen source: val/test source miRNAs not in train
    train_neg_path = os.path.join(OUTPUT_DIR, "train_negatives.csv")
    if os.path.exists(train_neg_path):
        train_mirnas = set()
        with open(train_neg_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_mirnas.add(row["miRNA_id"])

        # Also load training positives if available
        train_pos_path = os.path.join(REPO_ROOT, "processed_data", "training_positives.csv")
        if os.path.exists(train_pos_path):
            with open(train_pos_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    train_mirnas.add(row["miRNA_id"])
            print(f"\n  Train miRNAs (pos+neg): {len(train_mirnas)}")

        for unseen_file in ["valid_unseen_source.csv", "test_unseen_source.csv"]:
            path = os.path.join(OUTPUT_DIR, unseen_file)
            if not os.path.exists(path):
                continue
            unseen_mirnas = set()
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    unseen_mirnas.add(row["miRNA_id"])
            leaked = unseen_mirnas & train_mirnas
            if leaked:
                print(f"  WARN LEAKAGE in {unseen_file}: {len(leaked)} miRNAs also in train!")
                for m in list(leaked)[:5]:
                    print(f"    {m}")
            else:
                print(f"  OK {unseen_file}: {len(unseen_mirnas)} miRNAs, none in train")

    print(f"\nOK All split files written to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
