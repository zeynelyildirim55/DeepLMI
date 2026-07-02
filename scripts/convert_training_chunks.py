#!/usr/bin/env python3
"""
Step 1.1: Extract positive miRNA–lncRNA interactions from training_chunks.

Input:  training_chunks/chunk_*.jsonl
Output: processed_data/training_positives.csv  (miRNA_id, lncRNA_id)

Filters:
  - interaction_type == "rna-rna"
  - RNA_type == "miRNA"
  - target_RNA_type == "lncRNA"
  - interaction_label == 1
  - Human only: lncRNA IDs starting with "NONHSAG" (excludes mouse NONMMUG, etc.)
"""

import json
import os
import glob
import csv
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_DIR = os.path.join(REPO_ROOT, "training_chunks")
OUTPUT_DIR = os.path.join(REPO_ROOT, "processed_data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "training_positives.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    chunk_files = sorted(glob.glob(os.path.join(CHUNKS_DIR, "chunk_*.jsonl")))
    print(f"Found {len(chunk_files)} chunk files")

    # Counters
    total_records = 0
    rna_rna_records = 0
    mirna_lncrna_records = 0
    mirna_lncrna_positive = 0
    mirna_lncrna_negative = 0
    non_human_skipped = 0
    human_positive = 0

    positives = set()
    species_counter = Counter()

    for i, chunk_file in enumerate(chunk_files):
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_records += 1
                record = json.loads(line)

                if record.get("interaction_type") != "rna-rna":
                    continue
                rna_rna_records += 1

                rna_type = record.get("RNA_type", "")
                target_rna_type = record.get("target_RNA_type", "")

                if rna_type != "miRNA" or target_rna_type != "lncRNA":
                    continue
                mirna_lncrna_records += 1

                label = record.get("interaction_label")
                if label == 1:
                    mirna_lncrna_positive += 1
                else:
                    mirna_lncrna_negative += 1
                    continue

                mirna_id = record["RNA_id"]
                lncrna_id = record["target_id"]

                # Species filtering: keep only human lncRNAs (NONHSAG prefix)
                if lncrna_id.startswith("NONHSAG"):
                    species_counter["human"] += 1
                elif lncrna_id.startswith("NONMMUG"):
                    species_counter["mouse"] += 1
                    non_human_skipped += 1
                    continue
                elif lncrna_id.startswith("NON"):
                    # Other NONCODE species
                    prefix = lncrna_id[:7]
                    species_counter[prefix] += 1
                    non_human_skipped += 1
                    continue
                else:
                    # Non-NONCODE IDs (e.g., ENSG*, could be human)
                    # Keep if starts with ENSG (Ensembl human) or other human identifiers
                    if lncrna_id.startswith("ENSG") or lncrna_id.startswith("ENST"):
                        species_counter["human_ensembl"] += 1
                    else:
                        species_counter[f"other:{lncrna_id[:10]}"] += 1
                        # Still include — let validation catch non-FASTA IDs later

                human_positive += 1
                positives.add((mirna_id, lncrna_id))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(chunk_files)} chunks, {len(positives)} unique positives so far")

    # Write output
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["miRNA_id", "lncRNA_id"])
        for mirna_id, lncrna_id in sorted(positives):
            writer.writerow([mirna_id, lncrna_id])

    # Collect unique IDs
    unique_mirnas = set(p[0] for p in positives)
    unique_lncrnas = set(p[1] for p in positives)

    # Report
    print("\n" + "=" * 60)
    print("TRAINING CHUNKS CONVERSION REPORT")
    print("=" * 60)
    print(f"Total records scanned:           {total_records:>10,}")
    print(f"  rna-rna records:               {rna_rna_records:>10,}")
    print(f"  miRNA->lncRNA records:          {mirna_lncrna_records:>10,}")
    print(f"    positives (label=1):         {mirna_lncrna_positive:>10,}")
    print(f"    negatives (label=0):         {mirna_lncrna_negative:>10,}")
    print(f"  Non-human skipped:             {non_human_skipped:>10,}")
    print(f"  Human positives kept:          {human_positive:>10,}")
    print(f"  Unique positive pairs:         {len(positives):>10,}")
    print(f"  Unique miRNAs:                 {len(unique_mirnas):>10,}")
    print(f"  Unique lncRNAs:                {len(unique_lncrnas):>10,}")
    print(f"\nSpecies distribution:")
    for species, count in species_counter.most_common():
        print(f"  {species}: {count}")
    print(f"\nOutput written to: {OUTPUT_FILE}")
    print("=" * 60)

    # Self-interaction check
    self_interactions = [(m, l) for m, l in positives if m == l]
    if self_interactions:
        print(f"\nWARN WARNING: {len(self_interactions)} self-interactions found!")
    else:
        print("\nOK No self-interactions found")

    # Duplicate check (by construction, set guarantees uniqueness)
    print(f"OK All {len(positives)} pairs are unique (set-based deduplication)")


if __name__ == "__main__":
    main()
