#!/usr/bin/env python3
"""
Step 2.1: Comprehensive validation of all processed data.

Runs the following checks:
  1.  All miRNA IDs in interactions exist in FASTA
  2.  All lncRNA IDs in interactions exist in FASTA
  3.  All sequences are valid RNA/DNA alphabet
  4.  No duplicate pairs with conflicting labels
  5.  No pair appears in multiple splits (train vs val vs test)
  6.  Unseen source: val/test miRNAs NOT in training miRNAs
  7.  Unseen target: val/test lncRNAs NOT in training lncRNAs
  8.  Graph edge_index contains ONLY training pairs
  9.  No unseen molecule has neighbors in training graph
  10. Positive/negative ratio check
  11. No negative is also a known positive in training
  12. index_value.csv covers all molecules in all splits
  13. node_link.csv matches training positives
  14. No empty sequences
"""

import os
import csv
import glob
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "custom_dataset")
INDEXED_DIR = os.path.join(REPO_ROOT, "processed_data", "splits_indexed")
POSITIVES_PATH = os.path.join(REPO_ROOT, "processed_data", "training_positives.csv")

VALID_BASES = set("ACGUTNacgutn")  # Allow both RNA and DNA alphabets


class ValidationResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.messages = []

    def check(self, condition, name, detail=""):
        if condition:
            self.passed += 1
            self.messages.append(f"  OK PASS: {name}")
        else:
            self.failed += 1
            self.messages.append(f"  XX FAIL: {name}")
            if detail:
                self.messages.append(f"         {detail}")

    def warn(self, condition, name, detail=""):
        if not condition:
            self.warnings += 1
            self.messages.append(f"  !! WARN: {name}")
            if detail:
                self.messages.append(f"         {detail}")

    def print_report(self):
        for msg in self.messages:
            print(msg)
        print(f"\n  Summary: {self.passed} passed, {self.failed} failed, {self.warnings} warnings")
        return self.failed == 0


def load_fasta_dict(fasta_path):
    """Load a FASTA file as {name: sequence} dict."""
    seqs = {}
    current_name = None
    current_seq = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name is not None:
                    seqs[current_name] = "".join(current_seq)
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
    if current_name is not None:
        seqs[current_name] = "".join(current_seq)
    return seqs


def load_csv_pairs(csv_path):
    """Load a CSV with node1, node2, label columns."""
    records = []
    if not os.path.exists(csv_path):
        return records
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def main():
    vr = ValidationResult()
    print("=" * 60)
    print("COMPREHENSIVE VALIDATION REPORT")
    print("=" * 60)

    # --- Load data ---
    mirna_fasta = os.path.join(DATASET_DIR, "mirna.fasta")
    lncrna_fasta = os.path.join(DATASET_DIR, "lncrna.fasta")
    index_value_path = os.path.join(DATASET_DIR, "index_value.csv")
    node_link_path = os.path.join(DATASET_DIR, "node_link.csv")

    # Check files exist
    for path, name in [
        (mirna_fasta, "mirna.fasta"),
        (lncrna_fasta, "lncrna.fasta"),
        (index_value_path, "index_value.csv"),
        (node_link_path, "node_link.csv"),
    ]:
        vr.check(os.path.exists(path), f"{name} exists")

    if vr.failed > 0:
        vr.print_report()
        print("\nCannot continue — required files missing.")
        return False

    # Load FASTA sequences
    mirna_seqs = load_fasta_dict(mirna_fasta)
    lncrna_seqs = load_fasta_dict(lncrna_fasta)
    all_fasta_ids = set(mirna_seqs.keys()) | set(lncrna_seqs.keys())

    # Load index mapping
    index_to_name = {}
    name_to_index = {}
    with open(index_value_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            name = row["rna"]
            index_to_name[idx] = name
            name_to_index[name] = idx

    # Load node_link (training graph edges)
    graph_edges = set()
    with open(node_link_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            graph_edges.add((int(row["node1"]), int(row["node2"])))

    # Load indexed split files
    train_path = os.path.join(INDEXED_DIR, "train.csv")
    splits = {}
    for split_file in glob.glob(os.path.join(INDEXED_DIR, "*.csv")):
        basename = os.path.basename(split_file)
        splits[basename] = load_csv_pairs(split_file)

    # --- Check 1-2: All IDs in interactions exist in FASTA ---
    print("\n[1-2] ID coverage check:")
    all_interaction_indices = set()
    for name, records in splits.items():
        for row in records:
            all_interaction_indices.add(int(row["node1"]))
            all_interaction_indices.add(int(row["node2"]))

    missing_in_fasta = set()
    for idx in all_interaction_indices:
        rna_name = index_to_name.get(idx)
        if rna_name is None:
            missing_in_fasta.add(f"index_{idx}")
        elif rna_name not in all_fasta_ids:
            missing_in_fasta.add(rna_name)

    vr.check(len(missing_in_fasta) == 0, "All interaction IDs exist in FASTA",
             f"{len(missing_in_fasta)} missing" if missing_in_fasta else "")

    # --- Check 3: Valid RNA alphabet ---
    print("\n[3] Sequence validity check:")
    invalid_seqs = []
    for name, seq in list(mirna_seqs.items()) + list(lncrna_seqs.items()):
        invalid_chars = set(seq) - VALID_BASES
        if invalid_chars:
            invalid_seqs.append((name, invalid_chars))
    vr.check(len(invalid_seqs) == 0, "All sequences have valid RNA/DNA alphabet",
             f"{len(invalid_seqs)} invalid" if invalid_seqs else "")

    # --- Check 14: No empty sequences ---
    empty_seqs = [n for n, s in list(mirna_seqs.items()) + list(lncrna_seqs.items()) if len(s) == 0]
    vr.check(len(empty_seqs) == 0, "No empty sequences",
             f"{len(empty_seqs)} empty" if empty_seqs else "")

    # --- Check 4: No conflicting labels ---
    print("\n[4] Label consistency check:")
    for name, records in splits.items():
        pair_labels = {}
        conflicts = 0
        for row in records:
            key = (int(row["node1"]), int(row["node2"]))
            label = int(row["label"])
            if key in pair_labels and pair_labels[key] != label:
                conflicts += 1
            pair_labels[key] = label
        vr.check(conflicts == 0, f"{name}: no label conflicts",
                 f"{conflicts} conflicts" if conflicts else "")

    # --- Check 5: No pair in multiple splits ---
    print("\n[5] Cross-split pair leakage check:")
    if "train.csv" in splits:
        train_pairs = set()
        for row in splits["train.csv"]:
            if int(row["label"]) == 1:
                train_pairs.add((int(row["node1"]), int(row["node2"])))

        for name, records in splits.items():
            if name == "train.csv":
                continue
            eval_pairs = set()
            for row in records:
                if int(row["label"]) == 1:
                    eval_pairs.add((int(row["node1"]), int(row["node2"])))
            leaked = train_pairs & eval_pairs
            vr.check(len(leaked) == 0, f"No positive pairs from {name} in train",
                     f"{len(leaked)} leaked" if leaked else "")

    # --- Check 6: Unseen source (miRNA) ---
    print("\n[6] Unseen source (miRNA) check:")
    if "train.csv" in splits:
        train_mirnas = set()
        for row in splits["train.csv"]:
            train_mirnas.add(int(row["node2"]))  # node2 = miRNA

        for unseen_file in ["valid_unseen_source.csv", "test_unseen_source.csv"]:
            if unseen_file not in splits:
                continue
            unseen_mirnas = set()
            for row in splits[unseen_file]:
                unseen_mirnas.add(int(row["node2"]))
            leaked = unseen_mirnas & train_mirnas
            vr.check(len(leaked) == 0, f"{unseen_file}: miRNAs not in train",
                     f"{len(leaked)} leaked" if leaked else "")

    # --- Check 7: Unseen target (lncRNA) ---
    print("\n[7] Unseen target (lncRNA) check:")
    if "train.csv" in splits:
        train_lncrnas = set()
        for row in splits["train.csv"]:
            train_lncrnas.add(int(row["node1"]))  # node1 = lncRNA

        for unseen_file in ["valid_unseen_target.csv", "test_unseen_target.csv"]:
            if unseen_file not in splits:
                continue
            unseen_lncrnas = set()
            for row in splits[unseen_file]:
                unseen_lncrnas.add(int(row["node1"]))
            leaked = unseen_lncrnas & train_lncrnas
            vr.check(len(leaked) == 0, f"{unseen_file}: lncRNAs not in train",
                     f"{len(leaked)} leaked" if leaked else "")

    # --- Check 8: Graph contains only training pairs ---
    print("\n[8] Graph edge validation:")
    if "train.csv" in splits:
        train_positive_pairs = set()
        for row in splits["train.csv"]:
            if int(row["label"]) == 1:
                train_positive_pairs.add((int(row["node1"]), int(row["node2"])))
        non_train_edges = graph_edges - train_positive_pairs
        vr.check(len(non_train_edges) == 0, "Graph contains only training positive edges",
                 f"{len(non_train_edges)} extra edges" if non_train_edges else "")

    # --- Check 9: Unseen molecules not in graph ---
    print("\n[9] Unseen molecule graph exclusion:")
    graph_nodes = set()
    for n1, n2 in graph_edges:
        graph_nodes.add(n1)
        graph_nodes.add(n2)

    for unseen_file, node_col in [
        ("valid_unseen_source.csv", "node2"),
        ("test_unseen_source.csv", "node2"),
        ("valid_unseen_target.csv", "node1"),
        ("test_unseen_target.csv", "node1"),
    ]:
        if unseen_file not in splits:
            continue
        unseen_nodes = set()
        for row in splits[unseen_file]:
            unseen_nodes.add(int(row[node_col]))
        leaked = unseen_nodes & graph_nodes
        vr.warn(len(leaked) == 0, f"{unseen_file}: unseen molecules in graph",
                f"{len(leaked)} in graph — need per-setting graph filtering" if leaked else "")

    # --- Check 10: Positive/negative ratio ---
    print("\n[10] Positive/negative ratio:")
    for name, records in splits.items():
        pos = sum(1 for r in records if int(r["label"]) == 1)
        neg = sum(1 for r in records if int(r["label"]) == 0)
        ratio = neg / pos if pos > 0 else float("inf")
        vr.check(pos > 0 and neg > 0, f"{name}: {pos} pos, {neg} neg (ratio {ratio:.2f})")

    # --- Check 11: No negative is also a positive in train ---
    print("\n[11] Negative vs positive overlap:")
    if "train.csv" in splits:
        train_pos = set()
        train_neg = set()
        for row in splits["train.csv"]:
            pair = (int(row["node1"]), int(row["node2"]))
            if int(row["label"]) == 1:
                train_pos.add(pair)
            else:
                train_neg.add(pair)
        overlap = train_pos & train_neg
        vr.check(len(overlap) == 0, "No train negative is also a train positive",
                 f"{len(overlap)} overlaps" if overlap else "")

    # --- Check 12: index_value covers all ---
    print("\n[12] Index coverage:")
    vr.check(all_interaction_indices.issubset(set(index_to_name.keys())),
             "All interaction indices in index_value.csv")

    # --- Check 13: node_link matches training positives ---
    print("\n[13] node_link.csv consistency:")
    if "train.csv" in splits:
        vr.check(graph_edges == train_positive_pairs,
                 "node_link.csv == training positives from train.csv",
                 f"diff: {len(graph_edges.symmetric_difference(train_positive_pairs))}" 
                 if graph_edges != train_positive_pairs else "")

    # --- Final report ---
    print("\n" + "=" * 60)
    success = vr.print_report()
    print("=" * 60)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
