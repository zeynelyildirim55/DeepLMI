#!/usr/bin/env python3
"""
Step 1.3: Extract miRNA and lncRNA FASTA sequences from rna.fa.

Reads all unique miRNA and lncRNA IDs from:
  - processed_data/training_positives.csv
  - processed_data/splits/*.csv

Then looks up their sequences in fasta_files/rna.fa.

Output:
  - custom_dataset/mirna.fasta
  - custom_dataset/lncrna.fasta
  - processed_data/id_resolution_report.txt

NOTE: rna.fa is 1.8 GB. We first build a set of needed IDs, then do a single
pass through rna.fa to extract matching sequences.
"""

import os
import csv
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FASTA_PATH = os.path.join(REPO_ROOT, "fasta_files", "rna.fa")
POSITIVES_PATH = os.path.join(REPO_ROOT, "processed_data", "training_positives.csv")
SPLITS_DIR = os.path.join(REPO_ROOT, "processed_data", "splits")
OUTPUT_DIR = os.path.join(REPO_ROOT, "custom_dataset")

MIRNA_FASTA_OUT = os.path.join(OUTPUT_DIR, "mirna.fasta")
LNCRNA_FASTA_OUT = os.path.join(OUTPUT_DIR, "lncrna.fasta")
REPORT_PATH = os.path.join(REPO_ROOT, "processed_data", "id_resolution_report.txt")


def collect_all_ids():
    """Collect all unique miRNA and lncRNA IDs from all interaction files."""
    mirna_ids = set()
    lncrna_ids = set()

    # From training positives
    if os.path.exists(POSITIVES_PATH):
        with open(POSITIVES_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mirna_ids.add(row["miRNA_id"])
                lncrna_ids.add(row["lncRNA_id"])

    # From all split files
    for split_file in glob.glob(os.path.join(SPLITS_DIR, "*.csv")):
        with open(split_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mirna_ids.add(row["miRNA_id"])
                lncrna_ids.add(row["lncRNA_id"])

    return mirna_ids, lncrna_ids


def extract_sequences_from_fasta(fasta_path, needed_ids):
    """Single-pass extraction of sequences from a large FASTA file.
    
    Returns dict: {id: sequence}
    """
    found = {}
    current_id = None
    current_seq_parts = []

    print(f"  Scanning {fasta_path} ...")
    line_count = 0

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            line = line.strip()

            if line.startswith(">"):
                # Save previous record
                if current_id is not None and current_id in needed_ids:
                    found[current_id] = "".join(current_seq_parts)

                # Parse new header
                header = line[1:]  # Remove >
                # The ID is the full header before any whitespace
                # But some headers have :: separators, e.g. Human_RBP_POSTAR2update_1437::chr1:...
                # We want the full first token, or the part before ::
                current_id = header.split()[0] if " " in header else header
                # Also try without the :: suffix for matching
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

            if line_count % 5_000_000 == 0:
                print(f"    ... {line_count:,} lines, {len(found)} found so far")

    # Don't forget the last record
    if current_id is not None and current_id in needed_ids:
        found[current_id] = "".join(current_seq_parts)

    print(f"  Finished scanning: {line_count:,} lines, {len(found)} sequences found")
    return found


def write_fasta(sequences, output_path, rna_type):
    """Write sequences to a simple 2-line FASTA file."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for rna_id in sorted(sequences.keys()):
            seq = sequences[rna_id]
            f.write(f">{rna_id}\n")
            f.write(f"{seq}\n")
            count += 1
    print(f"  Wrote {count} {rna_type} sequences to {output_path}")
    return count


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("FASTA EXTRACTION REPORT")
    print("=" * 60)

    # Step 1: Collect all needed IDs
    mirna_ids, lncrna_ids = collect_all_ids()
    all_needed = mirna_ids | lncrna_ids
    print(f"\nNeeded IDs:")
    print(f"  miRNAs:  {len(mirna_ids):>8,}")
    print(f"  lncRNAs: {len(lncrna_ids):>8,}")
    print(f"  Total:   {len(all_needed):>8,}")

    # Step 2: Extract from rna.fa
    print(f"\nExtracting from {FASTA_PATH} ...")
    sequences = extract_sequences_from_fasta(FASTA_PATH, all_needed)

    # Step 3: Separate miRNA and lncRNA sequences
    mirna_seqs = {id_: sequences[id_] for id_ in mirna_ids if id_ in sequences}
    lncrna_seqs = {id_: sequences[id_] for id_ in lncrna_ids if id_ in sequences}

    # Step 4: Write output FASTAs
    print(f"\nWriting FASTA files:")
    mirna_count = write_fasta(mirna_seqs, MIRNA_FASTA_OUT, "miRNA")
    lncrna_count = write_fasta(lncrna_seqs, LNCRNA_FASTA_OUT, "lncRNA")

    # Step 5: Report missing IDs
    missing_mirnas = mirna_ids - set(sequences.keys())
    missing_lncrnas = lncrna_ids - set(sequences.keys())

    print(f"\nResolution summary:")
    print(f"  miRNAs found:    {len(mirna_seqs):>6} / {len(mirna_ids)}")
    print(f"  lncRNAs found:   {len(lncrna_seqs):>6} / {len(lncrna_ids)}")
    print(f"  miRNAs missing:  {len(missing_mirnas):>6}")
    print(f"  lncRNAs missing: {len(missing_lncrnas):>6}")

    # Write detailed report
    with open(REPORT_PATH, "w", encoding="utf-8") as report:
        report.write("ID RESOLUTION REPORT\n")
        report.write("=" * 60 + "\n\n")
        report.write(f"miRNAs found:    {len(mirna_seqs)} / {len(mirna_ids)}\n")
        report.write(f"lncRNAs found:   {len(lncrna_seqs)} / {len(lncrna_ids)}\n\n")

        if missing_mirnas:
            report.write(f"MISSING miRNAs ({len(missing_mirnas)}):\n")
            for m in sorted(missing_mirnas):
                report.write(f"  {m}\n")
            report.write("\n")

        if missing_lncrnas:
            report.write(f"MISSING lncRNAs ({len(missing_lncrnas)}):\n")
            for l in sorted(missing_lncrnas):
                report.write(f"  {l}\n")
            report.write("\n")

        # Sequence length stats
        report.write("miRNA sequence lengths:\n")
        if mirna_seqs:
            lengths = [len(s) for s in mirna_seqs.values()]
            report.write(f"  min={min(lengths)}, max={max(lengths)}, "
                        f"mean={sum(lengths)/len(lengths):.0f}\n\n")

        report.write("lncRNA sequence lengths:\n")
        if lncrna_seqs:
            lengths = [len(s) for s in lncrna_seqs.values()]
            report.write(f"  min={min(lengths)}, max={max(lengths)}, "
                        f"mean={sum(lengths)/len(lengths):.0f}\n\n")

    print(f"\nDetailed report: {REPORT_PATH}")

    if missing_mirnas or missing_lncrnas:
        print(f"\n  WARNING: {len(missing_mirnas) + len(missing_lncrnas)} IDs not found in rna.fa")
        if missing_mirnas:
            print(f"    Missing miRNAs: {len(missing_mirnas)}")
        if missing_lncrnas:
            print(f"    Missing lncRNAs: {len(missing_lncrnas)}")
        print(f"  See {REPORT_PATH} for details")
    else:
        print("\n  OK All IDs resolved successfully")

    print("=" * 60)


if __name__ == "__main__":
    main()
