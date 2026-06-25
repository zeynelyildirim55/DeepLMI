import os
import shutil
import numpy as np

def main():
    custom_dir = "dataset_custom"
    standard_contact_dir = "dataset/mirna_contact"
    custom_contact_dir = os.path.join(custom_dir, "mirna_contact")
    custom_fasta_path = os.path.join(custom_dir, "mirna.fasta")
    standard_fasta_path = "dataset/mirna.fasta"
    
    os.makedirs(custom_contact_dir, exist_ok=True)
    
    missing_fasta_path = os.path.join(custom_dir, "missing_mirna.fasta")
    if os.path.exists(missing_fasta_path):
        os.remove(missing_fasta_path)
    
    # 1. Parse standard fasta to map Sequence -> Standard ID
    print("Reading standard dataset/mirna.fasta to map sequences...")
    seq_to_standard_id = {}
    current_id = None
    if os.path.exists(standard_fasta_path):
        with open(standard_fasta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    current_id = line[1:]
                elif current_id:
                    # Append sequence
                    if current_id not in seq_to_standard_id:
                        seq_to_standard_id[line] = current_id
                    else:
                        seq_to_standard_id[line] = current_id # Overwrite or keep, usually 1:1
                    
    print(f"Mapped {len(seq_to_standard_id)} sequences from standard fasta.")
    
    # 2. Parse custom fasta
    print("Reading custom dataset_custom/mirna.fasta...")
    mirna_dict = {}
    current_id = None
    with open(custom_fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:]
                mirna_dict[current_id] = ""
            elif current_id:
                mirna_dict[current_id] += line
                
    # 3. Get available files
    available_files = set(os.listdir(standard_contact_dir)) if os.path.exists(standard_contact_dir) else set()
    
    # 4. Match and copy or create
    found_count = 0
    created_count = 0
    
    print(f"Total {len(mirna_dict)} miRNAs found in custom fasta.")
    
    # Clean up previous missing files
    missing_seqs_dir = os.path.join(custom_dir, "missing_mirna_seqs")
    if os.path.exists(missing_seqs_dir):
        shutil.rmtree(missing_seqs_dir)
    os.makedirs(missing_seqs_dir, exist_ok=True)
    
    missing_ids_path = os.path.join(custom_dir, "missing_ids.txt")
    if os.path.exists(missing_ids_path):
        os.remove(missing_ids_path)
        
    for mirna_id, seq in mirna_dict.items():
        found_file = None
        
        # Strategy A: Sequence match
        if seq in seq_to_standard_id:
            standard_id = seq_to_standard_id[seq]
            if f"{standard_id}.prob_single" in available_files:
                found_file = f"{standard_id}.prob_single"
        
        # Strategy B: ID match (if sequence match failed)
        if not found_file:
            candidates = [
                f"{mirna_id}.prob_single",
                f"{mirna_id.replace('hsa-', '')}.prob_single"
            ]
            for cand in candidates:
                if cand in available_files:
                    found_file = cand
                    break
                    
        target_path = os.path.join(custom_contact_dir, f"{mirna_id}.prob_single")
        
        if found_file:
            source_path = os.path.join(standard_contact_dir, found_file)
            shutil.copy2(source_path, target_path)
            found_count += 1
        else:
            # We don't create zero matrix anymore, we prepare them for SPOT-RNA-2D!
            # Create individual sequence file WITHOUT extension
            seq_file_path = os.path.join(missing_seqs_dir, f"{mirna_id}")
            with open(seq_file_path, "w", encoding="utf-8") as f_seq:
                f_seq.write(f">{mirna_id}\n{seq}\n")
                
            # Add to ID list
            with open(missing_ids_path, "a", encoding="utf-8") as f_ids:
                f_ids.write(f"{mirna_id}\n")
                
            created_count += 1
            
    print(f"Completed! Copied existing: {found_count}, Missing to process: {created_count}")
    print(f"Prepared {created_count} sequences for SPOT-RNA-2D inside {missing_seqs_dir}")
    print(f"ID list saved to {missing_ids_path}")

if __name__ == "__main__":
    main()
