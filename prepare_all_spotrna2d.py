import os
import shutil

def main():
    custom_dir = "dataset_custom"
    custom_fasta_path = os.path.join(custom_dir, "mirna.fasta")
    
    all_seqs_dir = os.path.join(custom_dir, "all_mirna_seqs")
    all_ids_path = os.path.join(custom_dir, "all_ids.txt")
    
    # 1. Klasörleri ve eski listeleri temizle
    if os.path.exists(all_seqs_dir):
        shutil.rmtree(all_seqs_dir)
    os.makedirs(all_seqs_dir, exist_ok=True)
    
    if os.path.exists(all_ids_path):
        os.remove(all_ids_path)
    
    # 2. Fasta dosyasını oku
    print(f"Reading {custom_fasta_path}...")
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
                
    # 3. Her bir dizilimi ayrı bir dosya olarak kaydet ve listeye ekle
    print(f"Total {len(mirna_dict)} miRNAs found. Preparing for SPOT-RNA-2D...")
    created_count = 0
    
    with open(all_ids_path, "w", encoding="utf-8") as f_ids:
        for mirna_id, seq in mirna_dict.items():
            # Create individual sequence file WITHOUT extension
            seq_file_path = os.path.join(all_seqs_dir, f"{mirna_id}")
            with open(seq_file_path, "w", encoding="utf-8") as f_seq:
                f_seq.write(f">{mirna_id}\n{seq}\n")
                
            f_ids.write(f"{mirna_id}\n")
            created_count += 1
            
    print(f"Completed! Prepared {created_count} sequences for SPOT-RNA-2D inside {all_seqs_dir}")
    print(f"ID list saved to {all_ids_path}")

if __name__ == "__main__":
    main()
