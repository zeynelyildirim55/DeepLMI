import os
import subprocess
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = r"c:\Users\zeynel\Desktop\ciceklab\DeepLMI"
FASTA_PATH = os.path.join(REPO_ROOT, "custom_dataset", "mirna.fasta")

SPOT2D_DIR = os.path.join(REPO_ROOT, "SPOT-RNA-2D")
INPUT_FEATS_DIR = os.path.join(REPO_ROOT, "custom_dataset", "spot2d_inputs")
OUTPUT_DIR = os.path.join(REPO_ROOT, "custom_dataset", "mirna_contact")
LIST_FILE = os.path.join(INPUT_FEATS_DIR, "rna_ids.txt")

def main():
    os.makedirs(INPUT_FEATS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Read sequences
    data = []
    with open(FASTA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            seq_id = lines[i].strip()[1:]
            seq = lines[i+1].strip()
            data.append((seq_id, seq))
            
    print(f"Preparing {len(data)} sequences for SPOT-RNA-2D...")
    
    # 2. Write individual files for SPOT-RNA-2D and run RNAfold
    valid_ids = []
    
    def process_sequence(item):
        seq_id, seq = item
        out_file = os.path.join(OUTPUT_DIR, f"{seq_id}.prob_single")
        ps_file = os.path.join(INPUT_FEATS_DIR, f"{seq_id}_dp.ps")
        
        if os.path.exists(out_file) and os.path.exists(ps_file):
            return seq_id
            
        if not os.path.exists(ps_file):
            # Write sequence file
            seq_file = os.path.join(INPUT_FEATS_DIR, seq_id)
            with open(seq_file, "w", encoding="utf-8") as f:
                f.write(f">{seq_id}\n{seq}\n")
                
            cmd = f"wsl RNAfold -p -d2 --noPS < {seq_id} > {seq_id}.out"
            subprocess.run(cmd, shell=True, cwd=INPUT_FEATS_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        if os.path.exists(ps_file):
            return seq_id
        return None

    print("Running RNAfold via multiprocessing pool...")
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = {executor.submit(process_sequence, item): item for item in data}
        for future in tqdm(as_completed(futures), total=len(data), desc="Running RNAfold"):
            res = future.result()
            if res:
                valid_ids.append(res)
            
    if not valid_ids:
        print("No new sequences to process.")
        return
        
    print(f"Writing list of {len(valid_ids)} IDs to process...")
    with open(LIST_FILE, "w", encoding="utf-8") as f:
        for seq_id in valid_ids:
            f.write(f"{seq_id}\n")
            
    # 3. Run SPOT-RNA-2D
    print("Running SPOT-RNA-2D...")
    # Convert windows paths to what SPOT-RNA-2D needs (relative or absolute, better absolute in WSL/Windows? SPOT is python in windows!)
    spot_cmd = [
        "conda", "run", "-n", "venv_spotrna_2d",
        "python", "run.py",
        "--list_rna_ids", LIST_FILE,
        "--input_feats", INPUT_FEATS_DIR,
        "--single_seq", "1",
        "--outputs", OUTPUT_DIR,
        "--gpu", "0"  # Assuming GPU 0 is available, but the log said fallback to CPU is fine
    ]
    
    # We must run it from inside the SPOT-RNA-2D directory
    subprocess.run(spot_cmd, cwd=SPOT2D_DIR)
    
    print("SPOT-RNA-2D processing complete.")

if __name__ == "__main__":
    main()
