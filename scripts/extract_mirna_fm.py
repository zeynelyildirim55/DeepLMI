import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add RNA-FM to path
REPO_ROOT = r"c:\Users\zeynel\Desktop\ciceklab\DeepLMI"
RNA_FM_DIR = os.path.join(REPO_ROOT, "RNA-FM")
sys.path.append(RNA_FM_DIR)

import fm

def extract_embeddings(fasta_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Load RNA-FM model (will automatically download weights to torch hub if not present)
    print("Loading RNA-FM model...")
    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # Load sequences
    print(f"Loading sequences from {fasta_path}")
    data = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            seq_id = lines[i].strip()[1:] # remove '>'
            seq = lines[i+1].strip()
            data.append((seq_id, seq))
            
    print(f"Loaded {len(data)} sequences.")

    # Process each sequence
    print("Extracting embeddings...")
    for seq_id, seq in tqdm(data):
        out_file = os.path.join(out_dir, f"{seq_id}.npy")
        if os.path.exists(out_file):
            continue # Skip if already generated
            
        # The model accepts sequences up to 1024, but usually miRNAs are very short (~22 nt)
        # DeepLMI's process_rna.py limits to 640. So we can just take the first 640 if it's long.
        if len(seq) > 640:
            seq = seq[:640]
            
        batch_labels, batch_strs, batch_tokens = batch_converter([(seq_id, seq)])
        batch_tokens = batch_tokens.to(device)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])
            
        # The output representations are of shape [batch_size, seq_len + 2, emb_dim] (due to <cls> and <eos> tokens)
        # We need to extract the actual sequence embeddings. 
        # By convention, token 0 is <cls>, token 1 to seq_len is the sequence, token seq_len+1 is <eos>.
        # So we slice [1:seq_len+1]
        token_embeddings = results["representations"][12]
        seq_len = len(seq)
        
        # token_embeddings[0] is the batch index. [1:seq_len+1] is the sequence.
        # Shape: [seq_len, 640]
        emb = token_embeddings[0, 1:seq_len+1, :].cpu().numpy()
        
        np.save(out_file, emb)

    print(f"All embeddings saved to {out_dir}")

if __name__ == "__main__":
    fasta = os.path.join(REPO_ROOT, "custom_dataset", "mirna.fasta")
    out = os.path.join(REPO_ROOT, "custom_dataset", "mirna_emb", "representations")
    extract_embeddings(fasta, out)
