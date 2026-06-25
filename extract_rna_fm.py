"""
RNA-FM Embedding Extraction Script
===================================
Downloads weights from HuggingFace, loads them into a manually-constructed
BertModel, and extracts embeddings. No dependency on rna-fm or multimolecule packages.
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm


def download_model():
    """Download model files from HuggingFace using huggingface_hub."""
    from huggingface_hub import hf_hub_download
    
    repo_id = "multimolecule/rnafm"
    config_path = hf_hub_download(repo_id, "config.json")
    
    # Try safetensors first, fallback to pytorch bin
    try:
        weights_path = hf_hub_download(repo_id, "model.safetensors")
        return config_path, weights_path, "safetensors"
    except Exception:
        weights_path = hf_hub_download(repo_id, "pytorch_model.bin")
        return config_path, weights_path, "bin"


def load_weights(weights_path, fmt):
    """Load weights from either safetensors or bin format."""
    if fmt == "safetensors":
        from safetensors.torch import load_file
        return load_file(weights_path)
    else:
        return torch.load(weights_path, map_location="cpu")


def remap_keys(state_dict):
    """
    Remap HuggingFace multimolecule/rnafm keys to standard BertModel keys.
    HF keys look like: rnafm.encoder.layer.0.attention...
    BertModel expects:  bert.encoder.layer.0.attention...
    """
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k
        # Strip the "rnafm." prefix and replace with "bert."
        if new_key.startswith("rnafm."):
            new_key = "bert." + new_key[len("rnafm."):]
        new_sd[new_key] = v
    return new_sd


def main():
    fasta_path = "dataset_custom/mirna.fasta"
    out_dir = "dataset_custom/mirna_emb/representations"

    if not os.path.exists(fasta_path):
        print(f"File not found: {fasta_path}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # 1. Read FASTA
    data = []
    current_id = None
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:]
                data.append((current_id, ""))
            elif current_id:
                data[-1] = (current_id, data[-1][1] + line)

    if not data:
        print("No sequences found.")
        return

    print(f"Loaded {len(data)} sequences.")

    # 2. Download model from HuggingFace
    print("Downloading RNA-FM weights from HuggingFace...")
    config_path, weights_path, fmt = download_model()
    print("Download complete.")

    # 3. Load config
    with open(config_path) as f:
        raw_config = json.load(f)

    # 4. Build a BertModel with the right dimensions
    from transformers import BertModel, BertConfig

    bert_config = BertConfig(
        vocab_size=raw_config.get("vocab_size", 28),
        hidden_size=raw_config.get("hidden_size", 640),
        num_hidden_layers=raw_config.get("num_hidden_layers", 12),
        num_attention_heads=raw_config.get("num_attention_heads", 20),
        intermediate_size=raw_config.get("intermediate_size", 5120),
        max_position_embeddings=raw_config.get("max_position_embeddings", 1026),
        hidden_act=raw_config.get("hidden_act", "gelu"),
        hidden_dropout_prob=raw_config.get("hidden_dropout", 0.1),
        attention_probs_dropout_prob=raw_config.get("attention_dropout", 0.1),
        layer_norm_eps=raw_config.get("layer_norm_eps", 1e-5),
        pad_token_id=raw_config.get("pad_token_id", 0),
        type_vocab_size=2,
    )

    model = BertModel(bert_config)

    # 5. Load weights with key remapping
    print("Loading weights...")
    state_dict = load_weights(weights_path, fmt)
    
    # Debug: print first 10 keys to understand the naming
    print("Weight keys sample:")
    for i, k in enumerate(sorted(state_dict.keys())):
        if i < 10:
            print(f"  {k}: {state_dict[k].shape}")

    # Try remapping keys
    remapped = remap_keys(state_dict)
    
    # Load with strict=False to skip mismatched keys (e.g. LM head weights)
    result = model.load_state_dict(remapped, strict=False)
    print(f"Loaded weights. Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
    
    if result.missing_keys:
        print(f"  Sample missing: {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"  Sample unexpected: {result.unexpected_keys[:5]}")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model ready on {device}.")

    # 6. Vocab mapping from HuggingFace vocab.txt
    # 0: <pad>, 1: <cls>, 2: <eos>, 3: <unk>, 4: <mask>, 5: <null>
    # 6: A, 7: C, 8: G, 9: U
    mapping = {'A': 6, 'C': 7, 'G': 8, 'U': 9, 'T': 9}

    # 7. Extract embeddings
    print("Extracting embeddings...")
    for rna_id, seq in tqdm(data):
        tokens = [1] + [mapping.get(c.upper(), 3) for c in seq] + [2]
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            # [1, seq_len+2, 640]
            hidden = outputs.last_hidden_state[0]
            # Strip <cls> and <eos>
            emb = hidden[1 : len(seq) + 1].cpu().numpy()

            out_path = os.path.join(out_dir, f"{rna_id}.npy")
            np.save(out_path, emb)

    print(f"\nDone! Generated embeddings for {len(data)} sequences in {out_dir}/")


if __name__ == "__main__":
    main()
