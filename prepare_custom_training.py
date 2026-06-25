"""
Prepare custom dataset for DeepLMI cold-start training.
1. Creates node_link.csv (positive-only, lncRNA=node1, miRNA=node2)
2. Generates 5-fold cold-start splits as CSVs
"""
import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch_geometric.utils import negative_sampling

import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch_geometric.utils import negative_sampling

DATASET = "dataset_custom"
SAVE_PATH = "cold_data_splits_custom"
SEED = 2026
N_SPLITS = 5
MODES = ["standard", "both", "lncRNA", "miRNA"]

os.makedirs(SAVE_PATH, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# 1. Create node_link.csv from node_link_train.csv
print("Creating node_link.csv (positive-only, swapped columns)...")
train_df = pd.read_csv(f"{DATASET}/node_link_train.csv")

# In convert_dataset.py: node1=miRNA, node2=lncRNA
# In original DeepLMI: node1=lncRNA, node2=miRNA
# So we swap!
positive_df = train_df[train_df['label'] == 1].copy()
node_link = pd.DataFrame({
    'node1': positive_df['node2'].values,  # lncRNA indices
    'node2': positive_df['node1'].values,  # miRNA indices
})
node_link.to_csv(f"{DATASET}/node_link.csv", index=False)
print(f"  Created node_link.csv with {len(node_link)} positive interactions")

# Verify the split
lncRNAs = sorted(node_link['node1'].unique().tolist())
miRNAs = sorted(node_link['node2'].unique().tolist())
print(f"  Unique lncRNAs (node1): {len(lncRNAs)}, Unique miRNAs (node2): {len(miRNAs)}")

num_nodes = int(max(node_link['node1'].max(), node_link['node2'].max()) + 1)

for mode in MODES:
    print(f"\nGenerating {N_SPLITS}-fold splits for mode: {mode}")
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    if mode == "standard":
        # Split by EDGES (Standard link prediction)
        edges = node_link.values
        for fold, (train_idx, val_idx) in enumerate(kfold.split(edges), start=1):
            train_set = pd.DataFrame(edges[train_idx], columns=['node1', 'node2'])
            val_pos = pd.DataFrame(edges[val_idx], columns=['node1', 'node2'])
            
            # Train CSV: positive edges only
            train_out = os.path.join(SAVE_PATH, f"{mode}_cold_train_{fold}.csv")
            train_set.to_csv(train_out, index=False)
            
            # Validation: need negative samples too
            u_train = torch.tensor(train_set['node1'].to_numpy(), dtype=torch.long)
            v_train = torch.tensor(train_set['node2'].to_numpy(), dtype=torch.long)
            edge_index_train = torch.stack([
                torch.cat([u_train, v_train]),
                torch.cat([v_train, u_train])
            ], dim=0)

            u_val = torch.tensor(val_pos['node1'].to_numpy(), dtype=torch.long)
            v_val = torch.tensor(val_pos['node2'].to_numpy(), dtype=torch.long)
            pos_label_index = torch.stack([u_val, v_val], dim=0)

            neg_edges = negative_sampling(
                edge_index=edge_index_train,
                num_nodes=num_nodes,
                num_neg_samples=pos_label_index.size(1),
                method='sparse'
            )

            val_node1 = torch.cat([u_val, neg_edges[0]]).numpy()
            val_node2 = torch.cat([v_val, neg_edges[1]]).numpy()
            val_labels = np.concatenate([np.ones(len(u_val)), np.zeros(neg_edges.size(1))])

            val_df = pd.DataFrame({'node1': val_node1, 'node2': val_node2, 'label': val_labels.astype(int)})
            val_out = os.path.join(SAVE_PATH, f"{mode}_cold_val_{fold}.csv")
            val_df.to_csv(val_out, index=False)
            
            print(f"  [Fold {fold}] train={len(train_set)}, val_pos={len(val_pos)}, val_neg={neg_edges.size(1)}")
    else:
        # Split by NODES (Cold Start)
        miRNA_folds = list(kfold.split(miRNAs))
        for fold, (train_lnc_idx, val_lnc_idx) in enumerate(kfold.split(lncRNAs), start=1):
            train_lncRNAs = set([lncRNAs[i] for i in train_lnc_idx])
            val_lncRNAs = set([lncRNAs[i] for i in val_lnc_idx])

            train_mi_idx, val_mi_idx = miRNA_folds[fold - 1]
            train_miRNAs = set([miRNAs[i] for i in train_mi_idx])
            val_miRNAs = set([miRNAs[i] for i in val_mi_idx])

            if mode == "both":
                train_mask = node_link["node1"].isin(train_lncRNAs) & node_link["node2"].isin(train_miRNAs)
                val_mask = node_link["node1"].isin(val_lncRNAs) & node_link["node2"].isin(val_miRNAs)
            elif mode == "lncRNA":
                train_mask = node_link["node1"].isin(train_lncRNAs)
                val_mask = node_link["node1"].isin(val_lncRNAs)
            elif mode == "miRNA":
                train_mask = node_link["node2"].isin(train_miRNAs)
                val_mask = node_link["node2"].isin(val_miRNAs)

            train_set = node_link[train_mask]
            val_pos = node_link[val_mask]

            # Train CSV: positive edges only
            train_out = os.path.join(SAVE_PATH, f"{mode}_cold_train_{fold}.csv")
            train_set[['node1', 'node2']].to_csv(train_out, index=False)

            # Validation: need negative samples too
            u_train = torch.tensor(train_set['node1'].to_numpy(), dtype=torch.long)
            v_train = torch.tensor(train_set['node2'].to_numpy(), dtype=torch.long)
            if u_train.size(0) > 0:
                edge_index_train = torch.stack([
                    torch.cat([u_train, v_train]),
                    torch.cat([v_train, u_train])
                ], dim=0)
            else:
                edge_index_train = torch.empty((2, 0), dtype=torch.long)

            u_val = torch.tensor(val_pos['node1'].to_numpy(), dtype=torch.long)
            v_val = torch.tensor(val_pos['node2'].to_numpy(), dtype=torch.long)
            pos_label_index = torch.stack([u_val, v_val], dim=0)

            if pos_label_index.size(1) > 0:
                neg_edges = negative_sampling(
                    edge_index=edge_index_train,
                    num_nodes=num_nodes,
                    num_neg_samples=pos_label_index.size(1),
                    method='sparse'
                )
                val_node1 = torch.cat([u_val, neg_edges[0]]).numpy()
                val_node2 = torch.cat([v_val, neg_edges[1]]).numpy()
                val_labels = np.concatenate([np.ones(len(u_val)), np.zeros(neg_edges.size(1))])
            else:
                val_node1 = np.array([])
                val_node2 = np.array([])
                val_labels = np.array([])

            val_df = pd.DataFrame({'node1': val_node1, 'node2': val_node2, 'label': val_labels.astype(int)})
            val_out = os.path.join(SAVE_PATH, f"{mode}_cold_val_{fold}.csv")
            val_df.to_csv(val_out, index=False)

            print(f"  [Fold {fold}] train={len(train_set)}, val_pos={len(val_pos)}")

print(f"\nDone! All splits saved to {SAVE_PATH}/")
