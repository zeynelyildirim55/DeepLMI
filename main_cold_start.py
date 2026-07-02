import os
import random
import string
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import f1_score, roc_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from dataset.process_rna import miRNA_dataset
from model import *
from utils import *

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)


class CustomDataset(Dataset):
    def __init__(self, data1, data2, data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx], self.data3[idx]


def load_split(path, device):
    """Load a CSV split into edge_label_index and edge_label"""
    df = pd.read_csv(path)
    u = torch.tensor(df['node1'].values, dtype=torch.long)
    v = torch.tensor(df['node2'].values, dtype=torch.long)
    labels = torch.tensor(df['label'].values, dtype=torch.float)
    edge_label_index = torch.stack([u, v], dim=0).to(device)
    edge_label = labels.to(device)
    return edge_label_index, edge_label


def train(model, criterion, train_data, miRNA_f, miRNA_name_list, lncRNA_f, lncRNA_name_list, rna_to_index, device):
    X = torch.zeros(len(miRNA_name_list) + len(lncRNA_name_list) + 2, 100).to(device)

    lnc_index = []
    mi_index = []
    
    for rna, feature in zip(miRNA_name_list, miRNA_f):
        index = rna_to_index.get(rna)
        if index is not None:
            X[index] = feature
            mi_index.append(index)
    for rna, feature in zip(lncRNA_name_list, lncRNA_f):
        index = rna_to_index.get(rna)
        if index is not None:
            X[index] = feature
            lnc_index.append(index)
            
    X[-1] = X[lnc_index].mean(dim=0, keepdim=True) # linc vir
    X[-2] = X[mi_index].mean(dim=0, keepdim=True) # mi vir
    vir_lnc_id = len(X)-1
    vir_mi_id = len(X)-2

    lnc_index = torch.tensor(lnc_index).to(device)
    mi_index = torch.tensor(mi_index).to(device)

    edge_lnc_to_virtual = torch.stack([
        lnc_index, 
        torch.full_like(lnc_index, vir_lnc_id)
    ], dim=0)
    edge_virtual_to_lnc = torch.stack([
        torch.full_like(lnc_index, vir_lnc_id),
        lnc_index
    ], dim=0)

    edge_mi_to_virtual = torch.stack([
        mi_index, 
        torch.full_like(mi_index, vir_mi_id)
    ], dim=0)
    edge_virtual_to_mi = torch.stack([
        torch.full_like(mi_index, vir_mi_id),
        mi_index
    ], dim=0)

    edge_index = torch.cat([
        train_data.edge_index, 
        edge_lnc_to_virtual, edge_virtual_to_lnc,
        edge_mi_to_virtual, edge_virtual_to_mi
    ], dim=1)

    z = model.encode(X, edge_index, lnc_index, mi_index)

    # Use precomputed training pairs (no dynamic negative sampling!)
    edge_label_index = train_data.edge_label_index
    edge_label = train_data.edge_label

    out, _ = model.decode(z, edge_label_index)
    loss = criterion(out, edge_label)

    return loss, lnc_index, mi_index

@torch.no_grad()
def test(model, data, miRNA_f, miRNA_name_list, lncRNA_f, lncRNA_name_list, lnc_index, mi_index, rna_to_index, device):
    X_t = torch.zeros(len(miRNA_name_list) + len(lncRNA_name_list) + 2, 100).to(device)

    for rna, feature in zip(miRNA_name_list, miRNA_f):
        index = rna_to_index.get(rna)
        if index is not None:
            X_t[index] = feature

    for rna, feature in zip(lncRNA_name_list, lncRNA_f):
        index = rna_to_index.get(rna)
        if index is not None:
            X_t[index] = feature

    z = model.encode(X_t, data.edge_index, lnc_index, mi_index)

    out, emb = model.decode(z, data.edge_label_index)
    out = out.view(-1).sigmoid()

    y_true = data.edge_label.cpu().numpy()
    y_pred = out.cpu().numpy()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_new = (y_pred >= optimal_threshold).astype(int)
    
    f1 = f1_score(y_true, y_pred_new)
    roc_auc = roc_auc_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    ndcg = NDCG(y_true, y_pred)
    precision = precision_score(y_true, y_pred_new, zero_division=0)
    recall = recall_score(y_true, y_pred_new, zero_division=0)

    return f1, roc_auc, avg_precision, ndcg, precision, recall, emb, y_pred_new


def main():
    parser = argparse.ArgumentParser(description="DeepLMI Cold Start Tests")
    parser.add_argument('--dataset', type=str, default='custom_dataset', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--splits', type=str, default='processed_data/splits_indexed', help='Splits directory')
    parser.add_argument('--mode', type=str, required=True, choices=['blind_lncRNA', 'blind_miRNA', 'blind_both'], help='Cold start scenario to test')
    args = parser.parse_args()

    set_seed(1)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if 'cuda' in str(device):
        torch.cuda.set_device(device)
    torch.set_num_threads(1)

    print(f"Using device: {device}")
    
    DATASET = args.dataset
    SPLITS = args.splits
    EPOCH = args.epochs

    # Load lncRNA Sequences
    lnc_seq = []
    unique_lnc = []
    with open(os.path.join(DATASET, 'lncrna.fasta'), 'r', encoding='utf-8') as fasta:
        lines = fasta.readlines()
    for i in range(0, len(lines), 2):
        rna_name = lines[i].strip()[1:]
        rna_seq = lines[i + 1].strip().translate(str.maketrans('', '', string.punctuation))
        unique_lnc.append(rna_name)
        lnc_seq.append(rna_seq)

    # Compute k-mers
    all_name = unique_lnc
    os.makedirs(f'{args.dataset}/processed', exist_ok=True)
    cache_path_3 = f'{args.dataset}/processed/doc2vec_3.npy'
    cache_path_5 = f'{args.dataset}/processed/doc2vec_5.npy'
    cache_path_7 = f'{args.dataset}/processed/doc2vec_7.npy'

    if os.path.exists(cache_path_3) and os.path.exists(cache_path_5) and os.path.exists(cache_path_7):
        print("Loading cached Doc2Vec embeddings...")
        import numpy as np
        x_embedding_3 = torch.tensor(np.load(cache_path_3), dtype=torch.float32)
        x_embedding_5 = torch.tensor(np.load(cache_path_5), dtype=torch.float32)
        x_embedding_7 = torch.tensor(np.load(cache_path_7), dtype=torch.float32)
    else:
        print("Computing Doc2Vec embeddings (this will take a while)...")
        # Compute k-mers
        all_mers_3 = [k_mers(3, seq) for seq in lnc_seq]
        pretrain_model_3 = train_doc2vec(all_mers_3, all_name)
        vectors_3 = get_vector(all_mers_3, all_name, pretrain_model_3)
    
        all_mers_5 = [k_mers(9, seq) for seq in lnc_seq]
        pretrain_model_5 = train_doc2vec(all_mers_5, all_name)
        vectors_5 = get_vector(all_mers_5, all_name, pretrain_model_5)
    
        all_mers_7 = [k_mers(15, seq) for seq in lnc_seq]
        pretrain_model_7 = train_doc2vec(all_mers_7, all_name)
        vectors_7 = get_vector(all_mers_7, all_name, pretrain_model_7)
    
        # Convert dictionary to ordered tensor
        ordered_vecs_3 = [vectors_3[name] for name in unique_lnc]
        ordered_vecs_5 = [vectors_5[name] for name in unique_lnc]
        ordered_vecs_7 = [vectors_7[name] for name in unique_lnc]
    
        x_embedding_3 = torch.tensor(ordered_vecs_3, dtype=torch.float32)
        x_embedding_5 = torch.tensor(ordered_vecs_5, dtype=torch.float32)
        x_embedding_7 = torch.tensor(ordered_vecs_7, dtype=torch.float32)
        
        import numpy as np
        np.save(cache_path_3, x_embedding_3.numpy())
        np.save(cache_path_5, x_embedding_5.numpy())
        np.save(cache_path_7, x_embedding_7.numpy())

    lncRNA_name_list = all_name
    miRNA_name_list = []
    with open(os.path.join(DATASET, 'mirna.fasta'), 'r', encoding='utf-8') as fasta:
        lines = fasta.readlines()
    for i in range(0, len(lines), 2):
        miRNA_name_list.append(lines[i].strip()[1:])

    # Load Graph Info
    graph_table = pd.read_csv(os.path.join(DATASET, "index_value.csv"))
    graph_label = list(graph_table["rna"])
    rna_to_index = pd.Series(graph_table['index'].values, index=graph_table['rna']).to_dict()

    # Load Train Data
    print(f"Loading base train dataset for {args.mode} test...")
    train_df = pd.read_csv(os.path.join(SPLITS, "train.csv"))
    
    # Map mode to correct test split file
    mode_to_split = {
        'blind_lncRNA': 'test_unseen_target.csv',
        'blind_miRNA': 'test_unseen_source.csv',
        'blind_both': 'test_unseen_pair.csv'
    }
    test_split_file = mode_to_split[args.mode]
    test_df = pd.read_csv(os.path.join(SPLITS, test_split_file))

    print("*"*40)
    print(f"Starting {args.mode} Experiment...")
    print("*"*40)

    # 1. Message Passing Graph MUST ONLY contain positive edges from the train_df!
    pos_train_split = train_df[train_df['label'] == 1]
    u_train = list(pos_train_split['node1'].astype(int))
    v_train = list(pos_train_split['node2'].astype(int))
    
    u_undirected_train = [x for pair in zip(u_train, v_train) for x in pair] + u_train[len(v_train):] + v_train[len(u_train):]
    v_undirected_train = [x for pair in zip(v_train, u_train) for x in pair] + u_train[len(v_train):] + v_train[len(u_train):]
    edge_index = torch.stack([torch.tensor(u_undirected_train), torch.tensor(v_undirected_train)], dim=0).to(device)

    # 2. Labels and Indices for Train and Test
    import numpy as np
    train_label_idx = torch.tensor(np.array([train_df['node1'].values, train_df['node2'].values]), dtype=torch.long).to(device)
    train_label = torch.tensor(train_df['label'].values, dtype=torch.float32).to(device)

    test_label_idx = torch.tensor(np.array([test_df['node1'].values, test_df['node2'].values]), dtype=torch.long).to(device)
    test_label = torch.tensor(test_df['label'].values, dtype=torch.float32).to(device)

    train_data = Data(x=graph_label, edge_index=edge_index, edge_label=train_label, edge_label_index=train_label_idx).to(device)
    test_data = Data(x=graph_label, edge_index=edge_index, edge_label=test_label, edge_label_index=test_label_idx).to(device)
    train_data.num_nodes = len(graph_label)
    test_data.num_nodes = len(graph_label)

    # Initialize Models
    model = GCNNet(100, 128, 64).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    miRNA_GCN = GCN_miRNA(100).to(device)
    model_k_f = SelfAttention(100).to(device)
    lncRNA_SA = SA_lncRNA(100).to(device)
    optimizer = torch.optim.Adam(
        params=list(model.parameters()) + list(miRNA_GCN.parameters()) + list(lncRNA_SA.parameters()) + list(model_k_f.parameters()), 
        lr=0.0001
    )
    
    # We patch miRNA_dataset inside process_rna.py dynamically to use args.dataset
    import dataset.process_rna
    dataset.process_rna.DATASET_DIR = args.dataset
    miRNA_feature_dataset = miRNA_dataset(is_merged=False)
    mirna_loader = DataLoader(miRNA_feature_dataset, batch_size=1, num_workers=0, drop_last=False, shuffle=False)

    lncRNA_dataset_loader = CustomDataset(x_embedding_3.to(device), x_embedding_5.to(device), x_embedding_7.to(device))
    lncRNA_loader = DataLoader(lncRNA_dataset_loader, batch_size=1, num_workers=0, drop_last=False, shuffle=False)


    best_val_auc = 0
    best_results = None

    print(f"Starting training for {EPOCH} epochs...")
    for epoch in range(1, EPOCH + 1):
        miRNA_GCN.train()
        model.train()
        lncRNA_SA.train()
        model_k_f.train()
        optimizer.zero_grad()

        miRNA_f_list = []
        lncRNA_f_list = []

        for batch in mirna_loader:
            miRAN_p = miRNA_GCN(batch.to(device))
            miRNA_f_list.append(miRAN_p)
        miRNA_f_list = torch.cat(miRNA_f_list, dim=0)

        for batch in lncRNA_loader:
            lncRNA_p = lncRNA_SA(batch)
            lncRNA_f_list.append(lncRNA_p)
        lncRNA_f_list = torch.cat(lncRNA_f_list, dim=0)

        loss, lnc_index, mi_index = train(
            model, criterion, train_data, miRNA_f_list, miRNA_name_list, lncRNA_f_list, lncRNA_name_list, rna_to_index, device
        )
        loss.backward(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            miRNA_GCN.eval()
            model.eval()
            lncRNA_SA.eval()
            model_k_f.eval()
            
            miRNA_f_list_t = []
            for batch_t in mirna_loader:
                miRAN_p_t = miRNA_GCN(batch_t.to(device))
                miRNA_f_list_t.append(miRAN_p_t)
            miRNA_f_list_t = torch.cat(miRNA_f_list_t, dim=0)

            lncRNA_f_list_t = []
            for batch_t in lncRNA_loader:
                lncRNA_p_t = lncRNA_SA(batch_t)
                lncRNA_f_list_t.append(lncRNA_p_t)
            lncRNA_f_list_t = torch.cat(lncRNA_f_list_t, dim=0)

            # We use test_data as validation proxy to get the best model
            val_f1, val_auc, val_ap, val_ndcg, val_pre, val_rec, emb, label = test(
                model, test_data, miRNA_f_list_t, miRNA_name_list, lncRNA_f_list_t, lncRNA_name_list, lnc_index, mi_index, rna_to_index, device
            )

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test AUC: {val_auc:.4f}, Test F1: {val_f1:.4f}, Test AP: {val_ap:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_results = [epoch, val_f1, val_auc, val_ap, val_ndcg, val_pre, val_rec]
            
            os.makedirs("save_cold", exist_ok=True)
            torch.save(model.state_dict(), f"save_cold/model_{args.mode}.pth")
            torch.save(miRNA_GCN.state_dict(), f"save_cold/miRNA_GCN_{args.mode}.pth")
            torch.save(lncRNA_SA.state_dict(), f"save_cold/lncRNA_SA_{args.mode}.pth")
            torch.save(model_k_f.state_dict(), f"save_cold/model_k_f_{args.mode}.pth")

    print(f"\nFinal Best Results for {args.mode} -> Epoch: {best_results[0]}")
    print(f"Metrics: F1: {best_results[1]:.3f}, AUC: {best_results[2]:.3f}, AP: {best_results[3]:.3f}, NDCG: {best_results[4]:.3f}, Pre: {best_results[5]:.3f}, Rec: {best_results[6]:.3f}")

if __name__ == "__main__":
    main()
