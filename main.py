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
    parser = argparse.ArgumentParser(description="DeepLMI Custom Training")
    parser.add_argument('--dataset', type=str, default='custom_dataset', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g. cuda:0 or cpu)')
    parser.add_argument('--splits', type=str, default='processed_data/splits_indexed', help='Splits directory')
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

    # Load Train Data for 5-Fold CV
    print("Loading base train dataset for CV...")
    base_train_df = pd.read_csv(os.path.join(SPLITS, "train.csv"))
    
    # 5-Fold Cross Validation Setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    best_epoches = []
    f1_scores = []
    auc_scores = []
    ap_scores = []
    ndcg_scores = []
    pre_scores = []
    rec_scores = []

    # We patch miRNA_dataset inside process_rna.py dynamically to use args.dataset
    import dataset.process_rna
    dataset.process_rna.DATASET_DIR = args.dataset
    miRNA_feature_dataset = miRNA_dataset(is_merged=False)
    mirna_loader = DataLoader(miRNA_feature_dataset, batch_size=1, num_workers=0, drop_last=False, shuffle=False)
        lncRNA_dataset_loader = CustomDataset(x_embedding_3.to(device), x_embedding_5.to(device), x_embedding_7.to(device))
    lncRNA_loader = DataLoader(lncRNA_dataset_loader, batch_size=1, num_workers=0, drop_last=False, shuffle=False)

    print("*"*40)
    f1_scores, auc_scores, ap_scores, ndcg_scores, pre_scores, rec_scores = [], [], [], [], [], []
    best_epoches = []

    checkpoint_path = "save/checkpoint_main.pth"
    start_fold = 0
    start_epoch = 1

    if os.path.exists(checkpoint_path):
        print(f"[*] Checkpoint bulundu! Kaldığı yerden devam ediliyor...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_fold = checkpoint['fold']
        start_epoch = checkpoint['epoch'] + 1
        
        # Eğer bir fold'un 150. epoch'u bittiyse, bir sonraki fold'a geç.
        if start_epoch > EPOCH:
            start_fold += 1
            start_epoch = 1
            
        f1_scores = checkpoint.get('f1_scores', [])
        auc_scores = checkpoint.get('auc_scores', [])
        ap_scores = checkpoint.get('ap_scores', [])
        ndcg_scores = checkpoint.get('ndcg_scores', [])
        pre_scores = checkpoint.get('pre_scores', [])
        rec_scores = checkpoint.get('rec_scores', [])
        best_epoches = checkpoint.get('best_epoches', [])
        print(f"[*] Fold {start_fold + 1}, Epoch {start_epoch} noktasından başlatılıyor...")

    for fold, (train_index, val_index) in enumerate(kf.split(base_train_df)):
        if fold < start_fold:
            continue
            
        print(f"\n--- FOLD {fold + 1}/5 ---")
        train_split = base_train_df.iloc[train_index]
        val_split = base_train_df.iloc[val_index]

        # 1. Message Passing Graph MUST ONLY contain positive edges from the train_split!
        pos_train_split = train_split[train_split['label'] == 1]
        u_train = list(pos_train_split['node1'].astype(int))
        v_train = list(pos_train_split['node2'].astype(int))
        
        u_undirected_train = [x for pair in zip(u_train, v_train) for x in pair] + u_train[len(v_train):] + v_train[len(u_train):]
        v_undirected_train = [x for pair in zip(v_train, u_train) for x in pair] + u_train[len(v_train):] + v_train[len(u_train):]
        edge_index = torch.stack([torch.tensor(u_undirected_train), torch.tensor(v_undirected_train)], dim=0).to(device)

        # 2. Labels and Indices for Train and Val
        import numpy as np
        train_label_idx = torch.tensor(np.array([train_split['node1'].values, train_split['node2'].values]), dtype=torch.long).to(device)
        train_label = torch.tensor(train_split['label'].values, dtype=torch.float32).to(device)

        val_label_idx = torch.tensor(np.array([val_split['node1'].values, val_split['node2'].values]), dtype=torch.long).to(device)
        val_label = torch.tensor(val_split['label'].values, dtype=torch.float32).to(device)

        # 3. Load Test Data
        test_df = pd.read_csv(os.path.join(SPLITS, "test_unseen_pair.csv"))
        test_label_idx = torch.tensor(np.array([test_df['node1'].values, test_df['node2'].values]), dtype=torch.long).to(device)
        test_label = torch.tensor(test_df['label'].values, dtype=torch.float32).to(device)

        train_data = Data(x=graph_label, edge_index=edge_index, edge_label=train_label, edge_label_index=train_label_idx).to(device)
        val_data = Data(x=graph_label, edge_index=edge_index, edge_label=val_label, edge_label_index=val_label_idx).to(device)
        test_data = Data(x=graph_label, edge_index=edge_index, edge_label=test_label, edge_label_index=test_label_idx).to(device)
        
        train_data.num_nodes = len(graph_label)
        val_data.num_nodes = len(graph_label)
        test_data.num_nodes = len(graph_label)

        # Initialize Models for this fold
        model = GCNNet(100, 128, 64).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        miRNA_GCN = GCN_miRNA(100).to(device)
        model_k_f = SelfAttention(100).to(device)
        lncRNA_SA = SA_lncRNA(100).to(device)
        optimizer = torch.optim.Adam(
            params=list(model.parameters()) + list(miRNA_GCN.parameters()) + list(lncRNA_SA.parameters()) + list(model_k_f.parameters()), 
            lr=0.0001
        )

        best_val_auc = 0
        best_results = None

        # Eğer tam bu fold'da kaldığımız yerden devam ediyorsak, ağırlıkları yükle
        if fold == start_fold and os.path.exists(checkpoint_path) and start_epoch > 1:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            miRNA_GCN.load_state_dict(checkpoint['miRNA_GCN'])
            lncRNA_SA.load_state_dict(checkpoint['lncRNA_SA'])
            model_k_f.load_state_dict(checkpoint['model_k_f'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_val_auc = checkpoint['best_val_auc']
            best_results = checkpoint['best_results']

        current_start = start_epoch if fold == start_fold else 1
        print(f"Starting training for {EPOCH} epochs (from epoch {current_start})...")
        for epoch in range(current_start, EPOCH + 1):
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

                val_f1, val_auc, val_ap, val_ndcg, val_pre, val_rec, emb, label = test(
                    model, val_data, miRNA_f_list_t, miRNA_name_list, lncRNA_f_list_t, lncRNA_name_list, lnc_index, mi_index, rna_to_index, device
                )

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}, Val AP: {val_ap:.4f}')

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_results = [epoch, val_f1, val_auc, val_ap, val_ndcg, val_pre, val_rec]
                
                # Test on test_unseen_pair
                test_f1, test_auc, test_ap, test_ndcg, test_pre, test_rec, _, _ = test(
                    model, test_data, miRNA_f_list_t, miRNA_name_list, lncRNA_f_list_t, lncRNA_name_list, lnc_index, mi_index, rna_to_index, device
                )
                print(f'   -> New Best! Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}')

                # Save models for the best fold (fold 1)
                if fold == 0:
                    os.makedirs("save", exist_ok=True)
                    torch.save(model.state_dict(), "save/model_best.pth")
                    torch.save(miRNA_GCN.state_dict(), "save/miRNA_GCN_best.pth")
                    torch.save(lncRNA_SA.state_dict(), "save/lncRNA_SA_best.pth")
                    torch.save(model_k_f.state_dict(), "save/model_k_f_best.pth")

            # --- CHECKPOINT KAYIT İŞLEMİ (Her Epoch Sonu) ---
            os.makedirs("save", exist_ok=True)
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model': model.state_dict(),
                'miRNA_GCN': miRNA_GCN.state_dict(),
                'lncRNA_SA': lncRNA_SA.state_dict(),
                'model_k_f': model_k_f.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'best_results': best_results,
                'f1_scores': f1_scores,
                'auc_scores': auc_scores,
                'ap_scores': ap_scores,
                'ndcg_scores': ndcg_scores,
                'pre_scores': pre_scores,
                'rec_scores': rec_scores,
                'best_epoches': best_epoches
            }, checkpoint_path)

        print(f"Fold {fold+1} Best Results -> Epoch: {best_results[0]}, Val F1: {best_results[1]:.3f}, Val AUC: {best_results[2]:.3f}, Test AUC: {test_auc:.4f}")
        
        best_epoches.append(best_results[0])
        f1_scores.append(best_results[1])
        auc_scores.append(best_results[2])
        ap_scores.append(best_results[3])
        ndcg_scores.append(best_results[4])
        pre_scores.append(best_results[5])
        rec_scores.append(best_results[6])

    print("\n" + "*"*20 + " Final 5-Fold Cross Validation Results " + "*"*20)
    print('F1 scores: mean {:.3f}, std {:.3f}'.format(np.mean(f1_scores), np.std(f1_scores)))
    print('AUC scores: mean {:.3f}, std {:.3f}'.format(np.mean(auc_scores), np.std(auc_scores)))
    print('AP scores: mean {:.3f}, std {:.3f}'.format(np.mean(ap_scores), np.std(ap_scores)))
    print('NDCG scores: mean {:.3f}, std {:.3f}'.format(np.mean(ndcg_scores), np.std(ndcg_scores)))
    print('Pre scores: mean {:.3f}, std {:.3f}'.format(np.mean(pre_scores), np.std(pre_scores)))
    print('Rec scores: mean {:.3f}, std {:.3f}'.format(np.mean(rec_scores), np.std(rec_scores)))
    print("*"*60)

if __name__ == "__main__":
    main()
