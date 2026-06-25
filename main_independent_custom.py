import os
import random
import string
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from sklearn.metrics import f1_score, roc_curve, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset_custom.process_rna import miRNA_dataset
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)
torch.set_num_threads(1)

set_seed(seed=2)
EPOCH = 200

# Datasets
TRAIN_DATASET = "dataset" # Original dataset for training
TEST_DATASET = "dataset_custom" # Custom dataset for independent testing

DOC2VEC_CACHE_TRAIN = f"{TRAIN_DATASET}/doc2vec_cache.pkl"

class CustomDataset(Dataset):
    def __init__(self, data1, data2, data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx], self.data3[idx]

def read_fasta(file_path):
    names = []
    sequences = []
    with open(file_path, 'r') as f:
        curr_name = None
        curr_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if curr_name is not None:
                    names.append(curr_name)
                    sequences.append("".join(curr_seq))
                curr_name = line[1:]
                curr_seq = []
            elif curr_name is not None:
                curr_seq.append(line)
        if curr_name is not None:
            names.append(curr_name)
            sequences.append("".join(curr_seq))
    return names, sequences

# --- Load Training Data (Original Dataset) ---
print(f"Loading Training Data from {TRAIN_DATASET}...")
train_graph_table = pd.read_csv(f"{TRAIN_DATASET}/index_value.csv")
train_graph_label = list(train_graph_table["rna"])

train_lncRNA_name_list, train_lnc_seq = read_fasta(f'{TRAIN_DATASET}/lncrna.fasta')
train_miRNA_name_list, _ = read_fasta(f'{TRAIN_DATASET}/mirna.fasta')

# Doc2Vec for Training
if os.path.exists(DOC2VEC_CACHE_TRAIN):
    print("Loading Train Doc2Vec embeddings from cache...")
    with open(DOC2VEC_CACHE_TRAIN, 'rb') as f:
        train_vectors_3, train_vectors_5, train_vectors_7, pretrain_model_3, pretrain_model_5, pretrain_model_7 = pickle.load(f)
else:
    print("Training Doc2Vec embeddings from scratch...")
    train_lnc_seq_mers_3 = [k_mers(3, i) for i in train_lnc_seq]
    pretrain_model_3 = train_doc2vec(train_lnc_seq_mers_3, train_lncRNA_name_list)
    train_vectors_3 = get_vector(train_lnc_seq_mers_3, train_lncRNA_name_list, pretrain_model_3)

    train_lnc_seq_mers_5 = [k_mers(9, i) for i in train_lnc_seq]
    pretrain_model_5 = train_doc2vec(train_lnc_seq_mers_5, train_lncRNA_name_list)
    train_vectors_5 = get_vector(train_lnc_seq_mers_5, train_lncRNA_name_list, pretrain_model_5)

    train_lnc_seq_mers_7 = [k_mers(15, i) for i in train_lnc_seq]
    pretrain_model_7 = train_doc2vec(train_lnc_seq_mers_7, train_lncRNA_name_list)
    train_vectors_7 = get_vector(train_lnc_seq_mers_7, train_lncRNA_name_list, pretrain_model_7)
    
    with open(DOC2VEC_CACHE_TRAIN, 'wb') as f:
        pickle.dump((train_vectors_3, train_vectors_5, train_vectors_7, pretrain_model_3, pretrain_model_5, pretrain_model_7), f)

# Map vectors to graph indices
train_graph_embedding_3 = np.zeros((len(train_lncRNA_name_list), 100))
train_graph_embedding_5 = np.zeros((len(train_graph_label), 100))
train_graph_embedding_7 = np.zeros((len(train_graph_label), 100))

for node, vec in train_vectors_3.items():
    position = train_lncRNA_name_list.index(node)
    train_graph_embedding_3[position] = vec
train_x_embedding_3 = torch.tensor(train_graph_embedding_3).float()

for node, vec in train_vectors_5.items():
    position = train_graph_label.index(node)
    train_graph_embedding_5[position] = vec
train_x_embedding_5 = torch.tensor(train_graph_embedding_5).float()

for node, vec in train_vectors_7.items():
    position = train_graph_label.index(node)
    train_graph_embedding_7[position] = vec
train_x_embedding_7 = torch.tensor(train_graph_embedding_7).float()

# Graph connectivity
train_node_table = pd.read_csv(f"{TRAIN_DATASET}/node_link.csv")
u_train = torch.tensor(train_node_table['node1'].to_numpy(), dtype=torch.long)
v_train = torch.tensor(train_node_table['node2'].to_numpy(), dtype=torch.long)

edge_index_train = torch.stack([
    torch.cat([u_train, v_train], dim=0),
    torch.cat([v_train, u_train], dim=0)
], dim=0)

edge_label_index_train = torch.stack([u_train, v_train], dim=0)
edge_label_train = torch.ones(u_train.size(0), dtype=torch.float)

num_nodes_train = int(max(train_node_table['node1'].max(), train_node_table['node2'].max()) + 1)
train_data = Data(x=None, edge_index=edge_index_train, edge_label=edge_label_train, edge_label_index=edge_label_index_train)
train_data.num_nodes = num_nodes_train
train_data = train_data.to(device)


# --- Load Testing Data (Custom Dataset) ---
print(f"\nLoading Testing Data from {TEST_DATASET}...")
test_graph_table = pd.read_csv(f"{TEST_DATASET}/index_value.csv")
test_graph_label = list(test_graph_table["rna"])

test_lncRNA_name_list, test_lnc_seq = read_fasta(f'{TEST_DATASET}/lncrna.fasta')
test_miRNA_name_list, _ = read_fasta(f'{TEST_DATASET}/mirna.fasta')

print("Inferring Doc2Vec embeddings for Custom Dataset...")
test_lnc_seq_mers_3 = [k_mers(3, i) for i in test_lnc_seq]
test_vectors_3 = get_vector(test_lnc_seq_mers_3, test_lncRNA_name_list, pretrain_model_3)

test_lnc_seq_mers_5 = [k_mers(9, i) for i in test_lnc_seq]
test_vectors_5 = get_vector(test_lnc_seq_mers_5, test_lncRNA_name_list, pretrain_model_5)

test_lnc_seq_mers_7 = [k_mers(15, i) for i in test_lnc_seq]
test_vectors_7 = get_vector(test_lnc_seq_mers_7, test_lncRNA_name_list, pretrain_model_7)

test_graph_embedding_3 = np.zeros((len(test_lncRNA_name_list), 100))
test_graph_embedding_5 = np.zeros((len(test_graph_label), 100))
test_graph_embedding_7 = np.zeros((len(test_graph_label), 100))

for node, vec in test_vectors_3.items():
    position = test_lncRNA_name_list.index(node[0]) if isinstance(node, list) else test_lncRNA_name_list.index(node)
    test_graph_embedding_3[position] = vec
test_x_embedding_3 = torch.tensor(test_graph_embedding_3).float()

for node, vec in test_vectors_5.items():
    position = test_graph_label.index(node[0]) if isinstance(node, list) else test_graph_label.index(node)
    test_graph_embedding_5[position] = vec
test_x_embedding_5 = torch.tensor(test_graph_embedding_5).float()

for node, vec in test_vectors_7.items():
    position = test_graph_label.index(node[0]) if isinstance(node, list) else test_graph_label.index(node)
    test_graph_embedding_7[position] = vec
test_x_embedding_7 = torch.tensor(test_graph_embedding_7).float()

# Custom Dataset Graph Connectivity
test_node_table = pd.read_csv(f"{TEST_DATASET}/node_link.csv")
u_test = torch.tensor(test_node_table['node1'].to_numpy(), dtype=torch.long)
v_test = torch.tensor(test_node_table['node2'].to_numpy(), dtype=torch.long)

edge_index_test = torch.stack([
    torch.cat([u_test, v_test], dim=0),
    torch.cat([v_test, u_test], dim=0)
], dim=0)
pos_label_index_test = torch.stack([u_test, v_test], dim=0)

num_nodes_test = int(max(test_node_table['node1'].max(), test_node_table['node2'].max()) + 1)

# Generate negative samples for test
print("Generating negative samples for test set...")
neg_edges_test = negative_sampling(
    edge_index=edge_index_test,
    num_nodes=num_nodes_test,
    num_neg_samples=pos_label_index_test.size(1),
    method='sparse'
)
test_edge_label_index = torch.cat([pos_label_index_test, neg_edges_test], dim=1)
test_edge_label = torch.cat([torch.ones(pos_label_index_test.size(1)), torch.zeros(neg_edges_test.size(1))], dim=0)

test_data = Data(x=None, edge_index=edge_index_test, edge_label=test_edge_label, edge_label_index=test_edge_label_index)
test_data.num_nodes = num_nodes_test
test_data = test_data.to(device)


# --- Models & Loaders ---
def run_once(seed):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Starting Run with seed={seed}")
    print(f"{'='*60}")

    model = GCNNet(100, 128, 64).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    model_k_f = SelfAttention(100).to(device)

    miRNA_GCN_train = GCN_miRNA(100).to(device)
    lncRNA_SA_train = SA_lncRNA(100).to(device)

    miRNA_feature_dataset_train = miRNA_dataset(is_merged=True, root_dir=TRAIN_DATASET)
    mirna_loader_train = DataLoader(miRNA_feature_dataset_train, batch_size=1, num_workers=0, drop_last=False, shuffle=False)

    lncRNA_dataset_train_ds = CustomDataset(train_x_embedding_3.to(device), train_x_embedding_5.to(device), train_x_embedding_7.to(device))
    lncRNA_loader_train = DataLoader(lncRNA_dataset_train_ds, batch_size=1, num_workers=0, drop_last=False, shuffle=False)

    miRNA_feature_dataset_test = miRNA_dataset(is_merged=True, root_dir=TEST_DATASET)
    mirna_loader_test = DataLoader(miRNA_feature_dataset_test, batch_size=1, num_workers=0, drop_last=False, shuffle=False)

    lncRNA_dataset_test_ds = CustomDataset(test_x_embedding_3.to(device), test_x_embedding_5.to(device), test_x_embedding_7.to(device))
    lncRNA_loader_test = DataLoader(lncRNA_dataset_test_ds, batch_size=1, num_workers=0, drop_last=False, shuffle=False)

    optimizer = torch.optim.Adam(params=list(model.parameters())+list(miRNA_GCN_train.parameters())+list(lncRNA_SA_train.parameters())+list(model_k_f.parameters()), lr=0.0001)

    def do_train(miRNA_f, miRNA_name_list_l, lncRNA_f, lncRNA_name_list_l, rna_to_index_dict):
        X = torch.zeros(len(miRNA_name_list_l) + len(lncRNA_name_list_l) + 2, 100).to(device)
        lnc_idx, mi_idx = [], []

        for rna, feature in zip(miRNA_name_list_l, miRNA_f):
            index = rna_to_index_dict.get(rna)
            if index is not None:
                X[index] = feature
                mi_idx.append(index)
        for rna, feature in zip(lncRNA_name_list_l, lncRNA_f):
            index = rna_to_index_dict.get(rna)
            if index is not None:
                X[index] = feature
                lnc_idx.append(index)

        X[-1] = X[lnc_idx].mean(dim=0, keepdim=True)
        X[-2] = X[mi_idx].mean(dim=0, keepdim=True)
        vir_lnc_id = len(X)-1
        vir_mi_id = len(X)-2

        lnc_idx_t = torch.tensor(lnc_idx).cuda()
        mi_idx_t = torch.tensor(mi_idx).cuda()

        edge_lnc_to_virtual = torch.stack([lnc_idx_t, torch.full_like(lnc_idx_t, vir_lnc_id)], dim=0)
        edge_virtual_to_lnc = torch.stack([torch.full_like(lnc_idx_t, vir_lnc_id), lnc_idx_t], dim=0)

        edge_mi_to_virtual = torch.stack([mi_idx_t, torch.full_like(mi_idx_t, vir_mi_id)], dim=0)
        edge_virtual_to_mi = torch.stack([torch.full_like(mi_idx_t, vir_mi_id), mi_idx_t], dim=0)

        edge_index = torch.cat([
            train_data.edge_index,
            edge_lnc_to_virtual, edge_virtual_to_lnc,
            edge_mi_to_virtual, edge_virtual_to_mi
        ], dim=1)

        z = model.encode(X, edge_index, lnc_idx_t, mi_idx_t)

        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([train_data.edge_label, train_data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)

        out, _ = model.decode(z, edge_label_index)
        loss = criterion(out, edge_label)

        return loss, lnc_idx_t, mi_idx_t

    @torch.no_grad()
    def do_test(data, miRNA_f, miRNA_name_list_l, lncRNA_f, lncRNA_name_list_l, rna_to_index_dict, t_lnc_index, t_mi_index):
        X_t = torch.zeros(len(miRNA_name_list_l) + len(lncRNA_name_list_l) + 2, 100).to(device)

        for rna, feature in zip(miRNA_name_list_l, miRNA_f):
            index = rna_to_index_dict.get(rna)
            if index is not None:
                X_t[index] = feature
        for rna, feature in zip(lncRNA_name_list_l, lncRNA_f):
            index = rna_to_index_dict.get(rna)
            if index is not None:
                X_t[index] = feature

        z = model.encode(X_t, data.edge_index, t_lnc_index, t_mi_index)

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
        precision = precision_score(y_true, y_pred_new)
        recall = recall_score(y_true, y_pred_new)
        acc = accuracy_score(y_true, y_pred_new)

        return f1, roc_auc, avg_precision, ndcg, precision, recall, acc

    # Pre-compute indices dict
    train_rna_to_index = pd.Series(train_graph_table['index'].values, index=train_graph_table['rna']).to_dict()
    test_rna_to_index = pd.Series(test_graph_table['index'].values, index=test_graph_table['rna']).to_dict()

    test_lnc_index = torch.tensor([test_rna_to_index.get(rna) for rna in test_lncRNA_name_list if test_rna_to_index.get(rna) is not None]).cuda()
    test_mi_index = torch.tensor([test_rna_to_index.get(rna) for rna in test_miRNA_name_list if test_rna_to_index.get(rna) is not None]).cuda()

    best_test_auc = 0
    best_test_result = []
    results = []

    for epoch in range(1, EPOCH + 1):
        miRNA_GCN_train.train()
        model.train()
        lncRNA_SA_train.train()
        model_k_f.train()
        optimizer.zero_grad()

        miRNA_f_list = []
        lncRNA_f_list = []

        for step, batch in enumerate(mirna_loader_train):
            miRAN_p = miRNA_GCN_train(batch.to(device))
            miRNA_f_list.append(miRAN_p.squeeze(0))

        for step, batch in enumerate(lncRNA_loader_train):
            lncRNA_p = lncRNA_SA_train(batch)
            lncRNA_f_list.append(lncRNA_p)
        lncRNA_f_list = torch.cat(lncRNA_f_list, dim=0)

        loss, lnc_index, mi_index = do_train(miRNA_f_list, train_miRNA_name_list, lncRNA_f_list, train_lncRNA_name_list, train_rna_to_index)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            miRNA_GCN_train.eval()
            model.eval()
            lncRNA_SA_train.eval()
            model_k_f.eval()

            miRNA_f_list_t = []
            for step, batch_t in enumerate(mirna_loader_test):
                miRAN_p_t = miRNA_GCN_train(batch_t.to(device))
                miRNA_f_list_t.append(miRAN_p_t.squeeze(0))

            lncRNA_f_list_t = []
            for step, batch_t in enumerate(lncRNA_loader_test):
                lncRNA_p_t = lncRNA_SA_train(batch_t)
                lncRNA_f_list_t.append(lncRNA_p_t)
            lncRNA_f_list_t = torch.cat(lncRNA_f_list_t, dim=0)

            f1, roc_auc, avg_precision, ndcg, precision, recall, acc = do_test(
                test_data, miRNA_f_list_t, test_miRNA_name_list, lncRNA_f_list_t, test_lncRNA_name_list, test_rna_to_index, test_lnc_index, test_mi_index
            )

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test AUC: {roc_auc:.4f}, Test Acc: {acc:.4f}')

        if roc_auc > best_test_auc:
            best_test_auc = roc_auc
            best_test_result = [f1, roc_auc, avg_precision, ndcg, precision, recall, acc]
            print(f" -> Best Test AUC so far: {best_test_auc:.4f}")

        results.append([epoch, f1, roc_auc, avg_precision, ndcg, precision, recall, acc])

    best_result = max(results, key=lambda x: x[2])
    print(f"\n[Seed {seed}] Best result:")
    print(f"  Epoch: {best_result[0]}, F1: {best_result[1]:.3f}, AUC: {best_result[2]:.3f}, AP: {best_result[3]:.3f}, NDCG: {best_result[4]:.3f}, Pre: {best_result[5]:.3f}, Rec: {best_result[6]:.3f}, Acc: {best_result[7]:.3f}")

    return best_test_result


# --- Run 5 seeds ---
all_test_metrics = []

for seed in [1, 2, 3, 4, 5]:
    test_result = run_once(seed)
    all_test_metrics.append(test_result)

test_results = np.array(all_test_metrics)
test_mean = test_results.mean(axis=0)
test_std = test_results.std(axis=0)

print(f"\n{'='*60}")
print("FINAL RESULTS (mean ± std over 5 seeds):")
print(f"{'='*60}")
metric_names = ['F1', 'AUC', 'AP', 'NDCG', 'Precision', 'Recall', 'Accuracy']
for i, name in enumerate(metric_names):
    print(f"Test {name}: {test_mean[i]:.3f} ± {test_std[i]:.3f}")

