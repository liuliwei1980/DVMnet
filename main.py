import ast
import pickle
import random
import string
from de_graph import de_bruijn_graph
from model import *
from utils import *
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.model_selection import KFold
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix,roc_curve,average_precision_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_num_threads(1)
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
def l2_norm(x):
    return torch.norm(x, p=2)

csv_file = './dataset/di.csv'
ser_di = pd.read_csv(csv_file)
csv_file1 = './dataset/sub.csv'
mi_sub = pd.read_csv(csv_file1)
match_table = pd.read_csv("dataset/mirna_lncrna_interaction.csv")
unique_lnc = list(set(match_table['lncrna']))
unique_mi = list(set(match_table['mirna']))
lnc_seq = []
mi_seq = []
def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return s
for i in unique_lnc:
    seq = match_table[match_table['lncrna'] == i]["lncrna_seq"]
    seq = list(seq)
    seq = seq[0]
    seq = seq.translate(str.maketrans('', '', string.punctuation))
    lnc_seq.append(seq)
for i in unique_mi:
    seq = match_table[match_table['mirna'] == i]["mirna_seq"]
    seq = list(seq)
    seq = seq[0]
    seq = seq.replace('.', '')
    if ',' in seq:
        seq = seq.split(',')
        seq = seq[0]
    mi_seq.append(seq)


lnc_seq_mers = []
mi_seq_mers = []
for i in lnc_seq:
    lnc_seq_mers.append(k_mers(3, i))
for i in mi_seq:
    mi_seq_mers.append(k_mers(3, i))

all_seq = lnc_seq + mi_seq
all_mers = lnc_seq_mers + mi_seq_mers
all_name = unique_lnc + unique_mi
pretrain_model = train_doc2vec_model(all_mers, all_name)
vectors = get_vector_embeddings(all_mers, all_name, pretrain_model)
# If you need to build an improved De Bruijn matrix for your own data, run the following two lines of code.
# de_vectors_lnc = de_bruijn_graph(lnc_seq,unique_lnc,3)
# de_vectors_lnc = de_bruijn_graph(mi_seq,unique_mi,2)
with open('vectors_lnc.pkl', 'rb') as f:
    vectors_lnc = pickle.load(f)
with open('vectors_mi.pkl', 'rb') as f:
    vectors_mi = pickle.load(f)
graph_table = pd.read_csv("dataset/index_value.csv")
graph_label = list(graph_table["rna"])
#vec_feature
graph_embedding = np.zeros((len(graph_label), 100))
for node, vec in vectors.items():
    position = graph_label.index(node)
    graph_embedding[position] = vec
x_embedding = torch.tensor(graph_embedding).float()
#lnc_feature
graph_lnc = np.zeros((284,64,64))
for node, vec in vectors_lnc.items():
    position = graph_label.index(node)
    graph_lnc[position] = vec
x_lnc = torch.tensor(graph_lnc).float()
#mi_feature
graph_mi = np.zeros((len(graph_label),16,16))
for node, vec in vectors_mi.items():
    position = graph_label.index(node)
    graph_mi[position] = vec
graph_mi = graph_mi[284:]
x_mi = torch.tensor(graph_mi).float()
node_table = pd.read_csv("dataset/node_link.csv")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


def train():
    model.train()
    optimizer.zero_grad()
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index.to(device)],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)
    # %%
    out,di_x,di_out,sub_x,sub_out = model.mainNet(train_data.x,train_data.x_lnc,train_data.x_mi, train_data.edge_index, train_data.xname,edge_label_index,train_data.di,train_data.sub)
    out = out.view(-1)
    loss1 = criterion(out, edge_label)
    di_out = di_out.to(device)
    sub_out = sub_out.to(device)
    loss2 = criterion2(di_x,di_out)
    loss3 = criterion2(sub_x, sub_out)
    loss_history1.append(loss1.item())
    loss_history2.append(loss2.item())
    loss_history3.append(loss3.item())
    if len(loss_history1) > T:
        loss_history1.pop(0)
    if len(loss_history2) > T:
        loss_history2.pop(0)
    if len(loss_history3) > T:
        loss_history3.pop(0)
    avg_loss1 = sum(loss_history1) / T
    avg_loss2 = sum(loss_history2) / T
    avg_loss3 = sum(loss_history2) / T
    weight1 = 1.0 / (avg_loss1 + 1e-8)
    weight2 = 1.0 / (avg_loss2 + 1e-8)
    weight3 = 1.0 / (avg_loss3 + 1e-8)
    total_weight = weight1 + weight2 + weight3
    weight1 = weight1 / total_weight
    weight2 = weight2 / total_weight
    weight3 = weight3 / total_weight
    loss = weight1 * loss1 + weight2 * loss2 *0.5 + weight3 * loss3*0.5
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    out,di_x,di_out,sub_x,sub_out = model.mainNet(data.x,train_data.x_lnc,train_data.x_mi, data.edge_index,data.xname,data.edge_label_index,data.di,data.sub)
    y_true = data.edge_label.cpu().numpy()
    y_pred = out.cpu().numpy()
    di_x = di_x.cpu().numpy().flatten()
    di_out = di_out.cpu().numpy().flatten()
    sub_x = sub_x.cpu().numpy().flatten()
    sub_out = sub_out.cpu().numpy().flatten()
    #Calculation of subtask indicators
    fpr_sub, tpr_sub, thresholds_sub = roc_curve(sub_out, sub_x)
    optimal_idx_sub = np.argmax(tpr_sub - fpr_sub)
    optimal_threshold_sub = thresholds_sub[optimal_idx_sub]
    di_pred_new_sub = (sub_x >= optimal_threshold_sub).astype(int)
    accuracy_sub = accuracy_score(sub_out, di_pred_new_sub)
    fpr_di, tpr_di, thresholds_di = roc_curve(di_out, di_x)
    optimal_idx_di = np.argmax(tpr_di - fpr_di)
    optimal_threshold_di = thresholds_di[optimal_idx_di]
    di_pred_new = (di_x >= optimal_threshold_di).astype(int)
    accuracy = accuracy_score(di_out, di_pred_new)
    #Main task index calculation
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_new = (y_pred >= optimal_threshold).astype(int)
    mat = confusion_matrix(y_true, y_pred_new)
    tp = float(mat[0][0])
    fp = float(mat[1][0])
    fn = float(mat[0][1])
    tn = float(mat[1][1])
    f1 = f1_score(y_true, y_pred_new)
    roc_auc = roc_auc_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    ndcg = NDCG(y_true, y_pred)
    return f1, roc_auc, avg_precision, ndcg,accuracy,accuracy_sub,tp,fp,fn,tn


best_epoches = []
f1_scores = []
auc_scores = []
ap_scores = []
ndcg_scores = []
di_scores=[]


for train_index, val_index in kfold.split(node_table):
    original_array = np.arange(187)
    np.random.shuffle(original_array)
    array1 = original_array[:157]
    array2 = original_array[157:]
    ser_di_t = ser_di.iloc[array1]
    ser_di_v = ser_di.iloc[array2]
    original_array1 = np.arange(454)
    np.random.shuffle(original_array1)
    subarray1 = original_array1[:379]
    subarray2 = original_array1[379:]
    mi_sub_t = mi_sub.iloc[subarray1]
    mi_sub_v = mi_sub.iloc[subarray2]
    train_set = node_table.iloc[train_index]
    val_set = node_table.iloc[val_index]
    u_float_train = list(train_set['node1'])
    u_train = [int(x) for x in u_float_train]
    v_float_train = list(train_set['node2'])
    v_train = [int(y) for y in v_float_train]
    u_undirected_train = [x for pair in zip(
        u_train, v_train) for x in pair] + u_train[len(v_train):] + v_train[len(u_train):]
    v_undirected_train = [x for pair in zip(
        v_train, u_train) for x in pair] + u_train[len(v_train):] + v_train[len(u_train):]
    u_undirected_train = torch.tensor(u_undirected_train)
    v_undirected_train = torch.tensor(v_undirected_train)
    u_float_val = list(val_set['node1'])
    u_val = [int(x) for x in u_float_val]
    v_float_val = list(val_set['node2'])
    v_val = [int(y) for y in v_float_val]
    u_undirected_val = [x for pair in zip(
        u_val, v_val) for x in pair] + u_val[len(v_val):] + v_val[len(u_val):]
    v_undirected_val = [x for pair in zip(
        v_val, u_val) for x in pair] + u_val[len(v_val):] + v_val[len(u_val):]
    u_undirected_val = torch.tensor(u_undirected_val)
    v_undirected_val = torch.tensor(v_undirected_val)
    edge_index = torch.stack(
        [u_undirected_train, v_undirected_train], dim=0)
    edge_train_index = torch.stack(
        [torch.tensor(u_train), torch.tensor(v_train)], dim=0)
    edge_val_index = torch.stack(
        [torch.tensor(u_val), torch.tensor(v_val)], dim=0)
    ser_di_t = ser_di_t.values
    mi_sub_t = mi_sub_t.values
    train_data = Data(x=x_embedding,x_lnc=x_lnc,x_mi=x_mi, edge_index=edge_index, edge_label=torch.ones(
        len(u_train)), edge_label_index=edge_train_index,xname = graph_label,di=ser_di_t,sub=mi_sub_t).to(device)

    val_data = Data(x=x_embedding,x_lnc=x_lnc,x_mi=x_mi, edge_index=edge_index, edge_label=torch.ones(
        len(u_val)), edge_label_index=edge_val_index,xname = graph_label).to(device)

    neg_edge_index = negative_sampling(
        edge_index=val_data.edge_index, num_nodes=val_data.num_nodes,
        num_neg_samples=val_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [val_data.edge_label_index, neg_edge_index.to(device)],
        dim=-1,
    )
    edge_label = torch.cat([
        val_data.edge_label,
        val_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)
    ser_di_v = ser_di_v.values
    mi_sub_v = mi_sub_v.values
    val_data = Data(x=x_embedding, edge_index=edge_index,
                    edge_label=edge_label, edge_label_index=edge_label_index,xname = graph_label,di=ser_di_v,sub=mi_sub_v ).to(device)

    transform = NormalizeFeatures()
    train_data = transform(train_data)
    val_data = transform(val_data)
    model = DVMNet(train_data.num_features, 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.002)
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion2 = torch.nn.CrossEntropyLoss()
    # Train and Validation
    results = []
    diACC=[]
    subACC = []
    best_val_auc = 0
    best_val_auc_di = 0
    loss_history1 = []
    loss_history2 = []
    loss_history3 = []
    alpha = 1.0
    T = 10

    for epoch in range(1, 20):
        loss = train()
        val_f1, val_auc, val_ap, val_ndcg,val_acc_di,val_acc_sub,tp,fp,fn,tn = test(val_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, di: {val_acc_di:.4f}, sub: {val_acc_sub:.4f}')
        results.append([epoch, val_f1, val_auc, val_ap, val_ndcg])
        diACC.append(val_acc_di)
        subACC.append(val_acc_sub)
    best_result = max(results, key=lambda x: x[2])
    best_acc_di =  max(diACC)
    best_acc_sub = max(subACC)
    log_and_print('Best result: Epoch: {}, F1: {:.3f}, AUC: {:.3f}, AP: {:.3f}, NDCG: {:.3f}'.format(
        *best_result))
    print("best_acc_di:",best_acc_di)
    print("best_acc_sub:", best_acc_sub)
    best_epoches.append(best_result[0])
    f1_scores.append(best_result[1])
    auc_scores.append(best_result[2])
    ap_scores.append(best_result[3])
    ndcg_scores.append(best_result[4])
    di_scores.append(best_acc_di)


