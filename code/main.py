import torch
from data import DDDataset_txt
from model import normalization, tensor_from_numpy, ModelA
from util import create_sparse_one_hot, test_evalute

if __name__ == '__main__':
    INPUT_DIM = 4 + pow(2,16)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 2
    HIDDEN_DIM = 64
    dataset_AES = DDDataset_txt('../data/Matrix-DeTrust')
    _, test_list, _, _ = dataset_AES.leaveone_split('AES-DeTrust')
    test_adjacency, test_node_features, test_graph_indicator, test_node_labels = dataset_AES.__getitem__(test_list)
    test_normalize_adjacency = normalization(test_adjacency).to(DEVICE)
    test_node_features = create_sparse_one_hot(test_node_features, INPUT_DIM, DEVICE)
    test_node_labels = tensor_from_numpy(test_node_labels, DEVICE)
    test_model = ModelA(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    test_model = torch.load('model.pth',map_location='cpu')
    test_model.eval()
    with torch.no_grad():
        logits = test_model(test_normalize_adjacency, test_node_features, None)
        test_logits = logits
        test_acc = torch.eq(
            test_logits.max(1)[1], test_node_labels
        ).float().mean()
        tpr,tnr,acc,pre,f1_,Auc = test_evalute(prob_labels=test_logits.max(1)[1], true_labels=test_node_labels)
        print("Tpr {:.6}, Tnr {:.6}, Acc {:.6}, Auc {:.6}".format(
                    tpr, tnr, acc, Auc))