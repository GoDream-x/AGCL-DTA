import os
import argparse

import torch
import json
import warnings
from collections import OrderedDict
from torch import nn
from itertools import chain
from data_process import load_data, process_data, get_drug_molecule_graph, get_target_molecule_graph
from utils import GraphDataset, collate, model_evaluate
from models import AGCL, PredictModule
import matplotlib.pyplot as plt

def train(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, lr, epoch,
          batch_size, affinity_graph, drug_pos, target_pos):
    global best_loss, best_epoch  # 声明为全局变量以便于外部访问
    best_loss = float('inf')
    best_epoch = None  # 用于记录达到最佳loss的epoch
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    predictor.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())), lr=lr, weight_decay=0)

    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        ssl_loss, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs,
                                                           target_graph_batchs, drug_pos, target_pos)
        # drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs,
        #                                                    target_graph_batchs, drug_pos, target_pos)
        output, _ = predictor(data.to(device), drug_embedding, target_embedding)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) + ssl_loss
        # 更新最佳损失和对应的epoch
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            torch.save(model.state_dict(), args.BEST_MODEL_PATH)
            torch.save(predictor.state_dict(), args.BEST_Predictor_PATH)

        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


def test(model, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_pos,
         target_pos):
    model.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        for data in loader:
            _,drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs,
                                                        target_graph_batchs, drug_pos, target_pos)
            output, _ = predictor(data.to(device), drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def train_predict():
    print("Data preparation in progress for the {} dataset...".format(args.dataset))
    affinity_mat = load_data(args.dataset)
    train_data, test_data, affinity_graph, drug_pos, target_pos = process_data(affinity_mat, args.dataset, args.num_pos,
                                                                               args.pos_threshold)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    drug_graphs_dict = get_drug_molecule_graph(
        json.load(open(f'data/{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=affinity_graph.num_drug)
    target_graphs_dict = get_target_molecule_graph(
        json.load(open(f'data/{args.dataset}/targets.txt'), object_pairs_hook=OrderedDict), args.dataset)
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=affinity_graph.num_target)

    print("Model preparation... ")
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    model = AGCL(tau=args.tau,
                   lam=args.lam,
                   ns_dims=[affinity_graph.num_drug + affinity_graph.num_target + 2, 512, 256],
                   d_ms_dims=[78, 78, 78 * 2, 256],
                   t_ms_dims=[54, 54, 54 * 2, 256],
                   embedding_dim=128,
                   dropout_rate=args.edge_dropout_rate)
    predictor = PredictModule()
    drug_pos = drug_pos.to(device)
    target_pos = target_pos.to(device)
    model.to(device)
    predictor.to(device)

    print("Start training...")
    for epoch in range(args.epochs):
        train(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, args.lr,
              epoch + 1,
              args.batch_size, affinity_graph, drug_pos, target_pos)
    # 训练结束后输出最终的最好损失值和对应的epoch
    print(f'\nTraining complete. Best model was saved at Epoch {best_epoch} with loss {best_loss:.4f}')
    # Load the saved best model after all epochs have completed
    model.load_state_dict(torch.load(args.BEST_MODEL_PATH, map_location=device), strict=True)
    predictor.load_state_dict(torch.load(args.BEST_Predictor_PATH, map_location=device), strict=True)
    model.to(device)
    predictor.to(device)
    model.eval()  # 确保模型处于评估模式
    predictor.eval()


    print('\npredicting for train data')
    L, P = test(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph,
                drug_pos, target_pos)
    MSE, MAE, R2 = model_evaluate(L, P)
    print('Test MSE score: ', MSE)
    print('Test MAE score: ', MAE)
    print('Test R2 score: ', R2)

    print('\npredicting for test data')
    L, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph,
                drug_pos, target_pos)
    MSE, MAE, R2 = model_evaluate(L, P)
    print('Test MSE score: ', MSE)
    print('Test MAE score: ', MAE)
    print('Test R2 score: ', R2)

    # 创建散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(L, P, color='blue', alpha=0.5, label='Prediction vs Actual')

    # 添加对角线
    min_val = min(min(L), min(P))
    max_val = max(max(L), max(P))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal Prediction')

    # 设置全局字体为新罗马字体、大小24
    plt.rcParams.update({'font.size': 24, 'font.family': 'Times New Roman'})  # 修改字体大小为24

    # 设置坐标轴标签和标题，显式调整标题大小为22
    plt.xlabel('Predicted affinity', fontsize=22)
    plt.ylabel('Actual affinity', fontsize=22)
    plt.title('Scatter Plot of Predicted vs Actual Affinity', fontsize=22)

    # 设置坐标轴的数字大小为18
    plt.tick_params(axis='both', labelsize=18)

    # 添加图例，调整图例位置
    plt.legend(loc='upper left', fontsize=20, frameon=False)  # 图例大小设置为20，避免遮挡

    # 显示网格
    plt.grid(True)

    # 保存图形为 SVG 格式
    plt.savefig("scatter_plot_affinity.svg", format='svg', bbox_inches='tight')

    # 显示图形
    plt.show()


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--epochs', type=int, default=2200)  # davis 2200; kiba 3000
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--edge_dropout_rate', type=float, default=0.2)  # --kiba 0.
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--num_pos', type=int, default=3)  # davis 3; kiba 10
    parser.add_argument('--pos_threshold', type=float, default=8.0)
    parser.add_argument('--BEST_MODEL_PATH', type=str,
                        default=r'C:/Users/86176/Desktop/AGCL-DTA/bestmodel/best_model_new.pth')
    parser.add_argument('--BEST_Predictor_PATH', type=str,
                        default=r'C:/Users/86176/Desktop/AGCL-DTA/bestmodel/best_predictor_new.pth')
    args, _ = parser.parse_known_args()

    train_predict()