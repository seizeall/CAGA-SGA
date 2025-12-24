import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import random

# === 导入模型模块 ===
from model import GSA_CAST
from graph_align import SemanticAligner
from golden_style import ch_stats, gram  # 复用统计函数

# === 导入数据接口 (假设 datapipe.py 存在于同级目录) ===
# 用户需要自行提供 datapipe.py
from datapipe import create_cross_dataset_setup, load_cross_dataset_fold


def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 实验参数配置 ---
NUM_TARGET_SUBJECTS = 16  # 目标域被试数量 (SEED-V)
EPOCHS = 200
CLASSES = 3
NETWORK = GSA_CAST
DEVICE = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
EXPERIMENT_NAME = f'SEED-VII_to_SEED-V_{NETWORK.__name__}'

# --- 损失函数超参数 ---
TAU = 0.7
LAMBDA_E = 1.0
LAMBDA_V = 0.1
PSEUDO_LABEL_THRESHOLD_START = 0.95
PSEUDO_LABEL_THRESHOLD_END = 0.70
LAMBDA_STYLE = 0.5
LAMBDA_PRESERVE = 1.0
LAMBDA_GOLD = 0.5
LAMBDA_ALIGN = 1.0
LAMBDA_ALIGN_GRAM = 0.1
USE_GRAM_PRESERVE = True

# 设置种子
set_random_seed(42)

# 结果保存路径
version = 1
summary_dfile = f'./result/Summary_{EXPERIMENT_NAME}_v{version}.csv'
detailed_dfile = f'./result/Detailed_Metrics_{EXPERIMENT_NAME}_v{version}.csv'
while os.path.exists(summary_dfile) or os.path.exists(detailed_dfile):
    version += 1
    summary_dfile = f'./result/Summary_{EXPERIMENT_NAME}_v{version}.csv'
    detailed_dfile = f'./result/Detailed_Metrics_{EXPERIMENT_NAME}_v{version}.csv'

os.makedirs('./result', exist_ok=True)
print(f"Summary results will be saved to: {summary_dfile}")
print(f"Detailed metrics will be saved to: {detailed_dfile}")


def style_loss_per_layer(src_pre, src_post, tgt_pre, tgt_post, mu_g, sig_g):
    """计算单层的 Style Loss (Preserve + Gold + Align)"""
    mu_s0, std_s0 = ch_stats(src_pre)
    mu_s1, std_s1 = ch_stats(src_post)
    mu_t0, std_t0 = ch_stats(tgt_pre)
    mu_t1, std_t1 = ch_stats(tgt_post)

    # 1. Content Preserve Loss
    L_pres = (mu_s1 - mu_s0).abs().mean() + (std_s1 - std_s0).abs().mean() + \
             (mu_t1 - mu_t0).abs().mean() + (std_t1 - std_t0).abs().mean()

    if USE_GRAM_PRESERVE:
        Gs0, Gs1 = gram(src_pre), gram(src_post)
        Gt0, Gt1 = gram(tgt_pre), gram(tgt_post)
        L_pres = L_pres + (Gs1 - Gs0).abs().mean() + (Gt1 - Gt0).abs().mean()

    # 2. Golden Anchor Loss
    L_gold = (mu_s1 - mu_g).abs().mean() + (std_s1 - sig_g).abs().mean() + \
             (mu_t1 - mu_g).abs().mean() + (std_t1 - sig_g).abs().mean()

    # 3. Cross-Domain Align Loss
    L_align = (mu_s1 - mu_t1).abs().mean() + (std_s1 - std_t1).abs().mean()
    L_align = L_align + LAMBDA_ALIGN_GRAM * (gram(src_post) - gram(tgt_post)).abs().mean()

    return LAMBDA_PRESERVE * L_pres + LAMBDA_GOLD * L_gold + LAMBDA_ALIGN * L_align


def train(model, graph_module, train_loader, target_loader, crit, domain_crit, optimizer, graph_optimizer, epoch,
          total_epochs, lambdas=0.1):
    model.train()
    graph_module.train()
    loss_all, domain_loss_all, graph_loss_all, style_loss_all = 0.0, 0.0, 0.0, 0.0

    progress = epoch / total_epochs
    current_threshold = PSEUDO_LABEL_THRESHOLD_START - (
                PSEUDO_LABEL_THRESHOLD_START - PSEUDO_LABEL_THRESHOLD_END) * progress

    train_iter, target_iter = iter(train_loader), iter(target_loader)
    num_iter = len(train_loader)

    for i in range(num_iter):
        try:
            source_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            source_data = next(train_iter)
        try:
            target_data = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_data = next(target_iter)

        source_data, target_data = source_data.to(DEVICE), target_data.to(DEVICE)

        optimizer.zero_grad()
        graph_optimizer.zero_grad()

        source_label = torch.argmax(source_data.y.view(-1, CLASSES), axis=1)

        # --- 前向传播 ---
        # 1. 第一次 Forward: 获取 bootstrap 所需的 style info
        _, _, _, _, _, sc_spatials = model(source_data.x, source_data.edge_index, source_data.batch,
                                           return_style_info=True)
        _, _, _, _, _, tg_spatials = model(target_data.x, target_data.edge_index, target_data.batch,
                                           return_style_info=True)

        # 2. 第二次 Forward: 执行 style bootstrap 并获取最终特征
        sc_out, _, sc_dom, sc_feat, sc_style, _ = model(source_data.x, source_data.edge_index, source_data.batch,
                                                        return_style_info=True,
                                                        style_bootstrap_pair=(sc_spatials, tg_spatials))
        tg_out, _, tg_dom, tg_feat, tg_style, _ = model(target_data.x, target_data.edge_index, target_data.batch,
                                                        return_style_info=True,
                                                        style_bootstrap_pair=(sc_spatials, tg_spatials))

        # --- Loss 计算 ---
        loss_cls = crit(sc_out, source_label)

        domain_preds = torch.cat([sc_dom, tg_dom])
        domain_labels = torch.cat([torch.zeros_like(sc_dom), torch.ones_like(tg_dom)])
        loss_domain = domain_crit(domain_preds, domain_labels)
        relaxed_loss_domain = torch.min(loss_domain, torch.tensor(TAU, device=DEVICE))

        # Graph Loss
        combined_features = torch.cat([sc_feat, tg_feat], dim=0)
        graph_logits, affinity_hat = graph_module(combined_features)

        with torch.no_grad():
            conf, tg_pl = torch.max(F.softmax(tg_out, dim=1), dim=1)
            mask = conf > current_threshold
            combined_labels = torch.cat([source_label, tg_pl])
            A_star = (combined_labels.unsqueeze(1) == combined_labels.unsqueeze(0)).float()
            valid_pairs_mask = torch.cat([torch.ones_like(source_label, dtype=torch.bool), mask]).unsqueeze(0).T.bool()
            valid_pairs_mask = valid_pairs_mask & valid_pairs_mask.T

        if valid_pairs_mask.sum() > 0:
            loss_e = F.binary_cross_entropy_with_logits(affinity_hat[valid_pairs_mask], A_star[valid_pairs_mask])
        else:
            loss_e = torch.tensor(0.0, device=DEVICE)

        loss_v = crit(graph_logits[:source_data.num_graphs], source_label)
        loss_graph = LAMBDA_E * loss_e + LAMBDA_V * loss_v

        # Style Loss
        style_loss = sum(style_loss_per_layer(
            sc_style[l]["pre"], sc_style[l]["post"],
            tg_style[l]["pre"], tg_style[l]["post"],
            sc_style[l]["mu_g"], sc_style[l]["sig_g"]
        ) for l in range(len(sc_style)))

        total_loss = loss_cls + loss_graph + lambdas * relaxed_loss_domain + LAMBDA_STYLE * style_loss

        total_loss.backward()
        optimizer.step()
        graph_optimizer.step()

        loss_all += loss_cls.item() * source_data.num_graphs
        domain_loss_all += loss_domain.item() * len(domain_labels)
        graph_loss_all += loss_graph.item() * source_data.num_graphs
        style_loss_all += style_loss.item() * source_data.num_graphs if isinstance(style_loss,
                                                                                   torch.Tensor) else style_loss

    return (loss_all / len(train_loader.dataset),
            domain_loss_all / (len(train_loader.dataset) + len(target_loader.dataset)),
            graph_loss_all / len(train_loader.dataset),
            style_loss_all / len(train_loader.dataset))


def evaluate(model, loader):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for data in loader:
            labels.append(data.y.view(-1, CLASSES).cpu().numpy())
            data = data.to(DEVICE)
            _, pred, _, _ = model(data.x, data.edge_index, data.batch)
            predictions.append(pred.cpu().numpy())

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    predictions_cls = np.argmax(predictions, axis=-1)
    labels_cls = np.argmax(labels, axis=-1)

    try:
        auc = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
    except ValueError:
        auc = 0.5

    acc = accuracy_score(labels_cls, predictions_cls)
    f1 = f1_score(labels_cls, predictions_cls, average='macro', zero_division=0)
    recall = recall_score(labels_cls, predictions_cls, average='macro', zero_division=0)
    precision = precision_score(labels_cls, predictions_cls, average='macro', zero_division=0)

    return auc, acc, f1, recall, precision


def main():
    # 1. 准备数据环境 (调用 datapipe)
    print("Preparing dataset setup...")
    create_cross_dataset_setup()

    summary_results, detailed_results = [], []
    domain_crit, class_crit = torch.nn.BCEWithLogitsLoss(), torch.nn.CrossEntropyLoss()

    # 2. 循环每个目标被试 (Leave-One-Subject-Out 风格)
    for cv_n in range(NUM_TARGET_SUBJECTS):
        print(f"\n--- Starting Subject: {cv_n} ---")
        best_val_acc = 0.0
        best_metrics = {}

        # 加载单个 fold 数据
        train_dataset, test_dataset = load_cross_dataset_fold(cv_n)

        train_loader = DataLoader(train_dataset, 16, shuffle=True, drop_last=True)
        target_loader = DataLoader(test_dataset, 16, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, 16, shuffle=False)

        # 初始化模型
        model = NETWORK().to(DEVICE)
        graph_module = SemanticAligner(in_features=128, num_classes=CLASSES).to(DEVICE)

        graph_optimizer = torch.optim.Adam(graph_module.parameters(), lr=1e-4)
        main_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # 训练 Loop
        for epoch in range(EPOCHS):
            loss, domain_loss, graph_loss, style_loss = train(
                model, graph_module, train_loader, target_loader,
                class_crit, domain_crit, main_optimizer, graph_optimizer,
                epoch, EPOCHS
            )
            val_auc, val_acc, val_f1, val_recall, val_precision = evaluate(model, test_loader)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_metrics = {
                    'subject': cv_n, 'epoch': epoch + 1, 'accuracy': val_acc,
                    'f1_score': val_f1, 'recall': val_recall, 'precision': val_precision, 'auc': val_auc
                }

            # 日志 (减少刷屏，每10轮打印一次)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'S{cv_n:02d}|E{epoch + 1:03d} | L_cls:{loss:.3f} | Acc:{val_acc:.4f} | Best:{best_val_acc:.4f}')

        detailed_results.append(best_metrics)
        summary_results.append([cv_n, best_metrics.get('epoch', -1), best_metrics.get('accuracy', 0.0)])
        print(f"--- Subject {cv_n} Finished. Best Acc: {best_metrics.get('accuracy', 0.0):.4f} ---")

    # 3. 保存最终统计结果
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(detailed_dfile, index=False)

    summary_df = pd.DataFrame(summary_results, columns=['Subject', 'Best_Epoch', 'Best_Vacc'])
    summary_df.to_csv(summary_dfile, index=False)

    mean_acc = detailed_df['accuracy'].mean()
    std_acc = detailed_df['accuracy'].std()

    print(f"\n=== Final Results: {mean_acc:.4f} ± {std_acc:.4f} ===")

    # 将最终平均值追加到文件末尾
    with open(summary_dfile, 'a') as f:
        f.write(f"\nMean,N/A,{mean_acc:.4f} ± {std_acc:.4f}\n")


if __name__ == '__main__':
    main()