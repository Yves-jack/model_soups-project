import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Compute diagonal Fisher Information Matrix approximation.
def compute_fisher(model, data_loader, device, criterion):
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)  

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        for name, param in model.named_parameters():
            fisher[name] += param.grad.data ** 2  # 累加grad²

    # Diagonal Fisher Approximation
    for name in fisher:
        fisher[name] /= len(data_loader.dataset)

    return fisher

# Compute cosine similarity between two Fisher Information Matrices.
def fisher_cosine_similarity(fisher1, fisher2):
    # Compute overall similarity 
    # 一个数字，越接近1说明两个模型的Fisher信息越相似，越接近0说明越不相似
    all_f1 = torch.cat([v.flatten() for v in fisher1.values()])
    all_f2 = torch.cat([v.flatten() for v in fisher2.values()])

    fisher_similarity = F.cosine_similarity(
        all_f1, all_f2, dim=0
    ).item()

    return fisher_similarity

def fisher_sim_matrix(models, train_loader, device, criterion):
    # Compute Fisher Information for all models
    fishers = [compute_fisher(model, train_loader, device, criterion) for model in models]
    models_num = len(fishers)

    # Compute serveral pairwise similarities between models
    fisher_sim = np.zeros((models_num, models_num))
    for i in range(models_num):
        for j in range(i + 1, models_num):
            sim = fisher_cosine_similarity(fishers[i], fishers[j])
            fisher_sim[i][j] = sim
            fisher_sim[j][i] = sim

    # 对角线填充为 1.0，表示自身相似度
    np.fill_diagonal(fisher_sim, 1.0)
    return fisher_sim

def fisher_sim_heatmap(fisher_sim):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    model_names = [f'Model {i+1}' for i in range(fisher_sim.shape[0])]
    df_sim = pd.DataFrame(fisher_sim, index=model_names, columns=model_names)
    # 构建下三角 mask
    mask = np.triu(np.ones_like(df_sim, dtype=bool), k=1)  # k=1 保留对角线

    sns.heatmap(
        df_sim,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        vmin=0, vmax=1,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Fisher Similarity', 'shrink': 0.8},
        ax=ax
    )

    ax.set_title('Model Fisher Similarity (Lower Triangle)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    plt.tight_layout()
    # plt.savefig('/mnt/data/similarity_heatmap_lower.png', dpi=150, bbox_inches='tight')
    plt.show()

def find_model_pairs(fisher_sims):
    fisher_sim_max, fisher_sim_min = 0.0, 1.0
    model_pairs_max, model_pairs_min = (0, 0), (0, 0)
    model_num = fisher_sims.shape[0]
    for i in range(model_num):
        for j in range(i+1, model_num):
            if fisher_sims[i][j] > fisher_sim_max:
                fisher_sim_max = fisher_sims[i][j]
                model_pairs_max = (i, j)
            if fisher_sims[i][j] < fisher_sim_min:
                fisher_sim_min = fisher_sims[i][j]
                model_pairs_min = (i, j)
    return model_pairs_max, model_pairs_min

def find_add_model(fisher_sims, model_idx_list):
    model_num = fisher_sims.shape[0]
    model_sim = {}
    for i in range(model_num):
        if i in model_idx_list:
            continue
        for j in model_idx_list:
            model_sim[i] = model_sim.get(i, 0.0) + fisher_sims[i][j]
    add_model_idx = max(model_sim, key=model_sim.get) # type: ignore
    model_idx_list.append(add_model_idx)
    return model_idx_list