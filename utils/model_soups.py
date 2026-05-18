import torch
from copy import deepcopy
from training import evaluate
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from fisher import find_add_model


def uniform_soup(models):
    uniform_soup_model = deepcopy(models[0])
    avg_state_dict =  uniform_soup_model.state_dict()  
    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.stack([model.state_dict()[key].float() for model in models]).mean(dim=0)
    uniform_soup_model.load_state_dict(avg_state_dict)
    return uniform_soup_model

def greedy_soup(models, valid_loader, device):
    valid_accs = [(i, evaluate(model, valid_loader, device)) for i, model in enumerate(models)]
    valid_accs.sort(key=lambda x: x[1], reverse=True)
    models_ordered = [models[i] for i, _ in valid_accs]
    
    greedy_soup_model = deepcopy(models_ordered[0])
    best_acc = evaluate(greedy_soup_model, valid_loader, device)
    ingredients = [models_ordered[0]]
    print(f"\nInitial greedy soup accuracy: {best_acc:.4f}")

    for i in range(1, len(models_ordered)):
        # add model to soup and evaluate
        model_idx = valid_accs[i][0]
        candidate_soup_model = uniform_soup(ingredients + [models_ordered[i]])
        candidate_acc = evaluate(candidate_soup_model, valid_loader, device)

        # evaluate on validation set
        if candidate_acc > best_acc:
            greedy_soup_model = candidate_soup_model
            best_acc = candidate_acc
            ingredients.append(models_ordered[i])
            
            print(f" -> Added model_{model_idx}, new valid acc:{best_acc:.4f}")

    return greedy_soup_model

def ensemble_evaluate(models, test_loader, device):
    for model in models:
        model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = [model(inputs) for model in models]
            avg_output = torch.mean(torch.stack(outputs), dim=0)
            _, predicted = torch.max(avg_output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 对一组 sampled_models：计算 Uniform Soup\Greedy Soup\Ensemble 的准确率
def run_single(sampled_models, valid_loader, test_loader, device):

    # Uniform Soup
    uniform_model = uniform_soup(sampled_models)
    uniform_acc = evaluate(uniform_model, test_loader, device)

    # Greedy Soup
    greedy_model = greedy_soup(sampled_models, valid_loader, device)
    greedy_acc = evaluate(greedy_model, test_loader, device)

    # Ensemble
    ensemble_acc = ensemble_evaluate(sampled_models, test_loader, device)

    return {
        "uniform": uniform_acc,
        "greedy": greedy_acc,
        "ensemble": ensemble_acc
    }

def soup_experiment(models, valid_loader, test_loader, device, sample_num_list, num_trials=5):
    random.seed(42)
    results = []
    total_models = len(models)

    for sample_num in sample_num_list:
        if sample_num > total_models:
            print(f"Skip sample number = {sample_num}, only {total_models} models available.")
            continue

        print(f"Running experiment for sample number = {sample_num}")
        current_trials = 1 if sample_num == total_models else num_trials

        uniform_accs = []
        greedy_accs = []
        ensemble_accs = []
        for trial in range(current_trials):
            sampled_models = random.sample(models, sample_num)
            result = run_single(sampled_models, valid_loader, test_loader, device)

            uniform_accs.append(result["uniform"])
            greedy_accs.append(result["greedy"])
            ensemble_accs.append(result["ensemble"])

        results.append({
            "Number of Models": sample_num,

            "Uniform Mean": np.mean(uniform_accs),
            "Uniform Std": np.std(uniform_accs),

            "Greedy Mean": np.mean(greedy_accs),
            "Greedy Std": np.std(greedy_accs),

            "Ensemble Mean": np.mean(ensemble_accs),
            "Ensemble Std": np.std(ensemble_accs),
        })

    return pd.DataFrame(results)

# 绘制 Same Init 与 Different Init 的 Model Soup 对比图
def plot_soup_comparison(same_init_df, diff_init_df, title, save_path, ylim=None):

    fig, ax = plt.subplots(figsize=(9, 6))

    x = same_init_df["Number of Models"]

    colors = {
        "Uniform": "#1f77b4",  
        "Greedy":  "#ff7f0e",  
        "Ensemble":"#2ca02c",  
    }
    markers = {
        "Uniform": "o",
        "Greedy":  "s",
        "Ensemble":"^",
    }

    # Same Init
    ax.errorbar(x, same_init_df["Uniform Mean"],  yerr=same_init_df["Uniform Std"],  marker=markers["Uniform"], capsize=4, color=colors["Uniform"],  linestyle='-',  linewidth=2, markersize=6)
    ax.errorbar(x, same_init_df["Greedy Mean"],   yerr=same_init_df["Greedy Std"],   marker=markers["Greedy"],  capsize=4, color=colors["Greedy"],   linestyle='-',  linewidth=2, markersize=6)
    ax.errorbar(x, same_init_df["Ensemble Mean"], yerr=same_init_df["Ensemble Std"], marker=markers["Ensemble"],capsize=4, color=colors["Ensemble"], linestyle='-',  linewidth=2, markersize=6)

    # Diff Init 
    ax.errorbar(x, diff_init_df["Uniform Mean"],  yerr=diff_init_df["Uniform Std"],  marker=markers["Uniform"], capsize=4, color=colors["Uniform"],  linestyle='--', linewidth=2, markersize=6)
    ax.errorbar(x, diff_init_df["Greedy Mean"],   yerr=diff_init_df["Greedy Std"],   marker=markers["Greedy"],  capsize=4, color=colors["Greedy"],   linestyle='--', linewidth=2, markersize=6)
    ax.errorbar(x, diff_init_df["Ensemble Mean"], yerr=diff_init_df["Ensemble Std"], marker=markers["Ensemble"],capsize=4, color=colors["Ensemble"], linestyle='--', linewidth=2, markersize=6)

    uniform_handle = mlines.Line2D([], [], color=colors["Uniform"], marker=markers["Uniform"], linestyle='-', linewidth=2, markersize=6, label='Uniform Soup')
    greedy_handle  = mlines.Line2D([], [], color=colors["Greedy"],  marker=markers["Greedy"],  linestyle='-', linewidth=2, markersize=6, label='Greedy Soup')
    ensemble_handle= mlines.Line2D([], [], color=colors["Ensemble"],marker=markers["Ensemble"],linestyle='-', linewidth=2, markersize=6, label='Ensemble')
    same_init_handle = mlines.Line2D([], [], color='black', linestyle='-',  linewidth=2, label='Same Init')
    diff_init_handle = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Different Init')
    handles = [uniform_handle, greedy_handle, ensemble_handle, same_init_handle, diff_init_handle]

    ax.set_xlabel("Number of Models", fontsize=13)
    ax.set_ylabel("Test Accuracy", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    
    if ylim:
        ax.set_ylim(ylim)

    ax.legend(handles=handles, fontsize=10, ncol=2, loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def add_uniform_soup(model_max_list, max_model_num, fisher_sim_matrix, models, test_loader, device):
    while len(model_max_list) <= max_model_num:
        uniform_soup_model = uniform_soup([models[i] for i in model_max_list]) # type: ignore
        uniform_soup_test_acc = evaluate(uniform_soup_model, test_loader, device)
        print(f"Uniform Soup test accuracy({model_max_list}): {uniform_soup_test_acc:.4f}")
        model_max_list = find_add_model(fisher_sim_matrix, model_max_list)