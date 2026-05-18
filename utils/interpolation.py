import torch
import torch.nn as nn
from training import evaluate
import matplotlib.pyplot as plt


# 用线性插值法来生成位于一条线上的多个model
def interpolate_models(model_A, model_B, lambdas, model_class, device):
    models = []
    for lam in lambdas:
        model = model_class().to(device)
        state = {}
        for key in model_A.state_dict():
            state[key] = (1 - lam) * model_A.state_dict()[key] + lam * model_B.state_dict()[key]
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models.append(model)
    return models

# 计算线性插值的valid loss和valid acc
def loss_acc(model_A, model_B, valid_loader, device, lambdas, model_class):
    models = interpolate_models(model_A, model_B, lambdas, model_class, device)
    
    valid_losses = []
    valid_accs = []
    criterion = nn.CrossEntropyLoss()
    
    for model, lam in zip(models, lambdas):
        torch.manual_seed(42) # 
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                count += inputs.size(0)
                
        valid_losses.append(total_loss / count)
        
        acc = evaluate(model, valid_loader, device)
        valid_accs.append(acc)
        print(f"λ={lam:.2f}: Valid Loss={valid_losses[-1]:.4f}, Valid Acc={acc:.4f}")

    return valid_losses, valid_accs

#barrier是一个判断两个模型是否在同一个盆地里的指标。
def barrier(valid_losses, lambdas):
    ref_line = [valid_losses[0] + (valid_losses[-1] - valid_losses[0]) * (lam / lambdas[-1]) for lam in lambdas]
    barriers = [valid_losses[i] - ref_line[i] for i in range(len(valid_losses))]
    barrier_value = max(barriers)
    idx = barriers.index(barrier_value)
    return ref_line, barrier_value, idx

# 可视化 线性插值的结果
def plot_interpolation(lambdas, valid_losses, valid_accs, ref_line, barrier_value, barrier_idx, title="Linear Interpolation"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Valid Loss and Reference Line
    ax1.plot(lambdas, valid_losses, 'o-', label='Valid Loss')
    ax1.plot(lambdas, ref_line, 'r-', label='Reference Line')
    ax1.plot([lambdas[barrier_idx], lambdas[barrier_idx]], [ref_line[barrier_idx], valid_losses[barrier_idx]], 'g--', label=f'Barrier Value: {barrier_value:.4f}')
    ax1.set_xlabel('λ')
    ax1.set_ylabel('Valid Loss')
    ax1.set_title("Valid Loss with Barrier")
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot Valid Accuracy
    ax2.plot(lambdas, valid_accs, 's-', color='tab:blue', label='Valid Acc')
    ax2.set_xlabel('λ')
    ax2.set_ylabel('Valid Accuracy')
    ax2.set_title("Valid Accuracy")
    ax2.legend(loc='upper right')
    ax2.grid(True)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()