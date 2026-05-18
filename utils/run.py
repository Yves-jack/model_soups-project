import torch
import torch.nn as nn
import numpy as np
from dataset import get_loaders
from models import LeNet5_Caffe, ResNet18
from training import train, evaluate
from model_soups import uniform_soup, greedy_soup, ensemble_evaluate, soup_experiment, plot_soup_comparison, add_uniform_soup
from fisher import fisher_sim_matrix, fisher_sim_heatmap, find_model_pairs
from interpolation import loss_acc, barrier, plot_interpolation


mnist_train_loader, mnist_valid_loader, mnist_test_loader = get_loaders(dataset_type="mnist")
cifar_train_loader, cifar_valid_loader, cifar_test_loader = get_loaders(dataset_type="cifar")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# training
## mnist
dataset_type = "mnist"
mnist_models_num = 15
mnist_result_diff_init = train(device, dataset_type, mnist_train_loader, mnist_test_loader, 
                               mnist_models_num, same_init_flag=False)
for item in mnist_result_diff_init["accs"]:
    print(f'{item["model_name"]} Test Accuracy: {item["acc"]:.4f}')

mnist_result_same_init = train(device, dataset_type, mnist_train_loader, mnist_test_loader, 
                               mnist_models_num, same_init_flag=True)
for item in mnist_result_same_init["accs"]:
    print(f'{item["model_name"]} Test Accuracy: {item["acc"]:.4f}')

## cifar10
dataset_type = "cifar"
cifar_models_num = 15
cifar_result_diff_init = train(device, dataset_type, cifar_train_loader, cifar_test_loader, 
                               cifar_models_num, same_init_flag=False)
for item in cifar_result_diff_init["accs"]:
    print(f'{item["model_name"]} Test Accuracy: {item["acc"]:.4f}')

cifar_result_same_init = train(device, dataset_type, cifar_train_loader, cifar_test_loader, 
                               cifar_models_num, same_init_flag=True)
for item in cifar_result_same_init["accs"]:
    print(f'{item["model_name"]} Test Accuracy: {item["acc"]:.4f}')


# model soup
## mnist
mnist_uniform_soup_model = uniform_soup(mnist_result_diff_init["models"])
mnist_uniform_soup_test_acc = evaluate(mnist_uniform_soup_model, mnist_test_loader, device) 
mnist_greedy_soup_model = greedy_soup(mnist_result_diff_init["models"], mnist_valid_loader, device)
mnist_greedy_soup_test_acc = evaluate(mnist_greedy_soup_model, mnist_test_loader, device) 
mnist_ensemble_test_acc = ensemble_evaluate(mnist_result_diff_init["models"], mnist_test_loader, device)


accs = [item["acc"] for item in mnist_result_diff_init["accs"]]
print(f"single model test accuracies: {min(accs):.4f} ~ {max(accs):.4f}")
print(f"Ensemble test accuracy: {mnist_ensemble_test_acc:.4f}")
print(f"Uniform Soup test accuracy: {mnist_uniform_soup_test_acc:.4f}")
print(f"Greedy Soup test accuracy: {mnist_greedy_soup_test_acc:.4f}")


mnist_uniform_soup_model = uniform_soup(mnist_result_same_init["models"])
mnist_uniform_soup_test_acc = evaluate(mnist_uniform_soup_model, mnist_test_loader, device) 
mnist_greedy_soup_model = greedy_soup(mnist_result_same_init["models"], mnist_valid_loader, device)
mnist_greedy_soup_test_acc = evaluate(mnist_greedy_soup_model, mnist_test_loader, device) 
mnist_ensemble_test_acc = ensemble_evaluate(mnist_result_same_init["models"], mnist_test_loader, device)

accs = [item["acc"] for item in mnist_result_same_init["accs"]]
print(f"single model test accuracies: {min(accs):.4f} ~ {max(accs):.4f}")
print(f"Ensemble test accuracy: {mnist_ensemble_test_acc:.4f}")
print(f"Uniform Soup test accuracy: {mnist_uniform_soup_test_acc:.4f}")
print(f"Greedy Soup test accuracy: {mnist_greedy_soup_test_acc:.4f}")

## cifar10
cifar_uniform_soup_model = uniform_soup(cifar_result_diff_init["models"])
cifar_uniform_soup_test_acc = evaluate(cifar_uniform_soup_model, cifar_test_loader, device) # type: ignore
cifar_greedy_soup_model = greedy_soup(cifar_result_diff_init["models"], cifar_valid_loader, device)
cifar_greedy_soup_test_acc = evaluate(cifar_greedy_soup_model, cifar_test_loader, device) # type: ignore
cifar_ensemble_test_acc = ensemble_evaluate(cifar_result_diff_init["models"], cifar_test_loader, device)

accs = [item["acc"] for item in cifar_result_diff_init["accs"]]
print(f"single model test accuracies: {min(accs):.4f} ~ {max(accs):.4f}")
print(f"Ensemble test accuracy: {cifar_ensemble_test_acc:.4f}")
print(f"Uniform Soup test accuracy: {cifar_uniform_soup_test_acc:.4f}")
print(f"Greedy Soup test accuracy: {cifar_greedy_soup_test_acc:.4f}")


cifar_uniform_soup_model = uniform_soup(cifar_result_same_init["models"])
cifar_uniform_soup_test_acc = evaluate(cifar_uniform_soup_model, cifar_test_loader, device)
cifar_greedy_soup_model = greedy_soup(cifar_result_same_init["models"], cifar_valid_loader, device)
cifar_greedy_soup_test_acc = evaluate(cifar_greedy_soup_model, cifar_test_loader, device)
cifar_ensemble_test_acc = ensemble_evaluate(cifar_result_same_init["models"], cifar_test_loader, device)

accs = [item["acc"] for item in cifar_result_same_init["accs"]]
print(f"single model test accuracies: {min(accs):.4f} ~ {max(accs):.4f}")
print(f"Ensemble test accuracy: {cifar_ensemble_test_acc:.4f}")
print(f"Uniform Soup test accuracy: {cifar_uniform_soup_test_acc:.4f}")
print(f"Greedy Soup test accuracy: {cifar_greedy_soup_test_acc:.4f}")


# comparaison of model numbers
sample_num_list = [1, 3, 6, 9, 12, 15]
## mnist

mnist_table_diff_init = soup_experiment(mnist_result_diff_init["models"], mnist_valid_loader, mnist_test_loader, device, sample_num_list, num_trials=10)
mnist_table_diff_init.to_csv("lenet5_soup_results_diff_init.csv", index=False)
print(mnist_table_diff_init)

mnist_table_same_init = soup_experiment(mnist_result_same_init["models"], mnist_valid_loader, mnist_test_loader, device, sample_num_list, num_trials=10)
mnist_table_same_init.to_csv("lenet5_soup_results_same_init.csv", index=False)
print(mnist_table_same_init)

## cifar10
cifar_table_diff_init = soup_experiment(cifar_result_diff_init["models"], cifar_valid_loader, cifar_test_loader, device, sample_num_list, num_trials=5)
cifar_table_diff_init.to_csv("resnet18_soup_results_diff_init.csv", index=False)
print(cifar_table_diff_init)

cifar_table_same_init = soup_experiment(cifar_result_same_init["models"], cifar_valid_loader, cifar_test_loader, device, sample_num_list, num_trials=5)
cifar_table_same_init.to_csv("resnet18_soup_results_same_init.csv", index=False)
print(cifar_table_same_init)

# MNIST (LeNet5) 
plot_soup_comparison(
    same_init_df=mnist_table_same_init, 
    diff_init_df=mnist_table_diff_init, 
    title="LeNet5 Soup: Same Init vs Different Init", 
    save_path="lenet5_soup_result.png",
    ylim=(0, 1)  
)

# CIFAR10 (ResNet18) 
plot_soup_comparison(
    same_init_df=cifar_table_same_init, 
    diff_init_df=cifar_table_diff_init, 
    title="ResNet18 Soup: Same Init vs Different Init", 
    save_path="resnet18_soup_result.png",
    ylim=(0.1, 0.90)  
)


# fisher heatmap
mnist_fisher_sim_matrix_diff_init = fisher_sim_matrix(mnist_result_diff_init["models"], mnist_train_loader, device, criterion=nn.CrossEntropyLoss())
fisher_sim_heatmap(mnist_fisher_sim_matrix_diff_init)

mnist_fisher_sim_matrix_same_init = fisher_sim_matrix(mnist_result_same_init["models"], mnist_train_loader, device, criterion=nn.CrossEntropyLoss())
fisher_sim_heatmap(mnist_fisher_sim_matrix_same_init)

cifar_fisher_sim_matrix_diff_init = fisher_sim_matrix(cifar_result_diff_init["models"], cifar_train_loader, device, criterion=nn.CrossEntropyLoss())
fisher_sim_heatmap(cifar_fisher_sim_matrix_diff_init)

cifar_fisher_sim_matrix_same_init = fisher_sim_matrix(cifar_result_same_init["models"], cifar_train_loader, device, criterion=nn.CrossEntropyLoss())
fisher_sim_heatmap(cifar_fisher_sim_matrix_same_init)


# interpolation
## mnist
mnist_model_pairs_max_diff, mnist_model_pairs_min_diff = find_model_pairs(mnist_fisher_sim_matrix_diff_init)
print(f"max: {mnist_model_pairs_max_diff}")
print(f"min: {mnist_model_pairs_min_diff}")

lambdas=np.linspace(0, 1, 11)

model_max_A = mnist_result_diff_init["models"][mnist_model_pairs_max_diff[0]]
model_max_B = mnist_result_diff_init["models"][mnist_model_pairs_max_diff[1]]

mnist_valid_losses, mnist_valid_accs = loss_acc(model_max_A, model_max_B, mnist_valid_loader, device, lambdas, model_class=LeNet5_Caffe)
mnist_barrier_ref, mnist_barrier_value, mnist_barrier_idx = barrier(mnist_valid_losses, lambdas)
plot_interpolation(lambdas, mnist_valid_losses, mnist_valid_accs, mnist_barrier_ref, mnist_barrier_value, mnist_barrier_idx)


model_min_A = mnist_result_diff_init["models"][mnist_model_pairs_min_diff[0]]
model_min_B = mnist_result_diff_init["models"][mnist_model_pairs_min_diff[1]]

mnist_valid_losses, mnist_valid_accs = loss_acc(model_min_A, model_min_B, mnist_valid_loader, device, lambdas, model_class=LeNet5_Caffe)
mnist_barrier_ref, mnist_barrier_value, mnist_barrier_idx = barrier(mnist_valid_losses, lambdas)
plot_interpolation(lambdas, mnist_valid_losses, mnist_valid_accs, mnist_barrier_ref, mnist_barrier_value, mnist_barrier_idx)


mnist_model_pairs_max_same, mnist_model_pairs_min_same = find_model_pairs(mnist_fisher_sim_matrix_same_init)
print(f"max: {mnist_model_pairs_max_same}")
print(f"min: {mnist_model_pairs_min_same}")

model_max_A = mnist_result_same_init["models"][mnist_model_pairs_max_same[0]]
model_max_B = mnist_result_same_init["models"][mnist_model_pairs_max_same[1]]

mnist_valid_losses, mnist_valid_accs = loss_acc(model_max_A, model_max_B, mnist_valid_loader, device, lambdas, model_class=LeNet5_Caffe)
mnist_barrier_ref, mnist_barrier_value, mnist_barrier_idx = barrier(mnist_valid_losses, lambdas)
plot_interpolation(lambdas, mnist_valid_losses, mnist_valid_accs, mnist_barrier_ref, mnist_barrier_value, mnist_barrier_idx)


model_min_A = mnist_result_same_init["models"][mnist_model_pairs_min_same[0]]
model_min_B = mnist_result_same_init["models"][mnist_model_pairs_min_same[1]]

mnist_valid_losses, mnist_valid_accs = loss_acc(model_min_A, model_min_B, mnist_valid_loader, device, lambdas, model_class=LeNet5_Caffe)
mnist_barrier_ref, mnist_barrier_value, mnist_barrier_idx = barrier(mnist_valid_losses, lambdas)
plot_interpolation(lambdas, mnist_valid_losses, mnist_valid_accs, mnist_barrier_ref, mnist_barrier_value, mnist_barrier_idx)


## cifar10
cifar_model_pairs_max_diff, cifar_model_pairs_min_diff = find_model_pairs(cifar_fisher_sim_matrix_diff_init)
print(f"max: {cifar_model_pairs_max_diff}")
print(f"min: {cifar_model_pairs_min_diff}")


model_max_A = cifar_result_diff_init["models"][cifar_model_pairs_max_diff[0]]
model_max_B = cifar_result_diff_init["models"][cifar_model_pairs_max_diff[1]]

cifar_valid_losses, cifar_valid_accs = loss_acc(model_max_A, model_max_B, cifar_valid_loader, device, lambdas, model_class=ResNet18)
cifar_barrier_ref, cifar_barrier_value, cifar_barrier_idx = barrier(cifar_valid_losses, lambdas)
plot_interpolation(lambdas, cifar_valid_losses, cifar_valid_accs, cifar_barrier_ref, cifar_barrier_value, cifar_barrier_idx)


model_min_A = cifar_result_diff_init["models"][cifar_model_pairs_min_diff[0]]
model_min_B = cifar_result_diff_init["models"][cifar_model_pairs_min_diff[1]]

cifar_valid_losses, cifar_valid_accs = loss_acc(model_min_A, model_min_B, cifar_valid_loader, device, lambdas, model_class=ResNet18)
cifar_barrier_ref, cifar_barrier_value, cifar_barrier_idx = barrier(cifar_valid_losses, lambdas)
plot_interpolation(lambdas, cifar_valid_losses, cifar_valid_accs, cifar_barrier_ref, cifar_barrier_value, cifar_barrier_idx)


cifar_model_pairs_max_same, cifar_model_pairs_min_same = find_model_pairs(cifar_fisher_sim_matrix_same_init)
print(f"max: {cifar_model_pairs_max_same}")
print(f"min: {cifar_model_pairs_min_same}")


model_max_A = cifar_result_same_init["models"][cifar_model_pairs_max_same[0]]
model_max_B = cifar_result_same_init["models"][cifar_model_pairs_max_same[1]]

cifar_valid_losses, cifar_valid_accs = loss_acc(model_max_A, model_max_B, cifar_valid_loader, device, lambdas, model_class=ResNet18)
cifar_barrier_ref, cifar_barrier_value, cifar_barrier_idx = barrier(cifar_valid_losses, lambdas)
plot_interpolation(lambdas, cifar_valid_losses, cifar_valid_accs, cifar_barrier_ref, cifar_barrier_value, cifar_barrier_idx)


model_min_A = cifar_result_same_init["models"][cifar_model_pairs_min_same[0]]
model_min_B = cifar_result_same_init["models"][cifar_model_pairs_min_same[1]]

cifar_valid_losses, cifar_valid_accs = loss_acc(model_min_A, model_min_B, cifar_valid_loader, device, lambdas, model_class=ResNet18)
cifar_barrier_ref, cifar_barrier_value, cifar_barrier_idx = barrier(cifar_valid_losses, lambdas)
plot_interpolation(lambdas, cifar_valid_losses, cifar_valid_accs, cifar_barrier_ref, cifar_barrier_value, cifar_barrier_idx)


# uniform soup for adding models
max_model_num = 5
add_uniform_soup(list(mnist_model_pairs_max_diff), max_model_num, mnist_fisher_sim_matrix_diff_init, mnist_result_diff_init["models"], mnist_test_loader, device)
add_uniform_soup(list(mnist_model_pairs_max_same), max_model_num, mnist_fisher_sim_matrix_same_init, mnist_result_same_init["models"], mnist_test_loader, device)
add_uniform_soup(list(cifar_model_pairs_max_diff), max_model_num, cifar_fisher_sim_matrix_diff_init, cifar_result_diff_init["models"], cifar_test_loader, device)
add_uniform_soup(list(cifar_model_pairs_max_same), max_model_num, cifar_fisher_sim_matrix_same_init, cifar_result_same_init["models"], cifar_test_loader, device)

