import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from data_loader_mnv2 import get_dataloader
import argparse
import timm
import seaborn as sns
import os
from tqdm import tqdm

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        eval_progress = tqdm(data_loader, desc="Evaluating", leave=True)
        for inputs, labels in eval_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            eval_progress.set_postfix(acc=accuracy_score(all_labels, all_preds))
    
    cm = confusion_matrix(all_labels, all_preds)
    metrics = {
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced accuracy': balanced_accuracy_score(all_labels, all_preds)
    }
    return cm, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='mobilenetv2_100')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = timm.create_model(args.model_name, num_classes=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Load data
    test_loader = get_dataloader(args.data_dir, args.batch_size, shuffle=False, phase='test')
    
    # Evaluate
    cm, metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nEvaluation Results:")
    for k,v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
    
    os.makedirs('results', exist_ok=True)
    
    # Confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0,1], ['Cancer', 'No Cancer'])
    plt.yticks([0,1], ['Cancer', 'No Cancer'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/cm.png')
    plt.close()
    
    # Heatmap
    group_names = ['True Pos','False Neg','False Pos','True Neg']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    x_ala = ['Cancer', 'No_Cancer']
    y_ala = ['Cancer', 'No_Cancer']
    hm = sns.heatmap(cm, annot=labels, xticklabels=x_ala, yticklabels=y_ala, fmt='', cmap='rocket', vmin=0, vmax=np.sum(cm))
    cbar = hm.collections[0].colorbar
    a = 0.2*np.sum(cm)
    b = 0.75*np.sum(cm)
    c = np.sum(cm)
    cbar.set_ticks([0, a, b, c])
    cbar.set_ticklabels(['low', '20%', '75%', '100%'])
    fig = hm.get_figure()
    fig.savefig(f'results/{args.model_name}_FIN_HEATMAP_EVALUATION.png')
    plt.close()

if __name__ == '__main__':
    main()