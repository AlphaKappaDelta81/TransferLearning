import torch
import torch.nn as nn
from termcolor import colored
import torch.optim as optim
from data_loader_effnet import get_dataloader
import timm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import time
import seaborn as sns
import numpy as np
from torchsummary import summary

def train_model(model_name, data_dir, num_classes, num_epochs, batch_size, pretrained_path=None, learning_rate=0.0003, checkpoint_path='effnet_checkpoints'):
    os.makedirs(checkpoint_path, exist_ok=True)
    train_losses, val_losses, precisions, recalls = [], [], [], []
    best_f1 = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = get_dataloader(os.path.join(data_dir, 'train'), batch_size, shuffle=True, phase='train')
    val_loader = get_dataloader(os.path.join(data_dir, 'val'), batch_size, shuffle=False, phase='val')

    print(f"Training Data: {len(train_loader.dataset)} samples")
    print(f"Validation Data: {len(val_loader.dataset)} samples")

    if pretrained_path:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(pretrained_path))
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True
    if hasattr(model, 'global_pool'):
        for param in model.global_pool.parameters():
            param.requires_grad = True

    summary(model, (3, 224, 224), device='cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    tic = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        model.eval()
        val_loss = 0.0
        all_labels, all_preds = [], []

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validating", leave=False)
            for inputs, labels in val_progress:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                val_progress.set_postfix(val_loss=loss.item())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        accu = accuracy_score(all_labels, all_preds)
        baccu = balanced_accuracy_score(all_labels, all_preds)
        precisions.append(precision)
        recalls.append(recall)

        scheduler.step(f1)
        print(f"Validation Loss: {avg_val_loss:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_path = os.path.join(checkpoint_path, f'{model_name}_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(colored(f'Best F1: {best_f1} in epoch {epoch+1}', 'red'))

            print(f'Saved Best Model at epoch {epoch+1} with F1 score: {best_f1:.4f}')
            plt.figure()
            plt.plot(train_losses, label='Train Loss', color='#B0AACE')
            plt.plot(val_losses, label='Val Loss', color='#01AEA7')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(checkpoint_path, f'epoch{epoch+1}_loss_plot.png'))
            plt.close()

        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'{model_name}_latest.pth'))

    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(train_losses, label='Train Loss', color='#B0AACE')
    plt.plot(val_losses, label='Val Loss', color='#01AEA7')
    plt.legend()

    plt.subplot(122)
    plt.plot(precisions, label='Precision', color='#C96868')
    plt.plot(recalls, label='Recall', color='#7EACB5')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_path, 'final_metrics.png'))
    plt.close()

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    plot_confusion_matrix(cm, ['Cancer(0)', 'No_Cancer(1)'])
    plt.savefig(os.path.join(checkpoint_path, 'confusion_matrix.png'))
    plt.close()

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
    fig.savefig(os.path.join(checkpoint_path, f'{model_name}_FIN_HEATMAP_confusion_matrix.png'))

    print(f"\nTraining completed in {time.perf_counter()-tic:.1f}s")
    print("\n--- Overall Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accu:.4f}")
    print(f"Balanced Accuracy: {baccu:.4f}")

def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max()/2.
    for i,j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i,j], fmt), 
                ha="center", va="center",
                color="white" if cm[i,j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EfficientNet')
    parser.add_argument('--model_name', default='efficientnet_b0', help='Timm model name')
    parser.add_argument('--data_dir', required=True, help='Dataset path')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--checkpoint_path', default='effnet_checkpoints')
    parser.add_argument('--pretrained_path', default=None)

    args = parser.parse_args()
    train_model(args.model_name, args.data_dir, args.num_classes, 
                args.num_epochs, args.batch_size, args.pretrained_path,
                args.learning_rate, args.checkpoint_path)