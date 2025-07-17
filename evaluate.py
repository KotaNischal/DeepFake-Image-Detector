import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from model import ModifiedMobileNetV2

def plot_class_losses(s0, s1):
    classes = ['Real (Class 0)', 'Fake (Class 1)']
    losses = [s0, s1]
    plt.figure(figsize=(6, 6))
    bars = plt.bar(classes, losses, color=['skyblue', 'salmon'])
    plt.title('Average Loss per Class')
    plt.ylabel('Loss')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', ha='center', va='bottom')
    plt.show()

def evaluate_model_with_class_scores(model_path, test_loader, criterion, device):
    model = ModifiedMobileNetV2().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    real_inputs, real_labels = [], []
    fake_inputs, fake_labels = [], []

    for inputs, labels in test_loader:
        for i in range(len(labels)):
            (real_inputs if labels[i] == 0 else fake_inputs).append(inputs[i])
            (real_labels if labels[i] == 0 else fake_labels).append(labels[i])

    def compute_avg_loss(inputs_list, labels_list):
        total_loss, total_samples = 0.0, 0
        for i in range(0, len(inputs_list), 32):
            batch_inputs = torch.stack(inputs_list[i:i + 32]).to(device)
            batch_labels = torch.tensor(labels_list[i:i + 32]).to(device)
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item() * batch_inputs.size(0)
            total_samples += batch_inputs.size(0)
        return total_loss / total_samples

    s0 = compute_avg_loss(real_inputs, real_labels)
    s1 = compute_avg_loss(fake_inputs, fake_labels)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    plot_class_losses(s0, s1)
    return s0, s1, acc, precision, recall, f1
