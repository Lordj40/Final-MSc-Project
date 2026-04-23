

# Comprehensive ROC/AUC Analysis
model.eval()

all_labels = []
all_probs = []


with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
         # convert to probabilities
        probs = torch.softmax(outputs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Concatenate all batches
all_probs = np.vstack(all_probs)
all_labels = np.concatenate(all_labels)

# Binarize labels for multi-class ROC
num_classes = all_probs.shape[1]
all_labels_bin = label_binarize(all_labels, classes=np.arange(num_classes))

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(12,8))

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc='lower right', fontsize='small', ncol=2)
plt.grid(True)
plt.show()

# Compute macro and weighted AUC
macro_auc = roc_auc_score(all_labels_bin, all_probs, average='macro')
weighted_auc = roc_auc_score(all_labels_bin, all_probs, average='weighted')

print(f"Macro AUC: {macro_auc:.4f}")
print(f"Weighted AUC: {weighted_auc:.4f}")
