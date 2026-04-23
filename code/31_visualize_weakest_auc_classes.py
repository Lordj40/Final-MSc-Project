
# Visualizing weakest classes

model.eval()
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(labels)

# Concatenate all batches
all_probs = torch.cat(all_probs).numpy()
all_labels = torch.cat(all_labels).numpy()

# Binarize labels for multi-class ROC
labels_binarized = label_binarize(all_labels, classes=np.arange(num_classes))

# Compute per-class ROC and AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(labels_binarized[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


macro_auc = np.mean(list(roc_auc.values()))
weights = np.sum(labels_binarized, axis=0) / labels_binarized.shape[0]
weighted_auc = np.sum([weights[i]*roc_auc[i] for i in range(num_classes)])

# Select 5 classes with lowest AUC
lowest5_classes = sorted(roc_auc, key=roc_auc.get)[:5]


plt.figure(figsize=(10, 8))

colors = ['r', 'g', 'b', 'c', 'm']
for idx, cls in enumerate(lowest5_classes):
    plt.plot(fpr[cls], tpr[cls], color=colors[idx],
             label=f'Class {cls} (AUC = {roc_auc[cls]:.2f})')


plt.plot([0,1], [0,1], 'k--', label='Random')
plt.title("Top 5 Lowest AUC Classes + Macro/Weighted ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend(loc='lower right')

plt.text(0.6, 0.2, f'Macro AUC = {macro_auc:.2f}\nWeighted AUC = {weighted_auc:.2f}',
         bbox=dict(facecolor='white', alpha=0.5))

plt.show()
