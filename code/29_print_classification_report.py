
# Printing Classification Report
report = classification_report(
    all_labels,
    all_preds,
    target_names=dataset.classes
)

print(report)
