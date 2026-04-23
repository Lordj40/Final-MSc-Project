
# Dual-Axis Training Plot
# Metrics calculated earlier
train_losses = [350.96, 160.00, 107.80, 92.10, 74.04]
val_accuracies = [78.44, 81.78, 82.53, 82.67, 83.73]
epochs = range(1, len(train_losses)+1)

fig, ax1 = plt.subplots(figsize=(10,6))

# Plot training loss on left y-axis
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color=color)
ax1.plot(epochs, train_losses, marker='o', color=color, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

# Create second y-axis for validation accuracy
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Validation Accuracy (%)', color=color)
ax2.plot(epochs, val_accuracies, marker='o', color=color, label='Val Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

# Add title and legends
plt.title("Training Loss and Validation Accuracy")
fig.tight_layout()
plt.show()
