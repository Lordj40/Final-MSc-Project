
# Training model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


train_losses = []
val_accuracies = []


epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss)


    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.2f} - Val Acc: {val_acc:.2f}%")

# Plot Training Loss
plt.figure(figsize=(10,4))
plt.plot(range(1, epochs+1), train_losses, marker='o', label='Train Loss')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

# Plot Validation Accuracy
plt.figure(figsize=(10,4))
plt.plot(range(1, epochs+1), val_accuracies, marker='o', color='orange', label='Validation Accuracy')
plt.title("Validation Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()
plt.show()
