# Saving the model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, "resnet50_best_epoch5_full.pth")
