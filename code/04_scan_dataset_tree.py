
dataset_path = "/content/dataset"

for root, dirs, files in os.walk(dataset_path):
    print(root, "→", len(files), "files")
