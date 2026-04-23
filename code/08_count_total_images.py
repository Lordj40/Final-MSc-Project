# Printing total images
images = glob.glob(dataset_path + "/**/*.png", recursive=True)
print("Total images:", len(images))
