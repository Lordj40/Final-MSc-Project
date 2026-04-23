# Checking if all images have the same dimensions (Height, Width, Channels)
sizes = []
# For 500 images
for img_path in random.sample(image_paths, 500):
    img = cv2.imread(img_path)
    sizes.append(img.shape)

print("Sample image sizes:")
print(set(sizes))
