# Checking for corrupt images
bad_images = []

for img_path in image_paths:
    try:
        img = cv2.imread(img_path)
        if img is None:
            bad_images.append(img_path)
    except:
        bad_images.append(img_path)

print("Corrupted images:", len(bad_images))
