
# Enhanced Batch Visualization (real_world and default)
dataset_path = "/content/dataset/images/images/"


image_paths = glob.glob(dataset_path + "/**/*.png", recursive=True)

plt.figure(figsize=(15,10))

sample_images = random.sample(image_paths, 12)

for i, img_path in enumerate(sample_images):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    class_name = img_path.split("/")[-2]

    plt.subplot(3,4,i+1)
    plt.imshow(img)
    plt.title(class_name)
    plt.axis("off")

plt.show()
