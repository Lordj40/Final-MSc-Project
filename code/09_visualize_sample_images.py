# Visualization of images
dataset_path = "/content/dataset/images/images/"

image_paths = glob.glob(dataset_path + "/**/*.png", recursive=True)
# Selecting random 9
sample_images = random.sample(image_paths, 9)
# Set up a 10x10 inch figure area for the plot
plt.figure(figsize=(10,10))
# Using OpenCV for reading and converting the images to RGB
for i, img_path in enumerate(sample_images):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(img_path.split("/")[-2])
    plt.axis("off")

plt.show()
