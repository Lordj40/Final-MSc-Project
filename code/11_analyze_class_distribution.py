
# Data distribution analysis
class_counts = Counter()

for img in image_paths:
    class_name = img.split("/")[-2]
    class_counts[class_name] += 1

print(class_counts)
