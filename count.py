import os
from collections import Counter

train_dir = 'data/train'
class_counts = {emotion: len(os.listdir(os.path.join(train_dir, emotion)))
                for emotion in os.listdir(train_dir)}
print("Image count per class:")
for emotion, count in class_counts.items():
    print(f"{emotion}: {count}")
