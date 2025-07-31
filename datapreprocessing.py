import os
import cv2
from tqdm import tqdm

# Path to your dataset
dataset_path = r"C:\Users\rohit\OneDrive\Desktop\dog_emotions"

# Emotions (folder names)
emotions = ['happy', 'angry', 'relaxed', 'sad']

# Log file to keep track of processed images
log_file = 'cleaning_log.txt'

# Load already processed files
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        processed = set(f.read().splitlines())
else:
    processed = set()

# Clean and verify dataset images
for emotion in emotions:
    emotion_dir = os.path.join(dataset_path, emotion)
    if not os.path.exists(emotion_dir):
        print(f"Directory {emotion_dir} not found, skipping...")
        continue

    files = os.listdir(emotion_dir)
    for img_name in tqdm(files, desc=f"Scanning {emotion}"):
        img_path = os.path.join(emotion_dir, img_name)

        if img_path in processed:
            continue  # Skip already processed

        try:
            img = cv2.imread(img_path)
            if img is None or img.shape[0] < 20 or img.shape[1] < 20:
                os.remove(img_path)
                print(f"Removed corrupt/invalid image: {img_path}")
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            try:
                os.remove(img_path)
            except:
                pass

        # Log the processed image path
        with open(log_file, 'a') as log:
            log.write(img_path + '\n')

print("✅ Dataset cleaning complete.")
import os

# Set the dataset path
dataset_path = r"C:\Users\rohit\OneDrive\Desktop\dog_emotions"

# List of emotion categories
emotions = ['happy', 'angry', 'relaxed', 'sad']

# Count images in each category
for emotion in emotions:
    folder_path = os.path.join(dataset_path, emotion)
    
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        image_count = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"{emotion.capitalize()}: {image_count} images")
    else:
        print(f"Folder not found: {folder_path}")
