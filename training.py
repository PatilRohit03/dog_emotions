import os
import time
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Label, Button
import numpy as np
import random
from PIL import Image, ImageTk, ImageDraw, ImageFont

# CONFIG
DATASET_DIR = 'dataset'
MODEL_PATH = 'model.pth'
BATCH_SIZE = 64
EPOCHS = 10
IMG_SIZE = 224
YOLO_CFG = r"C:\Users\rohit\OneDrive\Desktop\Yolo\yolov4.cfg.txt"
YOLO_WEIGHTS = r"C:\Users\rohit\OneDrive\Desktop\Yolo\yolov4.weights"
YOLO_NAMES = r"C:\Users\rohit\OneDrive\Desktop\Yolo\coco.names.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def load_yolo():
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    with open(YOLO_NAMES, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes

def detect_dogs_yolo(image):
    net, output_layers, classes = load_yolo()
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences = [], []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "dog":
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    dog_boxes = [boxes[i] for i in indices.flatten()]
    return dog_boxes

def is_image_quality_good(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    return brightness > 40 and blur > 100

def clean_and_save(image, label):
    cleaned = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    if is_image_quality_good(cleaned):
        save_path = os.path.join(DATASET_DIR, label)
        os.makedirs(save_path, exist_ok=True)
        filename = f"{label}_{int(time.time())}.jpg"
        cv2.imwrite(os.path.join(save_path, filename), cleaned)
# --- DATA LOADER ---
def load_data():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
    class_names = dataset.classes
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    return train_loader, val_loader, class_names

# --- MODEL ---
def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# --- TRAINING ---
def train_model(train_loader, val_loader, class_names, epochs=EPOCHS, light=False):
    model = build_model(len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(2 if light else epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, MODEL_PATH)
    return model

# --- LOAD MODEL ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet. Click 'Train Model Again' first.")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint['class_names']
    model = build_model(len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device), class_names

# --- ACCURACY CHECK ---
def check_model_accuracy():
    try:
        train_loader, val_loader, class_names = load_data()
        model, _ = load_model()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = (correct / total) * 100
        messagebox.showinfo("Model Accuracy", f"Validation Accuracy: {accuracy:.2f}%")
    except Exception as e:
        messagebox.showerror("Error", str(e))
# --- EMOJI MAPPING ---
EMOJI_MAP = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "relaxed": "😌"
}

# --- IMAGE QUALITY CHECK ---
def is_image_quality_good(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    return brightness > 40 and blur > 100

def clean_and_save(image, label):
    cleaned = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    if is_image_quality_good(cleaned):
        save_path = os.path.join(DATASET_DIR, label)
        os.makedirs(save_path, exist_ok=True)
        filename = f"{label}_{int(time.time())}.jpg"
        cv2.imwrite(os.path.join(save_path, filename), cleaned)

# --- YOLO SETUP ---
def load_yolo():
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    with open(YOLO_NAMES, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes

def detect_dogs_yolo(image):
    net, output_layers, classes = load_yolo()
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "dog":
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    dog_boxes = [boxes[i] for i in indices.flatten()]
    return dog_boxes

# --- PREDICTION & AUTO RETRAIN ---
def predict_and_handle(image, model, class_names):
    dog_boxes = detect_dogs_yolo(image)
    emoji_map = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "relaxed": "😌"
    }

    for (x, y, w, h) in dog_boxes:
        dog_crop = image[y:y+h, x:x+w]
        if dog_crop.size == 0:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(dog_crop, cv2.COLOR_BGR2RGB)).resize((IMG_SIZE, IMG_SIZE))
        tensor_img = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor_img)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred_idx = torch.max(probs, 0)
            label = class_names[pred_idx.item()]
            confidence = conf.item() * 100
            emoji = emoji_map.get(label, "")
            text = f"{confidence:.1f}% {label} {emoji}"

        # Draw with PIL to show emoji
        pil_frame = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_frame)

        # Use emoji-supporting font
        try:
            font_path = "C:\\Windows\\Fonts\\seguiemj.ttf"  # Change if needed
            font = ImageFont.truetype(font_path, 24)
        except:
            font = ImageFont.load_default()

        draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
        draw.text((x, y - 30), text, font=font, fill="green")

        image = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

        if confidence > 70:
            clean_and_save(dog_crop, label)
            train_loader, val_loader, class_names = load_data()
            train_model(train_loader, val_loader, class_names, light=True)
        else:
            response = messagebox.askyesno("Low Confidence", f"Predicted: {label} ({confidence:.1f}%). Is this correct?")
            if response:
                clean_and_save(dog_crop, label)
                train_loader, val_loader, class_names = load_data()
                train_model(train_loader, val_loader, class_names, light=True)

    return image

# --- GUI ---
def launch_gui():
    try:
        model, class_names = load_model()
    except Exception as e:
        model, class_names = None, None
        messagebox.showwarning("Warning", str(e))

    def train():
        train_loader, val_loader, class_names = load_data()
        train_model(train_loader, val_loader, class_names)
        messagebox.showinfo("Info", "✅ Model Trained")

    def upload_photo():
        nonlocal model, class_names
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            img = predict_and_handle(img, model, class_names)
            cv2.imshow("Prediction", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def use_camera():
        nonlocal model, class_names
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = predict_and_handle(frame.copy(), model, class_names)
            cv2.imshow("Live Prediction", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def preview_augmented_images():
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(),
            transforms.ToTensor()
        ])
        dataset = datasets.ImageFolder(DATASET_DIR)
        idx = random.randint(0, len(dataset) - 1)
        img_path, label = dataset.samples[idx]
        img = Image.open(img_path)
        transformed = transform(img)
        transformed_img = transforms.ToPILImage()(transformed)
        transformed_img.show()

    root = tk.Tk()
    root.title("🐶 Dog Emotion Detection")
    tk.Label(root, text="Dog Emotion Detection System", font=("Arial", 16)).pack(pady=10)

    Button(root, text="Train Model Again", command=train, width=30, height=2).pack(pady=5)
    Button(root, text="Upload a Photo", command=upload_photo, width=30, height=2).pack(pady=5)
    Button(root, text="Use Live Camera", command=use_camera, width=30, height=2).pack(pady=5)
    Button(root, text="Preview Augmented Images", command=preview_augmented_images, width=30, height=2).pack(pady=5)
    Button(root, text="Check Model Accuracy", command=check_model_accuracy, width=30, height=2).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    launch_gui()
