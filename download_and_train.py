import os
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np

CLASS_MAPPING = {
    'glioma': 'glioma',
    'meningioma': 'meningioma',
    'no-tumor': 'notumor',
    'pituitary': 'pituitary'
}

IMG_SIZE = (128, 128)
MODEL_FILENAME = "brain_tumor_multiclass_cnn.h5"

def download_file(item):
    url, out_path = item
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(data)
    except Exception as e:
        print(f"Failed {url}: {e}")

def prepare_dataset():
    print("Fetching dataset file lists from HuggingFace...")
    tasks = []
    
    # 75 train images per class, 20 test images per class
    for hf_cls, local_cls in CLASS_MAPPING.items():
        # Train
        train_url = f"https://huggingface.co/api/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor/tree/main/timri/train/{hf_cls}"
        req = urllib.request.Request(train_url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=15)
        files = json.loads(res.read().decode())
        jpg_files = [f['path'] for f in files if f.get('path', '').endswith('.jpg')][:75]
        for fpath in jpg_files:
            fname = os.path.basename(fpath)
            raw_url = f"https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor/resolve/main/{fpath}"
            tasks.append((raw_url, os.path.join("training", local_cls, fname)))
            
        # Test
        test_url = f"https://huggingface.co/api/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor/tree/main/timri/valid/{hf_cls}"
        req = urllib.request.Request(test_url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=15)
        vfiles = json.loads(res.read().decode())
        vjpg_files = [f['path'] for f in vfiles if f.get('path', '').endswith('.jpg')][:20]
        for fpath in vjpg_files:
            fname = os.path.basename(fpath)
            raw_url = f"https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor/resolve/main/{fpath}"
            tasks.append((raw_url, os.path.join("testing", local_cls, fname)))

    print(f"Total images to download: {len(tasks)}. Starting parallel download...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        list(executor.map(download_file, tasks))
    print("Dataset download complete!")

def train_and_save():
    from model import build_model
    model = build_model()
    
    print("Preparing datasets from disk...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "training",
        image_size=IMG_SIZE,
        batch_size=16,
        label_mode='categorical',
        shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "testing",
        image_size=IMG_SIZE,
        batch_size=16,
        label_mode='categorical'
    )
    
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    print("Training CNN model for 8 epochs...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=8)
    
    # Save model
    model.save(MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}!")
    
    # Verify with sample images
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    for sname in os.listdir("sample_images"):
        if not sname.endswith(('.png', '.jpg')):
            continue
        p = os.path.join("sample_images", sname)
        img = Image.open(p).convert("RGB").resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
        preds = model.predict(arr, verbose=0)[0]
        top_cls = classes[np.argmax(preds)]
        top_conf = np.max(preds) * 100
        print(f"Sample test: {sname} -> {top_cls} ({top_conf:.2f}%)")

if __name__ == "__main__":
    prepare_dataset()
    train_and_save()
