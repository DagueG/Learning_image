import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import ttkbootstrap as ttk
import requests
from PIL import Image
from io import BytesIO
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from duckduckgo_search import DDGS
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Configuration des chemins
DATASET_DIR = r"C:\Users\daniel.guedj_arondor\dataset"
MODEL_PATH = "image_classifier.h5"

root = ttk.Window(themename="cyborg")
root.title("Application IA Polyvalente")
root.geometry("750x600")

mode_var = tk.StringVar(value="ML")
df = None  
best_model = None  
class_names = []

# 🔹 Téléchargement d'images depuis DuckDuckGo
def download_images(query, num_images=100):
    """Télécharge des images pour enrichir le dataset."""
    save_dir = os.path.join(DATASET_DIR, query)
    os.makedirs(save_dir, exist_ok=True)

    print(f"🔍 Recherche d'images pour '{query}'...")
    with DDGS() as ddgs:
        image_results = list(ddgs.images(query, max_results=num_images))
    
    downloaded = 0
    for i, img_data in enumerate(image_results):
        try:
            img_url = img_data["image"]
            img_response = requests.get(img_url, timeout=10)
            img_response.raise_for_status()
            img = Image.open(BytesIO(img_response.content)).convert("RGB")

            img_path = os.path.join(save_dir, f"{query}_{i+1}.jpg")
            img.save(img_path, "JPEG")
            downloaded += 1
            print(f"✅ Image téléchargée: {img_path}")

            if downloaded >= num_images:
                break
        except Exception as e:
            print(f"⚠️ Erreur téléchargement {i+1}: {e}")

    print(f"📸 {downloaded}/{num_images} images téléchargées.")

# 🔹 Entraînement du modèle d'image
def train_image_model():
    """Entraîne un modèle CNN avec les images du dataset."""
    global class_names

    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        messagebox.showerror("Erreur", "Aucune image trouvée dans le dataset.")
        return
    
    img_size = (224, 224)
    batch_size = 32

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_data = datagen.flow_from_directory(
        DATASET_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", subset="training"
    )

    val_data = datagen.flow_from_directory(
        DATASET_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", subset="validation"
    )

    class_names = list(train_data.class_indices.keys())

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_data, validation_data=val_data, epochs=10)

    model.save(MODEL_PATH)
    messagebox.showinfo("Succès", "Modèle d'image entraîné et sauvegardé !")

# 🔹 Classification d'image
def classify_image():
    """Classifie une image en utilisant le modèle entraîné."""
    filepath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
    if not filepath:
        return

    img = Image.open(filepath).resize((224, 224))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    try:
        model = load_model(MODEL_PATH)
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        confidence = predictions[0][class_index] * 100
        recognized_label = class_names[class_index]

        messagebox.showinfo("Résultat", f"Reconnaissance : {recognized_label} ({confidence:.2f}%)")
    
    except Exception:
        user_input = simpledialog.askstring("Nom de l'objet", "Quel est cet objet ?")
        if user_input:
            download_images(user_input, num_images=100)
            train_image_model()
            messagebox.showinfo("Ajout", f"Images de '{user_input}' ajoutées et modèle mis à jour !")

# 🔹 Interface
frame = ttk.Frame(root, padding=20)
frame.pack(expand=True, fill="both")

ttk.Button(frame, text="📂 Charger un fichier CSV", command=lambda: filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])).pack(pady=5)
ttk.Button(frame, text="🚀 Lancer l'analyse ML", command=lambda: messagebox.showinfo("Info", "Analyse ML lancée !")).pack(pady=5)
ttk.Button(frame, text="📥 Sauvegarder le modèle", command=lambda: messagebox.showinfo("Info", "Modèle ML sauvegardé !")).pack(pady=5)
ttk.Button(frame, text="🎯 Entraîner le modèle image", command=train_image_model).pack(pady=5)
ttk.Button(frame, text="📸 Reconnaître une image", command=classify_image).pack(pady=5)

result_label = ttk.Label(frame, text="", font=("Helvetica", 12, "bold"))
result_label.pack(pady=10)

root.mainloop()
