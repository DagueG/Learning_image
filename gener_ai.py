import os
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import requests
from PIL import Image, ImageTk
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from duckduckgo_search import DDGS
import shutil
import matplotlib.pyplot as plt
from googletrans import Translator

# Configuration des chemins
DATASET_DIR = r"C:\Users\daniel.guedj_arondor\dataset"
MODEL_PATH = "image_classifier.h5"
CLASS_NAMES_PATH = "classes.json"
DOWNLOADED_IMAGES_PATH = "downloaded_images.json"

root = ttk.Window(themename="cyborg")
root.title("Application IA Polyvalente")
root.geometry("800x750")

class_names = []
last_image_path = None  
last_probabilities = None  
translator = Translator()

# 🔹 Sauvegarder et charger les classes
def save_class_names():
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)


def load_class_names():
    global class_names
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
    else:
        class_names = []

# 🔹 Charger la liste des images déjà téléchargées
def load_downloaded_images():
    if os.path.exists(DOWNLOADED_IMAGES_PATH):
        with open(DOWNLOADED_IMAGES_PATH, "r") as f:
            return set(json.load(f))
    return set()


# 🔹 Sauvegarder les nouvelles images téléchargées
def save_downloaded_images(urls):
    existing_urls = load_downloaded_images()
    updated_urls = existing_urls.union(urls)
    with open(DOWNLOADED_IMAGES_PATH, "w") as f:
        json.dump(list(updated_urls), f)


# 🔹 Traduire automatiquement les noms de classes en anglais avec confirmation si nécessaire
def translate_to_english(label):
    try:
        translation = translator.translate(label, dest='en')
        translated_text = translation.text.lower().strip()

        # 🔹 Si la traduction est identique, exécuter sans confirmation
        if translated_text == label.lower().strip():
            return translated_text

        # 🔹 Sinon, demander confirmation à l'utilisateur
        confirm = messagebox.askyesno("Confirmation de la traduction", 
            f"Le mot '{label}' a été traduit en '{translated_text}'. Voulez-vous l'utiliser ?")

        if confirm:
            return translated_text
        else:
            # L'utilisateur entre manuellement le mot correct
            corrected_text = simpledialog.askstring("Correction", "Entrez le mot correct :")
            return corrected_text.lower().strip() if corrected_text else label.lower().strip()

    except Exception as e:
        print(f"Erreur de traduction : {e}")
        return label.lower().strip()


# 🔹 Ajouter l'image mal reconnue avec correction des variantes linguistiques
def correct_recognition():
    def ask_correct_label():
        correct_label = simpledialog.askstring("Correction", "Quel est le bon objet ?")
        if correct_label:
            translated_label = translate_to_english(correct_label)
            threading.Thread(target=process_correction, args=(translated_label,)).start()

    def process_correction(correct_label):
        global last_image_path

        save_dir = os.path.join(DATASET_DIR, correct_label)
        os.makedirs(save_dir, exist_ok=True)

        # Copier l'image mal reconnue dans la bonne catégorie
        image_name = os.path.basename(last_image_path)
        new_image_path = os.path.join(save_dir, image_name)
        shutil.copy(last_image_path, new_image_path)  

        update_progress(f"L'image a été copiée dans '{correct_label}'. Téléchargement de nouvelles images...")

        # Télécharger des images sans doublons et entraîner le modèle
        download_images(correct_label, num_images=100)
        train_image_model()

    root.after(0, ask_correct_label)


# 🔹 Télécharger 100 nouvelles images sans doublons avec affichage du progrès (multithreaded)
def download_images(query, num_images=100):
    def task():
        save_dir = os.path.join(DATASET_DIR, query)
        os.makedirs(save_dir, exist_ok=True)

        existing_urls = load_downloaded_images()
        new_urls = set()

        with DDGS() as ddgs:
            image_results = list(ddgs.images(query, max_results=num_images * 3))

        downloaded = 0
        update_progress(f"Téléchargement des images pour '{query}': 0%")

        for i, img_data in enumerate(image_results):
            img_url = img_data["image"]
            if img_url in existing_urls:
                continue  

            try:
                img_response = requests.get(img_url, timeout=10)
                img_response.raise_for_status()
                img = Image.open(BytesIO(img_response.content)).convert("RGB")

                img_path = os.path.join(save_dir, f"{query}_{i+1}.jpg")
                img.save(img_path, "JPEG")
                new_urls.add(img_url)
                downloaded += 1

                percent_complete = int((downloaded / num_images) * 100)
                update_progress(f"Téléchargement des images pour '{query}': {percent_complete}%")

                if downloaded >= num_images:
                    break
            except Exception:
                pass

        save_downloaded_images(new_urls)
        update_progress(f"Téléchargement terminé : {downloaded}/{num_images} images pour '{query}'")

    threading.Thread(target=task).start()


# 🔹 Entraînement du modèle avec affichage du progrès (multithreaded)
def train_image_model():
    def task():
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
        save_class_names()

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

        epochs = 10
        for epoch in range(epochs):
            model.fit(train_data, validation_data=val_data, epochs=1, verbose=0)
            percent_complete = int(((epoch + 1) / epochs) * 100)
            update_progress(f"Entraînement du modèle : {percent_complete}%")

        model.save(MODEL_PATH)
        update_progress("Entraînement du modèle terminé !")

    threading.Thread(target=task).start()


# 🔹 Mise à jour de l'interface pour le progrès
def update_progress(message):
    progress_label.config(text=message)
    root.update_idletasks()


# 🔹 Chargement et affichage de l’image sélectionnée
def load_and_display_image():
    global last_image_path
    filepath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
    if not filepath:
        return
    
    last_image_path = filepath
    img = Image.open(filepath).resize((300, 300))  
    img = ImageTk.PhotoImage(img)
    
    image_label.config(image=img)
    image_label.image = img  


# 🔹 Affichage des statistiques et gestion des classes
def show_statistics():
    existing_classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    global class_names
    class_names = existing_classes
    save_class_names()

    class_counts = {
        cls: len(os.listdir(os.path.join(DATASET_DIR, cls)))
        for cls in class_names
    }

    stats_window = tk.Toplevel(root)
    stats_window.title("Gestion des classes")
    stats_window.geometry("600x600")

    tk.Label(stats_window, text="Liste des classes :", font=("Arial", 14, "bold")).pack(pady=10)

    listbox_frame = tk.Frame(stats_window)
    listbox_frame.pack(expand=True, fill="both", padx=10, pady=10)

    scrollbar = tk.Scrollbar(listbox_frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(listbox_frame, width=50, height=20, yscrollcommand=scrollbar.set)
    for cls, count in class_counts.items():
        listbox.insert(tk.END, f"{cls} ({count} images)")
    listbox.pack(expand=True, fill="both")

    scrollbar.config(command=listbox.yview)

    # 🔹 Fonction pour afficher le graphe des classes
    def show_class_graph():
        plt.figure(figsize=(10, 6))
        plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
        plt.xlabel("Classes")
        plt.ylabel("Nombre d'images")
        plt.title("Répartition des images par classe")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # 🔹 Fonction pour renommer une classe
    def rename_class():
        selection = listbox.curselection()
        if not selection:
            messagebox.showerror("Erreur", "Veuillez sélectionner une classe à renommer.")
            return

        old_class = list(class_counts.keys())[selection[0]]
        new_class = simpledialog.askstring("Renommer", f"Entrez le nouveau nom pour '{old_class}':")

        if new_class:
            new_class = translate_to_english(new_class)
            old_path = os.path.join(DATASET_DIR, old_class)
            new_path = os.path.join(DATASET_DIR, new_class)

            if os.path.exists(new_path):
                messagebox.showerror("Erreur", f"La classe '{new_class}' existe déjà.")
                return

            os.rename(old_path, new_path)
            messagebox.showinfo("Succès", f"Classe '{old_class}' renommée en '{new_class}'.")
            show_statistics()

    # 🔹 Fonction pour fusionner deux classes
    def merge_classes():
        selection = listbox.curselection()
        if not selection:
            messagebox.showerror("Erreur", "Veuillez sélectionner une classe à fusionner.")
            return

        source_class = list(class_counts.keys())[selection[0]]
        target_class = simpledialog.askstring("Fusionner", f"Dans quelle classe fusionner '{source_class}' ?")

        if target_class:
            target_class = translate_to_english(target_class)
            source_path = os.path.join(DATASET_DIR, source_class)
            target_path = os.path.join(DATASET_DIR, target_class)

            if not os.path.exists(target_path):
                os.makedirs(target_path)

            for filename in os.listdir(source_path):
                shutil.move(os.path.join(source_path, filename), os.path.join(target_path, filename))

            os.rmdir(source_path)
            messagebox.showinfo("Succès", f"Contenu de '{source_class}' fusionné dans '{target_class}'.")
            show_statistics()

    # 🔹 Fonction pour supprimer une classe
    def delete_class():
        selection = listbox.curselection()
        if not selection:
            messagebox.showerror("Erreur", "Veuillez sélectionner une classe à supprimer.")
            return

        class_to_delete = list(class_counts.keys())[selection[0]]
        confirm = messagebox.askyesno("Confirmation", f"Êtes-vous sûr de vouloir supprimer la classe '{class_to_delete}' ? Toutes les images seront définitivement perdues.")

        if confirm:
            class_path = os.path.join(DATASET_DIR, class_to_delete)
            shutil.rmtree(class_path)
            messagebox.showinfo("Succès", f"La classe '{class_to_delete}' a été supprimée avec succès.")
            show_statistics()

    # Boutons pour gérer les classes
    ttk.Button(stats_window, text="Renommer la classe", command=rename_class).pack(pady=5)
    ttk.Button(stats_window, text="Fusionner la classe", command=merge_classes).pack(pady=5)
    ttk.Button(stats_window, text="🗑️ Supprimer la classe", command=delete_class).pack(pady=5)
    ttk.Button(stats_window, text="📊 Voir le graphe des classes", command=show_class_graph).pack(pady=5)
    ttk.Button(stats_window, text="Fermer", command=stats_window.destroy).pack(pady=10)


# 🔹 Classification d'image avec affichage des probabilités
def classify_image():
    global last_image_path, last_probabilities
    if not os.path.exists(MODEL_PATH):
        response = messagebox.askyesno("Aucun modèle disponible", "Voulez-vous entraîner un modèle maintenant ?")
        if response:
            train_image_model()
        return

    if last_image_path is None:
        messagebox.showerror("Erreur", "Veuillez d'abord sélectionner une image.")
        return

    img = Image.open(last_image_path).resize((224, 224))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    try:
        model = load_model(MODEL_PATH)
        predictions = model.predict(img_array)[0]
        last_probabilities = predictions  

        class_index = np.argmax(predictions)
        confidence = predictions[class_index] * 100
        recognized_label = class_names[class_index]

        response = messagebox.askyesno("Résultat", f"Reconnaissance : {recognized_label} ({confidence:.2f}%)\nEst-ce correct ?")
        if not response:
            correct_recognition()

    except Exception as e:
        messagebox.showerror("Erreur", f"Problème de reconnaissance : {e}")


# Interface principale
frame = ttk.Frame(root, padding=20)
frame.pack(expand=True, fill="both")

image_label = tk.Label(frame)
image_label.pack(pady=10)

progress_label = tk.Label(frame, text="")
progress_label.pack(pady=5)

# 🔹 Boutons pour les fonctionnalités principales
ttk.Button(frame, text="📂 Sélectionner une image", command=load_and_display_image).pack(pady=5)
ttk.Button(frame, text="📸 Reconnaître l'image", command=classify_image).pack(pady=5)
ttk.Button(frame, text="📊 Gérer les classes", command=show_statistics).pack(pady=5)

# Chargement des noms de classes existants au démarrage
load_class_names()

# 🔹 Lancement de l'application
root.mainloop()
