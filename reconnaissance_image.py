import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sqlite3
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
import pickle

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Application de Reconnaissance Faciale")
        self.root.geometry("1200x750")
        
        # Charger le modèle de détection de visages
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialiser le recognizer LBPH
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer_trained = False
        
        # Initialisation de la base de données
        self.init_database()
        
        # Variables
        self.current_image_path = None
        self.current_frame = None
        self.camera = None
        self.is_camera_on = False
        
        # Création de l'interface
        self.create_widgets()
        self.load_known_faces()
        
    def init_database(self):
        """Initialise la base de données si elle n'existe pas"""
        self.conn = sqlite3.connect('face_recognition.db')
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS personnes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                matricule TEXT UNIQUE NOT NULL,
                nom TEXT NOT NULL,
                prenom TEXT NOT NULL,
                age INTEGER,
                email TEXT,
                telephone TEXT,
                face_data BLOB NOT NULL
            )
        ''')
        self.conn.commit()
        
    def create_widgets(self):
        """Crée l'interface graphique"""
        # Notebook pour les onglets
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Onglet 1: Enregistrement
        self.tab_register = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_register, text="📝 Enregistrement")
        self.create_register_tab()
        
        # Onglet 2: Gestion
        self.tab_manage = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_manage, text="📋 Gestion")
        self.create_manage_tab()
        
        # Onglet 3: Reconnaissance
        self.tab_recognition = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_recognition, text="🎥 Reconnaissance")
        self.create_recognition_tab()
        
    def create_register_tab(self):
        """Crée l'onglet d'enregistrement"""
        # Frame pour l'image
        frame_image = ttk.LabelFrame(self.tab_register, text="Photo")
        frame_image.pack(side='left', padx=10, pady=10, fill='both', expand=True)
        
        self.label_image = tk.Label(frame_image, text="Aucune image chargée", bg='#2c3e50', fg='white', 
                                     width=50, height=25, font=('Arial', 12))
        self.label_image.pack(padx=10, pady=10, fill='both', expand=True)
        
        btn_frame = ttk.Frame(frame_image)
        btn_frame.pack(pady=10)
        
        btn_load = ttk.Button(btn_frame, text="📁 Charger une image", command=self.load_image)
        btn_load.pack(side='left', padx=5)
        
        btn_capture = ttk.Button(btn_frame, text="📷 Capturer", command=self.open_capture_window)
        btn_capture.pack(side='left', padx=5)
        
        # Frame pour les informations
        frame_info = ttk.LabelFrame(self.tab_register, text="Informations personnelles")
        frame_info.pack(side='right', padx=10, pady=10, fill='both')
        
        # Champs de saisie
        fields = [
            ("Matricule:", "matricule"),
            ("Nom:", "nom"),
            ("Prénom:", "prenom"),
            ("Âge:", "age"),
            ("Email:", "email"),
            ("Téléphone:", "telephone")
        ]
        
        self.entries = {}
        for idx, (label_text, field_name) in enumerate(fields):
            ttk.Label(frame_info, text=label_text, font=('Arial', 10)).grid(
                row=idx, column=0, sticky='w', padx=10, pady=8
            )
            entry = ttk.Entry(frame_info, width=35, font=('Arial', 10))
            entry.grid(row=idx, column=1, padx=10, pady=8)
            self.entries[field_name] = entry
        
        # Bouton d'enregistrement
        btn_save = ttk.Button(frame_info, text="💾 Enregistrer la personne", command=self.save_person)
        btn_save.grid(row=len(fields), column=0, columnspan=2, pady=20)
        
    def create_manage_tab(self):
        """Crée l'onglet de gestion"""
        # Frame pour la liste
        frame_list = ttk.Frame(self.tab_manage)
        frame_list.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Treeview
        columns = ('Matricule', 'Nom', 'Prénom', 'Âge', 'Email', 'Téléphone')
        self.tree = ttk.Treeview(frame_list, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=180)
        
        self.tree.pack(side='left', fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame_list, orient='vertical', command=self.tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Frame pour les boutons
        frame_buttons = ttk.Frame(self.tab_manage)
        frame_buttons.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(frame_buttons, text="🔄 Rafraîchir", command=self.refresh_list).pack(side='left', padx=5)
        ttk.Button(frame_buttons, text="✏️ Modifier", command=self.modify_person).pack(side='left', padx=5)
        ttk.Button(frame_buttons, text="🗑️ Supprimer", command=self.delete_person).pack(side='left', padx=5)
        
        self.refresh_list()
        
    def create_recognition_tab(self):
        """Crée l'onglet de reconnaissance"""
        # Canvas pour permettre le défilement
        canvas = tk.Canvas(self.tab_recognition)
        canvas.pack(side='left', fill='both', expand=True)
        
        # Scrollbars
        scrollbar_v = ttk.Scrollbar(self.tab_recognition, orient='vertical', command=canvas.yview)
        scrollbar_v.pack(side='right', fill='y')
        
        scrollbar_h = ttk.Scrollbar(self.tab_recognition, orient='horizontal', command=canvas.xview)
        scrollbar_h.pack(side='bottom', fill='x')
        
        canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        # Frame principal dans le canvas
        main_frame = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=main_frame, anchor='nw')
        
        # Frame pour la caméra/image
        frame_camera = ttk.LabelFrame(main_frame, text="Reconnaissance")
        frame_camera.pack(side='left', padx=10, pady=10, fill='both', expand=True)
        
        self.label_camera = tk.Label(frame_camera, text="Caméra éteinte / Aucune image", bg='black', fg='white',
                                      width=60, height=28, font=('Arial', 14, 'bold'))
        self.label_camera.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Frame pour les boutons caméra
        btn_frame_camera = ttk.LabelFrame(frame_camera, text="Reconnaissance par Webcam")
        btn_frame_camera.pack(pady=5, padx=10, fill='x')
        
        btn_start = ttk.Button(btn_frame_camera, text="▶️ Démarrer la caméra", command=self.start_recognition)
        btn_start.pack(side='left', padx=5, pady=5)
        
        btn_stop = ttk.Button(btn_frame_camera, text="⏹️ Arrêter la caméra", command=self.stop_recognition)
        btn_stop.pack(side='left', padx=5, pady=5)
        
        # Frame pour les boutons image
        btn_frame_image = ttk.LabelFrame(frame_camera, text="Reconnaissance par Image")
        btn_frame_image.pack(pady=5, padx=10, fill='x')
        
        btn_load_recognize = ttk.Button(btn_frame_image, text="📁 Charger et reconnaître une image", 
                                        command=self.recognize_from_image)
        btn_load_recognize.pack(padx=5, pady=5)
        
        # Frame pour les résultats
        frame_results = ttk.LabelFrame(main_frame, text="Historique des reconnaissances")
        frame_results.pack(side='right', padx=10, pady=10, fill='both')
        
        self.text_results = tk.Text(frame_results, width=45, height=30, font=('Courier', 9))
        self.text_results.pack(side='left', padx=10, pady=10, fill='both', expand=True)
        
        scrollbar_text = ttk.Scrollbar(frame_results, orient='vertical', command=self.text_results.yview)
        scrollbar_text.pack(side='right', fill='y')
        self.text_results.configure(yscrollcommand=scrollbar_text.set)
        
        # Configurer le canvas pour qu'il s'adapte au contenu
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        main_frame.bind('<Configure>', configure_scroll_region)
        
        # Permettre le défilement avec la molette de la souris
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
    def open_capture_window(self):
        """Ouvre une fenêtre pour capturer une photo depuis la webcam"""
        capture_window = tk.Toplevel(self.root)
        capture_window.title("Capture Webcam")
        capture_window.geometry("800x650")
        
        # Label pour afficher le flux vidéo
        video_label = tk.Label(capture_window, bg='black')
        video_label.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Ouvrir la webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            messagebox.showerror("Erreur", "Impossible d'accéder à la webcam")
            capture_window.destroy()
            return
        
        def update_frame():
            ret, frame = cap.read()
            if ret:
                # Redimensionner pour l'affichage
                display_frame = cv2.resize(frame, (780, 585))
                
                # Détecter les visages
                gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                # Dessiner des rectangles autour des visages
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Visage detecte", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convertir pour Tkinter
                img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                photo = ImageTk.PhotoImage(img_pil)
                
                video_label.configure(image=photo)
                video_label.image = photo
                
                # Stocker le frame original pour la capture
                capture_window.current_frame = frame
            
            if cap.isOpened():
                capture_window.after(30, update_frame)
        
        def capture_photo():
            if hasattr(capture_window, 'current_frame'):
                # Sauvegarder l'image
                temp_path = "temp_capture.jpg"
                cv2.imwrite(temp_path, capture_window.current_frame)
                self.current_image_path = temp_path
                self.display_image(temp_path)
                
                # Fermer la fenêtre de capture
                cap.release()
                capture_window.destroy()
                messagebox.showinfo("Succès", "Photo capturée avec succès!")
        
        def close_capture():
            cap.release()
            capture_window.destroy()
        
        # Boutons
        btn_frame = ttk.Frame(capture_window)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="📸 Capturer cette photo", 
                  command=capture_photo).pack(side='left', padx=10)
        ttk.Button(btn_frame, text="❌ Annuler", 
                  command=close_capture).pack(side='left', padx=10)
        
        # Démarrer l'affichage
        update_frame()
        
        # Gérer la fermeture de la fenêtre
        capture_window.protocol("WM_DELETE_WINDOW", close_capture)
        
    def load_image(self):
        """Charge une image depuis le disque"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            
    def display_image(self, path):
        """Affiche une image dans le label"""
        img = cv2.imread(path)
        if img is None:
            return
            
        # Redimensionner pour l'affichage
        h, w = img.shape[:2]
        max_size = 500
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        
        img_resized = cv2.resize(img, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(img_pil)
        
        self.label_image.configure(image=photo, text="")
        self.label_image.image = photo
        
    def detect_face(self, image_path):
        """Détecte et extrait le visage d'une image"""
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) == 0:
            return None
        
        # Prendre le plus grand visage détecté
        faces_sorted = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces_sorted[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Redimensionner à une taille standard
        face_roi = cv2.resize(face_roi, (200, 200))
        
        return face_roi
        
    def save_person(self):
        """Enregistre une personne dans la base de données"""
        if not self.current_image_path:
            messagebox.showerror("Erreur", "Veuillez charger ou capturer une image")
            return
            
        matricule = self.entries['matricule'].get().strip()
        nom = self.entries['nom'].get().strip()
        prenom = self.entries['prenom'].get().strip()
        age = self.entries['age'].get().strip()
        email = self.entries['email'].get().strip()
        telephone = self.entries['telephone'].get().strip()
        
        if not all([matricule, nom, prenom]):
            messagebox.showerror("Erreur", "Matricule, nom et prénom sont obligatoires")
            return
            
        try:
            # Détecter le visage
            face_data = self.detect_face(self.current_image_path)
            
            if face_data is None:
                messagebox.showerror("Erreur", "Aucun visage détecté dans l'image.\nAssurez-vous que le visage est bien visible.")
                return
            
            # Convertir en bytes pour la base de données
            face_blob = pickle.dumps(face_data)
            
            # Insérer dans la base de données
            self.cursor.execute('''
                INSERT INTO personnes (matricule, nom, prenom, age, email, telephone, face_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (matricule, nom, prenom, age or None, email, telephone, face_blob))
            
            self.conn.commit()
            messagebox.showinfo("Succès", f"Personne {prenom} {nom} enregistrée avec succès!")
            
            # Réinitialiser les champs
            for entry in self.entries.values():
                entry.delete(0, tk.END)
            self.label_image.configure(image='', text="Aucune image chargée")
            self.current_image_path = None
            
            # Recharger les visages connus et réentraîner
            self.load_known_faces()
            
        except sqlite3.IntegrityError:
            messagebox.showerror("Erreur", "Ce matricule existe déjà dans la base de données")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'enregistrement: {str(e)}")
            
    def load_known_faces(self):
        """Charge tous les visages connus depuis la base de données et entraîne le modèle"""
        self.cursor.execute('SELECT id, matricule, nom, prenom, face_data FROM personnes')
        rows = self.cursor.fetchall()
        
        if len(rows) == 0:
            self.recognizer_trained = False
            return
        
        faces = []
        labels = []
        self.person_mapping = {}
        
        for row in rows:
            person_id, matricule, nom, prenom, face_blob = row
            face_data = pickle.loads(face_blob)
            
            faces.append(face_data)
            labels.append(person_id)
            
            self.person_mapping[person_id] = {
                'matricule': matricule,
                'nom': nom,
                'prenom': prenom
            }
        
        # Entraîner le recognizer
        try:
            self.face_recognizer.train(faces, np.array(labels))
            self.recognizer_trained = True
            print(f"✓ Modèle entraîné avec {len(faces)} visage(s)")
        except Exception as e:
            print(f"✗ Erreur lors de l'entraînement: {e}")
            self.recognizer_trained = False
            
    def refresh_list(self):
        """Rafraîchit la liste des personnes"""
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        self.cursor.execute('SELECT matricule, nom, prenom, age, email, telephone FROM personnes')
        rows = self.cursor.fetchall()
        
        for row in rows:
            self.tree.insert('', 'end', values=row)
            
    def modify_person(self):
        """Modifie les informations d'une personne"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Attention", "Veuillez sélectionner une personne")
            return
            
        item = self.tree.item(selected[0])
        values = item['values']
        matricule = values[0]
        
        # Fenêtre de modification
        modify_window = tk.Toplevel(self.root)
        modify_window.title("Modifier les informations")
        modify_window.geometry("450x350")
        
        fields = [
            ("Nom:", values[1]),
            ("Prénom:", values[2]),
            ("Âge:", values[3]),
            ("Email:", values[4]),
            ("Téléphone:", values[5])
        ]
        
        entries = []
        for idx, (label_text, value) in enumerate(fields):
            ttk.Label(modify_window, text=label_text).grid(row=idx, column=0, padx=10, pady=10, sticky='w')
            entry = ttk.Entry(modify_window, width=35)
            entry.insert(0, value)
            entry.grid(row=idx, column=1, padx=10, pady=10)
            entries.append(entry)
        
        def save_modifications():
            try:
                self.cursor.execute('''
                    UPDATE personnes 
                    SET nom=?, prenom=?, age=?, email=?, telephone=?
                    WHERE matricule=?
                ''', (entries[0].get(), entries[1].get(), entries[2].get(), 
                      entries[3].get(), entries[4].get(), matricule))
                self.conn.commit()
                messagebox.showinfo("Succès", "Modifications enregistrées avec succès")
                modify_window.destroy()
                self.refresh_list()
                self.load_known_faces()
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur: {str(e)}")
                
        ttk.Button(modify_window, text="💾 Enregistrer", command=save_modifications).grid(
            row=len(fields), column=0, columnspan=2, pady=20
        )
        
    def delete_person(self):
        """Supprime une personne de la base de données"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Attention", "Veuillez sélectionner une personne")
            return
            
        item = self.tree.item(selected[0])
        matricule = item['values'][0]
        nom = item['values'][1]
        prenom = item['values'][2]
        
        if messagebox.askyesno("Confirmation", f"Supprimer définitivement {prenom} {nom} (matricule: {matricule})?"):
            try:
                self.cursor.execute('DELETE FROM personnes WHERE matricule=?', (matricule,))
                self.conn.commit()
                messagebox.showinfo("Succès", "Personne supprimée avec succès")
                self.refresh_list()
                self.load_known_faces()
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur: {str(e)}")
                
    def start_recognition(self):
        """Démarre la reconnaissance faciale"""
        if not self.recognizer_trained:
            messagebox.showwarning("Attention", "Aucune personne enregistrée.\nVeuillez d'abord ajouter des personnes dans l'onglet Enregistrement.")
            return
            
        if self.is_camera_on:
            messagebox.showinfo("Info", "La caméra est déjà en marche")
            return
        
        # Essayer différents indices de caméra
        camera_found = False
        for camera_index in [0, 1, 2]:
            self.camera = cv2.VideoCapture(camera_index)
            if self.camera.isOpened():
                camera_found = True
                print(f"✓ Caméra trouvée à l'indice {camera_index}")
                break
            self.camera.release()
        
        if not camera_found:
            messagebox.showerror("Erreur", "Impossible d'accéder à la webcam.\n\nVérifiez que:\n- Votre webcam est branchée\n- Aucune autre application n'utilise la webcam\n- Les pilotes sont à jour")
            return
        
        # Configurer la caméra
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Attendre que la caméra s'initialise
        import time
        time.sleep(0.5)
        
        # Tester la lecture
        ret, test_frame = self.camera.read()
        if not ret or test_frame is None:
            messagebox.showerror("Erreur", "La caméra ne répond pas correctement")
            self.camera.release()
            return
            
        self.is_camera_on = True
        self.last_recognized = {}
        self.process_video()
        messagebox.showinfo("Succès", "Caméra démarrée avec succès!")
        
    def stop_recognition(self):
        """Arrête la reconnaissance faciale"""
        self.is_camera_on = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.label_camera.configure(image='', text="Caméra éteinte / Aucune image")
        messagebox.showinfo("Info", "Caméra arrêtée")
    
    def recognize_from_image(self):
        """Charge et reconnaît une personne depuis une image"""
        if not self.recognizer_trained:
            messagebox.showwarning("Attention", "Aucune personne enregistrée.\nVeuillez d'abord ajouter des personnes dans l'onglet Enregistrement.")
            return
        
        # Arrêter la caméra si elle est en cours
        if self.is_camera_on:
            self.stop_recognition()
        
        # Charger l'image
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image à reconnaître",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
        
        # Lire l'image
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Erreur", "Impossible de charger l'image")
            return
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Détecter les visages
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) == 0:
            messagebox.showwarning("Aucun visage", "Aucun visage détecté dans cette image")
            return
        
        # Traiter chaque visage détecté
        recognized_count = 0
        display_img = img.copy()
        
        for (x, y, w, h) in faces:
            # Extraire le visage
            face_roi = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (200, 200))
            
            try:
                # Reconnaître
                label, confidence = self.face_recognizer.predict(face_roi_resized)
                
                # Plus la confiance est basse, meilleure est la correspondance
                if confidence < 80:
                    person_data = self.person_mapping.get(label)
                    if person_data:
                        name = f"{person_data['prenom']} {person_data['nom']}"
                        recognized_count += 1
                        
                        # Enregistrer dans le fichier
                        self.log_recognition(person_data)
                        self.display_recognition(person_data, confidence)
                        
                        # Dessiner rectangle vert
                        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        cv2.putText(display_img, name, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.putText(display_img, f"Confiance: {int(100-confidence)}%", 
                                   (x, y+h+30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                        cv2.putText(display_img, "Inconnu", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 165, 255), 3)
                    cv2.putText(display_img, "Personne inconnue", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                    
            except Exception as e:
                print(f"Erreur de reconnaissance: {e}")
                cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(display_img, "Erreur", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Afficher l'image avec les résultats
        h, w = display_img.shape[:2]
        max_size = 700
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        
        img_resized = cv2.resize(display_img, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(img_pil)
        
        self.label_camera.configure(image=photo, text="")
        self.label_camera.image = photo
        
        # Message de résultat
        if recognized_count > 0:
            messagebox.showinfo("Résultat", f"{recognized_count} personne(s) reconnue(s) !\nConsultez l'historique pour les détails.")
        else:
            messagebox.showinfo("Résultat", "Aucune personne reconnue dans cette image.")
        
    def process_video(self):
        """Traite le flux vidéo pour la reconnaissance"""
        if not self.is_camera_on:
            return
            
        ret, frame = self.camera.read()
        if ret:
            # Redimensionner pour l'affichage
            display_frame = cv2.resize(frame, (780, 585))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Détecter les visages
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                # Extraire le visage
                face_roi = gray[y:y+h, x:x+w]
                face_roi_resized = cv2.resize(face_roi, (200, 200))
                
                # Calculer les coordonnées pour l'affichage
                scale_x = 780 / frame.shape[1]
                scale_y = 585 / frame.shape[0]
                x_display = int(x * scale_x)
                y_display = int(y * scale_y)
                w_display = int(w * scale_x)
                h_display = int(h * scale_y)
                
                try:
                    # Reconnaître
                    label, confidence = self.face_recognizer.predict(face_roi_resized)
                    
                    # Plus la confiance est basse, meilleure est la correspondance
                    if confidence < 80:
                        person_data = self.person_mapping.get(label)
                        if person_data:
                            name = f"{person_data['prenom']} {person_data['nom']}"
                            
                            # Vérifier si pas déjà reconnu récemment (dans les 3 dernières secondes)
                            current_time = datetime.now()
                            if label not in self.last_recognized or \
                               (current_time - self.last_recognized[label]).seconds > 3:
                                self.last_recognized[label] = current_time
                                self.log_recognition(person_data)
                                self.display_recognition(person_data, confidence)
                            
                            # Dessiner rectangle vert
                            cv2.rectangle(display_frame, (x_display, y_display), 
                                        (x_display+w_display, y_display+h_display), (0, 255, 0), 3)
                            cv2.putText(display_frame, name, (x_display, y_display-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(display_frame, f"Confiance: {int(100-confidence)}%", 
                                       (x_display, y_display+h_display+25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            cv2.rectangle(display_frame, (x_display, y_display), 
                                        (x_display+w_display, y_display+h_display), (0, 0, 255), 3)
                            cv2.putText(display_frame, "Inconnu", (x_display, y_display-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(display_frame, (x_display, y_display), 
                                    (x_display+w_display, y_display+h_display), (0, 165, 255), 3)
                        cv2.putText(display_frame, "Personne inconnue", (x_display, y_display-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                        
                except Exception as e:
                    print(f"Erreur de reconnaissance: {e}")
                    cv2.rectangle(display_frame, (x_display, y_display), 
                                (x_display+w_display, y_display+h_display), (0, 0, 255), 3)
            
            # Convertir pour Tkinter
            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            photo = ImageTk.PhotoImage(img_pil)
            self.label_camera.configure(image=photo, text="")
            self.label_camera.image = photo
            
        self.root.after(30, self.process_video)
        
    def log_recognition(self, person_data):
        """Enregistre la reconnaissance dans un fichier"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {person_data['nom']} {person_data['prenom']} (Matricule: {person_data['matricule']})\n"
        
        with open("reconnaissances.txt", "a", encoding='utf-8') as f:
            f.write(log_entry)
            
    def display_recognition(self, person_data, confidence):
        """Affiche les informations de reconnaissance"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        confidence_percent = int(100 - confidence)
        message = f"\n{'='*45}\n[{timestamp}]\n✓ PERSONNE RECONNUE\n"
        message += f"Nom: {person_data['nom']}\n"
        message += f"Prénom: {person_data['prenom']}\n"
        message += f"Matricule: {person_data['matricule']}\n"
        message += f"Confiance: {confidence_percent}%\n"
        message += f"{'='*45}\n"
        
        self.text_results.insert('1.0', message)
        
    def __del__(self):
        """Ferme la connexion à la base de données"""
        if hasattr(self, 'conn'):
            self.conn.close()
        if hasattr(self, 'camera') and self.camera:
            self.camera.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()