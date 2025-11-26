# Application de Reconnaissance Faciale en Python

Cette application permet d’enregistrer, gérer et reconnaître des personnes à partir d’images ou d’un flux webcam.
Elle utilise :

- <b>Tkinter</b> pour l’interface graphique

- <b>OpenCV</b> (LBPH) pour la reconnaissance faciale

- <b>SQLite</b> pour stocker les données des personnes

- <b>PIL</b> pour l'affichage des images

- <b>Pickle</b> pour sérialiser les visages dans la base de données

# Fonctionnalités
## 1. Enregistrement

- Chargement d’une image depuis le disque

- Capture de photo via la webcam

- Extraction automatique du visage

- Enregistrement dans une base SQLite avec :

  - Matricule

  - Nom / Prénom

  - Âge

  - Email

  - Téléphone

  - Données faciales sérialisées

## 2. Gestion

- Affichage de toutes les personnes enregistrées

- Modification des informations

- Suppression d’une fiche

- Rafraîchissement de la liste en temps réel

## 3. Reconnaissance

- Reconnaissance via webcam en direct

- Reconnaissance depuis une image chargée

- Historique détaillé des reconnaissances identifiées :

  - Nom et prénom

  - Matricule

  - Date et heure

  - Niveau de confiance

# Algorithme de Reconnaissance

L’application utilise l’algorithme LBPH (Local Binary Patterns Histograms) intégré à OpenCV :

- Robuste aux variations de lumière

- Très performant pour les reconnaissances en temps réel

- Idéal pour les visages frontaux

Chaque visage est redimensionné en 200×200 pixels, puis entraîné à chaque ajout dans la base.

# Base de Données

Le fichier SQLite face_recognition.db contient une table :

<pre>CREATE TABLE personnes ( id INTEGER PRIMARY KEY AUTOINCREMENT, matricule TEXT UNIQUE NOT NULL, nom TEXT NOT NULL, prenom TEXT NOT NULL, age INTEGER, email TEXT, telephone TEXT, face_data BLOB NOT NULL );</pre>

Chaque entrée contient à la fois les métadonnées et les données faciales.

# Installation
1) Installer Python ≥ 3.8
2) Installer les dépendances
- pip install <b>opencv-contrib-python</b>
- pip install <b>pillow</b>
- pip install <b>numpy</b>


## ⚠️ Attention :
Il est obligatoire d’installer opencv-contrib-python, car le module LBPH n’est pas présent dans la version standard d’OpenCV.

3) Lancer l’application
python reconnaissance_image.py


(renomme ton fichier si nécessaire)

# 📷 Utilisation
## ➤ Enregistrer une personne

- Aller dans l’onglet 📝 Enregistrement

- Importer une image ou capturer une photo

- Remplir les informations

- Cliquer sur 💾 Enregistrer

## ➤ Gérer le registre

- Onglet 📋 Gestion

- Modifier ou supprimer une personne facilement

## ➤ Reconnaître un visage

Deux options :

- Webcam
  - Démarrer la caméra → reconnaissance en temps réel

- Image
  - Charger une photo → détection et identification


# ⚙️ Points techniques importants

- La reconnaissance nécessite au moins 1 visage enregistré

- Le modèle LBPH est réentraîné automatiquement à chaque ajout

- Les visages sont triés par taille pour éviter les faux positifs

- L'application gère plusieurs caméras (indices 0,1,2)

- L’historique n'est pas stocké en base mais affiché dans l’interface

# 🛡️ Limites et améliorations possibles
## ✔️ Améliorations simples

- Ajouter une exportation CSV de la base

- Ajouter un système de logs persistants

- Intégrer un système d’authentification admin

## ✔️ Améliorations avancées

- Remplacer HaarCascade par un modèle DNN (plus précis)


- Gérer plusieurs visages par personne (multiple samples par personne)

# 🙌 Auteur

Projet réalisé par Cyr DJOKI pour démonstration d’une application Python complète combinant :

- Gestion de base de données

- Interface graphique avancée

- Traitement d’images

- Reconnaissance faciale en temps réel
