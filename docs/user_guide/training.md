# Entraînement du Modèle

Cette section décrit comment configurer et lancer le processus d'entraînement pour le U-Net à Deux Étapes.

## Configuration de l'Entraînement

L'entraînement est principalement contrôlé via le notebook `2_Stage_U-NET.ipynb`. Les paramètres clés à configurer avant de lancer l'entraînement incluent :

* **Chemins des Données :**
    * `DATA_DIR`: Répertoire racine des données.
    * `IMAGE_DIR_NAME`: Sous-répertoire des images.
    * `MASK_DIR_NAME`: Sous-répertoire des masques.
* **Paramètres du Modèle :**
    * `IMAGE_HEIGHT`, `IMAGE_WIDTH`: Dimensions des images d'entrée (par exemple, 128, 128).
* **Hyperparamètres d'Entraînement :**
    * `DEVICE`: `"cuda"` pour GPU, `"cpu"` pour CPU.
    * `BATCH_SIZE`: Taille du lot (par exemple, 4, 8, 16).
    * `NUM_EPOCHS`: Nombre total d'époques d'entraînement (par exemple, 100).
    * `LEARNING_RATE`: Taux d'apprentissage pour l'optimiseur (par exemple, 1e-4).
    * `VALIDATION_SPLIT`: Proportion des données à utiliser pour la validation (si un découpage est fait dynamiquement).
* **Paramètres de Sauvegarde :**
    * `MODEL_SAVE_PATH`: Chemin où sauvegarder les poids du meilleur modèle.

[**Détaillez ici la logique spécifique aux deux étapes : Y a-t-il des configurations différentes pour chaque étape ? Sont-elles entraînées séquentiellement ?**]

## Lancement de l'Entraînement

1.  Ouvrez le notebook `2_Stage_U-NET.ipynb` dans un environnement Jupyter.
2.  Configurez les paramètres mentionnés ci-dessus dans les cellules appropriées.
3.  Exécutez les cellules dans l'ordre :
    * Définition du modèle (`UNet`).
    * Définition du `CustomDataset` et des `DataLoader`.
    * Initialisation du modèle, de la fonction de perte ([**spécifiez la perte, par exemple, `BCEWithLogitsLoss`**]) et de l'optimiseur ([**spécifiez l'optimiseur, par exemple, `Adam`**]).
    * Exécution de la boucle d'entraînement.

## Suivi de l'Entraînement

Pendant l'entraînement, les informations suivantes sont généralement affichées pour chaque époque :
* Perte d'entraînement (`Loss`).
* Coefficient de Dice d'entraînement (`Dice`).
* Perte de validation.
* Coefficient de Dice de validation.

Le modèle avec le meilleur [**spécifiez le critère, par exemple, "coefficient de Dice de validation"**] est sauvegardé à l'emplacement spécifié (`MODEL_SAVE_PATH`). L'extrait de log montre :
D'accord, si vous visez une documentation de type "Read the Docs", cela implique généralement une structure plus formelle et souvent multi-pages, organisée pour une navigation facile. Read the Docs utilise souvent Sphinx, qui peut générer de la documentation à partir de fichiers reStructuredText (.rst) ou Markdown (.md), et peut extraire automatiquement la documentation des docstrings de votre code Python.

Voici une structure et un contenu que vous pourriez utiliser pour mettre en place une documentation de ce type pour votre projet 2_Stage_U-NET.ipynb. Je vais utiliser la syntaxe Markdown ici, car elle est plus simple à écrire directement et peut être convertie ou utilisée par Sphinx.

Imaginez que ce sont différents fichiers .md ou différentes sections d'une documentation plus large :

Structure des Fichiers de Documentation (Exemple)
docs/
├── index.md                # Page d'accueil
├── installation.md         # Guide d'installation
├── user_guide/
│   ├── index.md            # Introduction au guide utilisateur
│   ├── data_preparation.md
│   ├── training.md
│   ├── evaluation.md
│   └── inference.md
├── api_reference/
│   ├── index.md            # Introduction à la référence API
│   ├── unet_model.md       # Documentation de la classe UNet
│   ├── dataset.md          # Documentation de la classe Dataset
│   └── (autres_modules.md) # Pour les fonctions utilitaires, etc.
├── tutorials/
│   └── example_notebook.md # Liant ou intégrant votre .ipynb
├── contributing.md         # (Optionnel)
└── license.md              # (Optionnel)
index.md (Page d'Accueil)
Markdown

# Documentation du Projet U-Net à Deux Étapes

Bienvenue sur la documentation du projet de segmentation d'images utilisant un U-Net à deux étapes. Ce projet est conçu pour [**Décrivez brièvement l'objectif principal, par exemple, "la segmentation précise d'objets dans des images médicales en niveaux de gris."**].

Ce système de documentation vous guidera à travers l'installation, l'utilisation, et la compréhension des composants clés du projet.

## Vue d'ensemble

* **Architecture U-Net :** Utilise une architecture U-Net modifiée ou appliquée en deux phases distinctes pour une segmentation améliorée.
* **Traitement d'Images en Niveaux de Gris :** Optimisé pour les images à un seul canal.
* **Entraînement Robuste :** Inclut des augmentations de données, une fonction de perte adaptée et un suivi des métriques.
* **Implémentation PyTorch :** Entièrement codé en utilisant le framework PyTorch.

## Navigation

* [Installation](./installation.md) : Comment configurer votre environnement et installer les dépendances.
* [Guide de l'Utilisateur](./user_guide/index.md) : Instructions détaillées sur la préparation des données, l'entraînement, l'évaluation et l'inférence.
* [Référence API](./api_reference/index.md) : Documentation détaillée des modules, classes et fonctions du code.
* [Tutoriels](./tutorials/example_notebook.md) : Exemples pratiques, y compris l'utilisation du notebook principal.
* [Contribuer](./contributing.md) : (Si applicable) Comment contribuer au projet.
* [Licence](./license.md) : Informations sur la licence du projet.

Pour commencer, consultez le [Guide d'Installation](./installation.md).
installation.md
Markdown

# Guide d'Installation

Suivez ces étapes pour configurer votre environnement de développement et installer les dépendances nécessaires pour le projet U-Net à Deux Étapes.

## Prérequis

* Python (version >= 3.8 recommandée)
* Pip (gestionnaire de paquets Python)
* Git (pour cloner le dépôt, si applicable)
* (Optionnel mais recommandé) Un environnement avec support CUDA pour l'entraînement sur GPU.

## Étapes d'Installation

1.  **Cloner le Dépôt (si applicable)**
    Si le code est hébergé sur une plateforme comme GitHub :
    ```bash
    git clone [URL_DU_DEPOT]
    cd [NOM_DU_REPERTOIRE_DU_PROJET]
    ```
    Sinon, assurez-vous d'avoir tous les fichiers du projet dans un répertoire local.

2.  **Créer un Environnement Virtuel (Recommandé)**
    Il est fortement conseillé d'utiliser un environnement virtuel pour isoler les dépendances du projet.
    ```bash
    python -m venv env
    ```
    Activez l'environnement :
    * Sur macOS et Linux :
        ```bash
        source env/bin/activate
        ```
    * Sur Windows :
        ```bash
        env\Scripts\activate
        ```

3.  **Installer les Dépendances**
    Les bibliothèques requises sont listées ci-dessous. Vous pouvez les installer en utilisant pip :
    ```bash
    pip install torch torchvision torchaudio
    pip install opencv-python numpy scikit-learn albumentations matplotlib jupyter
    ```
    Si un fichier `requirements.txt` est fourni :
    ```bash
    pip install -r requirements.txt
    ```

    **Dépendances Principales :**
    * `torch` : Framework d'apprentissage profond.
    * `torchvision` : Utilitaires pour la vision par ordinateur avec PyTorch.
    * `opencv-python` : Bibliothèque pour le traitement d'images.
    * `numpy` : Pour les opérations numériques.
    * `scikit-learn` : Pour les métriques d'évaluation (precision, recall).
    * `albumentations` : Pour l'augmentation d'images.
    * `jupyter` : Pour exécuter les notebooks `.ipynb`.
    * `matplotlib` : Pour la visualisation (si utilisée).

Une fois ces étapes complétées, vous devriez être prêt à utiliser le projet. Consultez le [Guide de l'Utilisateur](./user_guide/index.md) pour les prochaines étapes.
user_guide/index.md
Markdown

# Guide de l'Utilisateur

Ce guide fournit des instructions détaillées sur l'utilisation du projet U-Net à Deux Étapes, de la préparation des données à l'exécution des prédictions.

## Sections

* [Préparation des Données](./data_preparation.md) : Comment formater et prétraiter vos données.
* [Entraînement du Modèle](./training.md) : Comment configurer et lancer le processus d'entraînement.
* [Évaluation du Modèle](./evaluation.md) : Comment évaluer les performances du modèle entraîné.
* [Inférence avec le Modèle](./inference.md) : Comment utiliser le modèle pour segmenter de nouvelles images.

Naviguez vers la section appropriée pour obtenir les informations dont vous avez besoin.
user_guide/data_preparation.md
Markdown

# Préparation des Données

La qualité et le format des données d'entrée sont cruciaux pour la performance du modèle U-Net.

## Format des Données Requis

* **Images :**
    * Type : Images en niveaux de gris (1 canal).
    * Format de fichier : `JPEG`, `PNG`, etc. (lisibles par OpenCV).
    * [**Précisez toute autre spécificité, par exemple, résolution attendue avant redimensionnement.**]
* **Masques de Segmentation (Vérité Terrain) :**
    * Type : Images binaires (ou multi-classes si applicable) où les pixels de l'objet d'intérêt ont une valeur (par exemple, 255) et le fond une autre (par exemple, 0).
    * Format de fichier : Identique aux images.
    * Doivent correspondre pixel à pixel aux images d'entrée après redimensionnement.

## Structure des Répertoires

Le code s'attend typiquement à la structure suivante (configurable via des variables dans le notebook) :

&lt;DATA_DIR>/
├── &lt;IMAGE_DIR_NAME>/       # Par défaut, 'images' ou 'train_images'
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── &lt;MASK_DIR_NAME>/        # Par défaut, 'masks' ou 'train_masks'
├── image_001_mask.png  # Assurez-vous que les noms correspondent
├── image_002_mask.png
└── ...

[**Adaptez ceci à la structure exacte attendue par votre `CustomDataset`.**]

## Prétraitement

Les étapes de prétraitement suivantes sont appliquées (principalement dans la classe `CustomDataset` et/ou avant l'entraînement) :
1.  **Lecture des Images et Masques :** Utilisation de `cv2.imread()`. Les images sont lues en niveaux de gris.
2.  **Redimensionnement :** Les images et les masques sont redimensionnés à la taille d'entrée définie pour le réseau (par exemple, `IMAGE_HEIGHT`, `IMAGE_WIDTH` comme 128x128).
3.  **Normalisation :** Les valeurs des pixels des images sont normalisées (par exemple, divisées par 255.0 pour être dans la plage [0, 1]). Les masques sont également normalisés si nécessaire (par exemple, valeurs de pixel à 0.0 ou 1.0).
4.  [**Ajoutez d'autres étapes spécifiques si elles existent, par exemple, filtrage, etc.**]

## Augmentation des Données

Pour améliorer la généralisation du modèle, des techniques d'augmentation de données sont appliquées à la volée pendant l'entraînement en utilisant la bibliothèque `albumentations`. Les transformations configurées incluent :
* [**Listez les transformations d'Albumentations utilisées, e.g., `HorizontalFlip`, `VerticalFlip`, `Rotate`, `RandomBrightnessContrast`. Référez-vous à la section `transform` dans votre `CustomDataset`.**]
* Les transformations sont appliquées de manière aléatoire à chaque image/masque du lot d'entraînement.
* Finalement, les images et masques sont convertis en tenseurs PyTorch via `ToTensorV2()`.
user_guide/training.md
Markdown

# Entraînement du Modèle

Cette section décrit comment configurer et lancer le processus d'entraînement pour le U-Net à Deux Étapes.

## Configuration de l'Entraînement

L'entraînement est principalement contrôlé via le notebook `2_Stage_U-NET.ipynb`. Les paramètres clés à configurer avant de lancer l'entraînement incluent :

* **Chemins des Données :**
    * `DATA_DIR`: Répertoire racine des données.
    * `IMAGE_DIR_NAME`: Sous-répertoire des images.
    * `MASK_DIR_NAME`: Sous-répertoire des masques.
* **Paramètres du Modèle :**
    * `IMAGE_HEIGHT`, `IMAGE_WIDTH`: Dimensions des images d'entrée (par exemple, 128, 128).
* **Hyperparamètres d'Entraînement :**
    * `DEVICE`: `"cuda"` pour GPU, `"cpu"` pour CPU.
    * `BATCH_SIZE`: Taille du lot (par exemple, 4, 8, 16).
    * `NUM_EPOCHS`: Nombre total d'époques d'entraînement (par exemple, 100).
    * `LEARNING_RATE`: Taux d'apprentissage pour l'optimiseur (par exemple, 1e-4).
    * `VALIDATION_SPLIT`: Proportion des données à utiliser pour la validation (si un découpage est fait dynamiquement).
* **Paramètres de Sauvegarde :**
    * `MODEL_SAVE_PATH`: Chemin où sauvegarder les poids du meilleur modèle.

[**Détaillez ici la logique spécifique aux deux étapes : Y a-t-il des configurations différentes pour chaque étape ? Sont-elles entraînées séquentiellement ?**]

## Lancement de l'Entraînement

1.  Ouvrez le notebook `2_Stage_U-NET.ipynb` dans un environnement Jupyter.
2.  Configurez les paramètres mentionnés ci-dessus dans les cellules appropriées.
3.  Exécutez les cellules dans l'ordre :
    * Définition du modèle (`UNet`).
    * Définition du `CustomDataset` et des `DataLoader`.
    * Initialisation du modèle, de la fonction de perte ([**spécifiez la perte, par exemple, `BCEWithLogitsLoss`**]) et de l'optimiseur ([**spécifiez l'optimiseur, par exemple, `Adam`**]).
    * Exécution de la boucle d'entraînement.

## Suivi de l'Entraînement

Pendant l'entraînement, les informations suivantes sont généralement affichées pour chaque époque :
* Perte d'entraînement (`Loss`).
* Coefficient de Dice d'entraînement (`Dice`).
* Perte de validation.
* Coefficient de Dice de validation.

Le modèle avec le meilleur [**spécifiez le critère, par exemple, "coefficient de Dice de validation"**] est sauvegardé à l'emplacement spécifié (`MODEL_SAVE_PATH`). L'extrait de log montre :
Epoch [1/10], Loss: 0.0664, Dice: 0.9589
Best model saved.
...
Epoch [10/10], Loss: 0.0444, Dice: 0.9723
[**Si l'entraînement se fait en deux étapes distinctes avec des scripts ou des sauvegardes de modèles intermédiaires, décrivez ce processus ici.**]
