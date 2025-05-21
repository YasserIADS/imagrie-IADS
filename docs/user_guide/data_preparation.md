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

