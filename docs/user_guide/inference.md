# Inférence avec le Modèle Entraîné

Une fois qu'un modèle U-Net a été entraîné et sauvegardé, il peut être utilisé pour segmenter de nouvelles images (inférence).

## Étapes pour l'Inférence

1.  **Initialiser le Modèle :**
    Créez une instance de votre classe `UNet`.
    ```python
    # from unet_model import UNet # Si UNet est dans un fichier séparé
    model = UNet()
    ```

2.  **Charger les Poids Entraînés :**
    Chargez les poids du meilleur modèle sauvegardé pendant l'entraînement. Assurez-vous de spécifier `map_location` si vous chargez un modèle entraîné sur GPU vers un environnement CPU.
    ```python
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "chemin/vers/votre/best_model.pth" # Adaptez le chemin
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Mettre le modèle en mode évaluation (important pour désactiver le dropout, etc.)
    ```

3.  **Prétraiter l'Image d'Entrée :**
    La nouvelle image doit subir les mêmes étapes de prétraitement que les images d'entraînement :
    * Lecture de l'image (par exemple, avec `cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)`).
    * Redimensionnement à la taille attendue par le réseau (`IMAGE_HEIGHT`, `IMAGE_WIDTH`).
    * Normalisation (par exemple, division par 255.0).
    * Conversion en tenseur PyTorch et ajout de la dimension du lot (`unsqueeze(0)`).
    ```python
    # Exemple de fonction de prétraitement
    # def preprocess_image(image_path, height, width):
    #     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     img = cv2.resize(img, (width, height))
    #     img = img / 255.0
    #     img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) # (batch, channel, H, W)
    #     return img_tensor.to(DEVICE)

    # input_tensor = preprocess_image("chemin/vers/nouvelle_image.png", IMAGE_HEIGHT, IMAGE_WIDTH)
    ```
    [**Utilisez les mêmes transformations (redimensionnement, normalisation) que celles définies dans votre `CustomDataset` mais sans l'augmentation aléatoire.**]

4.  **Effectuer la Prédiction :**
    Passez le tenseur de l'image prétraitée à travers le modèle.
    ```python
    with torch.no_grad(): # Désactiver le calcul du gradient pour l'inférence
        output = model(input_tensor)
    ```

5.  **Post-traiter la Sortie :**
    La sortie du modèle est généralement une carte de logits ou de probabilités.
    * Appliquez une fonction sigmoïde si la dernière couche n'en a pas et que vous utilisez `BCEWithLogitsLoss` à l'entraînement : `probs = torch.sigmoid(output)`.
    * Appliquez un seuil pour obtenir un masque binaire : `predicted_mask = (probs > 0.5).float()`.
    * Redimensionnez le masque prédit à la taille de l'image originale si nécessaire.
    * Convertissez le tenseur en image NumPy pour la visualisation ou la sauvegarde.

    ```python
    # Exemple
    # probs = torch.sigmoid(output)
    # predicted_mask_tensor = (probs > 0.5).cpu().squeeze() # Enlever les dimensions batch/channel
    # predicted_mask_numpy = predicted_mask_tensor.numpy()
    # # Optionnel: redimensionner à la taille originale, sauvegarder, ou visualiser
    ```

[**Décrivez ici la procédure d'inférence spécifique à l'approche à deux étapes :**]
* [**Comment la première étape est-elle exécutée ?**]
* [**Comment sa sortie est-elle utilisée comme entrée (ou pour guider) la deuxième étape ?**]
* [**Le post-traitement est-il différent ?**]

Ce processus peut être encapsulé dans une fonction ou un script pour une utilisation facile.
