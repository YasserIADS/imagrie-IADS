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
