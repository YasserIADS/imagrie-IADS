# Évaluation du Modèle

Après l'entraînement, il est essentiel d'évaluer les performances du modèle sur un ensemble de données non vu pendant l'entraînement (idéalement un ensemble de test distinct, ou l'ensemble de validation).

## Métriques d'Évaluation

Les métriques suivantes sont utilisées pour évaluer la qualité des segmentations produites par le modèle. Elles sont calculées en comparant les masques prédits aux masques de vérité terrain.

* **Coefficient de Dice (DSC) :**
    $DSC = \frac{2 \cdot |X \cap Y|}{|X| + |Y|}$
    Mesure le chevauchement entre la prédiction ($X$) et la vérité terrain ($Y$). Un score plus élevé est meilleur (max 1).

* **Précision (Precision) :**
    $Precision = \frac{TP}{TP + FP}$
    Proportion de pixels correctement identifiés comme positifs parmi tous les pixels prédits comme positifs.

* **Rappel (Recall / Sensitivity) :**
    $Recall = \frac{TP}{TP + FN}$
    Proportion de pixels correctement identifiés comme positifs parmi tous les pixels réellement positifs.

Où :
* `TP` (Vrais Positifs) : Pixels correctement classifiés comme appartenant à l'objet.
* `FP` (Faux Positifs) : Pixels incorrectement classifiés comme appartenant à l'objet (erreur de type I).
* `FN` (Faux Négatifs) : Pixels de l'objet que le modèle n'a pas réussi à identifier (erreur de type II).

## Processus d'Évaluation

1.  **Charger le Meilleur Modèle :** Chargez les poids du modèle sauvegardé (`.pth` ou `.pt` file) qui a obtenu les meilleures performances sur l'ensemble de validation.
    ```python
    # Exemple
    # model = UNet()
    # model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    # model.to(DEVICE)
    # model.eval()
    ```
2.  **Préparer l'Ensemble de Données d'Évaluation :** Utilisez un `DataLoader` pour l'ensemble de test ou de validation, sans augmentation de données (sauf le redimensionnement et la normalisation).
3.  **Itérer et Prédire :** Pour chaque image de l'ensemble d'évaluation :
    * Obtenez la prédiction du modèle.
    * Appliquez un seuil à la sortie (si ce sont des logits/probabilités) pour obtenir un masque binaire (par exemple, seuil à 0.5).
    * Calculez les métriques (Dice, Precision, Recall) en comparant avec le masque de vérité terrain.
4.  **Agréger les Scores :** Calculez la moyenne des métriques sur toutes les images de l'ensemble d'évaluation.

Les résultats de l'évaluation sont généralement imprimés ou peuvent être sauvegardés dans un fichier.

[**Si l'évaluation de l'approche à deux étapes est spécifique (par exemple, évaluer chaque étape ou l'ensemble), détaillez-le.**]
