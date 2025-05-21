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
