# docs/conf.py

import os
import sys
from unittest.mock import Mock

# Si votre code source est dans un répertoire parent (ex: projet/mon_code),
# ajoutez ce chemin pour que Sphinx puisse trouver vos modules.
# sys.path.insert(0, os.path.abspath('..')) # Décommenter et ajuster si nécessaire

# --- Section pour le "mocking" des imports lourds sur Read the Docs ---
# Ceci est crucial pour les bibliothèques comme torch, torchvision, etc.,
# qui sont souvent trop complexes ou lourdes à installer sur les environnements de build.
# Sphinx fera semblant que ces modules sont importés, permettant de documenter leurs objets
# (classes, fonctions) via autodoc sans avoir à les exécuter réellement.
autodoc_mock_imports = [
    'torch',
    'torchvision',
    'cv2', # C'est le module importé par opencv-python
    'numpy',
    'albumentations',
    'sklearn', # Pour scikit-learn
    # Ajoutez ici tout autre module de votre projet principal
    # qui pourrait causer des problèmes d'importation/installation
    # pendant la construction de la doc.
]

# --- Configurations générales de Sphinx (déjà présentes ou à vérifier) ---
project = 'Documentation du Projet U-Net à Deux Étapes'
copyright = '2023, Votre Nom ou Nom de l\'Organisation' # Mettez à jour
author = 'Votre Nom' # Mettez à jour
release = '0.1' # Version de votre projet

# Extensions Sphinx que vous utilisez
extensions = [
    'sphinx.ext.autodoc',  # Pour la génération auto depuis les docstrings
    'sphinx.ext.napoleon', # Pour supporter les docstrings de style NumPy/Google
    'sphinx.ext.todo',     # Pour les notes de type TODO
    'sphinx.ext.viewcode', # Pour lier le code source
    'myst_parser',         # Pour écrire en Markdown
]

# Configurez le thème Read the Docs
html_theme = 'sphinx_rtd_theme'

# Si vous utilisez myst_parser, vous devrez peut-être configurer ceci:
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# --- Autres configurations (si vous en avez) ---
# ...
