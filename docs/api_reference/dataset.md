# Gestion des Données (`CustomDataset`)

## Classe `CustomDataset`

```python
# Supposons une structure de classe comme celle-ci
# (adaptez selon votre implémentation exacte)
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_height=128, image_width=128):
        # self.image_dir = image_dir
        # self.mask_dir = mask_dir
        # self.images = os.listdir(image_dir)
        # self.transform = transform
        # self.image_height = image_height
        # self.image_width = image_width
        pass # Remplacez par la vraie logique

    def __len__(self):
        # return len(self.images)
        pass # Remplacez par la vraie logique

    def __getitem__(self, idx):
        # img_path = os.path.join(self.image_dir, self.images[idx])
        # mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".png", "_mask.png")) # Exemple de convention de nommage

        # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # image = cv2.resize(image, (self.image_width, self.image_height))
        # mask = cv2.resize(mask, (self.image_width, self.image_height))

        # # Normalisation et gestion des masques
        # image = image / 255.0
        # mask = (mask / 255.0 > 0.5).astype(np.float32) # Binarisation

        # if self.transform:
        #     augmented = self.transform(image=image, mask=mask)
        #     image = augmented['image']
        #     mask = augmented['mask']

        # # Ajout de la dimension canal pour l'image si nécessaire après Albumentations ToTensorV2
        # # mask = mask.unsqueeze(0) # Si le masque est (H, W) et doit être (1, H, W)

        # return image, mask # Doivent être des tenseurs
        pass # Remplacez par la vraie logique
