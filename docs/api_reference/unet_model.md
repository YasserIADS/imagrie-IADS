# Modèle U-Net (`UNet`)

## Classe `UNet`

```python
# Extrait partiel basé sur votre snippet
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder (Downsampling)
        self.enc1 = self.conv_block(1, 64)      # Input: 1 channel (grayscale)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        # ... (Reste de l'encodeur) ...

        # Bottleneck
        # self.bottleneck = self.conv_block(...)

        # Decoder (Upsampling)
        # self.upconv1 = nn.ConvTranspose2d(...)
        # self.dec1 = self.conv_block(...)
        # ... (Reste du décodeur) ...

        # Output layer
        # self.out_conv = nn.Conv2d(...)

    def conv_block(self, in_channels, out_channels):
        # ... (Définition du bloc convolutif, par ex. Conv2D -> ReLU -> Conv2D -> ReLU) ...
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Optionnel, mais commun
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Optionnel
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # ... (Définition du passage avant, incluant les skip connections) ...
        # Exemple simplifié:
        # e1 = self.enc1(x)
        # p1 = self.pool1(e1)
        # ...
        # d1 = self.upconv1(bottleneck_output)
        # d1 = torch.cat([d1, skip_connection_from_encoder], dim=1)
        # d1 = self.dec1(d1)
        # ...
        # return self.out_conv(final_decoder_output)
        pass # Remplacez par la vraie logique
