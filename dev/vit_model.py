class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size #16
        self.class_token = torch.randn(1,1,emb_size) #image representation
        self.pos_embedding = torch.randn((img_size // patch_size) **2 + 1, emb_size) #positional embedding of patch

        self.projection = nn.Sequential(
            # Convolution Layer to each patch
            nn.Conv2d(in_channels=in_channels, out_channels=emb_size, kernel_size=patch_size, stride=patch_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)

        #Flatten images after Convolution
        x = torch.flatten(x, start_dim=2, end_dim=3)
        x = torch.transpose(x, 1, 2)

        batch,_,_=x.shape # batch size

        # Concatenate Class Tensor to Projected Patches
        class_tensor = self.class_token.repeat(batch,1,1)
        x = torch.cat([class_tensor, x], dim=1)

        # Add Positional Embedding to Projected Patches
        x += self.pos_embedding

        return x
