class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size #16
        self.class_token = torch.randn(1,1,emb_size) #image representation
        self.pos_embedding = torch.randn((img_size // patch_size) **2, emb_size) #positional embedding of patch

        self.projection = nn.Sequential(
            # Convolution Layer to each patch
            nn.Conv2d(in_channels=in_channels, out_channels=emb_size, kernel_size=patch_size, stride=patch_size),
            # [batch, embedding_size, height, width]
            # Flatten the resulting images -> Multiply: Height*Width
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        batch,_,_=x.shape
        # Concatenate Class Tensor to Projected Patches
        class_tensor = self.class_token.repeat(batch,1,1)

        # Add Positional Embedding to Projected Patches
        x += self.pos_embedding

        return x
