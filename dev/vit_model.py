class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size #16
        self.class_token = torch.randn(1,1,emb_size) #image representation
        self.pos_embedding = torch.randn((img_size // patch_size)**2 + 1, emb_size) #positional embedding of patch

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


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.qs = nn.Linear(emb_size, emb_size)
        self.ks = nn.Linear(emb_size, emb_size)
        self.vs = nn.Linear(emb_size, emb_size)

        self.A_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        b,n,_ = x.shape #b: batch_size || n: input_size
        d = int(self.emb_size / self.num_heads) # d: sequence_length

        # Create 'num_heads' subsets of Query, Key and Values with shape-> (BATCH_SIZE:b, NUM_HEADS:h, SEQUENCE_LENGTH:qdk, EMBEDDING_SIZE:d )
        Q = self.qs(x).reshape((b, self.num_heads, n, d))
        K = self.ks(x).reshape((b, self.num_heads, n, d))
        V = self.vs(x).reshape((b, self.num_heads, n, d))

        # Basic Attention
        attention = torch.einsum('bhqd, bhkd -> bhqk', Q, K) / np.sqrt(Q.shape[-1]) # Matrix product between Q and K and Scaling

        if mask is not None:
            attention = attention.masked_fill(mask[None], -np.inf) #masked_fill -> torch function

        attention = torch.softmax(attention, dim=3)
        attention = self.A_drop(attention) #attention vector

        # output processing
        output = torch.einsum('bhal, bhlv -> bhav ', attention, V) #use attention to scale the values
        output = torch.transpose(output, 1, 2)
        output= torch.flatten(output, start_dim=2, end_dim=3)
        output = self.projection(output)

        return attention, output


class Residuals(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.ffnn = ffnn #Feed Forward Neural Network ???

    def forward(self, x, **kwargs):
        res = x
        x = self.ffnn(x, **kwargs)
        x += res
        return x


# The feed foeward network with 2 linear layers with a GELU nonlinearity
class MLP(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, dropout: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )
