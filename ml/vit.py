import torch

class ConvPatchEmbedder(torch.nn):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_size: int,
    ):
        super.__init__(self)
        self.patch_size = patch_size
        self.embed_size = embed_size
        num_patches = (image_size ** 2) // (patch_size ** 2)
        self.pos_embedding = nn.Parameter(
            torch.normal(
                mean=0,
                std=0.02,
                size=(1, num_patches, embed_size)
            )
        )
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=embed_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        ) # N, 3, H, W -> N, F, H // k, W // k
        self.cls_token = nn.Parameter((1, 1, embed_size))


    def forward(self, x: torch.Tensor):
        assert (
            x.size(dim=-2) % self.patch_size == 0
            and x.size(dim=-1) % self.patch_size == 0
        )
        out = self.conv1(x)
        out = out.transpose(0, 3, 1, 2) # N, F, H // k, W // k -> N, H // k * W // k, F
        out = out.reshape(out.size(dim=0), -1, self.embed_size) # N, H // k * W // k, F -> # N, num_patches, F
        out = torch.cat((self.cls_token.expand((out.size(dim=1), 1, self.embed_size)), out), dim=1) # N, num_patches + 1, F

        out = out + self.pos_embedding # N, num_patches + 1, F

        return out


class LinearPatchEmbedder(torch.nn):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_size: int,
    ):
        super.__init__(self)
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size ** 2) // (patch_size ** 2)
        self.embed_size = embed_size
        self.linear = nn.Linear(in_features=3 * patch_size ** 2, out_features=embed_size)
        self.pos_embedding = nn.Parameter(
            torch.normal(
                mean=0,
                std=0.02,
                size=(1, num_patches, embed_size)
            )
        )
        self.cls_token = nn.Parameter((1, 1, embed_size))



    def forward(self, x):
        x = x.view(
            *x.shape[:2],
            x.dim(2) // self.patch_size,
            self.patch_size,
            x.dim(3) // self.patch_size,
            self.patch_size
        ) # N, 3, H, W -> N, 3, H // patch_size, patch_size, W // patch_size, patch_size
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(x.size(dim=0), self.num_patches, self.patch_size ** 2 * 3)
        out = self.linear(x) # N, num_patches, F
        out = torch.cat((self.cls_token.expand((out.size(dim=1), 1, self.embed_size)), out), dim=1)
        out = torch.cat((self.cls_token.expand((x.size(dim=0), self.num_patches, self.embed_size)), out))
        out = out + self.pos_embedding
        return out


class MultiHeadAttention(torch.nn):
    def __init__(self, hidden_size: int, d_k: int, dropout: float = 0.1, n_heads: int = 4):
        self.d_k = d_k
        self.hidden_size = hidden_size
        self.qkv_linear = nn.Linear(
            in_features=hidden_size,
            out_features=3 * d_k
        )
        self.scale = d_k ** -0.5
        self.dropout = nn.Dropout(p=dropout)
        self.n_heads = n_heads
        self.projection = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Dropout(p=dropout)
        )


    def project(self, x):
        B, S, _ = x.shape
        out = self.qkv_linear(x) # N, S, H -> N, S, d_k * 3
        q, k, v = out.split(self.d_k, dim=2)
        q, k, v = (
            q.view(B, S, self.n_heads, self.d_k // self.n_heads).transpose(0, 2, 1, 3),
            k.view(B, S, self.n_heads, self.d_k // self.n_heads).transpose(0, 2, 1, 3),
            v.view(B, S, self.n_heads, self.d_k // self.n_heads).transpose(0, 2, 1, 3),
        ) # N, num_heads, S, d_k // num_heads
        return q, k, v


    def forward(self, x):
        q, k, v = self.project(x)
        out = (
            self.dropout(torch.softmax((q @ k.T) * self.scale, dim=-1)) # N, num_heads, S_source, S_target
            @ v # N, num_heads, S_target, d_k // num_heads
        ) # N, num_heads, S_target, dk // num_heads
        out = out.transpose(1, 2) # N, S_target, num_heads, dk // num_heads
        out = out.contiguous().reshape(x.shape[:2], self.d_k) # N, S_target, num_heads, dk
        return self.projection(out)

class MLP(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_hidden_size: int = 3072,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=embed_dim,
                out_features=mlp_hidden_size,
            ),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(
                in_features=mlp_hidden_size,
                out_features=embed_dim,
            ),
            torch.nn.Dropout(dropout_rate),
        )

    @typechecked
    def forward(self, tensor: EmbeddedPatches) -> EmbeddedPatches:
        return self.mlp(tensor)



class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        n_heads: int = 12,
        qkv_dim: int = 64,
        embed_dim: int = 768,
        mlp_hidden_size: int = 3072,
        attention_dropout_rate: float = 0.1,
        mlp_dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(normalized_shape=embed_dim)
        self.mhsa = MultiHeadAttention(
            n_heads, embed_dim, qkv_dim, attention_dropout_rate
        )
        self.ln2 = torch.nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_size, mlp_dropout_rate)

    @typechecked
    def forward(self, tensor: EmbeddedPatches) -> EmbeddedPatches:
        tensor += self.mhsa(self.ln1(tensor))
        tensor += self.mlp(self.ln2(tensor))
        return tensor


class ViT(torch.nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 3072,
        n_layers: int = 12,
        n_heads: int = 12,
        n_classes: int = 1_000,
    ):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            ConvPatchEmbedder(image_size, patch_size, embed_dim),
            *[EncoderBlock(n_heads, qkv_dim, embed_dim, mlp_hidden_size) for _ in range(n_layers)]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(in_features=embed_dim, out_features=n_classes)
        )

    def forward(self, x):
        features = self.encoder(tensor)
        return self.classifier(features[:, 0, :])
