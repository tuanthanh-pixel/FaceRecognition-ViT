import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.number_of_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images):
        patches = self.projection(images)
        patches = patches.flatten(start_dim=2)
        return patches.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        return self.layers(tokens)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm_attention = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, hidden_dim, dropout)

    def forward(self, tokens):
        normalized_tokens = self.norm_attention(tokens)
        attention_output, _ = self.attention(
            normalized_tokens,
            normalized_tokens,
            normalized_tokens,
            need_weights=False,
        )
        tokens = tokens + attention_output
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return tokens


class FaceVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout,
        face_embedding_dim,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size,
            patch_size,
            in_channels,
            embed_dim,
        )
        number_of_patches = self.patch_embedding.number_of_patches
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, number_of_patches + 1, embed_dim)
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.embedding_head = nn.Linear(embed_dim, face_embedding_dim)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward_features(self, images):
        patch_tokens = self.patch_embedding(images)
        batch_size = patch_tokens.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat((class_tokens, patch_tokens), dim=1)
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        tokens = self.encoder_blocks(tokens)
        tokens = self.final_norm(tokens)
        return tokens[:, 0]

    def forward(self, images):
        class_features = self.forward_features(images)
        embeddings = self.embedding_head(class_features)
        return F.normalize(embeddings, p=2, dim=1)


def build_model(cfg):
    return FaceVisionTransformer(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_channels=3,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        face_embedding_dim=cfg.face_embedding_dim,
    )


if __name__ == "__main__":
    from config import get_parser

    cfg = get_parser()
    model = build_model(cfg)
    images = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    embeddings = model(images)
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    parameters = sum(parameter.numel() for parameter in model.parameters())

    print(f"Input image shape: {images.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding norms: {norms}")
    print(f"Total parameters: {parameters:,}")
