import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils.OutputTrainingHook import OutputTrainingHook
from .augdfa_nets import AlignmentProbe, AuxDFANetwork

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def build_dfa_layer_indices(depth, dfa_num):
    if depth <= 0 or dfa_num <= 0:
        return set()

    dfa_num = min(dfa_num, depth)
    return {
        ((hook_idx + 1) * depth - 1) // dfa_num
        for hook_idx in range(dfa_num)
    }

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, train_mode, B_dim, class_num, dropout = 0., use_dfa = True):
        super().__init__()
        self.train_mode = train_mode
        self.use_dfa = use_dfa
        self.dfa1 = AuxDFANetwork(class_num, B_dim, train_mode) if use_dfa else None
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, target, grad_at_output):
        x = self.net(x)
        if not self.use_dfa:
            return x.new_zeros(()), x

        loss, _ = self.dfa1(x, target, grad_at_output)
        if self.training:
            x = x.detach()
        return loss, x

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, train_mode, B_dim, class_num, dfa_num, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])
        dfa_layer_indices = build_dfa_layer_indices(depth, dfa_num)

        for layer_idx in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(
                    dim,
                    mlp_dim,
                    train_mode,
                    B_dim,
                    class_num,
                    dropout = dropout,
                    use_dfa = layer_idx in dfa_layer_indices,
                )
            ]))

    def forward(self, x, target, grad_at_output):
        aug_loss = x.new_zeros(())
        for attn, ff in self.layers:
            x = attn(x) + x
            loss, ff_out = ff(x, target, grad_at_output)
            if self.training:
                aug_loss = aug_loss + loss
            x = ff_out + x

        return aug_loss, self.norm(x)

class ViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, train_mode, class_num, B_dim, dfa_num = 12, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.num_classes = num_classes
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.train_mode = train_mode
        self.B_dim = B_dim
        self.dfa_num = dfa_num
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, train_mode, B_dim, class_num, dfa_num, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

        self.output_hook = OutputTrainingHook()

    def forward(self, img, target):
        batch = img.shape[0]
        grad_at_output = torch.zeros([batch, self.num_classes], requires_grad=False).to(img.device)
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        aug_loss, x = self.transformer(x, target, grad_at_output)
        if self.mlp_head is None:
            return x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        x = self.output_hook(x, grad_at_output)
        return x, aug_loss
