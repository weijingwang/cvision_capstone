from ml_depth_pro.src.depth_pro.depth_pro import DepthPro
import copy, math
import torch
import torch.nn as nn
from timm.models.vision_transformer import LayerScale
from utils import interpolate_pos_embed
from config import GRID_SIZES, STUDENT_DIM, STUDENT_HEADS
from ml_depth_pro.src.depth_pro.depth_pro import DepthPro
import copy, math
import torch
import torch.nn as nn
from timm.models.vision_transformer import LayerScale
from utils import interpolate_pos_embed
from config import GRID_SIZES, STUDENT_DIM, STUDENT_HEADS

class DepthProStudent(nn.Module):
    """
    Student that runs at 384×384 input with fixed 96×96 patches at each scale
    (no 1536px upsampling), but still uses the multi‑scale pipeline.
    """

    def __init__(
        self,
        teacher: DepthPro,
        grid_sizes=GRID_SIZES,    # [96, 192, 384]
        student_dim=STUDENT_DIM,  # e.g. 192
        heads=STUDENT_HEADS,      # e.g. 3
    ):
        super().__init__()
        # 1) copy teacher’s encoder & decoder
        self.encoder = copy.deepcopy(teacher.encoder)
        self.decoder = copy.deepcopy(teacher.decoder)

        # 2) force all the multiscale bookkeeping to your student sizes
        student_sz = grid_sizes[-1]
        enc = self.encoder
        enc.img_size        = student_sz
        enc.scales          = list(grid_sizes)
        enc.multiscale_size = list(grid_sizes)

        # 3) shorthands
        pe    = enc.patch_encoder   # the timm.VisionTransformer
        ie    = enc.image_encoder
        ps_pe = pe.patch_embed.patch_size[0]
        ps_ie = ie.patch_embed.patch_size[0]

        # 4) --- shrink the patch_encoder to student_dim ---
        # 4a) conv→student_dim
        old_proj = pe.patch_embed.proj
        pe.patch_embed.proj = nn.Conv2d(
            old_proj.in_channels,
            student_dim,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
        )
        pe.embed_dim = pe.patch_embed.embed_dim = student_dim

        # 4b) transformer blocks
        for blk in pe.blocks:
            if hasattr(blk, 'attn'):
                blk.attn.qkv = nn.Linear(student_dim, student_dim*3,
                                         bias=blk.attn.qkv.bias is not None)
                blk.attn.proj = nn.Linear(student_dim, student_dim,
                                          bias=blk.attn.proj.bias is not None)
                blk.attn.num_heads = heads
                blk.attn.head_dim  = student_dim // heads
                blk.attn.scale     = (student_dim//heads)**-0.5
            if hasattr(blk, 'mlp'):
                blk.mlp.fc1 = nn.Linear(student_dim, student_dim*4,
                                        bias=blk.mlp.fc1.bias is not None)
                blk.mlp.fc2 = nn.Linear(student_dim*4, student_dim,
                                        bias=blk.mlp.fc2.bias is not None)
            if hasattr(blk, 'norm1'): blk.norm1 = nn.LayerNorm(student_dim)
            if hasattr(blk, 'norm2'): blk.norm2 = nn.LayerNorm(student_dim)
            if hasattr(blk, 'ls1'):
                init = float(blk.ls1.gamma.mean().detach().cpu())
                blk.ls1 = LayerScale(student_dim, init_values=init)
            if hasattr(blk, 'ls2'):
                init = float(blk.ls2.gamma.mean().detach().cpu())
                blk.ls2 = LayerScale(student_dim, init_values=init)
        if hasattr(pe, 'norm'):
            pe.norm = nn.LayerNorm(student_dim)

        # 4c) now re‑project teacher’s cls_token & pos_embed → student_dim
        old_dim    = teacher.encoder.patch_encoder.pos_embed.shape[-1]
        grid_last  = student_sz // ps_pe
        proj_token = nn.Linear(old_dim, student_dim, bias=False).to(
            pe.cls_token.device
        )
        # CLS token
        pe.cls_token = nn.Parameter(
            proj_token(teacher.encoder.patch_encoder.cls_token.data)
        )
        # POS embed (interpolate then project)
        pe.pos_embed = nn.Parameter(
            proj_token(
                interpolate_pos_embed(
                    teacher.encoder.patch_encoder.pos_embed.data,
                    grid_last
                )
            )
        )

        # 5) shrink the image_encoder the same way (for the fov head)
        grid_ie = grid_sizes[0] // ps_ie
        ie.pos_embed.data = interpolate_pos_embed(ie.pos_embed.data, grid_ie)
        old_proj = ie.patch_embed.proj
        ie.patch_embed.proj = nn.Conv2d(
            old_proj.in_channels, student_dim,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
        )
        ie.embed_dim = ie.patch_embed.embed_dim = student_dim
        for blk in ie.blocks:
            if hasattr(blk, 'attn'):
                blk.attn.qkv = nn.Linear(student_dim, student_dim*3,
                                         bias=blk.attn.qkv.bias is not None)
                blk.attn.proj = nn.Linear(student_dim, student_dim,
                                          bias=blk.attn.proj.bias is not None)
                blk.attn.num_heads = heads
                blk.attn.head_dim  = student_dim // heads
                blk.attn.scale     = (student_dim//heads)**-0.5
            if hasattr(blk, 'mlp'):
                blk.mlp.fc1 = nn.Linear(student_dim, student_dim*4,
                                        bias=blk.mlp.fc1.bias is not None)
                blk.mlp.fc2 = nn.Linear(student_dim*4, student_dim,
                                        bias=blk.mlp.fc2.bias is not None)
            if hasattr(blk, 'norm1'): blk.norm1 = nn.LayerNorm(student_dim)
            if hasattr(blk, 'norm2'): blk.norm2 = nn.LayerNorm(student_dim)
            if hasattr(blk, 'ls1'):
                init = float(blk.ls1.gamma.mean().detach().cpu())
                blk.ls1 = LayerScale(student_dim, init_values=init)
            if hasattr(blk, 'ls2'):
                init = float(blk.ls2.gamma.mean().detach().cpu())
                blk.ls2 = LayerScale(student_dim, init_values=init)
        if hasattr(ie, 'norm'):
            ie.norm = nn.LayerNorm(student_dim)

        # 6) shrink all 1×1 convs in the decoder
        def shrink_conv1x1(m):
            for name, ch in m.named_children():
                if isinstance(ch, nn.Conv2d) and ch.kernel_size==(1,1):
                    in_c  = max(1, int(ch.in_channels  * student_dim/pe.embed_dim))
                    out_c = max(1, int(ch.out_channels * student_dim/pe.embed_dim))
                    setattr(m, name, nn.Conv2d(in_c, out_c, 1))
                else:
                    shrink_conv1x1(ch)
        shrink_conv1x1(self.decoder)

        # 7) fixed 96×96 split override
        patch_size = grid_sizes[0]
        def split_fixed(this, x, overlap_ratio=0.5):
            B,C,H,W = x.shape
            ps      = patch_size
            stride  = int(ps*(1-overlap_ratio))
            steps   = int(math.ceil((H-ps)/stride))+1
            patches = []
            for j in range(steps):
                for i in range(steps):
                    j0,j1 = j*stride, j*stride+ps
                    i0,i1 = i*stride, i*stride+ps
                    patches.append(x[:,:,j0:j1,i0:i1])
            return torch.cat(patches, dim=0)
        enc.split = split_fixed.__get__(enc, enc.__class__)

        # 8) reshape_feature for a single scale (B, H*W, C) → (B, C, H, W)
        def reshape_student(this, embeddings, height, width):
            B, N, C = embeddings.shape
            assert N == height*width
            return embeddings.permute(0,2,1).view(B, C, height, width)
        enc.reshape_feature = reshape_student.__get__(enc, enc.__class__)

        # 9) distillation hooks
        self.student_feats = {}
        for idx in teacher.encoder.hook_block_ids:
            pe.blocks[idx].register_forward_hook(
                lambda m,i,o,n=idx: self.student_feats.update({f'patch{n}': o})
            )
        if hasattr(ie, 'norm'):
            ie.norm.register_forward_hook(
                lambda m,i,o: self.student_feats.update({'img_feat': o})
            )

    def forward(self, x):
        self.student_feats.clear()
        feats = self.encoder(x)   # runs split_fixed + reshape_student
        return self.decoder(feats)


class FeatureProjector1x1(nn.Module):
    """1×1 conv mapping student→teacher feature channels."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.proj = nn.Conv2d(in_c, out_c, 1)
    def forward(self, x):
        return self.proj(x)
