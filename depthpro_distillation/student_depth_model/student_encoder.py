# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# DepthProEncoder combining patch and image encoders.

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthProEncoder(nn.Module):
    """DepthPro Encoder.

    An encoder aimed at creating multi-resolution encodings from Vision Transformers.
    """

    def __init__(
        self,
        dims_encoder: Iterable[int],
        patch_encoder: nn.Module,
        image_encoder: nn.Module,
        hook_block_ids: Iterable[int],
        decoder_features: int,
    ):
        """Initialize DepthProEncoder.

        The framework
            1. creates an image pyramid,
            2. generates overlapping patches with a sliding window at each pyramid level,
            3. creates batched encodings via vision transformer backbones,
            4. produces multi-resolution encodings.

        Args:
        ----
            img_size: Backbone image resolution.
            dims_encoder: Dimensions of the encoder at different layers.
            patch_encoder: Backbone used for patches.
            image_encoder: Backbone used for global image encoder.
            hook_block_ids: Hooks to obtain intermediate features for the patch encoder model.
            decoder_features: Number of feature output in the decoder.

        """
        super().__init__()

        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder
        self.hook_block_ids = list(hook_block_ids)

        patch_encoder_embed_dim = patch_encoder.embed_dim
        image_encoder_embed_dim = image_encoder.embed_dim

        self.out_size = int(
            patch_encoder.patch_embed.img_size[0] // patch_encoder.patch_embed.patch_size[0]
        )

        def _create_project_upsample_block(
            dim_in: int,
            dim_out: int,
            upsample_layers: int,
            dim_int: Optional[int] = None,
            input_size: Optional[int] = None,
            target_sizes: Optional[List[int]] = None,
        ) -> nn.Module:
            """
            Create an upsampling block with optional custom first upsampling step.
            
            Args:
                dim_in: Input dimension (channels)
                dim_out: Output dimension (channels)
                upsample_layers: Number of upsampling layers
                dim_int: Intermediate dimension after projection (defaults to dim_out)
                input_size: Current spatial size (e.g., 36 or 24)
                target_sizes: List of target sizes after each upsampling [e.g., 96, 192, 384]
                            If not provided, will use standard 2x upsampling
            """
            if dim_int is None:
                dim_int = dim_out
            
            # Standard case: just do projection and then regular upsampling
            if input_size is None or target_sizes is None:
                # Projection
                blocks = [
                    nn.Conv2d(
                        in_channels=dim_in,
                        out_channels=dim_int,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    )
                ]
                
                # Standard 2x upsampling
                blocks += [
                    nn.ConvTranspose2d(
                        in_channels=dim_int if i == 0 else dim_out,
                        out_channels=dim_out,
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        bias=False,
                    )
                    for i in range(upsample_layers)
                ]
                
                return nn.Sequential(*blocks)
            
            # Custom case: handle specific input and target sizes
            blocks = [
                # Projection
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_int,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ]
            
            # Check if first upsampling needs special handling
            if len(target_sizes) > 0:
                first_target = target_sizes[0]
                scale_factor = first_target / input_size
                
                # If scale factor is close to 2 (standard case), use ConvTranspose2d
                if abs(scale_factor - 2) < 0.1:
                    blocks.append(
                        nn.ConvTranspose2d(
                            in_channels=dim_int,
                            out_channels=dim_out,
                            kernel_size=2,
                            stride=2,
                            padding=0,
                            bias=False,
                        )
                    )
                # Otherwise use custom bilinear upsampling
                else:
                    blocks.append(
                        nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
                    )
                    blocks.append(
                        nn.Conv2d(
                            in_channels=dim_int,
                            out_channels=dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        )
                    )
            
            # Remaining upsampling layers (if any)
            if len(target_sizes) > 1:
                for i in range(len(target_sizes) - 1):
                    scale_factor = target_sizes[i+1] / target_sizes[i]
                    
                    # If scale factor is close to 2 (standard case), use ConvTranspose2d
                    if abs(scale_factor - 2) < 0.1:
                        blocks.append(
                            nn.ConvTranspose2d(
                                in_channels=dim_out,
                                out_channels=dim_out,
                                kernel_size=2,
                                stride=2,
                                padding=0,
                                bias=False,
                            )
                        )
                    # Otherwise use custom bilinear upsampling
                    else:
                        blocks.append(
                            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
                        )
                        blocks.append(
                            nn.Conv2d(
                                in_channels=dim_out,
                                out_channels=dim_out,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            )
                        )
            
            return nn.Sequential(*blocks)

        self.upsample_latent0 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim,
            dim_int=self.dims_encoder[0],
            dim_out=decoder_features,
            upsample_layers=3,
            input_size=36,
            target_sizes=[96,192,384]
        )
        self.upsample_latent1 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[0], upsample_layers=2, input_size=36, target_sizes=[96,192]
        )

        self.upsample0 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[1], upsample_layers=1, input_size = 36, target_sizes=[96] # CHANGED FROM 2: CHECK
        )
        self.upsample1 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[2], upsample_layers=1 # CHANGED FROM 1: CHECK 
        )
        self.upsample2 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[3], upsample_layers=1
        )

        self.upsample_lowres = nn.ConvTranspose2d(
            in_channels=image_encoder_embed_dim,
            out_channels=self.dims_encoder[3],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.fuse_lowres = nn.Conv2d(
            in_channels=(self.dims_encoder[3] + self.dims_encoder[3]),
            out_channels=self.dims_encoder[3],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Obtain intermediate outputs of the blocks.
        self.patch_encoder.blocks[self.hook_block_ids[0]].register_forward_hook(
            self._hook0
        )
        self.patch_encoder.blocks[self.hook_block_ids[1]].register_forward_hook(
            self._hook1
        )

    def _hook0(self, model, input, output):
        self.backbone_highres_hook0 = output

    def _hook1(self, model, input, output):
        self.backbone_highres_hook1 = output

    @property
    def img_size(self) -> int:
        """Return the full image size of the SPN network."""
        return self.patch_encoder.patch_embed.img_size[0] * 4

    def _create_pyramid(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a 3-level image pyramid."""
        # Original resolution: 1536 by default. changed to 384
        x0 = x

        # Middle resolution: 768 by default. changed to 192
        x1 = F.interpolate(
            x, size=None, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        # Low resolution: 384 by default, corresponding to the backbone resolution. change to 96
        x2 = F.interpolate(
            x, size=None, scale_factor=0.25, mode="bilinear", align_corners=False
        )

        return x0, x1, x2

    def split(self, x: torch.Tensor, overlap_ratio: float = 0.25) -> torch.Tensor:
        """Split the input into small patches with sliding window."""
        patch_size = 96 #changed from 384
        patch_stride = int(patch_size * (1 - overlap_ratio))

        image_size = x.shape[-1]
        steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

        x_patch_list = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size

            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                x_patch_list.append(x[..., j0:j1, i0:i1])

        return torch.cat(x_patch_list, dim=0)

    def merge(self, x: torch.Tensor, batch_size: int, padding: int = 3) -> torch.Tensor:
        """Merge the patched input into a image with sliding window."""
        steps = int(math.sqrt(x.shape[0] // batch_size))

        idx = 0

        output_list = []
        for j in range(steps):
            output_row_list = []
            for i in range(steps):
                output = x[batch_size * idx : batch_size * (idx + 1)]

                if j != 0:
                    output = output[..., padding:, :]
                if i != 0:
                    output = output[..., :, padding:]
                if j != steps - 1:
                    output = output[..., :-padding, :]
                if i != steps - 1:
                    output = output[..., :, :-padding]

                output_row_list.append(output)
                idx += 1

            output_row = torch.cat(output_row_list, dim=-1)
            output_list.append(output_row)
        output = torch.cat(output_list, dim=-2)
        return output

    def reshape_feature(
        self, embeddings: torch.Tensor, width, height, cls_token_offset=1
    ):
        """Discard class token and reshape 1D feature map to a 2D grid."""
        b, hw, c = embeddings.shape

        # Remove class token.
        if cls_token_offset > 0:
            embeddings = embeddings[:, cls_token_offset:, :]

        # Shape: (batch, height, width, dim) -> (batch, dim, height, width)
        embeddings = embeddings.reshape(b, height, width, c).permute(0, 3, 1, 2)
        return embeddings

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode input at multiple resolutions.

        Args:
        ----
            x (torch.Tensor): Input image.

        Returns:
        -------
            Multi resolution encoded features.

        """
        batch_size = x.shape[0]
        #print(f"[Debug] Input x        : {x.shape}")

        # Step 0: create a 3-level image pyramid.
        x0, x1, x2 = self._create_pyramid(x)    
        #print(f"[Debug] Pyramid x0 (384×384) : {x0.shape}")
        #print(f"[Debug] Pyramid x1 (192×192) : {x1.shape}")
        #print(f"[Debug] Pyramid x2 ( 96×96) : {x2.shape}")
        

        # Step 1: split to create batched overlapped mini-images at the backbone (BeiT/ViT/Dino)
        # resolution.
        # 5x5 @ 384x384 at the highest resolution (1536x1536).
        x0_patches = self.split(x0, overlap_ratio=0.25)
        # 3x3 @ 384x384 at the middle resolution (768x768).
        x1_patches = self.split(x1, overlap_ratio=0.5)
        # 1x1 # 384x384 at the lowest resolution (384x384).
        x2_patches = x2

        #print(f"[Debug] x0_patches (5×5 @96)  : {x0_patches.shape}")
        #print(f"[Debug] x1_patches (3×3 @96)  : {x1_patches.shape}")
        #print(f"[Debug] x2_patches (1×1 @96)  : {x2_patches.shape}")

        # Concatenate all the sliding window patches and form a batch of size (35=5x5+3x3+1x1).
        x_pyramid_patches = torch.cat(
            (x0_patches, x1_patches, x2_patches),
            dim=0,
        )
        #print(f"[Debug] Cat patches batch     : {x_pyramid_patches.shape}")

        # Step 2: Run the backbone (BeiT) model and get the result of large batch size.
        x_pyramid_encodings = self.patch_encoder(x_pyramid_patches)
        x_pyramid_encodings = self.reshape_feature(
            x_pyramid_encodings, self.out_size, self.out_size
        )
        #print(f"[Debug] Encodings reshaped    : {x_pyramid_encodings.shape}")

        # Step 3: merging.
        # Merge highres latent encoding.
        x_latent0_encodings = self.reshape_feature(
            self.backbone_highres_hook0,
            self.out_size,
            self.out_size,
        )
        #print(f"[Debug] x_latent0_encodings shape: {x_latent0_encodings.shape}")
        x_latent0_features = self.merge(
            x_latent0_encodings[: batch_size * 4 * 4], batch_size=batch_size, padding=2
        )
        #print(f"[Debug] x_latent0_features (after merge) shape: {x_latent0_features.shape}")

        x_latent1_encodings = self.reshape_feature(
            self.backbone_highres_hook1,
            self.out_size,
            self.out_size,
        )
        #print(f"[Debug] x_latent1_encodings shape: {x_latent1_encodings.shape}")
        x_latent1_features = self.merge(
            x_latent1_encodings[: batch_size * 4 * 4], batch_size=batch_size, padding=2
        )
        #print(f"[Debug] x_latent1_features (after merge) shape: {x_latent1_features.shape}")

        # Split the 35 batch size from pyramid encoding back into 5x5+3x3+1x1.
        x0_encodings, x1_encodings, x2_encodings = torch.split(
            x_pyramid_encodings,    
            [len(x0_patches), len(x1_patches), len(x2_patches)],
            dim=0,
        )
        #print(f"[Debug] x0_encodings batch shape: {x0_encodings.shape}")
        #print(f"[Debug] x1_encodings batch shape: {x1_encodings.shape}")
        #print(f"[Debug] x2_encodings batch shape: {x2_encodings.shape}")    

        # 96x96 feature maps by merging 5x5 @ 24x24 patches with overlaps. #NONE OF THESE DESCRIPTIONS MATCH (FROM TEACHER)
        x0_features = self.merge(x0_encodings, batch_size=batch_size, padding=3) # CHANGED FROM 3
        #print(f"[Debug] x0_features (after merge) shape: {x0_features.shape}")
        # 48x84 feature maps by merging 3x3 @ 24x24 patches with overlaps.
        x1_features = self.merge(x1_encodings, batch_size=batch_size, padding=3) # PADDING CHANGED FROM 6
        #print(f"[Debug] x1_features (after merge) shape: {x1_features.shape}")
        # 24x24 feature maps.
        x2_features = x2_encodings
        #print(f"[Debug] x2_features (no merge) shape: {x2_features.shape}")
        # Apply the image encoder model.
        x_global_features = self.image_encoder(x2_patches)
        x_global_features = self.reshape_feature(
            x_global_features, self.out_size, self.out_size
        )
        #print(f"[Debug] x_global_features (before upsample) shape: {x_global_features.shape}")
        # Upsample feature maps.
        x_latent0_features = self.upsample_latent0(x_latent0_features)
        #print(f"[Debug] x_latent0_features (after upsample) shape: {x_latent0_features.shape}")
        x_latent1_features = self.upsample_latent1(x_latent1_features)
        #print(f"[Debug] x_latent1_features (after upsample) shape: {x_latent1_features.shape}")
        x0_features = self.upsample0(x0_features)
        #print(f"[Debug] x0_features (after upsample) shape: {x0_features.shape}")
        x1_features = self.upsample1(x1_features)
        #print(f"[Debug] x1_features (after upsample) shape: {x1_features.shape}")
        x2_features = self.upsample2(x2_features)
        #print(f"[Debug] x2_features (after upsample) shape: {x2_features.shape}")
        x_global_features = self.upsample_lowres(x_global_features)
        #print(f"[Debug] x_global_features (after upsample) shape: {x_global_features.shape}")
        x_global_features = self.fuse_lowres(
            torch.cat((x2_features, x_global_features), dim=1)
        )
        ##prin(f"[Debug] x_global_features (after fuse) shape: {x_global_features.shape}")

        return [
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_global_features,
        ]
