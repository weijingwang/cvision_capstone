import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


from ml_depth_pro.src.depth_pro.depth_pro import create_model_and_transforms, DepthPro, DepthProConfig, DEFAULT_MONODEPTH_CONFIG_DICT
from ml_depth_pro.src.depth_pro.network.decoder import MultiresConvDecoder
from ml_depth_pro.src.depth_pro.network.encoder import DepthProEncoder
from ml_depth_pro.src.depth_pro.network.vit_factory import VIT_CONFIG_DICT, create_vit

def create_student_model(scale_factor=0.25):
    """
    Create student model at 384×384 resolution with reduced channel width
    """
    # Start with the 384×384 configuration
    student_config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=None,  # We don't load weights for the student initially
        decoder_features=int(256 * scale_factor),  # Scale down decoder features
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384"
    )
    
    # Create backbone models with reduced dimensions
    patch_encoder, patch_encoder_config = create_backbone_model(
        preset=student_config.patch_encoder_preset,
        scale_factor=scale_factor
    )
    image_encoder, _ = create_backbone_model(
        preset=student_config.image_encoder_preset,
        scale_factor=scale_factor
    )
    
    fov_encoder = None
    if student_config.use_fov_head and student_config.fov_encoder_preset is not None:
        fov_encoder, _ = create_backbone_model(
            preset=student_config.fov_encoder_preset,
            scale_factor=scale_factor
        )
    
    # Adjust encoder feature dimensions
    dims_encoder = [int(dim * scale_factor) for dim in patch_encoder_config.encoder_feature_dims]
    hook_block_ids = patch_encoder_config.encoder_feature_layer_ids
    
    # Create encoder with scaled dimensions
    encoder = DepthProEncoder(
        dims_encoder=dims_encoder,
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        hook_block_ids=hook_block_ids,
        decoder_features=student_config.decoder_features,
    )
    
    # Create decoder with scaled dimensions
    decoder = MultiresConvDecoder(
        dims_encoder=[student_config.decoder_features] + list(encoder.dims_encoder),
        dim_decoder=student_config.decoder_features,  
    )
    
    # Assemble the student model
    model = DepthPro(
        encoder=encoder,
        decoder=decoder,
        last_dims=(int(32 * scale_factor), 1),  # Scale down the last dimensions
        use_fov_head=student_config.use_fov_head,
        fov_encoder=fov_encoder,
    )
    
    return model


def create_backbone_model(preset, scale_factor=1.0):
    """Create and load a backbone model with optional scaling for feature dimensions"""
    # Get the original config
    if preset in VIT_CONFIG_DICT:
        config = VIT_CONFIG_DICT[preset].copy()  # Make a copy to avoid modifying the original
        
        if scale_factor != 1.0:
            # Scale down embedding dimensions and number of heads
            config.embed_dim = int(config.embed_dim * scale_factor)
            config.num_heads = max(1, int(config.num_heads * scale_factor))
            config.mlp_ratio = config.mlp_ratio  # Keep the same ratio
            
            # Scale down encoder feature dimensions
            config.encoder_feature_dims = tuple(int(dim * scale_factor) for dim in config.encoder_feature_dims)
            
        # Create ViT with scaled dimensions
        model = create_vit(preset=preset, use_pretrained=False, config_override=config)
    else:
        raise KeyError(f"Preset {preset} not found.")

    return model, config


class FeatureProjector(nn.Module):
    """Projects student features to teacher feature space"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        # Handle different tensor dimensions
        original_shape = x.shape
        # Flatten features to apply linear projection
        if len(original_shape) > 2:
            x = x.reshape(original_shape[0], original_shape[1], -1)
            x = x.permute(0, 2, 1)  # [B, HW, C]
            x = self.projection(x)  # Project features
            x = x.permute(0, 2, 1)  # [B, C, HW]
            x = x.reshape(original_shape[0], -1, original_shape[2], original_shape[3])
        else:
            x = self.projection(x)
        return x


