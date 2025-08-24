"""
Token-only modules for MTRRNet refactor
Implements frequency-split encoder stages, token fusion, and unified decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
from timm.models.swin_transformer import SwinTransformerBlock
from timm.layers import DropPath, LayerNorm2d
try:
    from mamba_ssm import Mamba
except ImportError:
    from mamba_ssm_mock import Mamba

class FrequencySplit(nn.Module):
    """Split input into low and high frequency components using learnable blur"""
    def __init__(self, channels=3, blur_kernel_size=5, sigma=1.0):
        super().__init__()
        self.channels = channels
        self.blur_kernel_size = blur_kernel_size
        
        # Create gaussian blur kernel
        kernel = self._create_gaussian_kernel(blur_kernel_size, sigma)
        kernel = kernel.repeat(channels, 1, 1, 1)  # (C, 1, K, K)
        self.register_buffer('blur_kernel', kernel)
        
        # Learnable adjustment for the blur (small perturbations)
        self.blur_adjust = nn.Parameter(torch.zeros_like(kernel) * 0.01)
        
    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create a 2D Gaussian kernel"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        kernel = g[:, None] * g[None, :]
        return kernel[None, None, :, :]  # (1, 1, K, K)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            low: Low frequency component (B, C, H, W)
            high: High frequency component (B, C, H, W)
        """
        # Apply learnable blur
        effective_kernel = self.blur_kernel + self.blur_adjust
        padding = self.blur_kernel_size // 2
        
        # Group convolution for per-channel blur
        low = F.conv2d(x, effective_kernel, padding=padding, groups=self.channels)
        high = x - low
        
        return low, high


class ResidualMambaTokenStage(nn.Module):
    """Mamba processing stage that operates on tokens with proper residuals"""
    def __init__(self, in_channels, embed_dim, patch_size, img_size, num_blocks=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_blocks = num_blocks
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            flatten=False,
            norm_layer=LayerNorm2d
        )
        
        # Mamba blocks with proper residual connections
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(embed_dim),
                Mamba(embed_dim)
            ))
            
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            tokens: Token tensor (B, N, D) where N = (H//patch_size) * (W//patch_size)
        """
        # Patch embedding: (B, C, H, W) -> (B, D, H//P, W//P)
        x_emb = self.patch_embed(x)
        B, D, H, W = x_emb.shape
        
        # Convert to token format: (B, D, H, W) -> (B, N, D)
        tokens = x_emb.flatten(2).transpose(1, 2)  # (B, H*W, D)
        
        # Apply Mamba blocks with residual connections
        for block in self.blocks:
            residual = tokens
            tokens = block(tokens) + residual  # Proper residual: x = x + f(LN(x))
            
        return tokens  # (B, N, D)


class ResidualSwinTokenStage(nn.Module):
    """Swin Transformer processing stage that operates on tokens with proper residuals"""
    def __init__(self, in_channels, embed_dim, patch_size, img_size, window_size=8, num_blocks=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.window_size = window_size
        self.num_blocks = num_blocks
        
        # Calculate spatial resolution after patching
        self.input_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            flatten=False,
            norm_layer=LayerNorm2d
        )
        
        # Swin transformer blocks with proper residual connections
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(embed_dim),
                SwinTransformerBlock(
                    dim=embed_dim,
                    input_resolution=self.input_resolution,
                    num_heads=embed_dim // 32,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=4.0
                )
            ))
            
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            tokens: Token tensor (B, N, D) where N = (H//patch_size) * (W//patch_size)
        """
        # Patch embedding: (B, C, H, W) -> (B, D, H//P, W//P)
        x_emb = self.patch_embed(x)
        B, D, H, W = x_emb.shape
        
        # Convert to Swin format: (B, D, H, W) -> (B, H, W, D)
        x_swin = x_emb.permute(0, 2, 3, 1)  # (B, H, W, D)
        
        # Apply Swin blocks with residual connections
        for block in self.blocks:
            residual = x_swin
            x_swin = block(x_swin) + residual  # Proper residual: x = x + f(LN(x))
            
        # Convert back to token format: (B, H, W, D) -> (B, N, D)
        tokens = x_swin.flatten(1, 2)  # (B, H*W, D)
        
        return tokens  # (B, N, D)


class TokenMerge(nn.Module):
    """Merge tokens from Mamba and Swin branches"""
    def __init__(self, embed_dim, merge_type='concat_linear'):
        super().__init__()
        self.merge_type = merge_type
        
        if merge_type == 'concat_linear':
            # Concatenate and project back to embed_dim
            self.proj = nn.Linear(embed_dim * 2, embed_dim)
        elif merge_type == 'average':
            # Simple average (requires same dimensions)
            pass
        else:
            raise ValueError(f"Unknown merge_type: {merge_type}")
            
    def forward(self, mamba_tokens, swin_tokens):
        """
        Args:
            mamba_tokens: (B, N, D)
            swin_tokens: (B, N, D)
        Returns:
            merged_tokens: (B, N, D)
        """
        if self.merge_type == 'concat_linear':
            # Concatenate along feature dimension and project
            concat_tokens = torch.cat([mamba_tokens, swin_tokens], dim=-1)  # (B, N, 2D)
            merged_tokens = self.proj(concat_tokens)  # (B, N, D)
        elif self.merge_type == 'average':
            # Simple average
            merged_tokens = (mamba_tokens + swin_tokens) / 2.0
        
        return merged_tokens


class TokenSubNet(nn.Module):
    """Token-space fusion subnet that operates on multi-scale token features"""
    def __init__(self, token_dims, num_iterations=2):
        super().__init__()
        self.token_dims = token_dims  # List of embedding dimensions for each scale
        self.num_iterations = num_iterations
        
        # Cross-scale attention for token fusion
        self.cross_attentions = nn.ModuleList()
        for _ in range(num_iterations):
            self.cross_attentions.append(
                TokenCrossScaleAttention(token_dims)
            )
        
        # Learnable residual scaling parameters
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5) for _ in range(len(token_dims))
        ])
    
    def forward(self, token_list):
        """
        Args:
            token_list: List of [T0, T1, T2, T3] where Ti is (B, Ni, Di)
        Returns:
            refined_tokens: List of refined token tensors
        """
        refined_tokens = token_list.copy()
        
        for iteration in range(self.num_iterations):
            # Apply cross-scale attention
            updated_tokens = self.cross_attentions[iteration](refined_tokens)
            
            # Residual connection with learnable scaling
            for i in range(len(refined_tokens)):
                refined_tokens[i] = self.alphas[i] * refined_tokens[i] + updated_tokens[i]
        
        return refined_tokens


class TokenCrossScaleAttention(nn.Module):
    """Cross-scale attention mechanism for token fusion"""
    def __init__(self, token_dims, target_spatial_size=16):
        super().__init__()
        self.token_dims = token_dims
        self.target_spatial_size = target_spatial_size
        
        # Project all scales to same dimension for attention
        common_dim = max(token_dims)
        self.scale_projections = nn.ModuleList([
            nn.Linear(dim, common_dim) for dim in token_dims
        ])
        
        # Multi-head cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Project back to original dimensions
        self.output_projections = nn.ModuleList([
            nn.Linear(common_dim, dim) for dim in token_dims
        ])
    
    def forward(self, token_list):
        """Apply cross-scale attention fusion"""
        # Project all tokens to common dimension and spatial size
        projected_tokens = []
        for i, tokens in enumerate(token_list):
            # Project to common dimension
            proj_tokens = self.scale_projections[i](tokens)  # (B, Ni, common_dim)
            
            # Resize spatially if needed (simple interpolation in token space)
            if proj_tokens.shape[1] != self.target_spatial_size ** 2:
                B, N, D = proj_tokens.shape
                H = W = int(N ** 0.5)  # Assume square spatial layout
                proj_tokens = proj_tokens.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
                proj_tokens = F.interpolate(
                    proj_tokens, 
                    size=(self.target_spatial_size, self.target_spatial_size), 
                    mode='bilinear', 
                    align_corners=False
                )
                proj_tokens = proj_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # (B, target_size^2, D)
            
            projected_tokens.append(proj_tokens)
        
        # Concatenate all scales as context
        all_context = torch.cat(projected_tokens, dim=1)  # (B, sum(Ni), common_dim)
        
        # Apply cross attention for each scale
        updated_tokens = []
        for i, proj_tokens in enumerate(projected_tokens):
            # Cross attention: query from current scale, key/value from all scales
            attn_output, _ = self.cross_attn(proj_tokens, all_context, all_context)
            
            # Project back to original dimension
            output_tokens = self.output_projections[i](attn_output)
            
            # Resize back to original spatial size if needed
            if output_tokens.shape[1] != token_list[i].shape[1]:
                B, _, D = output_tokens.shape
                H = W = int(token_list[i].shape[1] ** 0.5)
                output_tokens = output_tokens.view(B, self.target_spatial_size, self.target_spatial_size, D)
                output_tokens = output_tokens.permute(0, 3, 1, 2)  # (B, D, target_size, target_size)
                output_tokens = F.interpolate(output_tokens, size=(H, W), mode='bilinear', align_corners=False)
                output_tokens = output_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, D)
            
            updated_tokens.append(output_tokens)
        
        return updated_tokens


class UnifiedDecoder(nn.Module):
    """Unified decoder that processes fused tokens to final 6-channel output"""
    def __init__(self, token_dims, common_dim=128, target_spatial_size=64, output_size=(256, 256)):
        super().__init__()
        self.common_dim = common_dim
        self.target_spatial_size = target_spatial_size
        self.output_size = output_size
        
        # Project all scales to common dimension
        self.scale_projections = nn.ModuleList([
            nn.Linear(dim, common_dim) for dim in token_dims
        ])
        
        # Fusion layers after concatenation
        total_dim = common_dim * len(token_dims)
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, common_dim * 2),
            nn.GELU(),
            nn.Linear(common_dim * 2, common_dim),
            nn.GELU()
        )
        
        # Convert tokens back to spatial feature map
        self.to_spatial = nn.Sequential(
            nn.Linear(common_dim, common_dim * 4),
            nn.GELU(),
            nn.Linear(common_dim * 4, common_dim)
        )
        
        # Convolutional decoder
        self.conv_decoder = nn.Sequential(
            nn.Conv2d(common_dim, common_dim // 2, 3, padding=1),
            nn.BatchNorm2d(common_dim // 2),
            nn.GELU(),
            nn.Conv2d(common_dim // 2, common_dim // 4, 3, padding=1),
            nn.BatchNorm2d(common_dim // 4),
            nn.GELU(),
        )
        
        # Final output head for 6 channels (fake_T + fake_R)
        self.output_head = nn.Sequential(
            nn.Conv2d(common_dim // 4, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 6, 1)  # 6 channels: 3 for fake_T, 3 for fake_R
        )
        
        # Optional c_map head
        self.c_map_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(common_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, token_list):
        """
        Args:
            token_list: List of [T0, T1, T2, T3] where Ti is (B, Ni, Di)
        Returns:
            output: (B, 6, H, W) - concatenated fake_T and fake_R
            c_map: (B, 1) - confidence map
        """
        # Project all tokens to common dimension and spatial size
        unified_tokens = []
        for i, tokens in enumerate(token_list):
            # Project to common dimension
            proj_tokens = self.scale_projections[i](tokens)  # (B, Ni, common_dim)
            
            # Resize to target spatial size
            B, N, D = proj_tokens.shape
            current_spatial_size = int(N ** 0.5)
            if current_spatial_size != self.target_spatial_size:
                proj_tokens = proj_tokens.view(B, current_spatial_size, current_spatial_size, D)
                proj_tokens = proj_tokens.permute(0, 3, 1, 2)  # (B, D, H, W)
                proj_tokens = F.interpolate(
                    proj_tokens, 
                    size=(self.target_spatial_size, self.target_spatial_size),
                    mode='bilinear',
                    align_corners=False
                )
                proj_tokens = proj_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # (B, target_size^2, D)
            
            unified_tokens.append(proj_tokens)
        
        # Concatenate along feature dimension
        fused_tokens = torch.cat(unified_tokens, dim=-1)  # (B, target_size^2, total_dim)
        
        # Apply fusion layers
        fused_tokens = self.fusion_layers(fused_tokens)  # (B, target_size^2, common_dim)
        
        # Final token processing
        spatial_tokens = self.to_spatial(fused_tokens)  # (B, target_size^2, common_dim)
        
        # Convert back to spatial feature map
        B = spatial_tokens.shape[0]
        spatial_features = spatial_tokens.view(B, self.target_spatial_size, self.target_spatial_size, self.common_dim)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # (B, common_dim, H, W)
        
        # Apply convolutional decoder
        conv_features = self.conv_decoder(spatial_features)  # (B, common_dim//4, H, W)
        
        # Generate c_map
        c_map = self.c_map_head(conv_features)  # (B, 1)
        
        # Upsample to target output size
        upsampled_features = F.interpolate(
            conv_features, 
            size=self.output_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Generate final 6-channel output
        output = self.output_head(upsampled_features)  # (B, 6, output_H, output_W)
        
        return output, c_map


class AuxiliaryHead(nn.Module):
    """Auxiliary supervision head for intermediate scales"""
    def __init__(self, token_dim, output_size=(256, 256)):
        super().__init__()
        self.output_size = output_size
        
        # Simple projection to 3 channels
        self.proj = nn.Linear(token_dim, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 1)  # 3 channels for RGB supervision
        )
    
    def forward(self, tokens):
        """
        Args:
            tokens: (B, N, D)
        Returns:
            output: (B, 3, H, W)
        """
        B, N, D = tokens.shape
        spatial_size = int(N ** 0.5)
        
        # Project tokens
        proj_tokens = self.proj(tokens)  # (B, N, 64)
        
        # Reshape to spatial
        spatial_features = proj_tokens.view(B, spatial_size, spatial_size, 64)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # (B, 64, H, W)
        
        # Apply convolution
        conv_output = self.conv(spatial_features)  # (B, 3, H, W)
        
        # Upsample to target size
        output = F.interpolate(conv_output, size=self.output_size, mode='bilinear', align_corners=False)
        
        return output