"""
Token-only MTRRNet architecture implementing the refactored design:
- Multi-scale encoder with frequency split (Mamba + Swin token branches)
- Token fusion subnet
- Unified decoder
- Base residual scaling
- Auxiliary supervision
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

from token_modules import (
    FrequencySplit, ResidualMambaTokenStage, ResidualSwinTokenStage,
    TokenMerge, TokenSubNet, UnifiedDecoder, AuxiliaryHead
)

# Import RDM and other utilities from original MTRRNet
from MTRRNet import RDM, Conv2DLayer


class TokenMTRRNet(nn.Module):
    """Token-only MTRRNet with multi-scale encoder, token fusion, and unified decoder"""
    
    def __init__(self):
        super().__init__()
        
        # RDM module for reflection prior (unchanged)
        self.rdm = RDM()
        
        # Multi-scale token encoder configuration
        self.scales = [
            {'size': (256, 256), 'patch_size': 4, 'embed_dim': 96},   # S0: 1/4 resolution
            {'size': (128, 128), 'patch_size': 4, 'embed_dim': 128},  # S1: 1/8 resolution  
            {'size': (64, 64), 'patch_size': 4, 'embed_dim': 160},    # S2: 1/16 resolution
            {'size': (32, 32), 'patch_size': 2, 'embed_dim': 192},    # S3: 1/32 resolution
        ]
        
        # Frequency split modules for each scale
        self.freq_splits = nn.ModuleList([
            FrequencySplit(channels=3, blur_kernel_size=5, sigma=1.0 + i * 0.5)
            for i in range(len(self.scales))
        ])
        
        # Mamba and Swin token stages for each scale
        self.mamba_stages = nn.ModuleList()
        self.swin_stages = nn.ModuleList()
        self.token_mergers = nn.ModuleList()
        
        for scale in self.scales:
            # Mamba branch (low frequency)
            self.mamba_stages.append(
                ResidualMambaTokenStage(
                    in_channels=3,
                    embed_dim=scale['embed_dim'],
                    patch_size=scale['patch_size'],
                    img_size=scale['size'],
                    num_blocks=3
                )
            )
            
            # Swin branch (high frequency)
            self.swin_stages.append(
                ResidualSwinTokenStage(
                    in_channels=3,
                    embed_dim=scale['embed_dim'],
                    patch_size=scale['patch_size'],
                    img_size=scale['size'],
                    window_size=8,
                    num_blocks=3
                )
            )
            
            # Token merger
            self.token_mergers.append(
                TokenMerge(embed_dim=scale['embed_dim'], merge_type='concat_linear')
            )
        
        # Token SubNet for fusion (2 iterations)
        token_dims = [scale['embed_dim'] for scale in self.scales]
        self.token_subnet = TokenSubNet(token_dims=token_dims, num_iterations=2)
        
        # Unified decoder
        self.unified_decoder = UnifiedDecoder(
            token_dims=token_dims,
            common_dim=128,
            target_spatial_size=64,
            output_size=(256, 256)
        )
        
        # Auxiliary supervision heads for deeper scales (T2, T3)
        self.aux_head_t2 = AuxiliaryHead(token_dims[2], output_size=(256, 256))
        self.aux_head_t3 = AuxiliaryHead(token_dims[3], output_size=(256, 256))
        
        # Base residual scaling (learnable)
        self.base_scale = nn.Parameter(torch.tensor(0.3))
        
        # Delta scaling for output (with warmup mechanism)
        self.delta_scale = nn.Parameter(torch.tensor(0.05))
        
        # Debug outputs storage
        self.debug = {}
        
    def forward(self, x_in):
        """
        Args:
            x_in: Input image (B, 3, 256, 256)
        Returns:
            rmap: Reflection map from RDM (B, 1, 256, 256)
            output: Final 6-channel output (B, 6, 256, 256) [fake_T, fake_R]
        """
        batch_size = x_in.shape[0]
        
        # Clear debug outputs
        self.debug = {}
        
        # 1. Extract reflection prior using RDM
        rmap, x_down8, x_down4, x_down2 = self.rdm(x_in)
        self.debug['rmap'] = rmap.detach()
        
        # 2. Prepare multi-scale inputs
        scale_inputs = [
            x_in,      # S0: 256x256
            x_down2,   # S1: 128x128 
            x_down4,   # S2: 64x64
            x_down8    # S3: 32x32
        ]
        
        # 3. Multi-scale token encoding with frequency split
        scale_tokens = []
        for i, (x_scale, freq_split, mamba_stage, swin_stage, token_merger) in enumerate(
            zip(scale_inputs, self.freq_splits, self.mamba_stages, self.swin_stages, self.token_mergers)
        ):
            # Frequency split
            low_freq, high_freq = freq_split(x_scale)
            
            # Store debug outputs
            self.debug[f'low_s{i}'] = low_freq.detach()
            self.debug[f'high_s{i}'] = high_freq.detach()
            
            # Mamba branch (low frequency)
            mamba_tokens = mamba_stage(low_freq)  # (B, N_i, D_i)
            
            # Swin branch (high frequency)  
            swin_tokens = swin_stage(high_freq)   # (B, N_i, D_i)
            
            # Merge branches
            merged_tokens = token_merger(mamba_tokens, swin_tokens)  # (B, N_i, D_i)
            
            # Store debug outputs
            self.debug[f'tok_s{i}'] = merged_tokens.detach()
            
            scale_tokens.append(merged_tokens)
        
        # 4. Token SubNet fusion
        fused_tokens = self.token_subnet(scale_tokens)  # List of refined tokens
        
        # 5. Auxiliary supervision (during training)
        aux_outputs = {}
        if self.training:
            aux_outputs['aux_T2'] = self.aux_head_t2(fused_tokens[2])  # (B, 3, 256, 256)
            aux_outputs['aux_T3'] = self.aux_head_t3(fused_tokens[3])  # (B, 3, 256, 256)
            
            # Store debug outputs
            self.debug['aux_T2'] = aux_outputs['aux_T2'].detach()
            self.debug['aux_T3'] = aux_outputs['aux_T3'].detach()
        
        # 6. Unified decoder
        delta_output, c_map = self.unified_decoder(fused_tokens)  # (B, 6, 256, 256), (B, 1)
        
        # 7. Base residual path with learnable scaling
        base = self.base_scale * x_in  # Scaled base instead of cat(x, 0.1*x)
        base_6ch = torch.cat([base, base], dim=1)  # (B, 6, 256, 256)
        
        # 8. Final output with delta scaling
        delta = torch.tanh(delta_output) * self.delta_scale
        final_output = base_6ch + delta
        
        # Store debug outputs
        self.debug['c_map'] = c_map.detach()
        self.debug['final_T'] = final_output[:, 0:3, :, :].detach()
        self.debug['final_R'] = final_output[:, 3:6, :, :].detach()
        
        return rmap, final_output, aux_outputs if self.training else {}
    
    def get_debug_outputs(self):
        """Return dictionary of debug outputs for visualization"""
        return self.debug.copy()
    
    def log_grad_norms(self):
        """Log gradient norms for token branches (debug utility)"""
        grad_info = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if 'mamba' in name or 'swin' in name or 'token' in name:
                    grad_info[name] = grad_norm
                    
        return grad_info


class TokenMTRREngine(nn.Module):
    """Engine wrapper for TokenMTRRNet with training utilities"""
    
    def __init__(self, opts, device):
        super().__init__()
        self.device = device
        self.opts = opts
        self.visual_names = ['fake_T', 'fake_R', 'c_map', 'I', 'T', 'R']
        
        # Main network
        self.netG_T = TokenMTRRNet().to(device)
        self.netG_T.apply(self.init_weights)
        
        # Note: net_c is disabled as per requirements
        # self.net_c = None  # Disabled initially
        
    def set_input(self, input):
        """Load images from dataloader"""
        self.I = input['input'].to(self.device)
        self.T = input['target_t'].to(self.device) 
        self.R = input['target_r'].to(self.device)
    
    def forward(self):
        """Forward pass (net_c disabled)"""
        # self.Ic = self.I  # Use input directly (no net_c)
        self.c_map, self.out, self.aux_outputs = self.netG_T(self.I)
        self.fake_T, self.fake_R = self.out[:, 0:3, :, :], self.out[:, 3:6, :, :]
    
    def inference(self):
        """Inference wrapper"""
        self.forward()
    
    def get_current_visuals(self):
        """Get current visual results"""
        visual_result = OrderedDict()
        for name in self.visual_names:
            if hasattr(self, name):
                visual_result[name] = getattr(self, name)
        return visual_result
    
    def get_aux_outputs(self):
        """Get auxiliary outputs for supervision"""
        return getattr(self, 'aux_outputs', {})
    
    def get_debug_outputs(self):
        """Get debug outputs from the network"""
        return self.netG_T.get_debug_outputs()
    
    def log_grad_norms(self):
        """Log gradient norms for debugging"""
        return self.netG_T.log_grad_norms()
    
    def load_checkpoint(self, optimizer, model_path=None):
        """Load checkpoint with migration support for old model weights"""
        if model_path is None:
            model_path = self.opts.model_path
            
        if model_path is not None:
            print(f'Loading model from {model_path}')
            try:
                model_state = torch.load(model_path, map_location=str(self.device))
                
                # Try to load state dict with graceful handling of missing keys
                if 'netG_T' in model_state:
                    old_state_dict = model_state['netG_T']
                    new_state_dict = self.netG_T.state_dict()
                    
                    # Filter compatible keys
                    compatible_keys = []
                    incompatible_keys = []
                    
                    for key in old_state_dict:
                        if key in new_state_dict and old_state_dict[key].shape == new_state_dict[key].shape:
                            compatible_keys.append(key)
                        else:
                            incompatible_keys.append(key)
                    
                    # Load compatible keys
                    filtered_state_dict = {k: old_state_dict[k] for k in compatible_keys}
                    self.netG_T.load_state_dict(filtered_state_dict, strict=False)
                    
                    print(f"Loaded {len(compatible_keys)} compatible parameters")
                    print(f"Ignored {len(incompatible_keys)} incompatible parameters")
                    if incompatible_keys:
                        print("Incompatible keys:", incompatible_keys[:10])  # Show first 10
                
                # Load optimizer if available
                if 'optimizer_state_dict' in model_state and optimizer is not None:
                    try:
                        optimizer.load_state_dict(model_state['optimizer_state_dict'])
                    except Exception as e:
                        print(f"Warning: Could not load optimizer state: {e}")
                
                epoch = model_state.get('epoch', None)
                print(f'Loaded model at epoch {epoch+1}' if epoch is not None else 'Loaded model without epoch info')
                return epoch
                
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Continuing with randomly initialized weights")
                return None
        
        return None
    
    def eval(self):
        """Set to evaluation mode"""
        self.netG_T.eval()
    
    @staticmethod
    def init_weights(m):
        """Initialize weights for the network"""
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# Test forward pass function
def forward_smoke_test():
    """Smoke test to verify forward pass shapes"""
    print("Running forward smoke test...")
    
    device = torch.device("cpu")  # Use CPU for testing
    model = TokenMTRRNet().to(device)
    model.eval()
    
    # Test input
    x = torch.randn(2, 3, 256, 256).to(device)
    
    with torch.no_grad():
        rmap, output, aux_outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"RMap shape: {rmap.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary outputs: {list(aux_outputs.keys()) if aux_outputs else 'None (eval mode)'}")
    
    # Verify shapes
    assert rmap.shape == (2, 3, 256, 256), f"Expected rmap shape (2, 3, 256, 256), got {rmap.shape}"
    assert output.shape == (2, 6, 256, 256), f"Expected output shape (2, 6, 256, 256), got {output.shape}"
    
    # Test debug outputs
    debug_outputs = model.get_debug_outputs()
    print(f"Debug outputs: {list(debug_outputs.keys())}")
    
    print("✅ Forward smoke test passed!")
    return True


if __name__ == "__main__":
    # Run smoke test
    forward_smoke_test()