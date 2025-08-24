# TokenMTRRNet: Token-Only Multi-Scale Architecture

This is the refactored MTRRNet architecture implementing a token-only design with frequency-split multi-scale encoding and unified decoding.

## Architecture Overview

### Key Changes from Original MTRRNet

1. **No Early RGB Decodes**: Encoder stages operate purely on tokens, removing early RGB conversions that caused information loss
2. **Frequency Split Encoding**: Each scale splits input into low/high frequency components fed to Mamba/Swin branches respectively  
3. **Token-Space Fusion**: SubNet operates on token features rather than upsampled image tensors
4. **Unified Decoder**: Single decoder processes all scale tokens to generate final 6-channel output
5. **Base Residual Scaling**: Learnable base scaling instead of fixed `cat(x, 0.1*x)`
6. **Disabled net_c**: Color adjustment disabled initially for cleaner gradient flow

### Multi-Scale Token Encoder

- **S0 (1/4)**: 256×256 → 96-dim tokens via 4×4 patches
- **S1 (1/8)**: 128×128 → 128-dim tokens via 4×4 patches  
- **S2 (1/16)**: 64×64 → 160-dim tokens via 4×4 patches
- **S3 (1/32)**: 32×32 → 192-dim tokens via 2×2 patches

Each scale:
1. **Frequency Split**: `low = blur(x)`, `high = x - low`
2. **Mamba Branch**: Processes low-frequency component → tokens
3. **Swin Branch**: Processes high-frequency component → tokens  
4. **Token Merge**: Concatenate + linear projection to unified tokens

### Token SubNet Fusion

- **Input**: List of token tensors `[T0, T1, T2, T3]`
- **Cross-Scale Attention**: Tokens from different scales attend to each other
- **2 Iterations**: Non-shared parameters for refinement
- **Output**: Refined token list with enhanced multi-scale features

### Unified Decoder

- **Token Unification**: Project all scales to common dimension (128)
- **Spatial Alignment**: Resize to common spatial size (64×64)  
- **Feature Fusion**: Concatenate + MLP processing
- **Conv Decoder**: Convert tokens → spatial features → 6-channel output
- **Final Upsampling**: 64×64 → 256×256 via bilinear interpolation

## Usage

### Training with Token Architecture

```python
# Use the new token-based training script
python train_token.py --epoch 40 --batch_size 4

# Key differences from original:
# - net_c is disabled (uses input directly)  
# - Auxiliary supervision on intermediate scales
# - Debug visualization outputs
# - Learnable base/delta scaling
```

### Model Creation

```python
from MTRRNet_token import TokenMTRRNet, TokenMTRREngine

# Create model
model = TokenMTRRNet()

# Forward pass  
rmap, output, aux_outputs = model(input_image)
fake_T = output[:, 0:3, :, :]  # Transmission layer
fake_R = output[:, 3:6, :, :]  # Reflection layer

# Debug outputs
debug_info = model.get_debug_outputs()
# Available: low_s0, high_s0, tok_s0, ..., final_T, final_R, c_map
```

### Migration from Original MTRRNet

```python
# Original model weights can be partially loaded
engine = TokenMTRREngine(opts, device)
epoch = engine.load_checkpoint(optimizer, 'old_model.pth')
# Compatible parameters will be loaded, incompatible ones ignored
```

## Testing

Run comprehensive tests to verify architecture:

```bash
python test_forward.py
```

Tests include:
- Forward pass shape verification
- No early RGB decode confirmation  
- Token flow validation
- Gradient flow verification
- Auxiliary supervision functionality
- Engine interface compatibility

## Debug Outputs

The architecture provides extensive debug outputs for research:

```python
debug = model.get_debug_outputs()

# Frequency components per scale
debug['low_s0'], debug['high_s0']   # Scale 0 frequency split
debug['low_s1'], debug['high_s1']   # Scale 1 frequency split
# ... etc

# Token features per scale  
debug['tok_s0'], debug['tok_s1']    # Merged tokens per scale

# Auxiliary predictions (training mode)
debug['aux_T2'], debug['aux_T3']    # Intermediate supervision

# Final outputs
debug['final_T'], debug['final_R']  # Final transmission/reflection
debug['c_map']                      # Confidence map
```

## Key Advantages

1. **Improved Gradient Flow**: No early RGB decodes prevent gradient degradation
2. **Better Multi-Scale Fusion**: Token-space operations preserve semantic information
3. **Frequency-Aware Processing**: Explicit low/high frequency separation  
4. **Unified Architecture**: Single decoder handles all scales consistently
5. **Research Friendly**: Extensive debug outputs for analysis

## Parameters

- **base_scale**: Learnable base residual scaling (init: 0.3)
- **delta_scale**: Output delta scaling with warmup (init: 0.05)  
- **Auxiliary weights**: 0.03 for intermediate supervision
- **Token dimensions**: [96, 128, 160, 192] for scales S0-S3

## Files

- `MTRRNet_token.py`: Main token architecture
- `token_modules.py`: Token processing components  
- `train_token.py`: Updated training script
- `test_forward.py`: Comprehensive test suite
- `MTRRNet_legacy.py`: Original architecture backup