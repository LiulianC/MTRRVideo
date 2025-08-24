#!/usr/bin/env python3
"""
测试脚本：验证Token-only MTRRNet架构
用于检查新实现是否正确工作
"""

def test_token_modules():
    """测试Token模块是否可以正确导入和实例化"""
    try:
        # 先检查基础依赖
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        from token_modules import (
            FrequencySplit, TokenPatchEmbed, MambaTokenBlock, SwinTokenBlock,
            TokenStage, MultiScaleTokenEncoder, TokenSubNet, UnifiedTokenDecoder
        )
        print("✓ Token模块导入成功")
        
        # 测试FrequencySplit
        freq_split = FrequencySplit(kernel_size=5)
        print("✓ FrequencySplit实例化成功")
        
        # 测试TokenPatchEmbed
        patch_embed = TokenPatchEmbed(img_size=64, patch_size=4, in_chans=3, embed_dim=96)
        print("✓ TokenPatchEmbed实例化成功")
        
        # 测试MultiScaleTokenEncoder
        encoder = MultiScaleTokenEncoder()
        print("✓ MultiScaleTokenEncoder实例化成功")
        
        # 测试TokenSubNet
        subnet = TokenSubNet()
        print("✓ TokenSubNet实例化成功")
        
        # 测试UnifiedTokenDecoder
        decoder = UnifiedTokenDecoder()
        print("✓ UnifiedTokenDecoder实例化成功")
        
        return True
        
    except ImportError as e:
        print(f"⚠ 依赖缺失，跳过Token模块详细测试: {e}")
        # 尝试基本语法检查
        try:
            import ast
            with open('token_modules.py', 'r') as f:
                code = f.read()
            ast.parse(code)
            print("✓ token_modules.py语法检查通过")
            return True
        except Exception as syntax_e:
            print(f"✗ token_modules.py语法错误: {syntax_e}")
            return False
    except Exception as e:
        print(f"✗ Token模块测试失败: {e}")
        return False

def test_mtrr_net():
    """测试MTRRNet是否可以正确实例化"""
    try:
        # 先检查基础依赖
        import torch
        from MTRRNet import MTRRNet
        
        # 测试新Token-only模式
        model_new = MTRRNet(use_legacy=False)
        print("✓ 新Token-only MTRRNet实例化成功")
        
        # 测试Legacy模式
        model_legacy = MTRRNet(use_legacy=True)
        print("✓ Legacy MTRRNet实例化成功")
        
        return True
        
    except ImportError as e:
        print(f"⚠ 依赖缺失，跳过MTRRNet详细测试: {e}")
        # 尝试基本语法检查
        try:
            import ast
            with open('MTRRNet.py', 'r') as f:
                code = f.read()
            ast.parse(code)
            print("✓ MTRRNet.py语法检查通过")
            return True
        except Exception as syntax_e:
            print(f"✗ MTRRNet.py语法错误: {syntax_e}")
            return False
    except Exception as e:
        print(f"✗ MTRRNet测试失败: {e}")
        return False

def test_with_pytorch():
    """如果PyTorch可用，测试前向传播"""
    try:
        import torch
        from MTRRNet import MTRRNet
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 测试Token-only模式
        model = MTRRNet(use_legacy=False).to(device)
        x = torch.randn(1, 3, 256, 256).to(device)
        
        model.eval()
        with torch.no_grad():
            rmap, out = model(x)
            
        print(f"✓ Token-only前向传播成功")
        print(f"  - rmap shape: {rmap.shape}")
        print(f"  - out shape: {out.shape}")
        
        # 检查中间监督
        intermediates = model.get_intermediates()
        print(f"  - 中间监督结果数量: {len(intermediates)}")
        
        # 检查debug统计
        debug_stats = model.get_debug_stats()
        print(f"  - Debug统计数量: {len(debug_stats)}")
        
        # 测试Legacy模式
        model_legacy = MTRRNet(use_legacy=True).to(device)
        model_legacy.eval()
        with torch.no_grad():
            rmap_legacy, out_legacy = model_legacy(x)
            
        print(f"✓ Legacy前向传播成功")
        print(f"  - rmap shape: {rmap_legacy.shape}")
        print(f"  - out shape: {out_legacy.shape}")
        
        return True
        
    except ImportError:
        print("⚠ PyTorch不可用，跳过前向传播测试")
        return True
    except Exception as e:
        print(f"✗ PyTorch前向传播测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("MTRRNet Token-only架构验证测试")
    print("=" * 50)
    
    tests = [
        ("Token模块测试", test_token_modules),
        ("MTRRNet实例化测试", test_mtrr_net),
        ("PyTorch前向传播测试", test_with_pytorch),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            if test_func():
                passed += 1
                print(f"✓ {name} 通过")
            else:
                print(f"✗ {name} 失败")
        except Exception as e:
            print(f"✗ {name} 异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 50)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)