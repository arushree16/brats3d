import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from unet3d import UNet3D
from attention import SEBlock3D, CBAM3D

def test_attention_blocks():
    print("Testing attention blocks...")
    
    x = torch.randn(1, 64, 32, 32, 32)
    
    # Test SE Block
    se_block = SEBlock3D(64)
    se_out = se_block(x)
    assert se_out.shape == x.shape
    print("✓ SE Block works")
    
    # Test CBAM Block
    cbam_block = CBAM3D(64)
    cbam_out = cbam_block(x)
    assert cbam_out.shape == x.shape
    print("✓ CBAM Block works")

def test_unet_variants():
    print("\nTesting UNet variants...")
    
    # Test input
    x = torch.randn(1, 4, 128, 128, 128)
    
    # Test baseline UNet
    model_none = UNet3D(attention_type='none')
    out_none = model_none(x)
    assert out_none.shape == (1, 3, 128, 128, 128)
    print("✓ Baseline UNet works")
    
    # Test SE-UNet
    model_se = UNet3D(attention_type='se')
    out_se = model_se(x)
    assert out_se.shape == (1, 3, 128, 128, 128)
    print("✓ SE-UNet works")
    
    # Test CBAM-UNet
    model_cbam = UNet3D(attention_type='cbam')
    out_cbam = model_cbam(x)
    assert out_cbam.shape == (1, 3, 128, 128, 128)
    print("✓ CBAM-UNet works")
    
    # Compare parameter counts
    params_none = sum(p.numel() for p in model_none.parameters())
    params_se = sum(p.numel() for p in model_se.parameters())
    params_cbam = sum(p.numel() for p in model_cbam.parameters())
    
    print(f"\nParameter counts:")
    print(f"Baseline UNet: {params_none:,}")
    print(f"SE-UNet: {params_se:,} (+{params_se-params_none:,})")
    print(f"CBAM-UNet: {params_cbam:,} (+{params_cbam-params_none:,})")

if __name__ == "__main__":
    test_attention_blocks()
    test_unet_variants()
    print("\n✅ All tests passed!")