#!/usr/bin/env python3
"""
Training script for all 6 model variants
Comprehensive ablation study with controlled experiments
"""

import subprocess
import os
import time
from pathlib import Path

def run_training(script_name, model_name):
    """Run training script and capture output"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting {model_name} Training")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run training script
        result = subprocess.run(
            ['python3', script_name], 
            cwd='.', 
            capture_output=True, 
            text=True
        )
        
        end_time = time.time()
        duration = (end_time - start_time) / 3600  # Convert to hours
        
        if result.returncode == 0:
            print(f"✅ {model_name} completed successfully!")
            print(f"⏱️ Training time: {duration:.2f} hours")
            print(f"📁 Output saved to: outputs/{model_name.lower().replace(' ', '_')}/")
        else:
            print(f"❌ {model_name} failed!")
            print(f"Error: {result.stderr}")
            
        return result.returncode == 0, duration
        
    except Exception as e:
        print(f"❌ Error running {model_name}: {e}")
        return False, 0

def main():
    """Train all 6 model variants"""
    
    print("🧠 3D Brain Tumor Segmentation - Comprehensive Training")
    print("=" * 60)
    
    # Define all model configurations
    models = [
        {
            'script': 'train.py',
            'name': 'Baseline',
            'attention_type': 'none',
            'description': 'Standard 3D U-Net (no attention)'
        },
        {
            'script': 'train.py', 
            'name': 'SE-UNet',
            'attention_type': 'se',
            'description': 'SE blocks in encoder, bottleneck, decoder'
        },
        {
            'script': 'train.py',
            'name': 'CBAM-UNet', 
            'attention_type': 'cbam',
            'description': 'CBAM blocks in encoder, bottleneck, decoder'
        },
        {
            'script': 'train.py',
            'name': 'Hybrid',
            'attention_type': 'hybrid', 
            'description': 'SE in encoder + CBAM in bottleneck (proposed)'
        },
        {
            'script': 'train_se_encoder.py',
            'name': 'SE-Encoder Only',
            'attention_type': 'se_encoder_only',
            'description': 'SE blocks only in encoder'
        },
        {
            'script': 'train_cbam_bottleneck.py',
            'name': 'CBAM-Bottleneck Only', 
            'attention_type': 'cbam_bottleneck_only',
            'description': 'CBAM blocks only in bottleneck'
        }
    ]
    
    print(f"📋 Training {len(models)} model variants:")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model['name']}: {model['description']}")
    
    print(f"\n🎯 Experimental Matrix:")
    print(f"   Model | Encoder | Bottleneck | Decoder | Strategy")
    print(f"   -------|---------|------------|---------|----------")
    
    strategies = {
        'Baseline': ('None', 'None', 'None', 'No attention'),
        'SE-UNet': ('SE', 'SE', 'SE', 'Full SE'),
        'CBAM-UNet': ('CBAM', 'CBAM', 'CBAM', 'Full CBAM'),
        'Hybrid': ('SE', 'CBAM', 'None', 'Proposed hybrid'),
        'SE-Encoder Only': ('SE', 'None', 'None', 'Encoder-only'),
        'CBAM-Bottleneck Only': ('None', 'CBAM', 'None', 'Bottleneck-only')
    }
    
    for model, (enc, bot, dec, strategy) in strategies.items():
        print(f"   {model:<20} | {enc:<7} | {bot:<10} | {dec:<7} | {strategy}")
    
    # Check if data exists
    data_dir = Path('data/processed/brats128')
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("Please ensure the dataset is prepared before training.")
        return
    
    # Training results
    results = []
    total_start_time = time.time()
    
    # Train each model
    for model in models:
        success, duration = run_training(model['script'], model['name'])
        results.append({
            'model': model['name'],
            'success': success,
            'duration_hours': duration,
            'attention_type': model['attention_type']
        })
        
        if not success:
            print(f"⚠️ Skipping remaining models due to failure")
            break
    
    total_duration = (time.time() - total_start_time) / 3600
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*60}")
    
    successful_models = [r for r in results if r['success']]
    failed_models = [r for r in results if not r['success']]
    
    print(f"✅ Successful: {len(successful_models)}/{len(models)} models")
    print(f"❌ Failed: {len(failed_models)}/{len(models)} models")
    print(f"⏱️ Total time: {total_duration:.2f} hours")
    
    if successful_models:
        print(f"\n🎯 Successful Models:")
        for r in successful_models:
            print(f"   ✓ {r['model']}: {r['duration_hours']:.2f}h")
    
    if failed_models:
        print(f"\n⚠️ Failed Models:")
        for r in failed_models:
            print(f"   ✗ {r['model']}")
    
    # Check outputs
    print(f"\n📁 Generated Outputs:")
    for r in successful_models:
        output_dir = f"outputs/{r['model'].lower().replace(' ', '_').replace('-', '_')}"
        if os.path.exists(output_dir):
            print(f"   ✓ {output_dir}/")
            if os.path.exists(f"{output_dir}/best.pth"):
                print(f"     - best.pth")
            if os.path.exists(f"{output_dir}/training_log.json"):
                print(f"     - training_log.json")
    
    print(f"\n🎉 Training pipeline completed!")
    print(f"📈 Ready for analysis with phase4_visualization.ipynb and phase5_ablation.ipynb")
    
    return results

if __name__ == "__main__":
    main()
