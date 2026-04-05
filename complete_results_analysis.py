#!/usr/bin/env python3
"""
Complete Analysis of All 4 Training Results
Analyzes Baseline, SE-UNet, CBAM-UNet, and Hybrid models
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_all_training_logs():
    """Load training logs from all 4 completed models"""
    
    # Load all models from final results directory
    logs = {}
    
    # Expected model directories
    model_dirs = {
        'baseline': 'Baseline',
        'se': 'SE-UNet', 
        'cbam': 'CBAM-UNet',
        'hybrid': 'Hybrid'
    }
    
    # Load from final results directory
    for model_key, model_name in model_dirs.items():
        model_log = None
        try:
            log_path = f'final results/{model_key}/training_log.json'
            with open(log_path, 'r') as f:
                model_log = json.load(f)
            print(f"✅ Loaded {model_name} training log")
            logs[model_key] = model_log
        except:
            print(f"❌ {model_name} log not found at {log_path}")
            logs[model_key] = None
    
    # Also check Downloads directory for any additional logs
    downloads_logs = {
        'baseline': None,
        'se': None,
        'cbam': None, 
        'hybrid': None
    }
    
    # Check if there are logs in Downloads that aren't in final results
    for model_key in model_dirs.keys():
        try:
            log_path = f'/Users/arushreemishra/Downloads/training_log_{model_key}.json'
            with open(log_path, 'r') as f:
                model_log = json.load(f)
            if logs[model_key] is None:  # Only use if not already loaded
                logs[model_key] = model_log
                print(f"✅ Loaded {model_dirs[model_key]} from Downloads")
        except:
            pass  # Ignore if not found
    
    return logs

def analyze_performance_comparison(logs):
    """Create comprehensive performance comparison"""
    
    results = []
    
    for model_name, log in logs.items():
        if log is None:
            continue
            
        # Extract final metrics
        final_wt_dice = log.get('final_wt_dice', 0)
        final_tc_dice = log.get('final_tc_dice', 0)
        final_wt_hd95 = log.get('final_wt_hd95', 0)
        final_tc_hd95 = log.get('final_tc_hd95', 0)
        training_time = log.get('total_training_time', 0)
        
        # Determine model type and parameters
        if model_name == 'baseline':
            model_type = 'Baseline'
            params = 5661347  # From your training output
        elif model_name == 'se':
            model_type = 'SE-UNet'
            params = 5660000  # From your training output (estimate)
        elif model_name == 'cbam':
            model_type = 'CBAM-UNet'
            params = 5667521  # From your training output
        elif model_name == 'hybrid':
            model_type = 'Hybrid'
            params = 5860000  # Estimate
        else:
            model_type = model_name.upper()
            params = 0
        
        results.append({
            'Model': model_type,
            'Parameters': params,
            'WT_Dice': final_wt_dice,
            'TC_Dice': final_tc_dice,
            'WT_HD95': final_wt_hd95,
            'TC_HD95': final_tc_hd95,
            'Training_Time': training_time,
            'Efficiency_Score': (final_wt_dice + final_tc_dice) / (params / 1e6) if params > 0 else 0
        })
    
    return pd.DataFrame(results)

def create_convergence_plots(logs):
    """Create training convergence comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold')
    
    colors = {
        'baseline': 'blue', 
        'se': 'orange', 
        'cbam': 'green', 
        'hybrid': 'red'
    }
    labels = {
        'baseline': 'Baseline', 
        'se': 'SE-UNet', 
        'cbam': 'CBAM-UNet', 
        'hybrid': 'Hybrid'
    }
    
    # Training Loss
    ax1 = axes[0, 0]
    for model_name, log in logs.items():
        if log and 'train_loss' in log:
            epochs = range(1, len(log['train_loss']) + 1)
            ax1.plot(epochs, log['train_loss'], 
                    color=colors.get(model_name, 'black'),
                    label=labels.get(model_name, model_name),
                    linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation Loss
    ax2 = axes[0, 1]
    for model_name, log in logs.items():
        if log and 'val_loss' in log:
            epochs = range(1, len(log['val_loss']) + 1)
            ax2.plot(epochs, log['val_loss'],
                    color=colors.get(model_name, 'black'),
                    label=labels.get(model_name, model_name),
                    linewidth=2)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # WT Dice
    ax3 = axes[1, 0]
    for model_name, log in logs.items():
        if log and 'wt_dice' in log:
            epochs = range(1, len(log['wt_dice']) + 1)
            ax3.plot(epochs, log['wt_dice'],
                    color=colors.get(model_name, 'black'),
                    label=labels.get(model_name, model_name),
                    linewidth=2)
    ax3.set_title('Whole Tumor Dice')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Dice Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # TC Dice
    ax4 = axes[1, 1]
    for model_name, log in logs.items():
        if log and 'tc_dice' in log:
            epochs = range(1, len(log['tc_dice']) + 1)
            ax4.plot(epochs, log['tc_dice'],
                    color=colors.get(model_name, 'black'),
                    label=labels.get(model_name, model_name),
                    linewidth=2)
    ax4.set_title('Tumor Core Dice')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Dice Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_convergence_speed(logs):
    """Analyze convergence speed and stability"""
    
    convergence_analysis = []
    
    for model_name, log in logs.items():
        if log is None:
            continue
            
        # Calculate convergence metrics
        wt_dice = log.get('wt_dice', [])
        tc_dice = log.get('tc_dice', [])
        
        if not wt_dice or not tc_dice:
            continue
        
        # Find epochs where models reach certain performance thresholds
        wt_80_epoch = None
        tc_80_epoch = None
        
        for i, dice in enumerate(wt_dice):
            if dice >= 0.80 and wt_80_epoch is None:
                wt_80_epoch = i + 1
        
        for i, dice in enumerate(tc_dice):
            if dice >= 0.80 and tc_80_epoch is None:
                tc_80_epoch = i + 1
        
        # Calculate stability (variance in last 10 epochs)
        if len(wt_dice) >= 10:
            wt_stability = np.std(wt_dice[-10:])
            tc_stability = np.std(tc_dice[-10:])
        else:
            wt_stability = np.std(wt_dice)
            tc_stability = np.std(tc_dice)
        
        convergence_analysis.append({
            'Model': model_name.upper(),
            'WT_80_Epoch': wt_80_epoch,
            'TC_80_Epoch': tc_80_epoch,
            'WT_Stability': wt_stability,
            'TC_Stability': tc_stability,
            'Total_Epochs': len(wt_dice)
        })
    
    return pd.DataFrame(convergence_analysis)

def generate_comprehensive_report():
    """Generate complete analysis report"""
    
    print("🔬 Comprehensive Training Results Analysis")
    print("=" * 60)
    
    # Load all logs
    logs = load_all_training_logs()
    
    # Performance comparison
    print("\n📊 Final Performance Comparison:")
    print("-" * 40)
    performance_df = analyze_performance_comparison(logs)
    print(performance_df.to_string(index=False))
    
    # Key findings
    print("\n🎯 Key Performance Findings:")
    print("-" * 40)
    
    if not performance_df.empty:
        best_wt = performance_df.loc[performance_df['WT_Dice'].idxmax()]
        best_tc = performance_df.loc[performance_df['TC_Dice'].idxmax()]
        best_hd95_wt = performance_df.loc[performance_df['WT_HD95'].idxmin()]
        best_hd95_tc = performance_df.loc[performance_df['TC_HD95'].idxmin()]
        most_efficient = performance_df.loc[performance_df['Efficiency_Score'].idxmax()]
        
        print(f"🏆 Best WT Dice: {best_wt['Model']} ({best_wt['WT_Dice']:.3f})")
        print(f"🏆 Best TC Dice: {best_tc['Model']} ({best_tc['TC_Dice']:.3f})")
        print(f"🏆 Best WT HD95: {best_hd95_wt['Model']} ({best_hd95_wt['WT_HD95']:.1f})")
        print(f"🏆 Best TC HD95: {best_hd95_tc['Model']} ({best_hd95_tc['TC_HD95']:.1f})")
        print(f"🏆 Most Efficient: {most_efficient['Model']} ({most_efficient['Efficiency_Score']:.3f})")
        
        # Attention mechanism analysis
        print(f"\n🔍 Attention Mechanism Analysis:")
        print("-" * 40)
        
        baseline_row = performance_df[performance_df['Model'] == 'Baseline']
        se_row = performance_df[performance_df['Model'] == 'SE-UNet']
        cbam_row = performance_df[performance_df['Model'] == 'CBAM-UNet']
        
        if not baseline_row.empty and not se_row.empty:
            se_improvement = se_row['WT_Dice'].iloc[0] - baseline_row['WT_Dice'].iloc[0]
            print(f"SE vs Baseline WT Dice: +{se_improvement:.3f}")
        
        if not baseline_row.empty and not cbam_row.empty:
            cbam_improvement = cbam_row['WT_Dice'].iloc[0] - baseline_row['WT_Dice'].iloc[0]
            print(f"CBAM vs Baseline WT Dice: +{cbam_improvement:.3f}")
        
        if not se_row.empty and not cbam_row.empty:
            cbam_vs_se = cbam_row['WT_Dice'].iloc[0] - se_row['WT_Dice'].iloc[0]
            print(f"CBAM vs SE WT Dice: +{cbam_vs_se:.3f}")
    
    # Convergence analysis
    print(f"\n📈 Convergence Speed Analysis:")
    print("-" * 40)
    convergence_df = analyze_convergence_speed(logs)
    if not convergence_df.empty:
        print(convergence_df.to_string(index=False))
    
    # Training time analysis
    print(f"\n⏱️ Training Time Analysis:")
    print("-" * 40)
    for model_name, log in logs.items():
        if log and 'total_training_time' in log:
            time_hours = log['total_training_time'] / 3600
            print(f"{model_name.upper()}: {time_hours:.1f} hours")
    
    # Create visualizations
    print(f"\n📊 Generating Visualizations...")
    
    # Convergence plots
    conv_fig = create_convergence_plots(logs)
    conv_fig.savefig('results/complete_convergence_analysis.png', dpi=300, bbox_inches='tight')
    
    # Performance comparison chart
    if not performance_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Dice comparison
        ax1 = axes[0]
        x = np.arange(len(performance_df))
        width = 0.35
        
        ax1.bar(x - width/2, performance_df['WT_Dice'], width, label='WT Dice', alpha=0.8)
        ax1.bar(x + width/2, performance_df['TC_Dice'], width, label='TC Dice', alpha=0.8)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Dice Score')
        ax1.set_title('Dice Score Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(performance_df['Model'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # HD95 comparison
        ax2 = axes[1]
        ax2.bar(x - width/2, performance_df['WT_HD95'], width, label='WT HD95', alpha=0.8)
        ax2.bar(x + width/2, performance_df['TC_HD95'], width, label='TC HD95', alpha=0.8)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('HD95 (voxels)')
        ax2.set_title('Boundary Accuracy (HD95)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(performance_df['Model'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    
    # Save results
    if not performance_df.empty:
        performance_df.to_csv('results/complete_performance_summary.csv', index=False)
        if not convergence_df.empty:
            convergence_df.to_csv('results/convergence_analysis.csv', index=False)
    
    plt.close('all')
    print(f"✅ Analysis complete! Results saved to results/ directory")
    
    return performance_df, convergence_df

if __name__ == "__main__":
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    
    performance_df, convergence_df = generate_comprehensive_report()
