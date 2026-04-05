import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def training_dynamics_analysis():
    """Analyze training behavior and stability"""
    
    # Load training logs
    models = ['baseline', 'se', 'cbam', 'hybrid']
    logs = {}
    
    for model in models:
        try:
            with open(f'final results/{model}/training_log.json', 'r') as f:
                logs[model] = json.load(f)
        except:
            print(f"⚠️ Could not load {model} log")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Dynamics Analysis', fontsize=16, fontweight='bold')
    
    colors = {'baseline': 'blue', 'se': 'orange', 'cbam': 'green', 'hybrid': 'red'}
    
    # 1. Loss stability (variance)
    ax1 = axes[0, 0]
    for model, log in logs.items():
        if log and 'val_loss' in log:
            val_loss = log['val_loss']
            # Rolling variance
            window = min(5, len(val_loss))
            variance = [np.var(val_loss[max(0,i-window):i+1]) for i in range(len(val_loss))]
            epochs = range(1, len(variance) + 1)
            ax1.plot(epochs, variance, label=model, color=colors[model], linewidth=2)
    ax1.set_title('Validation Loss Variance (Stability)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Variance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Convergence speed
    ax2 = axes[0, 1]
    convergence_epochs = {}
    for model, log in logs.items():
        if log and 'wt_dice' in log:
            wt_dice = log['wt_dice']
            for i, dice in enumerate(wt_dice):
                if dice >= 0.8:
                    convergence_epochs[model] = i + 1
                    break
    
    if convergence_epochs:
        models_conv = list(convergence_epochs.keys())
        epochs_conv = list(convergence_epochs.values())
        colors_conv = [colors[m] for m in models_conv]
        bars = ax2.bar(models_conv, epochs_conv, color=colors_conv, alpha=0.7)
        ax2.set_title('Convergence Speed (Epochs to 80% WT Dice)')
        ax2.set_ylabel('Epochs')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, epoch in zip(bars, epochs_conv):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(epoch), ha='center', va='bottom')
    
    # 3. Final performance comparison
    ax3 = axes[0, 2]
    final_metrics = {}
    for model, log in logs.items():
        if log:
            final_metrics[model] = {
                'wt_dice': log.get('final_wt_dice', 0),
                'tc_dice': log.get('final_tc_dice', 0),
                'wt_hd95': log.get('final_wt_hd95', 0),
                'tc_hd95': log.get('final_tc_hd95', 0)
            }
    
    if final_metrics:
        models_list = list(final_metrics.keys())
        wt_dices = [final_metrics[m]['wt_dice'] for m in models_list]
        tc_dices = [final_metrics[m]['tc_dice'] for m in models_list]
        
        x = np.arange(len(models_list))
        width = 0.35
        
        ax3.bar(x - width/2, wt_dices, width, label='WT Dice', alpha=0.8)
        ax3.bar(x + width/2, tc_dices, width, label='TC Dice', alpha=0.8)
        ax3.set_title('Final Performance Comparison')
        ax3.set_ylabel('Dice Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models_list, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Parameter efficiency
    ax4 = axes[1, 0]
    param_efficiency = {}
    for model, log in logs.items():
        if log:
            wt_dice = log.get('final_wt_dice', 0)
            tc_dice = log.get('final_tc_dice', 0)
            # Estimate parameters
            params = {'baseline': 5.66e6, 'se': 5.66e6, 'cbam': 5.67e6, 'hybrid': 5.86e6}
            efficiency = (wt_dice + tc_dice) / (params.get(model, 5.66e6) / 1e6)
            param_efficiency[model] = efficiency
    
    if param_efficiency:
        models_eff = list(param_efficiency.keys())
        efficiencies = list(param_efficiency.values())
        colors_eff = [colors[m] for m in models_eff]
        bars = ax4.bar(models_eff, efficiencies, color=colors_eff, alpha=0.7)
        ax4.set_title('Parameter Efficiency (Dice per M params)')
        ax4.set_ylabel('Efficiency Score')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, eff in zip(bars, efficiencies):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{eff:.3f}', ha='center', va='bottom')
    
    # 5. Training time vs performance
    ax5 = axes[1, 1]
    time_performance = {}
    for model, log in logs.items():
        if log:
            wt_dice = log.get('final_wt_dice', 0)
            training_time = log.get('total_training_time', 0) / 3600  # Convert to hours
            time_performance[model] = (training_time, wt_dice)
    
    if time_performance:
        for model, (time_val, dice_val) in time_performance.items():
            ax5.scatter(time_val, dice_val, s=100, c=colors[model], 
                      label=model, alpha=0.7)
            ax5.annotate(model, (time_val, dice_val), 
                       xytext=(5, 5), textcoords='offset points')
    
    ax5.set_title('Training Time vs Final Performance')
    ax5.set_xlabel('Training Time (hours)')
    ax5.set_ylabel('Final WT Dice')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Hide empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/training_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Training dynamics analysis completed")

if __name__ == "__main__":
    training_dynamics_analysis()
