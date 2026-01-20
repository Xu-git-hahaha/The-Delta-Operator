import re
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




LOSS_SMOOTH_WINDOW = 50  
GRAD_SMOOTH_WINDOW = 100  
FIGURE_DPI = 300  


def parse_log_file(filepath):
    """解析单个日志文件"""
    print(f"📄 Processing: {filepath}")

    step_data = []
    epoch_data = []

    
    
    step_pattern = re.compile(
        r"Step\s+(\d+)\s+\|\s+Loss:\s+([\d\.]+)\s+\|\s+Grad:\s+([\d\.]+)\s+\|\s+Gamma:\s+([\d\.]+)\s+\|\s+Rho:\s+([\d\.]+)")

    
    
    
    epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+Done\.\s+Avg Loss:\s+([\d\.]+)")
    time_pattern = re.compile(r"Time:\s+([\d\.]+)s")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            
            step_match = step_pattern.search(line)
            if step_match:
                step_data.append({
                    'Step': int(step_match.group(1)),
                    'Loss': float(step_match.group(2)),
                    'Grad': float(step_match.group(3)),
                    'Gamma': float(step_match.group(4)),
                    'Rho': float(step_match.group(5))
                })
                continue

            
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                epoch_info = {
                    'Epoch': int(epoch_match.group(1)),
                    'AvgLoss': float(epoch_match.group(2)),
                    'Time': 0.0  
                }
                
                time_match = time_pattern.search(line)
                if time_match:
                    epoch_info['Time'] = float(time_match.group(1))

                epoch_data.append(epoch_info)

    return pd.DataFrame(step_data), pd.DataFrame(epoch_data)


def analyze_and_plot(filename, df_step, df_epoch):
    """生成图表和分析报告"""
    if df_step.empty:
        print(f"⚠️ Warning: No step data found in {filename}")
        return

    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_dir = os.path.join(os.path.dirname(filename), f"{base_name}_analysis")
    os.makedirs(output_dir, exist_ok=True)

    
    plt.style.use('seaborn-v0_8-whitegrid')

    
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_step['Step'], df_step['Loss'], alpha=0.2, color='gray', label='Raw Loss', linewidth=0.5)

    if len(df_step) > LOSS_SMOOTH_WINDOW:
        df_step['Loss_Smooth'] = df_step['Loss'].rolling(window=LOSS_SMOOTH_WINDOW, min_periods=1).mean()
        plt.plot(df_step['Step'], df_step['Loss_Smooth'], color='#1f77b4', linewidth=2,
                 label=f'Smoothed (MA-{LOSS_SMOOTH_WINDOW})')

    plt.title(f'Training Loss Curve (Tiny ImageNet) - {base_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=FIGURE_DPI)
    plt.close()

    
    
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_gamma = '#d62728'  
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Gamma (Kernel Scale)', color=color_gamma, fontweight='bold')
    ax1.plot(df_step['Step'], df_step['Gamma'], color=color_gamma, linewidth=2, label='Gamma')
    ax1.tick_params(axis='y', labelcolor=color_gamma)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_rho = '#2ca02c'  
    ax2.set_ylabel('Rho (Tail Heaviness)', color=color_rho, fontweight='bold')
    ax2.plot(df_step['Step'], df_step['Rho'], color=color_rho, linewidth=2, linestyle='--', label='Rho')
    ax2.tick_params(axis='y', labelcolor=color_rho)

    plt.title(f'Geometric Kernel Parameters - {base_name}', fontsize=12, fontweight='bold')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gamma_rho_evolution.png'), dpi=FIGURE_DPI)
    plt.close()

    
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_step['Step'], df_step['Grad'], color='#9467bd', alpha=0.25, linewidth=0.5, label='Raw Grad Norm')

    if len(df_step) > GRAD_SMOOTH_WINDOW:
        df_step['Grad_Smooth'] = df_step['Grad'].rolling(window=GRAD_SMOOTH_WINDOW, min_periods=1).mean()
        plt.plot(df_step['Step'], df_step['Grad_Smooth'], color='#4a148c', linewidth=1.5,
                 label=f'Smoothed (MA-{GRAD_SMOOTH_WINDOW})')

    plt.title(f'Gradient Norm Stability - {base_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Steps')
    plt.ylabel('Gradient Norm (L2)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grad_norm.png'), dpi=FIGURE_DPI)
    plt.close()

    
    
    
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write(f"=== Analysis Report for {base_name} ===\n\n")
        f.write(f"Dataset: Tiny ImageNet (Inferred)\n")
        f.write(f"Total Steps: {df_step['Step'].max()}\n")
        f.write(f"Total Epochs: {df_epoch['Epoch'].max() if not df_epoch.empty else 0}\n")

        if not df_epoch.empty:
            best_epoch = df_epoch.loc[df_epoch['AvgLoss'].idxmin()]
            f.write(f"Best Epoch: {int(best_epoch['Epoch'])} (Loss: {best_epoch['AvgLoss']:.6f})\n")
            if df_epoch['Time'].sum() > 0:
                f.write(f"Total Training Time: {df_epoch['Time'].sum() / 3600:.2f} hours\n")
            else:
                f.write(f"Total Training Time: Not logged\n")

        f.write("\n--- Delta Parameter Convergence ---\n")
        f.write(f"Gamma: Start={df_step['Gamma'].iloc[0]:.4f} -> End={df_step['Gamma'].iloc[-1]:.4f}\n")
        f.write(f"Rho:   Start={df_step['Rho'].iloc[0]:.4f}   -> End={df_step['Rho'].iloc[-1]:.4f}\n")

        f.write("\n--- Stability Statistics ---\n")
        f.write(f"Max Grad Norm: {df_step['Grad'].max():.4f}\n")
        f.write(f"Mean Grad Norm: {df_step['Grad'].mean():.4f}\n")

    print(f"✅ Analysis saved to: {output_dir}/")


def main():
    log_files = glob.glob("*.txt")
    if not log_files:
        print("❌ No .txt log files found.")
        return

    print(f"Found {len(log_files)} log files.")
    for log_file in log_files:
        try:
            df_step, df_epoch = parse_log_file(log_file)
            analyze_and_plot(log_file, df_step, df_epoch)
        except Exception as e:
            print(f"❌ Error processing {log_file}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()