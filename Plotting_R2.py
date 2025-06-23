import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import numpy as np

def plot_pred_vs_actual(preds, actuals, title="Prediction vs Actual", save_path="pred_vs_actual.png"):
    r2 = r2_score(actuals, preds)
    r, _ = pearsonr(actuals, preds)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, preds, alpha=0.5, edgecolors='k', label='Predicted vs Actual')
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=2, label='Ideal Line')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Add R² and r to plot
    text = f"$R^2$ = {r2:.5f}, $r$ = {r:.5f}"
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Scatter plot saved to: {save_path}")
