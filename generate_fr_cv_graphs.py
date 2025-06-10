import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Set style for better-looking plots
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3

# Load the three datasets
print("Loading datasets...")
df_normal = pd.read_csv('fr_cv_normal.csv')
df_low_affected = pd.read_csv('fr_cv_low_affected.csv')
df_severe = pd.read_csv('fr_cv_severe.csv')

# Add condition labels
df_normal['condition'] = 'Normal'
df_low_affected['condition'] = 'Low Affected'
df_severe['condition'] = 'Severe'

# Combine all datasets
df_combined = pd.concat([df_normal, df_low_affected, df_severe], ignore_index=True)

print(f"Data loaded successfully!")
print(f"Normal: {len(df_normal)} samples")
print(f"Low Affected: {len(df_low_affected)} samples") 
print(f"Severe: {len(df_severe)} samples")
print(f"Combined: {len(df_combined)} samples")

# Create a comprehensive figure with multiple subplots
fig = plt.figure(figsize=(20, 16))

# 1. Scatter plot: Firing Rate vs ISI CV for all conditions with trend lines
ax1 = plt.subplot(3, 3, 1)
colors = {'Normal': '#1f77b4', 'Low Affected': '#ff7f0e', 'Severe': '#2ca02c'}

for condition in ['Normal', 'Low Affected', 'Severe']:
    data = df_combined[df_combined['condition'] == condition]
    x = data['firing_rate'].values
    y = data['ISI_CV'].values

    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    # Scatter plot
    plt.scatter(x_clean, y_clean, alpha=0.6, label=condition,
               color=colors[condition], s=20)

    # Calculate and plot trend line (linear regression)
    if len(x_clean) > 1:
        # Calculate linear regression coefficients
        coeffs = np.polyfit(x_clean, y_clean, 1)
        poly_func = np.poly1d(coeffs)

        # Generate points for trend line
        x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
        y_trend = poly_func(x_trend)

        # Plot trend line
        plt.plot(x_trend, y_trend, '--', color=colors[condition],
                linewidth=2, alpha=0.8,
                label=f'{condition} trend (R²={np.corrcoef(x_clean, y_clean)[0,1]**2:.3f})')

plt.xlabel('Firing Rate (Hz)')
plt.ylabel('ISI CV')
plt.title('Firing Rate vs ISI CV - All Conditions with Trend Lines')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# 2. Box plot: Firing Rate distribution by condition
ax2 = plt.subplot(3, 3, 2)
conditions = ['Normal', 'Low Affected', 'Severe']
fr_data = [df_combined[df_combined['condition'] == c]['firing_rate'].values for c in conditions]
bp1 = plt.boxplot(fr_data, tick_labels=conditions, patch_artist=True)
colors_box = ['lightblue', 'lightcoral', 'lightgreen']
for patch, color in zip(bp1['boxes'], colors_box):
    patch.set_facecolor(color)
plt.title('Firing Rate Distribution by Condition')
plt.ylabel('Firing Rate (Hz)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 3. Box plot: ISI CV distribution by condition
ax3 = plt.subplot(3, 3, 3)
cv_data = [df_combined[df_combined['condition'] == c]['ISI_CV'].values for c in conditions]
bp2 = plt.boxplot(cv_data, tick_labels=conditions, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors_box):
    patch.set_facecolor(color)
plt.title('ISI CV Distribution by Condition')
plt.ylabel('ISI CV')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 4. Mean Firing Rate by neuron index and condition
ax4 = plt.subplot(3, 3, 4)
neuron_indices = sorted(df_combined['neuron_index'].unique())
x = np.arange(len(neuron_indices))
width = 0.25

for i, condition in enumerate(conditions):
    means = []
    for neuron_idx in neuron_indices:
        data = df_combined[(df_combined['condition'] == condition) &
                          (df_combined['neuron_index'] == neuron_idx)]['firing_rate']
        means.append(data.mean() if len(data) > 0 else 0)

    plt.bar(x + i*width, means, width, label=condition, alpha=0.8,
            color=colors[condition])

plt.title('Mean Firing Rate by Neuron Type and Condition')
plt.xlabel('Neuron Index')
plt.ylabel('Mean Firing Rate (Hz)')
plt.xticks(x + width, [f'{int(idx)}' for idx in neuron_indices])
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Mean ISI CV by neuron index and condition
ax5 = plt.subplot(3, 3, 5)
for i, condition in enumerate(conditions):
    means = []
    for neuron_idx in neuron_indices:
        data = df_combined[(df_combined['condition'] == condition) &
                          (df_combined['neuron_index'] == neuron_idx)]['ISI_CV']
        means.append(data.mean() if len(data) > 0 else 0)

    plt.bar(x + i*width, means, width, label=condition, alpha=0.8,
            color=colors[condition])

plt.title('Mean ISI CV by Neuron Type and Condition')
plt.xlabel('Neuron Index')
plt.ylabel('Mean ISI CV')
plt.xticks(x + width, [f'{int(idx)}' for idx in neuron_indices])
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Histogram: Firing Rate distribution
ax6 = plt.subplot(3, 3, 6)
for condition in ['Normal', 'Low Affected', 'Severe']:
    data = df_combined[df_combined['condition'] == condition]['firing_rate']
    plt.hist(data, alpha=0.7, label=condition, bins=30, color=colors[condition])
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Frequency')
plt.title('Firing Rate Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# 7. Histogram: ISI CV distribution
ax7 = plt.subplot(3, 3, 7)
for condition in ['Normal', 'Low Affected', 'Severe']:
    data = df_combined[df_combined['condition'] == condition]['ISI_CV']
    plt.hist(data, alpha=0.7, label=condition, bins=30, color=colors[condition])
plt.xlabel('ISI CV')
plt.ylabel('Frequency')
plt.title('ISI CV Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# 8. Mean values comparison
ax8 = plt.subplot(3, 3, 8)
mean_stats = df_combined.groupby('condition')[['firing_rate', 'ISI_CV']].mean()
x = np.arange(len(mean_stats.index))
width = 0.35

bars1 = plt.bar(x - width/2, mean_stats['firing_rate'], width, 
                label='Firing Rate', alpha=0.8)
bars2 = plt.bar(x + width/2, mean_stats['ISI_CV'] * 100, width, 
                label='ISI CV (×100)', alpha=0.8)

plt.xlabel('Condition')
plt.ylabel('Mean Value')
plt.title('Mean Firing Rate and ISI CV by Condition')
plt.xticks(x, mean_stats.index, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height/100:.3f}', ha='center', va='bottom')

# 9. Correlation analysis
ax9 = plt.subplot(3, 3, 9)
correlations = []
conditions = ['Normal', 'Low Affected', 'Severe']
for condition in conditions:
    data = df_combined[df_combined['condition'] == condition]
    corr = data['firing_rate'].corr(data['ISI_CV'])
    correlations.append(corr)

bars = plt.bar(conditions, correlations, color=[colors[c] for c in conditions], alpha=0.8)
plt.ylabel('Correlation Coefficient')
plt.title('Firing Rate vs ISI CV Correlation')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, corr in zip(bars, correlations):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top')

plt.tight_layout()
plt.savefig('fr_cv_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

summary_stats = df_combined.groupby('condition').agg({
    'firing_rate': ['count', 'mean', 'std', 'min', 'max'],
    'ISI_CV': ['mean', 'std', 'min', 'max']
}).round(4)

print("\nFiring Rate Statistics:")
print(summary_stats['firing_rate'])
print("\nISI CV Statistics:")
print(summary_stats['ISI_CV'])

# Neuron type analysis
print("\n" + "="*60)
print("NEURON TYPE ANALYSIS")
print("="*60)

neuron_stats = df_combined.groupby(['condition', 'neuron_index']).agg({
    'firing_rate': ['count', 'mean', 'std'],
    'ISI_CV': ['mean', 'std']
}).round(4)

print(neuron_stats)

print("\nGraphs saved as 'fr_cv_comprehensive_analysis.png'")
