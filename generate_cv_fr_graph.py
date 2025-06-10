import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Define fitting functions
def linear_func(x, a, b):
    return a * x + b

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

def logarithmic_func(x, a, b):
    return a * np.log(x) + b

def power_func(x, a, b):
    return a * np.power(x, b)

def polynomial_func(x, a, b, c):
    return a * x**2 + b * x + c

def fit_best_curve(x, y):
    """Find the best fitting curve among different models"""
    results = {}
    
    # Remove any invalid values
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        return None, None, None, None
    
    # Try linear fit
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        r_squared = r_value**2
        results['linear'] = {
            'params': [slope, intercept],
            'r_squared': r_squared,
            'func': linear_func,
            'name': 'Linear'
        }
    except:
        pass
    
    # Try exponential fit
    try:
        popt, _ = curve_fit(exponential_func, x_clean, y_clean, 
                           p0=[1, -0.1, 0], maxfev=2000)
        y_pred = exponential_func(x_clean, *popt)
        r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
        if r_squared > 0:
            results['exponential'] = {
                'params': popt,
                'r_squared': r_squared,
                'func': exponential_func,
                'name': 'Exponential'
            }
    except:
        pass
    
    # Try logarithmic fit
    try:
        popt, _ = curve_fit(logarithmic_func, x_clean, y_clean, maxfev=2000)
        y_pred = logarithmic_func(x_clean, *popt)
        r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
        if r_squared > 0:
            results['logarithmic'] = {
                'params': popt,
                'r_squared': r_squared,
                'func': logarithmic_func,
                'name': 'Logarithmic'
            }
    except:
        pass
    
    # Try power fit
    try:
        popt, _ = curve_fit(power_func, x_clean, y_clean, 
                           p0=[1, -0.5], maxfev=2000)
        y_pred = power_func(x_clean, *popt)
        r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
        if r_squared > 0:
            results['power'] = {
                'params': popt,
                'r_squared': r_squared,
                'func': power_func,
                'name': 'Power'
            }
    except:
        pass
    
    # Try polynomial fit (degree 2)
    try:
        popt, _ = curve_fit(polynomial_func, x_clean, y_clean, maxfev=2000)
        y_pred = polynomial_func(x_clean, *popt)
        r_squared = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
        if r_squared > 0:
            results['polynomial'] = {
                'params': popt,
                'r_squared': r_squared,
                'func': polynomial_func,
                'name': 'Polynomial (2nd degree)'
            }
    except:
        pass
    
    # Find best fit
    if not results:
        return None, None, None, None
    
    best_fit = max(results.items(), key=lambda x: x[1]['r_squared'])
    best_name = best_fit[0]
    best_data = best_fit[1]
    
    return best_data['func'], best_data['params'], best_data['r_squared'], best_data['name']

def main():
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

    # Create figure
    plt.figure(figsize=(12, 8))

    # Define colors for each condition
    colors = {'Normal': '#1f77b4', 'Low Affected': '#ff7f0e', 'Severe': '#2ca02c'}
    conditions = ['Normal', 'Low Affected', 'Severe']

    # Store best fit information for each condition
    best_fits = {}

    # Plot data and trend lines for each condition (ISI_CV on X-axis, Firing_Rate on Y-axis)
    for condition in conditions:
        data = df_combined[df_combined['condition'] == condition]
        # X = ISI_CV, Y = Firing_Rate
        x = data['ISI_CV'].values
        y = data['firing_rate'].values
        
        # Remove any NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Scatter plot
        plt.scatter(x_clean, y_clean, alpha=0.5, label=f'{condition} (n={len(x_clean)})', 
                   color=colors[condition], s=15)
        
        # Find best fitting curve
        if len(x_clean) > 3:
            best_func, best_params, best_r2, best_name = fit_best_curve(x_clean, y_clean)
            
            if best_func is not None:
                # Generate points for trend line
                x_trend = np.linspace(x_clean.min(), x_clean.max(), 200)
                
                try:
                    y_trend = best_func(x_trend, *best_params)
                    
                    # Plot trend line
                    plt.plot(x_trend, y_trend, '--', color=colors[condition], 
                            linewidth=2.5, alpha=0.9)
                    
                    # Store information
                    best_fits[condition] = {
                        'name': best_name,
                        'r_squared': best_r2,
                        'params': best_params
                    }
                    
                    # Print statistics
                    print(f"\n{condition}:")
                    print(f"  Best fit: {best_name}")
                    print(f"  R²: {best_r2:.4f}")
                    print(f"  Parameters: {best_params}")
                    
                except Exception as e:
                    print(f"Error plotting {condition}: {e}")
        else:
            print(f"\n{condition}: Not enough data points for fitting")

    # Customize the plot
    plt.xlabel('ISI CV (Coefficient of Variation)', fontsize=12)
    plt.ylabel('Firing Rate (Hz)', fontsize=12)
    plt.title('ISI CV vs Firing Rate with Best-Fit Trend Lines\nAcross Different Conditions', fontsize=14, pad=20)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)

    # Add text box with fitting info
    fit_info = "Best Fit Models:\n"
    for condition in conditions:
        if condition in best_fits:
            fit_info += f"{condition}: {best_fits[condition]['name']} (R²={best_fits[condition]['r_squared']:.3f})\n"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, fit_info.strip(), transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    # Improve layout
    plt.tight_layout()

    # Save the plot
    plt.savefig('cv_fr_scatter_best_fit.png', dpi=300, bbox_inches='tight')
    print(f"\nGraph saved as 'cv_fr_scatter_best_fit.png'")

    # Show the plot
    plt.show()

    # Print detailed equations
    print("\n" + "="*60)
    print("BEST FIT EQUATIONS")
    print("="*60)

    for condition in conditions:
        if condition in best_fits:
            fit_data = best_fits[condition]
            print(f"\n{condition}:")
            print(f"  Model: {fit_data['name']}")
            print(f"  R²: {fit_data['r_squared']:.4f}")
            
            # Print equation based on model type
            if 'Linear' in fit_data['name']:
                a, b = fit_data['params']
                print(f"  Equation: Firing_Rate = {a:.6f} * ISI_CV + {b:.6f}")
            elif 'Exponential' in fit_data['name']:
                a, b, c = fit_data['params']
                print(f"  Equation: Firing_Rate = {a:.6f} * exp({b:.6f} * ISI_CV) + {c:.6f}")
            elif 'Logarithmic' in fit_data['name']:
                a, b = fit_data['params']
                print(f"  Equation: Firing_Rate = {a:.6f} * ln(ISI_CV) + {b:.6f}")
            elif 'Power' in fit_data['name']:
                a, b = fit_data['params']
                print(f"  Equation: Firing_Rate = {a:.6f} * ISI_CV^{b:.6f}")
            elif 'Polynomial' in fit_data['name']:
                a, b, c = fit_data['params']
                print(f"  Equation: Firing_Rate = {a:.6f} * ISI_CV² + {b:.6f} * ISI_CV + {c:.6f}")

if __name__ == "__main__":
    main()
