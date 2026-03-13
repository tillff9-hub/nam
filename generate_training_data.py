import numpy as np
import pandas as pd
import random

# Parameters
n_samples = 1000  # Number of samples to generate
start_date = '2020-01-01'
end_date = '2026-03-13'

# Generate dates
date_range = pd.date_range(start=start_date, end=end_date, freq='H')[:n_samples]

# Generate synthetic price data
random.seed(42)  # For reproducibility
prices = np.cumsum(np.random.randn(n_samples) * random.uniform(0.5, 2)) + 1800  # Starting around 1800

# Create DataFrame
xauusd_data = pd.DataFrame({'Date': date_range, 'Price': prices})

# Save to CSV (optional)
# xauusd_data.to_csv('synthetic_xauusd_data.csv', index=False)

# Print first few rows
print(xauusd_data.head())