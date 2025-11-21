import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("real_estate_data_chicago.csv")

# -----------------------------
# Remove description column
# -----------------------------
for col in ["description", "Description"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# -----------------------------
# Label encode non-numeric columns
# -----------------------------
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == object or df[col].dtype == "object":
        df[col] = df[col].astype(str)   # convert to string to avoid errors
        df[col] = le.fit_transform(df[col])

# -----------------------------
# Compute correlation matrix
# -----------------------------
corr = df.corr()

# -----------------------------
# Plot heatmap
# -----------------------------
plt.figure(figsize=(12, 10))
plt.imshow(corr, interpolation="nearest")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.title("Correlation Heatmap of Features Affecting Price")
plt.tight_layout()
plt.show()
