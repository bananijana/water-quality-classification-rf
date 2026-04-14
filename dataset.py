import numpy as np
import pandas as pd

np.random.seed(42)
n = 80

# Base geochemical/hydrochemical parameters
data = {
    # Major ions (mg/L)
    "Ca":     np.round(np.random.uniform(10, 120, n), 2),
    "Mg":     np.round(np.random.uniform(5, 60, n), 2),
    "Na":     np.round(np.random.uniform(8, 200, n), 2),
    "K":      np.round(np.random.uniform(1, 20, n), 2),
    "HCO3":   np.round(np.random.uniform(50, 400, n), 2),
    "Cl":     np.round(np.random.uniform(10, 250, n), 2),
    "SO4":    np.round(np.random.uniform(5, 180, n), 2),
    "NO3":    np.round(np.random.uniform(0.5, 45, n), 2),
    # Physical parameters
    "pH":     np.round(np.random.uniform(6.5, 8.8, n), 2),
    "EC":     np.round(np.random.uniform(200, 2800, n), 1),   # µS/cm
    "TDS":    np.round(np.random.uniform(130, 1800, n), 1),   # mg/L
    "Temp":   np.round(np.random.uniform(18, 32, n), 1),      # °C
    # Heavy metals (mg/L)
    "Fe":     np.round(np.random.uniform(0.01, 2.5, n), 3),
    "Mn":     np.round(np.random.uniform(0.001, 0.8, n), 3),
}

df = pd.DataFrame(data)

# Derived: TH (Total Hardness)
df["TH"] = np.round(2.497 * df["Ca"] + 4.116 * df["Mg"], 2)

# Water type label based on dominant ion logic (simplified)
def water_type(row):
    if row["Ca"] + row["Mg"] > row["Na"] + row["K"]:
        if row["HCO3"] > row["Cl"] + row["SO4"]:
            return "Ca-HCO3"
        else:
            return "Ca-Cl"
    else:
        if row["HCO3"] > row["Cl"] + row["SO4"]:
            return "Na-HCO3"
        else:
            return "Na-Cl"

df["WaterType"] = df.apply(water_type, axis=1)

# Sample ID
df.insert(0, "SampleID", [f"S{str(i+1).zfill(3)}" for i in range(n)])

df.to_csv("/home/claude/projects/dataset.csv", index=False)
print(df.head())
print(df["WaterType"].value_counts())
