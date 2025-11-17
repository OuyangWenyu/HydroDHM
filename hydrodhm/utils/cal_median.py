import pandas as pd

# not neuralhydrology
# read the CSV file
# df = pd.read_csv(
#     "/home/xushuolong1/hydro2/HydroDHM/results/neuralhydrology_results/camels_all/metrics_all.csv",
# )

# neuralhydrology
df = pd.read_csv(
    "/home/xushuolong1/hydro2/HydroDHM/results/neuralhydrology_results/camels_all/train_metrics_all.csv",
    index_col=0,
)

for col in df.columns:
    df[col] = df[col].str.strip("[]").astype(float)

# calculate the median
median_values = df[["NSE", "RMSE", "Corr", "KGE"]].median()


print(f"NSE: {median_values['NSE']:.4f}")
print(f"RMSE: {median_values['RMSE']:.4f}")
print(f"Corr: {median_values['Corr']:.4f}")
print(f"KGE: {median_values['KGE']:.4f}")
