# EDA  results 
# ea results performed on the f1 dataset to make the predictions 
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

DATA_PATH = os.environ.get("F1_CLEAN_RESULTS", "data/clean/results.parquet")
df = pd.read_parquet(DATA_PATH)
plt.rcParams["figure.figsize"]=(9,6)

# structure and quality 
print(df.shape)
print(df.dtypes)
print(df.head())

# missing values 
print(df.isnull().sum()[lambda s: s > 0])
print(df[["grid", "grid_clean", "position"]].describe())

# Races and drivers per season (a sudden drop means missing data)
per_season = df.groupby("season").agg(
    races=("round", "nunique"),
    drivers=("driver", "nunique"),
    rows=("driver", "size"),
)

# target: how imbalanced
print("podium rate:", round(float(df["is_podium"].mean()), 3))
print("winner rate:", round(float(df["is_winner"].mean()), 3))

# grid position vs result
grid = df[df["grid_clean"] <= 20].groupby("grid_clean").agg(
    win_rate=("is_winner", "mean"),
    podium_rate=("is_podium", "mean"),
)
grid.plot(kind="bar", title="Win and podium rate by starting position")
plt.ylabel("rate")
plt.tight_layout()
plt.show()

# how strong is the grid -> finish relationship season by season
# A drop in a season can signal a regulation or competitive-order change
corr = df.groupby("season").apply(
    lambda d: d["grid_clean"].corr(d["position"], method="spearman")
)
corr.plot(marker="o", title="Spearman correlation: grid vs finish, by season")
plt.ylabel("correlation")
plt.tight_layout()
plt.show()

#Baseline: "podium = starts in top 3"
base = (df["grid_clean"] <= 3).astype(int)
precision = df.loc[base == 1, "is_podium"].mean()
recall = df.loc[df["is_podium"] == 1, "grid_clean"].le(3).mean()
print(f"Top-3-grid baseline  precision={precision:.3f}  recall={recall:.3f}")
team_pod = (
    df[df["is_podium"] == 1]
    .groupby(["season", "team"]).size().unstack(fill_value=0)
)
team_share = team_pod.div(team_pod.sum(axis=1), axis=0)
top_teams = team_share.sum().nlargest(6).index
team_share[top_teams].plot(kind="bar", stacked=True, figsize=(11, 4),
                           title="Share of podium finishes by team")
plt.tight_layout()
plt.show()
 
# Drivers with the most podiums in the data
df.groupby("driver")["is_podium"].sum().nlargest(10).plot(
    kind="barh", title="Most podium finishes").invert_yaxis()
plt.tight_layout()
plt.show()
  
dnf_by_season = df.groupby("season")[["dnf", "dnf_mechanical", "dnf_accident"]].mean()
dnf_by_season.plot(marker="o", title="DNF rate by season")
plt.tight_layout()
plt.show()

dnf_by_team = (
    df.groupby("team")
    .agg(rows=("dnf", "size"), dnf_rate=("dnf", "mean"), mech_rate=("dnf_mechanical", "mean"))
    .query("rows >= 50")
    .sort_values("dnf_rate", ascending=False)
)
dnf_by_team.head(10)
pole = df[df["grid_clean"] == 1]
circuit_conv = (
    pole.groupby("circuit")
    .agg(races=("is_winner", "size"), pole_win_rate=("is_winner", "mean"))
    .query("races >= 8")
    .sort_values("pole_win_rate")
)
circuit_conv.plot(kind="barh", y="pole_win_rate", figsize=(9, 8),
                  title="How often pole wins, by circuit", legend=False)
plt.tight_layout()
plt.show()
