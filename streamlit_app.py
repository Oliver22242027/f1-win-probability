import streamlit as st
import matplotlib.pyplot as plt
import fastf1 as ff1
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# make sure fastf1 has somewhere to cache data
import os
os.makedirs(".fastf1cache", exist_ok=True)
ff1.Cache.enable_cache(".fastf1cache")



# ---------- helper functions ----------
def load_race_laps(year: int, gp_name: str):
    """Load lap data for one race and create features + labels"""
    s = ff1.get_session(year, gp_name, "R")
    s.load(telemetry=False, weather=True, laps=True)
    df = s.laps.reset_index(drop=True).copy()

    winner_num = str(s.results.sort_values("Position").iloc[0]["DriverNumber"])
    df["y_win"] = (df["DriverNumber"].astype(str) == winner_num).astype(int)

    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()
    df["CumTimeSeconds"] = df.groupby("DriverNumber")["LapTimeSeconds"].cumsum()
    leader_cum = df.groupby("LapNumber").apply(
        lambda g: g.sort_values("Position").iloc[0]["CumTimeSeconds"]
    )
    leader_cum.name = "LeaderCumTime"
    df = df.merge(leader_cum, left_on="LapNumber", right_index=True, how="left")
    df["GapToLeader"] = df["CumTimeSeconds"] - df["LeaderCumTime"]

    df["IsPitLap"] = df["PitOutTime"].notna() | df["PitInTime"].notna()
    df["TyreAgeLaps"] = df.groupby("DriverNumber")["IsPitLap"].cumsum()
    total_laps = int(df["LapNumber"].max())
    df["LapsRemaining"] = total_laps - df["LapNumber"]
    df["RaceFracComplete"] = df["LapNumber"] / total_laps
    df["IsSafetyCar"] = df["TrackStatus"].fillna("").str.contains("4|5").astype(int)
    return df


def make_X_y(df: pd.DataFrame):
    out = df.copy()
    out["GapToLeader"] = out["GapToLeader"].fillna(out["GapToLeader"].median())
    out["IsPitLap"] = out["IsPitLap"].astype(int)
    out["PosRankInv"] = 1.0 / out["Position"].replace(0, np.nan)
    out["PitCount"] = out.groupby("DriverNumber")["IsPitLap"].cumsum()
    out["PosTimesRem"] = out["Position"] * out["LapsRemaining"]

    X = out[[
        "LapNumber","Position","GapToLeader","TyreAgeLaps",
        "RaceFracComplete","LapsRemaining","IsSafetyCar",
        "PosRankInv","PitCount","PosTimesRem"
    ]].fillna(0)

    y = out["y_win"].values
    meta = out[["DriverNumber","Driver","LapNumber"]].reset_index(drop=True)
    return X, y, meta


# ---------- streamlit app ----------
st.title("🏎️ F1 Win Probability Model")

year = st.selectbox("Choose Year", [2023, 2022], index=0)
gp = st.selectbox("Choose Grand Prix", ["Bahrain", "Saudi Arabia", "Australia",
                                        "Azerbaijan", "Miami", "Monaco",
                                        "Spain", "Canada", "Austria",
                                        "Great Britain"], index=9)

laps = load_race_laps(year, gp)
X, y, meta = make_X_y(laps)

pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", LogisticRegression(max_iter=300))
])
pipe.fit(X, y)
probs = pipe.predict_proba(X)[:, 1]

trace = pd.concat([meta, pd.Series(probs, name="WinProb")], axis=1)
top_drivers = trace.groupby("DriverNumber")["WinProb"].max().sort_values(ascending=False).head(5).index.tolist()

fig, ax = plt.subplots(figsize=(10, 5))
for drv in top_drivers:
    d = trace[trace["DriverNumber"] == drv]
    ax.plot(d["LapNumber"], d["WinProb"], label=f"{d['Driver'].iloc[0]}")
ax.set_xlabel("Lap")
ax.set_ylabel("Win Probability")
ax.set_title(f"{year} {gp} Grand Prix")
ax.legend()
st.pyplot(fig)
