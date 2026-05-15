import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "recorded_data_direct.csv"

JOINTS = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

df = pd.read_csv(CSV_PATH)
t = df["sim_time"] - df["sim_time"].iloc[0]

fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
for ax, joint in zip(axes.flat, JOINTS):
    ax.plot(t, df[f"target__{joint}"], label="commanded")
    ax.plot(t, df[f"qpos__{joint}"], label="actual")
    ax.set_title(joint)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

for ax in axes[-1]:
    ax.set_xlabel("time (s)")

fig.tight_layout()
plt.show()
