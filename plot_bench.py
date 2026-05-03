import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

labels = ["baseline", "+ kernels", "+ GNS", "+ split heads", "but dion", "dion .5", "dion .25", "dion .125", "adamw"]
gpu8 = pd.Series([98, 86, 65, 47, 59, 35, 27, 28, 14], index=labels)
gpu1 = pd.Series([654, 579, 449, 331, 417, 184, 120, 98, 105], index=labels)

gpu8 = gpu8 / gpu8.loc["adamw"]
gpu1 = gpu1 / gpu1.loc["adamw"]


fig, ax1 = plt.subplots(figsize=(12, 6))

ax2 = ax1.twinx()

sns.barplot(x=gpu8.index, y=gpu8.values, ax=ax1, color="steelblue", alpha=0.7, label="GPU8")
sns.barplot(x=gpu1.index, y=gpu1.values, ax=ax2, color="coral", alpha=0.7, label="GPU1")

ax1.set_ylabel("GPU8")
ax2.set_ylabel("GPU1")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()