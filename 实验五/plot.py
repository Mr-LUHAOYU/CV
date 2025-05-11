import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


f = open('./log.txt', 'r')
lines = f.readlines()
f.close()

BSD_values = []
Manga_values = []
Urban_values = []

for line in lines:
    file_name, _, v = line.split()
    v = float(v)
    if 'BSD' in file_name:
        BSD_values.append(v)
    elif 'Manga' in file_name:
        Manga_values.append(v)
    elif 'Urban' in file_name:
        Urban_values.append(v)

all_values = BSD_values + Manga_values + Urban_values
values = [BSD_values, Manga_values, Urban_values, all_values]
names = ['BSD', 'Manga', 'Urban', 'All']

for psnr_values, name in zip(values, names):
    psnr_values = np.array(psnr_values)

    plt.figure(figsize=(10, 6))
    sns.histplot(psnr_values, bins=30, kde=True)
    plt.xlabel("PSNR (dB)")
    plt.ylabel("Count")
    plt.title(name)
    plt.savefig(f"../assets/{name}_count.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=psnr_values)
    plt.xlabel("PSNR (dB)")
    plt.title(name)
    plt.savefig(f"../assets/{name}_Boxplot.png")

