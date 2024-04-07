import matplotlib.pyplot as plt
import csv
import sys
import os

filepath = sys.argv[1]
data = []
sizes = [128, 256, 512, 1024]

if os.path.isfile(filepath):
    csvfile = open(filepath, 'r', newline='')

    reader = csv.reader(csvfile)
    data = [[float(n) if n != "" else 0 for n in row][1:] for row in reader]
print(data)

labels = ["Sinkhorn-Knopp", "Feature Extraction", "Gradient", "Hungarian"]
plt.stackplot(
    sizes,
    data,
    labels=labels,
    baseline ='zero',
)
plt.xlabel("Graph size")
plt.ylabel("Time (seconds)")
plt.legend()
plt.savefig("timeplot.jpg")
