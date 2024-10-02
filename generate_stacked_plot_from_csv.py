import matplotlib.pyplot as plt
import csv
import sys
import os
import numpy as np

filepath = sys.argv[1]
data = []
sizes = [128, 256, 512, 1024]

if os.path.isfile(filepath):
    csvfile = open(filepath, 'r', newline='')

    reader = csv.reader(csvfile)
    data = [[float(n) if n != "" else 0 for n in row][1:] for row in reader]
print(data)


labels = ["Sinkhorn-Knopp", "Feature Extraction", "Gradient", "Hungarian"]
#plt.stackplot(
#    sizes,
#    data,
#    labels=labels,
#    baseline ='zero'
#)

n = len(data[0])

fig, ax = plt.subplots()
bottom = np.zeros(n)

print(len(data))
for i in range(len(labels)):
    print(i)
    print(data[:][i])
    p = ax.bar(range(n), data[:][i], 0.5, label=labels[i], bottom=bottom)
    bottom += data[i]

plt.show()

#plt.yscale("log")
plt.xlabel("Graph size")
plt.ylabel("Time (seconds)")
#plt.xticks(range(n), [2**(n+6) for n in range(7)]*2)
#slant xticks
plt.xticks(rotation=45)
plt.legend()
plt.savefig("timeplot.jpg")
