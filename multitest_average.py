import numpy as np
import sys
import os

# Get path from command line
path = sys.argv[1]

# Load the res.npy files
res = []
for f in os.listdir(path):
    if f.endswith(".npy") and f.startswith("_res"):
        res.append(np.load(path + "/" + f))

# Load the time.npy files
time = []
for f in os.listdir(path):
    if f.endswith(".npy") and f.startswith("_time"):
        time.append(np.load(path + "/" + f))

# fix format
print(res)
res = [r.squeeze() for r in res]
time = [t.squeeze() for t in time]


# Average the files
res = np.array(res)
res = np.mean(res, axis=0)

time = np.array(time)
time = np.mean(time, axis=0)

time = time[:, :, 0]
# Save the averaged files as csv
np.savetxt(path + "/avg_res.csv", res, delimiter=",")
np.savetxt(path + "/avg_time.csv", time, delimiter=",")