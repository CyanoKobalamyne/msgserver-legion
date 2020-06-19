import os
import sys

count = int(sys.argv[1])
data = [0, 0, 0, 0]

for i in range(count):
    stream = os.popen("./messaging -n 1 -k 5 -m 1000 -t 1000 -r 1 -ll:cpu 2")
    res = stream.read().strip().splitlines()
    for j in range(len(data)):
        data[j] += float(res[j + 1].split()[2])
for j in range(len(data)):
    data[j] /= count
print(data)
