#!/usr/bin/env python3
import os
import os.path

ITERATIONS = 30

USERS = [1, 2, 5, 10, 20, 50, 100]
CHANNELS = [5, 10, 20, 50, 100]
MESSAGES = [500]

REQUESTS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
RATIOS = [1, 10]
CPUS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


for n in USERS:
    for k in CHANNELS:
        for m in MESSAGES:
            print("# users:", n)
            print("# channels:", k)
            print("# messages:", m)
            print()
            print("requests ratio  " + " ".join(f"{c:4d}" for c in CPUS))
            for t in REQUESTS:
                for r in RATIOS:
                    print(f"{t:{len('requests')}d} {r:{len('ratio')}d}  ",
                          end="")
                    results = []
                    for cpu in CPUS:
                        times = []
                        for _ in range(ITERATIONS):
                            stream = os.popen(
                                f"{os.path.join(os.getcwd(),'messaging')} "
                                f"-n {n} -k {k} -m {m} -t {t} -r {r} "
                                f"-ll:cpu {cpu} -level 5")
                            times.append(int(stream.readline().strip()))
                        results.append(sum(times) / len(times))
                    print(" ".join(f"{r/1e6:4.0f}" for r in results))
            print("\n")
