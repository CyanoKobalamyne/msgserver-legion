#!/usr/bin/env python3
import os.path
import subprocess

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
                    print(f"{t:{len('requests')}d} {r:{len('ratio')}d}  ", end="")
                    for cpu in CPUS:
                        times = []
                        for _ in range(ITERATIONS):
                            prog = subprocess.run(
                                [
                                    os.path.join(os.getcwd(), "messaging"),
                                    "-n",
                                    str(n),
                                    "-k",
                                    str(k),
                                    "-m",
                                    str(m),
                                    "-t",
                                    str(t),
                                    "-r",
                                    str(r),
                                    "-ll:cpu",
                                    str(cpu),
                                    "-level",
                                    "5",
                                ],
                                bufsize=0,
                                capture_output=True,
                                text=True,
                            )
                            if prog.returncode != 0:
                                continue
                            time = prog.stdout.splitlines()[0].split()[1]
                            times.append(int(time))
                        if not times:
                            print(" " * 5, end="")
                            continue
                        print(f"{sum(times) / len(times) / 1e6 : 4.0f} ", end="")
                    print()
