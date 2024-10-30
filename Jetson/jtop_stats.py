import jtop
import matplotlib.pyplot as plt
import time

def collect_stats(duration=60, interval=1):
    cpu_usage = []
    memory_usage = []
    gpu_usage = []
    timestamps = []

    jtop_instance = jtop.Jtop()
    start_time = time.time()

    try:
        while (time.time() - start_time) < duration:
            stats = jtop_instance.get_stats()

            cpu_usage.append(stats["cpu"]["usage"])
            memory_usage.append(stats["memory"]["used"] / (1024 ** 2))  # Convert to MiB
            gpu_usage.append(stats["gpu"]["usage"])
            timestamps.append(time.time() - start_time)

            time.sleep(interval)

    finally:
        jtop_instance.close()  # Ensure jtop is properly closed

    return timestamps, cpu_usage, memory_usage, gpu_usage

def plot_stats(timestamps, cpu_usage, memory_usage, gpu_usage):
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(timestamps, cpu_usage, label='CPU Usage (%)', color='orange')
    plt.title('CPU Usage over Time')
    plt.ylabel('CPU Usage (%)')
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(timestamps, memory_usage, label='Memory Usage (MiB)', color='green')
    plt.title('Memory Usage over Time')
    plt.ylabel('Memory Usage (MiB)')
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(timestamps, gpu_usage, label='GPU Usage (%)', color='blue')
    plt.title('GPU Usage over Time')
    plt.ylabel('GPU Usage (%)')
    plt.grid()

    plt.xlabel('Time (seconds)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    duration = 60  # Collect stats for 60 seconds
    interval = 1   # Collect every second

    timestamps, cpu_usage, memory_usage, gpu_usage = collect_stats(duration, interval)
    plot_stats(timestamps, cpu_usage, memory_usage, gpu_usage)


# #!/usr/bin/python3
# # -*- coding: UTF-8 -*-

# from jtop import jtop, JtopException
# import csv
# import argparse


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Simple jtop logger')
#     # Standard file to store the logs
#     parser.add_argument('--file', action="store", dest="file", default="log.csv")
#     args = parser.parse_args()

#     print("Simple jtop logger")
#     print("Saving log on {file}".format(file=args.file))

#     try:
#         with jtop() as jetson:
#             # Make csv file and setup csv
#             with open(args.file, 'w') as csvfile:
#                 stats = jetson.stats
#                 # Initialize cws writer
#                 writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
#                 # Write header
#                 writer.writeheader()
#                 # Write first row
#                 writer.writerow(stats)
#                 # Start loop
#                 while jetson.ok():
#                     stats = jetson.stats
#                     # Write row
#                     writer.writerow(stats)
#                     print("Log at {time}".format(time=stats['time']))
#     except JtopException as e:
#         print(e)
#     except KeyboardInterrupt:
#         print("Closed with CTRL-C")
#     except IOError:
#         print("I/O error")
