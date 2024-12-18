import matplotlib.pyplot as plt

# Data for timing summary
sizes = [64, 128, 256, 512, 1024]
fast_times = [0.00378, 0.01482, 0.07476, 0.63413, 4.36493]
gpu_times = [0.00672, 0.01506, 0.05186, 0.20378, 0.84500]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(sizes, fast_times, label="Fast Operations", marker='o')
plt.plot(sizes, gpu_times, label="GPU Operations", marker='o')

# Adding labels, title, and legend
plt.xlabel("Matrix Size", fontsize=12)
plt.ylabel("Time (seconds)", fontsize=12)
plt.title("Comparison of Fast vs GPU Operations", fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig("3.4 graph.png")
# Display the graph
plt.show()
