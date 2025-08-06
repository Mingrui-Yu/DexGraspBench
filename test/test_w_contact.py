import numpy as np
import matplotlib.pyplot as plt

# Define the parameter a
a = 1.0  # You can change this to any value

# Define the input range
x = np.linspace(-3, 3, 500)

# Define the function
y = 2 ** (a * x) - 1

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=rf"$2^{{{a}x}} - 1$", color="blue")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.title("Plot of $2^{ax} - 1$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
