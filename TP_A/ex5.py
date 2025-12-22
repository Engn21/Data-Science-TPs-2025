import numpy as np
import matplotlib.pyplot as plt

# Problem data: 
# Line alpha: 3x + 4y = -6  <=> n·x + d = 0 with n=(3,4), d=6
n = np.array([3.0, 4.0])
d = 6.0
A = np.array([-1.0, 3.0])  # point A

# to equalize ||n|| to norm_n
norm_n = np.linalg.norm(n)

# Distance via scalar product 
signed_distance = (n @ A + d) / norm_n
distance = abs(signed_distance)

# Foot of the perpendicular ,orthogonal projection of A onto the line
H = A - ((n @ A + d) / (norm_n**2)) * n

# Also pick a concrete p0 on the line, this is better to use for the alt formula thanks to dot with n
# Set y=0 -> 3x + 4*0 = -6 => x=-2, so p0 = (-2,0)
p0 = np.array([-2.0, 0.0])

#Here the code checks the alternative (projection length) formula
alt_num = (A - p0) @ n
alt_dist = abs(alt_num) / norm_n

#Outputs 
print(f"Normal vector n = {n}, ||n|| = {norm_n}")
print(f"Line in normal form: n·x + d = 0 with d = {d}")
print(f"Point A = {A}")
print(f"Signed distance = {signed_distance}")
print(f"Distance (point to line) = {distance}")
print(f"Foot of perpendicular H = {H}")
print(f"Check via (A-p0)·n/||n|| with p0={p0}: {alt_dist}")

#  Sketch of the question
# Line: 3x + 4y + 6 = 0 -> y = (-3x - 6)/4
x_vals = np.linspace(-6, 4, 300)
y_vals = (-3*x_vals - 6) / 4

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.plot(x_vals, y_vals, label="line 3x + 4y = -6")
ax.scatter([A[0]], [A[1]], s=70, label="A (-1, 3)")
ax.scatter([H[0]], [H[1]], s=70, label=f"Foot H ({H[0]:.2f}, {H[1]:.2f})")
ax.plot([A[0], H[0]], [A[1], H[1]], linestyle="--", label="Perpendicular AH")

# normal direction arrow at H
n_unit = n / norm_n
ax.arrow(H[0], H[1], n_unit[0]*0.6, n_unit[1]*0.6, head_width=0.18, length_includes_head=True)

# simple annotations
ax.annotate("A", xy=(A[0], A[1]), xytext=(A[0]+0.2, A[1]+0.2))
ax.annotate("H", xy=(H[0], H[1]), xytext=(H[0]+0.2, H[1]+0.2))

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

ax.set_xlim(-6, 4)
ax.set_ylim(-4, 6)

plt.show()
