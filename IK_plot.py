# Output: q1high, q1medium, q1low in degrees each is 1xN vector!!!
# Input: 3xN matrix where each column N is a 3D point in the space of the desired path

#120-Angled: High Low Medium create a circle with center in Z=19.06 and radius R in the XY plane w.r.t. RF0fixed

#TOP (Middle and Bottom form the same circle)


import numpy as np
import matplotlib.pyplot as plt

def rotx(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def roty(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rd2tom(R, t):
    return np.vstack([np.hstack([R, t.reshape(-1, 1)]), [0, 0, 0, 1]])

def plot_ref_frame(R, t, colors, scale, alpha, length):
    axes = plt.gca()
    for i in range(3):
        end = t + scale * R[:, i] * length
        axes.plot3D([t[0], end[0]], [t[1], end[1]], [t[2], end[2]], color=colors[i], alpha=alpha)

def plot_Plane(t, n, x_range, y_range, fig_handler):
    d = -np.dot(n, t)
    xx, yy = np.meshgrid(x_range, y_range)
    z = (-n[0] * xx - n[1] * yy - d) * 1.0 / n[2]
    fig_handler.plot_surface(xx, yy, z, color='cyan', alpha=0.3)


# Fixed Point in Space
P4 = np.array([0, 0, 36.57])

# Desired pose (in rad)
roll = np.pi / 6
pitch = -np.pi / 5
yaw = 0

# R=Rz*Ry*Rx
R0fixedtoP4 = np.dot(rotz(np.rad2deg(yaw)), np.dot(roty(np.rad2deg(pitch)), rotx(np.rad2deg(roll))))

# Homogeneous transformation
T0fixedtoP4 = rd2tom(R0fixedtoP4, P4)

# Triangle Vertices
RadiusL90 = 15

# TOP
p4top_rfp4 = np.array([0, -RadiusL90, 0, 1])
p4top_rf0 = np.dot(T0fixedtoP4, p4top_rfp4)

# MIDDLE
p4middle_rfp4 = np.array([-RadiusL90 * np.cos(np.deg2rad(30)), RadiusL90 * np.sin(np.deg2rad(30)), 0, 1])
p4p4middle_rf0 = np.dot(T0fixedtoP4, p4middle_rfp4)

# BOTTOM
p4bottom_rfp4 = np.array([RadiusL90 * np.cos(np.deg2rad(30)), RadiusL90 * np.sin(np.deg2rad(30)), 0, 1])
p4bottom_rf0 = np.dot(T0fixedtoP4, p4bottom_rfp4)

# Plane normal given by the desired pose
plane_n = R0fixedtoP4[:3, 2]

# Inertial frame
RF0fixed = np.eye(3)

# Equations Parameters definition
l1a = 19.33
l2a = 22.01
psi = np.deg2rad(120)
Rxy = l1a - l2a * np.cos(psi)
centerXY = np.array([0, 0, 19.06])

# Vertices of the equilateral triangle
Rl2 = 35.03
Rplane_middle = np.dot(R0fixedtoP4, rotz(-120))
Rplane_bottom = np.dot(R0fixedtoP4, rotz(120))

# Plot parameters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot functions (plot_ref_frame and plot_Plane functions need to be implemented)
# ...

# Plot P4 and the triangle vertices
ax.scatter(*P4, color='k', marker='o')
ax.scatter(*p4top_rf0[:3], color='b', marker='*')
ax.scatter(*p4p4middle_rf0[:3], color='r', marker='o')
ax.scatter(*p4bottom_rf0[:3], color='g', marker='+')

# Draw lines connecting the triangle vertices
t_vertices = np.array([p4top_rf0[:3], p4p4middle_rf0[:3], p4bottom_rf0[:3], p4top_rf0[:3]])
ax.plot(t_vertices[:, 0], t_vertices[:, 1], t_vertices[:, 2])

# Display the plot
plt.show()