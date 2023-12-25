
"""
Purpose of the Function
The primary goal of the SPM_IK function is to calculate the inverse kinematics (IK) of a robotic arm consisting of three segments connected by revolute joints. Specifically, this function deals with solving for joint angles theta1_upper, theta2_middle, and theta3_lower when given desired Cartesian poses for the endpoint. The desired_poses argument contains a list of N target locations in 3D space for the robot's end effector.

Shape of the Desired Poses Matrix
As mentioned earlier, the desired_poses matrix has dimensions (3, N). Each row corresponds to one spatial dimension (X, Y, and Z), while each column represents a unique configuration in the workspace. When working with robotic arms, usually, N denotes the number of discrete target points specified for the end effector. For instance, N = 4 means that the robot needs to reach four distinct positions in its environment.

Return Values
The function computes and returns three arrays (theta1_upper, theta2_middle, and theta3_lower) representing the required joint angles for each of the N target poses. Since the task requires finding three separate joint angles, the resulting arrays will also have shapes (N,). By convention, the first element refers to the initial position, the next elements correspond to subsequent targets, and so forth.

Structure of the Code
Here's a brief overview of the main steps inside the SPM_IK function:

First, the function sets up several constant parameters describing geometric properties of the robotic arm, including segment lengths, reference frames, etc.
Next, it iterates over columns in the desired_poses matrix, computing joint angles for each individual target location. During this process, the function performs the following subtasks:
Determines the plane containing the target point and the base frame origin.
Computes the rotation matrix relating both coordinate systems.
Extracts angular displacements from the rotation matrix (roll, pitch, and yaw).
Solves algebraic equations to determine possible candidate solutions for theta1_upper, theta2_middle, and theta3_lower.
Selects the most suitable solution among candidates based on predefined criteria (proximity to default joint angles).
Stores the selected joint angle combination in the corresponding arrays (theta1_upper, theta2_middle, and theta3_lower).
Finally, upon completing the loop, the function returns the generated arrays holding computed joint angles for each target location.
"""

import numpy as np
import math

# locus_of_desired_pose: Input desired poses matrix, with dimensions (3, N) where each column holds the XYZ coordinates of the desired end effector pose.
def SPM_IK(locus_of_desired_pose):
    """IK Penguin Wrist without plots. Returns theta1_upper, theta2_middle, and theta3_lower."""
    
    # Kinematics constants
    P4 = np.array([0, 0, 36.57])
    
    def calc_circle_params():
        global l1a, l2a, psi, link_center_xyz, Rl2
        # l1a: Length of linkage L1 
        l1a = 19.33
        # l2a: Length of linkage L2 
        l2a = 22.01 
        # psi: Angle between linkages L1 and L2. Typically, this value is 120 degrees in PUMA-like arms. 
        # Ensure conversion to radians for use in the provided code.
        psi = math.radians(120)
        # link_center_xyz: Center point of the circular motion of linkages L1 and L2. 
        # Measure the XY coordinates relative to the chosen base frame origin and then construct the homogeneous transformation matrix.
        link_center_xyz = np.array([0, 0, 19.06])
        # Rl2: Radius of curvature associated with the circular movement of linkages L1 and L2. 
        # Obtain this value by measuring the distance between the center point (link_center_xyz) and the point where the two linkages connect.
        Rl2 = 35.03

    calc_circle_params()

    num_poses = desired_poses.shape[1]

    # vectorCircle_P4: Vector difference between the desired poses and the known point P4. 
    # Used to compute the normal vector of the plane containing the target point and the base frame origin.
    vectorCircle_P4 = np.zeros((3, num_poses))
    theta1_upper = np.zeros(num_poses)
    theta2_middle = np.zeros(num_poses)
    theta3_lower = np.zeros(num_poses)

    for pose_number in range(num_poses):
        vec_diff = locus_of_desired_pose[:, pose_number] - P4
        norm_vec_diff = np.linalg.norm(vec_diff)
        vectorCircle_P4[:, pose_number] = vec_diff / norm_vec_diff

        imin = 2  # bias towards the y-axis
        y_vec = np.zeros(3)
        y_vec[imin] = 1
        y_vec = normalize_vector(y_vec - vectorCircle_P4[:, pose_number] * np.dot(y_vec, vectorCircle_P4[:, pose_number]))
        rotation_matrix = get_rotation_matrix(y_vec)

        roll, pitch, yaw = decompose_rotation_matrix(rotation_matrix)

        A = link_center_xyz[2] - P4[2]
        B = Rl2 * np.sin(pitch)
        C = Rl2 * np.cos(pitch) * np.cos(roll)
        D = l1a - l2a * np.cos(psi)
        E = Rl2 * (np.cos(yaw) * np.sin(roll) - np.cos(roll) * np.sin(pitch) * np.sin(yaw))
        F = Rl2 * np.cos(pitch) * np.sin(yaw)
        H = Rl2 * (np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.cos(yaw) * np.sin(pitch))
        I = Rl2 * np.cos(pitch) * np.cos(yaw)
        L = Rl2 * (np.sin(pitch) / 2 - (3 ** (1 / 2) * np.cos(pitch) * np.sin(roll)) / 2)
        M = Rl2 * ((np.cos(pitch) * np.sin(yaw)) / 2 + (3 ** (1 / 2) * (np.cos(roll) * np.cos(yaw) + np.sin(pitch) * np.sin(roll) * np.sin(yaw))) / 2)
        N = Rl2 * ((np.cos(pitch) * np.cos(yaw)) / 2 - (3 ** (1 / 2) * (np.cos(roll) * np.sin(yaw) - np.cos(yaw) * np.sin(pitch) * np.sin(roll))) / 2)
        O = Rl2 * (np.sin(pitch) / 2 + (3 ** (1 / 2) * np.cos(pitch) * np.sin(roll)) / 2)
        P = Rl2 * ((np.cos(pitch) * np.sin(yaw)) / 2 - (3 ** (1 / 2) * (np.cos(roll) * np.cos(yaw) + np.sin(pitch) * np.sin(roll) * np.sin(yaw))) / 2)
        R = Rl2 * ((np.cos(pitch) * np.cos(yaw)) / 2 + (3 ** (1 / 2) * (np.cos(roll) * np.sin(yaw) - np.cos(yaw) * np.sin(pitch) * np.sin(roll))) / 2)

        # Calculate theta1_upper
        temp = -A**2 + B**2 + C**2
        if temp >= 0:
            sqrt_term = math.sqrt(temp)
            yplus = L + sqrt_term
            yminus = L - sqrt_term
            x = A + C
            angle_candidate_up = []
            
            if yplus != 0 and x != 0:
                angle_candidate_up.append(-2 * math.atan2(yplus, x))
            if yminus != 0 and x != 0:
                angle_candidate_up.append(-2 * math.atan2(yminus, x))
                
            if len(angle_candidate_up) > 0:
                for s3 in angle_candidate_up:
                    y1 = -E * np.cos(s3) + F * np.sin(s3)
                    x1 = H * np.cos(s3) + I * np.sin(s3)
                    if x1 != 0 or y1 != 0:
                        theta1_upper[pose_number] = math.degrees(math.atan2(y1, x1))

        # Calculate theta2_middle
        temp = -A**2 + C**2 + L**2
        if temp >= 0:
            sqrt_term = math.sqrt(temp)
            yplus = L + sqrt_term
            yminus = L - sqrt_term
            x = A + C
            angle_candidate_mid = []
            
            if yplus != 0 and x != 0:
                angle_candidate_mid.append(-2 * math.atan2(yplus, x))
            if yminus != 0 and x != 0:
                angle_candidate_mid.append(-2 * math.atan2(yminus, x))
                
            if len(angle_candidate_mid) > 0:
                for s3 in angle_candidate_mid:
                    y1 = -E * np.cos(s3) + M * np.sin(s3)
                    x1 = H * np.cos(s3) + N * np.sin(s3)
                    if x1 != 0 or y1 != 0:
                        theta2_middle[pose_number] = math.degrees(math.atan2(y1, x1))

        # Calculate theta3_lower
        temp = -A**2 + C**2 + O**2
        if temp >= 0:
            sqrt_term = math.sqrt(temp)
            yplus = O + sqrt_term
            yminus = O - sqrt_term
            x = A + C
            angle_candidate_down = []
            
            if yplus != 0 and x != 0:
                angle_candidate_down.append(2 * math.atan2(yplus, x))
            if yminus != 0 and x != 0:
                angle_candidate_down.append(2 * math.atan2(yminus, x))
                
            if len(angle_candidate_down) > 0:
                for s3 in angle_candidate_down:
                    y1 = -E * np.cos(s3) - P * np.sin(s3)
                    x1 = H * np.cos(s3) - R * np.sin(s3)
                    if x1 != 0 or y1 != 0:
                        theta3_lower[pose_number] = math.degrees(math.atan2(y1, x1))

    return theta1_upper, theta2_middle, theta3_lower

def normalize_vector(v):
    norm = math.sqrt(sum(x**2 for x in v))
    return tuple(x/norm for x in v)

def get_rotation_matrix(n):
    n = np.array(n)
    n_hat = np.array([0, 0, 1]) if np.isclose(n, [0, 0, 0]).all() else n
    a = np.array([n_hat[1], -n_hat[0], 0])
    b = np.cross(n_hat, a)
    a = np.cross(n_hat, b)
    a /= math.sqrt(sum(x**2 for x in a))
    b /= math.sqrt(sum(x**2 for x in b))
    c = np.cross(a, b)
    d = np.eye(3)
    d[:, 0] = a
    d[:, 1] = b
    d[:, 2] = c
    return d

def decompose_rotation_matrix(rm):
    sy = math.sqrt(rm[0][0]**2 + rm[1][0]**2)
    singular = sy < 1e-6

    x = rm[0][0]/sy if not singular else 1.0
    y = rm[1][0]/sy if not singular else 0.0
    z = rm[2][0]

    ry = -math.asin(x) if singular else math.atan2(-rm[0][0], sy)
    rx = 0.0  # Initialize rx
    rz = math.atan2(rx, ry)
    rx = math.atan2(ry, rz)

    return rx, ry, rz

# these are just random values for testing

v1x = 0.23463833986802626
v1y = 0.9717645982306949
v1z = 0.024867953062384623
v2x = 0.6968541171282346
v2y = -0.6767067021251991
v2z = -0.2376181363874942
v3x = -0.9314924537032421
v3y = -0.29505789722967923
v3z = 0.21275019618396085

desired_poses = np.array([
    [v2x],
    [v2y],
    [v2z]
])

theta1_upper, theta2_middle, theta3_lower = SPM_IK(desired_poses)

# Print the results
print("theta1_upper:", theta1_upper)
print("theta2_middle:", theta2_middle)
print("theta3_lower:", theta3_lower)