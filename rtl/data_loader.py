import numpy as np
import os


def get_lidar_data(filename="kitti_sample.bin", use_synthetic=True):
    """
    Returns an (N, 3) array of scaled integer LiDAR points.
    Also returns a floor threshold and the original float points.
    """
    points = []

    # If synthetic data is enabled or file does not exist
    if use_synthetic or not os.path.exists(filename):
        print("- Generating synthetic")

        # Ground plane
        # Around 40x40 meters with small noise on Z axis
        g_x = np.random.uniform(0, 40, 5000)
        g_y = np.random.uniform(-20, 20, 5000)
        g_z = np.random.normal(-1.73, 0.05, 5000)  # LiDAR height ~1.73m

        # Car model
        # Centered around X=15m, Y=-2m
        c1_x = np.random.uniform(13, 17, 500)
        c1_y = np.random.uniform(-3, -1, 500)
        c1_z = np.random.uniform(-1.7, 0, 500)  # From ground upwards

        # Pedestrian or pole
        # Small vertical cluster at X=10m, Y=3m
        p1_x = np.random.normal(10, 0.2, 200)
        p1_y = np.random.normal(3, 0.2, 200)
        p1_z = np.random.uniform(-1.7, 0, 200)

        # Combine all points into one array
        points = np.vstack([
            np.column_stack([g_x, g_y, g_z]),
            np.column_stack([c1_x, c1_y, c1_z]),
            np.column_stack([p1_x, p1_y, p1_z])
        ]).astype(np.float32)


    # Load binary file if available
    else:
        print(f"- Loading LiDAR file: {filename} ---")

        # KITTI format: [x, y, z, intensity]
        scan = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))
        points = scan[:, :3]  # Keep only XYZ

        # Filter region of interest (front area only)
        mask = (
            (points[:, 0] > 0) &
            (points[:, 0] < 40) &
            (abs(points[:, 1]) < 20)
        )
        points = points[mask]


    # Prepare data for hardware processing
    # Scaling: convert meters to centimeters
    SCALE = 100.0
    points_int = (points * SCALE).astype(int)

    # Offset: move all coordinates to positive range
    # This simplifies unsigned hardware arithmetic
    OFFSET_Z = 200
    OFFSET_X = 0
    OFFSET_Y = 2000  # Center Y axis in positive range

    points_int[:, 0] += OFFSET_X
    points_int[:, 1] += OFFSET_Y
    points_int[:, 2] += OFFSET_Z


    # Floor threshold for hardware
    # Threshold is set a bit higher to remove ground points
    floor_threshold = 35

    return points_int, floor_threshold, points
