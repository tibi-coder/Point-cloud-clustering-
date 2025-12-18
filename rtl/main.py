from pymtl3 import *
from pymtl3.passes.backends.verilog import VerilogTranslationPass
from rtl.LidarAccelerator import LidarCore
from data_loader import get_lidar_data
import numpy as np
import open3d as o3d
import time
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


# --- HELPER: GROUND GRID ---
def create_ground_grid(size=60, step=5, z_level=-2.0):
    points = []
    lines = []
    idx = 0
    # Lines X
    for y in range(-size, size + step, step):
        points.append([-size, y, z_level]);
        points.append([size, y, z_level])
        lines.append([idx, idx + 1]);
        idx += 2
    # Lines Y
    for x in range(-size, size + step, step):
        points.append([x, -size, z_level]);
        points.append([x, size, z_level])
        lines.append([idx, idx + 1]);
        idx += 2

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0.5, 0.5, 0.5])
    return line_set


def main():
    # --- CONFIG ---
    SKIP_FACTOR_SIM = 5
    SCALE_VISUAL = 0.01

    # 1. LOAD DATA
    print("- Loading raw LiDAR data...")
    data_int, floor_th, data_raw_float = get_lidar_data("000000.bin", use_synthetic=True)

    # Vizualizare Raw
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(data_raw_float)
    pcd_raw.paint_uniform_color([0.6, 0.6, 0.6])
    print(f"Loaded raw points: {len(data_raw_float)}")

    # 2. HARDWARE SETUP
    print("\n- Setting up Hardware Model...")
    data_to_process = data_int[::SKIP_FACTOR_SIM]

    # DUT
    dut = LidarCore(nbits=16)

    # verilog generation
    print("- Translating to Verilog...")
    dut.set_metadata( VerilogTranslationPass.enable, True )
    dut.apply( VerilogTranslationPass() )
    print(" -> Generated 'LidarCore.v' successfully!")


    # Simulation
    dut.apply(DefaultPassGroup())
    dut.sim_reset()

    # hardware
    dut.floor_thresh @= floor_th
    # Cluster centers (hardcoded)
    dut.c1_x @= 1500;
    dut.c1_y @= 2000;
    dut.c1_z @= 250
    dut.c2_x @= 1000;
    dut.c2_y @= 2300;
    dut.c2_z @= 250

    # 3. SIMULATION LOOP
    print("- Running Simulation...")
    valid_voxel_centers = []
    labels = []

    start = time.time()
    for pt in data_to_process:
        if np.max(pt) > 65000 or np.min(pt) < 0: continue

        # in pin
        dut.in_x @= int(pt[0])
        dut.in_y @= int(pt[1])
        dut.in_z @= int(pt[2])

        dut.sim_eval_combinational()

        # out pin
        if dut.out_valid:
            vx = int(dut.voxelizer.out_x)
            vy = int(dut.voxelizer.out_y)
            vz = int(dut.voxelizer.out_z)
            valid_voxel_centers.append([vx, vy, vz])
            labels.append(int(dut.out_cluster))

        # clock
        dut.sim_tick()

    duration = time.time() - start
    print(f"Simulation finished in {duration:.2f}s. Output voxels: {len(valid_voxel_centers)}")

    # 4. PREPARE OUTPUT GEOMETRY
    voxel_mesh = None
    if len(valid_voxel_centers) > 0:
        np_voxels = np.array(valid_voxel_centers) * SCALE_VISUAL
        pcd_processed = o3d.geometry.PointCloud()
        pcd_processed.points = o3d.utility.Vector3dVector(np_voxels)

        # Colors
        np_colors = np.zeros((len(valid_voxel_centers), 3))
        np_labels = np.array(labels)
        np_colors[np_labels == 0] = [1, 0, 0]  # Red
        np_colors[np_labels == 1] = [0, 0, 1]  # Blue
        pcd_processed.colors = o3d.utility.Vector3dVector(np_colors)

        # Voxels
        voxel_mesh = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_processed, voxel_size=0.15)

    # Grid
    grid_lines = create_ground_grid(size=50, step=5, z_level=-1.75)

    # 5. OPEN GUI
    print("\n- Opening Visualization...")
    app = gui.Application.instance
    app.initialize()
    window = app.create_window("Lidar FPGA: Input (Left) vs Output (Right)", 1280, 720)

    # Left Scene
    widget_left = gui.SceneWidget()
    widget_left.scene = rendering.Open3DScene(window.renderer)
    widget_left.scene.set_background([0.1, 0.1, 0.1, 1])
    mat_raw = rendering.MaterialRecord();
    mat_raw.shader = "defaultUnlit";
    mat_raw.point_size = 3.0
    mat_lines = rendering.MaterialRecord();
    mat_lines.shader = "unlitLine";
    mat_lines.line_width = 2.0

    widget_left.scene.add_geometry("Raw", pcd_raw, mat_raw)
    widget_left.scene.add_geometry("GridL", grid_lines, mat_lines)
    widget_left.setup_camera(60, pcd_raw.get_axis_aligned_bounding_box(), [0, 0, 0])

    # Right Scene
    widget_right = gui.SceneWidget()
    widget_right.scene = rendering.Open3DScene(window.renderer)
    widget_right.scene.set_background([0.15, 0.15, 0.15, 1])
    mat_voxel = rendering.MaterialRecord();
    mat_voxel.shader = "defaultLit"

    if voxel_mesh: widget_right.scene.add_geometry("Voxels", voxel_mesh, mat_voxel)
    widget_right.scene.add_geometry("GridR", grid_lines, mat_lines)

    target_bbox = voxel_mesh.get_axis_aligned_bounding_box() if voxel_mesh else pcd_raw.get_axis_aligned_bounding_box()
    widget_right.setup_camera(60, target_bbox, [0, 0, 0])

    # Split Screen Layout
    def on_layout(ctx):
        r = window.content_rect
        widget_left.frame = gui.Rect(0, 0, r.width / 2, r.height)
        widget_right.frame = gui.Rect(r.width / 2, 0, r.width / 2, r.height)

    window.set_on_layout(on_layout)
    window.add_child(widget_left)
    window.add_child(widget_right)
    app.run()


if __name__ == "__main__":
    main()