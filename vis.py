import open3d as o3d
import argparse

def visualize_ply(file_path):
    """
    Visualizes a PLY file using Open3D.

    Args:
        file_path (str): Path to the PLY file.
    """
    # Load the PLY file
    pcd = o3d.io.read_point_cloud(file_path)

    # Check if the point cloud is loaded successfully
    if pcd.is_empty():
        print(f"Failed to load point cloud from {file_path}. Please check the file path.")
        return

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="PLY Point Cloud Visualization")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a PLY file.")
    parser.add_argument("--file_path", type=str, required=True,
                        help="Path to the PLY file to visualize.")

    args = parser.parse_args()

    # Visualize the PLY file
    visualize_ply(args.file_path)