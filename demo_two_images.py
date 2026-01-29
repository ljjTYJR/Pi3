import torch
import torch.nn.functional as F
import argparse
import math
import numpy as np
from PIL import Image
from torchvision import transforms
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import time


def load_intrinsics(path):
    with open(path, 'r') as f:
        fx, fy, cx, cy = map(float, f.read().strip().split())
    return fx, fy, cx, cy


def load_single_image_as_tensor(path, pixel_limit=255000):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    scale = math.sqrt(pixel_limit / (w * h)) if w * h > 0 else 1
    k, m = round(w * scale / 14), round(h * scale / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        k, m = (k - 1, m) if k / m > w / h else (k, m - 1)
    tw, th = max(1, k) * 14, max(1, m) * 14
    print(f"Loading {path}: ({w}, {h}) -> ({tw}, {th})")
    return transforms.ToTensor()(img.resize((tw, th), Image.Resampling.LANCZOS)).unsqueeze(0)


def depth_to_pointcloud(depth, colors, fx, fy, cx, cy):
    h, w = depth.shape
    y, x = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing='ij')
    z = depth
    x3d = (x - cx) * z / fx
    y3d = (y - cy) * z / fy
    points = torch.stack([x3d, y3d, z], dim=-1).reshape(-1, 3).cpu().numpy()
    colors_flat = colors.reshape(-1, 3)
    mask = points[:, 2] > 0
    return points[mask], colors_flat[mask]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", type=str, required=True)
    parser.add_argument("--image2", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--intrinsics", type=str, default="/home/shuo/ssd2/vision_task/neural_rendering/mul-view-recon/calib/kinect_crop.txt")
    parser.add_argument("--scale_factor", type=int, default=8)
    args = parser.parse_args()

    t_start = time.time()

    scale_factor = args.scale_factor
    fx, fy, cx, cy = load_intrinsics(args.intrinsics)
    fx, fy, cx, cy = fx / scale_factor, fy / scale_factor, cx / scale_factor, cy / scale_factor
    print(f"Scaled intrinsics (1/{scale_factor}): fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    device = torch.device(args.device)

    t0 = time.time()
    if args.ckpt:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    t1 = time.time()
    print(f"[Time] Model loading: {t1-t0:.3f}s")

    t0 = time.time()
    imgs = torch.cat([load_single_image_as_tensor(args.image1),
                      load_single_image_as_tensor(args.image2)], dim=0).to(device)
    t1 = time.time()
    print(f"[Time] Image loading: {t1-t0:.3f}s")

    t0 = time.time()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        res = model(imgs[None])
    t1 = time.time()
    print(f"[Time] Model inference: {t1-t0:.3f}s")
    print(f"Result keys: {list(res.keys())}")

    t0 = time.time()
    depth0 = F.interpolate(res['local_points'][0, 0:1, :, :, 2:3].permute(0, 3, 1, 2),
                           size=(384, 512), mode='area')
    depth0 = F.interpolate(depth0, scale_factor=1/scale_factor, mode='area').squeeze()

    depth1 = F.interpolate(res['local_points'][0, 1:2, :, :, 2:3].permute(0, 3, 1, 2),
                           size=(384, 512), mode='area')
    depth1 = F.interpolate(depth1, scale_factor=1/scale_factor, mode='area').squeeze()

    colors0 = F.interpolate(imgs[0:1], size=(384, 512), mode='area')
    colors0 = F.interpolate(colors0, scale_factor=1/scale_factor, mode='area')
    colors0 = colors0[0].permute(1, 2, 0).cpu().numpy()

    colors1 = F.interpolate(imgs[1:2], size=(384, 512), mode='area')
    colors1 = F.interpolate(colors1, scale_factor=1/scale_factor, mode='area')
    colors1 = colors1[0].permute(1, 2, 0).cpu().numpy()
    t1 = time.time()
    print(f"[Time] Depth extraction and downsampling: {t1-t0:.3f}s")

    t0 = time.time()
    # Convert depth to local point clouds
    pts0_local, cols0 = depth_to_pointcloud(depth0, colors0, fx, fy, cx, cy)
    pts1_local, cols1 = depth_to_pointcloud(depth1, colors1, fx, fy, cx, cy)
    t1 = time.time()
    print(f"[Time] Depth to point cloud conversion: {t1-t0:.3f}s")

    t0 = time.time()
    # Get camera poses and transform to global frame
    poses = res['camera_poses'][0].cpu().numpy()  # (2, 4, 4)

    # Transform local points to global frame
    pts0_homo = np.concatenate([pts0_local, np.ones((pts0_local.shape[0], 1))], axis=1)  # (N, 4)
    pts0_global = (poses[0] @ pts0_homo.T).T[:, :3]  # (N, 3)

    pts1_homo = np.concatenate([pts1_local, np.ones((pts1_local.shape[0], 1))], axis=1)  # (N, 4)
    pts1_global = (poses[1] @ pts1_homo.T).T[:, :3]  # (N, 3)
    t1 = time.time()
    print(f"[Time] Transform to global frame: {t1-t0:.3f}s")

    t0 = time.time()
    # Create Open3D point clouds
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts0_global)
    pcd0.colors = o3d.utility.Vector3dVector(cols0)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1_global)
    pcd1.colors = o3d.utility.Vector3dVector(cols1)
    t1 = time.time()
    print(f"[Time] Create Open3D point clouds: {t1-t0:.3f}s")

    t0 = time.time()
    # Find nearest neighbor correspondences: pcd0 -> pcd1
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
    nn_pcd0_to_pcd1 = np.full(len(pts0_global), -1, dtype=np.int32)
    dist_pcd0_to_pcd1 = np.full(len(pts0_global), np.inf)

    for i, point in enumerate(pts0_global):
        k, idx, dist = pcd1_tree.search_knn_vector_3d(point, 1)
        if k > 0:
            nn_pcd0_to_pcd1[i] = idx[0]
            dist_pcd0_to_pcd1[i] = dist[0]

    # Find nearest neighbor correspondences: pcd1 -> pcd0
    pcd0_tree = o3d.geometry.KDTreeFlann(pcd0)
    nn_pcd1_to_pcd0 = np.full(len(pts1_global), -1, dtype=np.int32)
    dist_pcd1_to_pcd0 = np.full(len(pts1_global), np.inf)

    for j, point in enumerate(pts1_global):
        k, idx, dist = pcd0_tree.search_knn_vector_3d(point, 1)
        if k > 0:
            nn_pcd1_to_pcd0[j] = idx[0]
            dist_pcd1_to_pcd0[j] = dist[0]

    # Find mutual correspondences
    mutual_corr = []
    mutual_dist = []

    for i in range(len(pts0_global)):
        j = nn_pcd0_to_pcd1[i]
        if j >= 0 and nn_pcd1_to_pcd0[j] == i:
            mutual_corr.append((i, j))
            mutual_dist.append(dist_pcd0_to_pcd1[i])

    mutual_corr = np.array(mutual_corr)
    mutual_dist = np.array(mutual_dist)
    t1 = time.time()
    print(f"[Time] Nearest neighbor correspondence search: {t1-t0:.3f}s")

    print(f"Found {len(mutual_corr)} mutual correspondences")
    if len(mutual_dist) > 0:
        print(f"Distance stats - min: {mutual_dist.min():.4f}, max: {mutual_dist.max():.4f}, mean: {mutual_dist.mean():.4f}")

        # Prune correspondences with distance > mean
        mean_dist = mutual_dist.mean()
        valid_mask = mutual_dist <= mean_dist
        mutual_corr = mutual_corr[valid_mask]
        mutual_dist = mutual_dist[valid_mask]
        print(f"After pruning by mean distance: {len(mutual_corr)} correspondences retained")

    t0 = time.time()
    # ===== 2D Visualization: Draw correspondences on images =====
    if len(mutual_corr) > 0:
        # Convert depth back to pixel coordinates at downscaled resolution
        h_ds, w_ds = depth0.shape

        # Create pixel coordinate grids
        y_grid, x_grid = np.meshgrid(np.arange(h_ds), np.arange(w_ds), indexing='ij')

        # Map global point indices back to pixel coordinates
        valid_mask0 = depth0.cpu().numpy() > 0
        pixel_coords0_ds = np.column_stack([x_grid[valid_mask0], y_grid[valid_mask0]])

        valid_mask1 = depth1.cpu().numpy() > 0
        pixel_coords1_ds = np.column_stack([x_grid[valid_mask1], y_grid[valid_mask1]])

        # Rescale coordinates to original resolution (512x384)
        h_orig, w_orig = 384, 512
        pixel_coords0 = pixel_coords0_ds * scale_factor
        pixel_coords1 = pixel_coords1_ds * scale_factor

        # Load and resize images to original resolution for visualization
        img0_vis = cv2.imread(args.image1)
        img0_vis = cv2.resize(img0_vis, (w_orig, h_orig))
        img1_vis = cv2.imread(args.image2)
        img1_vis = cv2.resize(img1_vis, (w_orig, h_orig))

        # Draw correspondences (sample to avoid clutter)
        n_samples = min(100, len(mutual_corr))
        sample_idx = np.random.choice(len(mutual_corr), n_samples, replace=False)

        # Combine images side by side
        vis_img = np.hstack([img0_vis, img1_vis])

        for idx in sample_idx:
            i, j = mutual_corr[idx]
            pt0 = pixel_coords0[i].astype(int)
            pt1 = pixel_coords1[j].astype(int)

            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(vis_img, tuple(pt0), 3, color, -1)
            cv2.circle(vis_img, (pt1[0] + w_orig, pt1[1]), 3, color, -1)
            cv2.line(vis_img, tuple(pt0), (pt1[0] + w_orig, pt1[1]), color, 1)

        plt.figure(figsize=(15, 7))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"2D Correspondences at 512x384 ({n_samples} sampled from {len(mutual_corr)})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("correspondences_2d.png", dpi=200, bbox_inches='tight')
        print("Saved 2D correspondences to correspondences_2d.png")
        plt.close()
    t1 = time.time()
    print(f"[Time] 2D visualization: {t1-t0:.3f}s")

    t0 = time.time()
    # ===== 3D Visualization: Draw correspondence lines =====
    print(f"Visualizing {len(pts0_global)} + {len(pts1_global)} = {len(pts0_global) + len(pts1_global)} points")

    # Create line set for correspondences
    if len(mutual_corr) > 0:
        # Sample correspondences for 3D visualization
        n_samples_3d = min(100, len(mutual_corr))
        sample_idx_3d = np.random.choice(len(mutual_corr), n_samples_3d, replace=False)

        lines = []
        for idx in sample_idx_3d:
            i, j = mutual_corr[idx]
            lines.append([i, len(pts0_global) + j])

        # Combine points
        all_points = np.vstack([pts0_global, pts1_global])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(all_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(lines), 1)))

        o3d.visualization.draw_geometries([pcd0, pcd1, line_set],
                                         window_name="3D Correspondences", width=1024, height=768)
    else:
        o3d.visualization.draw_geometries([pcd0, pcd1],
                                         window_name="Global Point Cloud", width=1024, height=768)
    t1 = time.time()
    print(f"[Time] 3D visualization: {t1-t0:.3f}s")

    t_end = time.time()
    print(f"\n[Time] Total time: {t_end-t_start:.3f}s")