import torch
import numpy as np
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation
from pi3.utils.basic import load_images_as_tensor
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_demo.py <input_dir_or_video>")
        sys.exit(1)

    data_path = sys.argv[1]
    out_dir = Path(data_path) if Path(data_path).is_dir() else Path(data_path).parent
    out_dir = out_dir / "pi3_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    # If the given directory has no images directly, look for image files recursively
    p = Path(data_path)
    if p.is_dir():
        direct_imgs = sorted(p.glob('*.jpg')) + sorted(p.glob('*.jpeg')) + sorted(p.glob('*.png'))
        if not direct_imgs:
            all_imgs = sorted(p.rglob('*.jpg')) + sorted(p.rglob('*.jpeg')) + sorted(p.rglob('*.png'))
            if not all_imgs:
                print(f"Error: no images found in {data_path}")
                sys.exit(1)
            # Use parent dir containing most images
            from collections import Counter
            parent_counts = Counter(f.parent for f in all_imgs)
            data_path = str(max(parent_counts, key=parent_counts.get))
            print(f"No images in root dir, using: {data_path}")

    interval = 10 if data_path.endswith('.mp4') else 1
    print(f"Loading data from {data_path} (interval={interval})...")
    imgs = load_images_as_tensor(data_path, interval=interval).to(device)  # (N, 3, H, W)
    if imgs.shape[0] == 0:
        print(f"Error: no images loaded from {data_path}")
        sys.exit(1)
    print(f"Loaded {imgs.shape[0]} frames")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print("Running inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None])

    # Masks
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # Save poses (TUM format)
    poses_path = out_dir / "poses.txt"
    camera_poses = res['camera_poses'][0].cpu().numpy()  # (N, 4, 4)
    with open(poses_path, 'w') as f:
        f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
        for i, pose in enumerate(camera_poses):
            tx, ty, tz = pose[:3, 3]
            qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
            f.write(f"{i * interval} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    print(f"Poses saved to {poses_path}")

    # Save depth maps
    depth_dir = out_dir / "depth"
    depth_dir.mkdir(exist_ok=True)
    depth_maps = res['local_points'][0, :, :, :, 2].cpu().numpy()  # (N, H, W)
    for i, d in enumerate(depth_maps):
        np.save(depth_dir / f"{i * interval:06d}.npy", d)
    print(f"Depth maps saved to {depth_dir} ({len(depth_maps)} files)")

    print(f"\nDone! Point cloud shape: {res['points'][0][masks].shape}")
    print(f"Output directory: {out_dir}")
