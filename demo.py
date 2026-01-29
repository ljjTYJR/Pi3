import torch
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from pi3.utils.basic import load_images_as_tensor
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")

    parser.add_argument("--data_path", type=str, nargs='+', default=['examples/skating.mp4'],
                        help="Path(s) to the input image directory or video file(s). Multiple paths can be provided.")
    parser.add_argument("--save_poses", type=str, default=None,
                        help="Path to save camera poses in TUM format (.txt). Default: None (not saved)")
    parser.add_argument("--save_depth", action='store_true',
                        help="Save depth maps as .npy files in the same directory as poses. Default: False")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")

    args = parser.parse_args()

    # Ensure data_path is a list
    if isinstance(args.data_path, str):
        args.data_path = [args.data_path]

    # from pi3.utils.debug import setup_debug
    # setup_debug()

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)

        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input data - load all data paths and concatenate
    print(f"\n{'='*60}")
    print(f"Loading {len(args.data_path)} data path(s)...")
    print(f"{'='*60}")

    all_imgs = []
    frame_counts = []  # Track number of frames per data path
    frame_intervals = []  # Track interval used for each data path
    camera_names = []  # Track camera directory names
    for idx, data_path in enumerate(args.data_path):
        # Determine interval for this data path
        if args.interval < 0:
            interval = 10 if data_path.endswith('.mp4') else 1
        else:
            interval = args.interval

        # Extract camera name from the path
        path_obj = Path(data_path)
        if path_obj.is_dir():
            # If it's a directory, use the directory name (e.g., "cam03", "cam05")
            cam_name = path_obj.name
        else:
            # If it's a video file, use the stem (filename without extension)
            cam_name = path_obj.stem

        print(f"[{idx+1}/{len(args.data_path)}] Loading: {data_path} (camera={cam_name}, interval={interval})")
        imgs = load_images_as_tensor(data_path, interval=interval) # (N, 3, H, W)
        all_imgs.append(imgs)
        frame_counts.append(imgs.shape[0])
        frame_intervals.append(interval)
        camera_names.append(cam_name)
        print(f"  Loaded {imgs.shape[0]} frames from {data_path}")

    # Concatenate all images along the frame dimension
    imgs = torch.cat(all_imgs, dim=0).to(device) # (Total_N, 3, H, W)
    print(f"\nTotal frames: {imgs.shape[0]}")

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None]) # Add batch dimension

    # 4. process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Save camera poses if requested
    if args.save_poses is not None:
        camera_poses = res['camera_poses'][0].cpu().numpy()  # Shape: (N, 4, 4)

        # Determine output directory
        save_dir = Path(args.save_poses)
        if not save_dir.suffix:  # No file extension means it's a directory
            save_dir.mkdir(parents=True, exist_ok=True)
        else:  # Has extension, use parent directory
            save_dir = save_dir.parent
            save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving camera poses...")

        # Split poses by data path and save separately
        frame_start = 0
        for data_idx, num_frames in enumerate(frame_counts):
            frame_end = frame_start + num_frames
            poses_subset = camera_poses[frame_start:frame_end]  # Poses for this data path
            interval = frame_intervals[data_idx]

            # Generate output filename using actual camera name: cam03_poses.txt, cam05_poses.txt, etc.
            cam_name = camera_names[data_idx]
            cam_filename = f"{cam_name}_poses.txt"
            save_path = save_dir / cam_filename

            # Convert 4x4 matrices to TUM format (timestamp tx ty tz qx qy qz qw)
            with open(save_path, 'w') as f:
                f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
                f.write("# Camera-to-world transformation\n")
                f.write(f"# Data source: {args.data_path[data_idx]}\n")
                f.write(f"# Sampling interval: {interval}\n")
                for local_idx in range(poses_subset.shape[0]):
                    pose = poses_subset[local_idx]  # (4, 4)

                    # Extract translation (camera-to-world)
                    tx, ty, tz = pose[:3, 3]

                    # Extract rotation matrix and convert to quaternion
                    rotation_matrix = pose[:3, :3]
                    rotation = Rotation.from_matrix(rotation_matrix)
                    qx, qy, qz, qw = rotation.as_quat()  # scipy returns [x, y, z, w]

                    # Write in TUM format: timestamp (original frame index) tx ty tz qx qy qz qw
                    original_frame_idx = local_idx * interval
                    f.write(f"{original_frame_idx} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

            print(f"  Saved {save_path.name}: {num_frames} frames from {args.data_path[data_idx]}")
            frame_start = frame_end

        print(f"  Total pose files saved: {len(frame_counts)}")
        print(f"  Output directory: {save_dir}")
        print(f"  Format: TUM (timestamp tx ty tz qx qy qz qw)")
        print(f"  Representation: Camera-to-world transformation")

    # 6. Save depth maps if requested
    if args.save_depth:
        if args.save_poses is None:
            print("\nWarning: --save_depth requires --save_poses to be specified. Skipping depth saving.")
        else:
            depth_maps = res['local_points'][0, :, :, :, 2].cpu().numpy()  # Shape: (N, H, W) - extract z component

            # Use the same directory as poses
            depth_base_dir = Path(args.save_poses)
            if not depth_base_dir.suffix:  # No file extension means it's a directory
                depth_base_dir = depth_base_dir
            else:  # Has extension, use parent directory
                depth_base_dir = depth_base_dir.parent

            print(f"\nSaving depth maps...")

            # Split depth maps by data path and save separately
            frame_start = 0
            for data_idx, num_frames in enumerate(frame_counts):
                frame_end = frame_start + num_frames
                depth_subset = depth_maps[frame_start:frame_end]  # Depth maps for this data path
                interval = frame_intervals[data_idx]

                # Create subdirectory for each camera using actual camera name: cam03_depth, cam05_depth, etc.
                cam_name = camera_names[data_idx]
                cam_depth_dir = depth_base_dir / f"{cam_name}_depth"
                cam_depth_dir.mkdir(parents=True, exist_ok=True)

                # Save each depth map as a separate .npy file
                for local_idx in range(depth_subset.shape[0]):
                    original_frame_idx = local_idx * interval
                    depth_filename = f"{original_frame_idx:06d}.npy"
                    depth_path = cam_depth_dir / depth_filename

                    np.save(depth_path, depth_subset[local_idx])

                print(f"  Saved {cam_name}_depth: {num_frames} depth maps from {args.data_path[data_idx]}")
                frame_start = frame_end

            print(f"  Total depth maps saved: {depth_maps.shape[0]}")
            print(f"  Output directory: {depth_base_dir}")
            print(f"  Format: .npy files (H, W) in float32")

    print(f"\n{'='*60}")
    print(f"Done! Processed {len(args.data_path)} data path(s) with {imgs.shape[0]} total frames.")
    print(f"Point cloud shape: {res['points'][0][masks].shape}")
    print(f"{'='*60}")