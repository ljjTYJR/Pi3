#!/usr/bin/env python3
"""
Automatic batch processing script for Pi3 reconstruction.
Processes all subdirectories containing cam01 and cam02 folders.
"""

import subprocess
import argparse
from pathlib import Path


def find_multi_camera_scenes(base_dir):
    """
    Find all directories containing multiple camera folders (cam01, cam02, etc.).
    Filters out directories ending with _depth (not image directories).

    Args:
        base_dir: Base directory to search

    Returns:
        List of tuples: (scene_path, [cam_paths])
    """
    base_path = Path(base_dir)
    scenes = []

    # Find all subdirectories in base_dir (skip directories ending with _depth)
    for scene_dir in sorted(base_path.iterdir()):
        if not scene_dir.is_dir():
            continue

        # Skip directories that end with _depth (these are output directories, not scenes)
        if scene_dir.name.endswith('_depth'):
            continue

        # Look for camera folders (cam01, cam02, etc.), excluding _depth folders
        cam_dirs = sorted([d for d in scene_dir.iterdir()
                          if d.is_dir() and d.name.startswith('cam') and not d.name.endswith('_depth')])

        if len(cam_dirs) >= 2:  # Only process scenes with at least 2 cameras
            scenes.append((scene_dir, cam_dirs))

    return scenes


def process_scene(scene_path, cam_paths, interval=2, ckpt=None, device='cuda'):
    """
    Process a single scene with multiple cameras.

    Args:
        scene_path: Path to the scene directory
        cam_paths: List of camera directory paths
        interval: Frame sampling interval
        ckpt: Path to model checkpoint (optional)
        device: Device to run on ('cuda' or 'cpu')
    """
    # Prepare output directory
    output_dir = scene_path / "pi_poses"

    # Build command
    cmd = [
        'python', 'demo.py',
        '--data_path', *[str(p) for p in cam_paths],
        '--interval', str(interval),
        '--save_poses', str(output_dir),
        '--save_depth',
        '--device', device
    ]

    if ckpt is not None:
        cmd.extend(['--ckpt', ckpt])

    print(f"\n{'='*80}")
    print(f"Processing: {scene_path.name}")
    print(f"Cameras: {[p.name for p in cam_paths]}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")

    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Successfully processed {scene_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to process {scene_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch process multiple scenes with Pi3 reconstruction.")

    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing scene subdirectories")
    parser.add_argument("--interval", type=int, default=2,
                        help="Frame sampling interval. Default: 2")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to model checkpoint file. Default: None (use pretrained)")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--dry_run", action='store_true',
                        help="Print scenes to process without actually processing them")

    args = parser.parse_args()

    # Find all scenes
    print(f"Searching for multi-camera scenes in: {args.base_dir}")
    scenes = find_multi_camera_scenes(args.base_dir)

    if not scenes:
        print("\nNo multi-camera scenes found!")
        return

    print(f"\nFound {len(scenes)} scene(s) to process:")
    for scene_path, cam_paths in scenes:
        print(f"  - {scene_path.name}: {len(cam_paths)} cameras ({', '.join(p.name for p in cam_paths)})")

    if args.dry_run:
        print("\nDry run mode - no processing performed.")
        return

    # Process each scene
    print(f"\n{'='*80}")
    print("Starting batch processing...")
    print(f"{'='*80}")

    success_count = 0
    failed_scenes = []

    for scene_path, cam_paths in scenes:
        success = process_scene(
            scene_path,
            cam_paths,
            interval=args.interval,
            ckpt=args.ckpt,
            device=args.device
        )

        if success:
            success_count += 1
        else:
            failed_scenes.append(scene_path.name)

    # Print summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total scenes: {len(scenes)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(failed_scenes)}")

    if failed_scenes:
        print("\nFailed scenes:")
        for scene_name in failed_scenes:
            print(f"  - {scene_name}")

    print(f"{'='*80}")


if __name__ == '__main__':
    main()
