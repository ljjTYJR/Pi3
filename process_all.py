#!/usr/bin/env python3
"""
Automatic batch processing script for Pi3 reconstruction.
Processes all subdirectories containing cam01 and cam02 folders.
"""

import subprocess
import argparse
import time
import json
import re
import sys
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

    # Run the command and capture output
    try:
        t_scene_start = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        t_scene_end = time.time()
        subprocess_time = t_scene_end - t_scene_start

        # Print captured output
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Extract timing JSON from subprocess output
        timing_data = None
        match = re.search(r'===TIMING_JSON_START===\n(.*?)\n===TIMING_JSON_END===',
                          result.stdout, re.DOTALL)
        if match:
            try:
                timing_data = json.loads(match.group(1))
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse timing data for {scene_path.name}")

        print(f"✓ Successfully processed {scene_path.name}")
        return True, timing_data, subprocess_time
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to process {scene_path.name}: {e}")
        return False, None, 0


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
    all_timing_data = []

    for idx, (scene_path, cam_paths) in enumerate(scenes, 1):
        print(f"\n[{idx}/{len(scenes)}] Processing {scene_path.name}...")

        success, timing_data, subprocess_time = process_scene(
            scene_path,
            cam_paths,
            interval=args.interval,
            ckpt=args.ckpt,
            device=args.device
        )

        if success:
            success_count += 1
            if timing_data:
                timing_data['scene_name'] = scene_path.name
                timing_data['subprocess_time'] = subprocess_time
                all_timing_data.append(timing_data)
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

    # Timing Summary
    if all_timing_data:
        print(f"\n{'='*80}")
        print("TIMING SUMMARY")
        print(f"{'='*80}")

        # Per-scene breakdown
        print("\nPer-Scene Breakdown:")
        print(f"{'Scene':<30} {'Frames':>8} {'Model':>8} {'Load':>8} {'Infer':>8} {'Save':>8} "
              f"{'Total':>8} {'FPS':>8}")
        print("-" * 104)

        for td in all_timing_data:
            timings = td['timings']
            print(f"{td['scene_name']:<30} "
                  f"{td['total_frames']:>8} "
                  f"{timings['model_loading']:>7.2f}s "
                  f"{timings['data_loading']:>7.2f}s "
                  f"{timings['inference']:>7.2f}s "
                  f"{timings['saving']:>7.2f}s "
                  f"{timings['total']:>7.2f}s "
                  f"{td['fps']:>7.1f}")

        # Aggregate statistics
        total_frames = sum(td['total_frames'] for td in all_timing_data)
        total_model_time = sum(td['timings']['model_loading'] for td in all_timing_data)
        total_data_time = sum(td['timings']['data_loading'] for td in all_timing_data)
        total_infer_time = sum(td['timings']['inference'] for td in all_timing_data)
        total_save_time = sum(td['timings']['saving'] for td in all_timing_data)
        total_proc_time = sum(td['timings']['total'] for td in all_timing_data)
        avg_fps = sum(td['fps'] for td in all_timing_data) / len(all_timing_data)

        print("-" * 104)
        print(f"{'TOTAL':<30} "
              f"{total_frames:>8} "
              f"{total_model_time:>7.2f}s "
              f"{total_data_time:>7.2f}s "
              f"{total_infer_time:>7.2f}s "
              f"{total_save_time:>7.2f}s "
              f"{total_proc_time:>7.2f}s "
              f"{avg_fps:>7.1f}")

        print(f"\nOverall Statistics:")
        print(f"  Average processing time per scene: {total_proc_time / len(all_timing_data):.2f}s")
        print(f"  Average frames per scene: {total_frames / len(all_timing_data):.1f}")
        print(f"  Total throughput: {total_frames / total_infer_time:.2f} frames/sec")
        print(f"  Time breakdown:")
        print(f"    - Model loading: {total_model_time / total_proc_time * 100:.1f}%")
        print(f"    - Data loading: {total_data_time / total_proc_time * 100:.1f}%")
        print(f"    - Inference: {total_infer_time / total_proc_time * 100:.1f}%")
        print(f"    - Saving: {total_save_time / total_proc_time * 100:.1f}%")

    print(f"{'='*80}")


if __name__ == '__main__':
    main()
