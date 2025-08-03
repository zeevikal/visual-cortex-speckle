#!/usr/bin/env python3
"""
Simple script to generate synthetic shape dataset.

This script only generates the video files without any machine learning dependencies.
It's designed to be as simple as possible with minimal library requirements.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.shape_dataset_generator import ShapeDatasetGenerator


def main():
    """Main function to generate synthetic shape dataset."""
    parser = argparse.ArgumentParser(description="Generate synthetic shape dataset")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_shapes",
                       help="Output directory for generated videos")
    parser.add_argument("--videos-per-type", type=int, default=50,
                       help="Number of videos to generate per type")
    parser.add_argument("--frame-width", type=int, default=640,
                       help="Frame width in pixels")
    parser.add_argument("--frame-height", type=int, default=480,
                       help="Frame height in pixels")
    parser.add_argument("--frames-per-video", type=int, default=100,
                       help="Number of frames per video")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second for output videos")
    parser.add_argument("--colorized", action="store_true",
                       help="Use colorized shapes instead of monochrome")
    parser.add_argument("--motion-speed", type=float, default=4.0,
                       help="Speed multiplier for shape motion")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SYNTHETIC SHAPE DATASET GENERATOR")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Videos per type: {args.videos_per_type}")
    print(f"Frame size: {args.frame_width}x{args.frame_height}")
    print(f"Frames per video: {args.frames_per_video}")
    print(f"FPS: {args.fps}")
    print(f"Colorized shapes: {args.colorized}")
    print(f"Motion speed: {args.motion_speed}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    # Initialize generator
    generator = ShapeDatasetGenerator(
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        frames_per_video=args.frames_per_video,
        fps=args.fps,
        use_colorized_shapes=args.colorized,
        motion_speed=args.motion_speed
    )
    
    # Generate dataset
    stats = generator.generate_dataset(
        output_dir=args.output_dir,
        videos_per_type=args.videos_per_type,
        seed=args.seed
    )
    
    print(f"\nDataset generated successfully!")
    print(f"Total videos: {stats['total_videos']}")
    print(f"Video types: {len(stats['video_types'])}")
    
    # Save generation statistics
    stats_file = os.path.join(args.output_dir, "generation_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Generation statistics saved to: {stats_file}")
    print("\nGeneration completed!")


if __name__ == "__main__":
    main()
