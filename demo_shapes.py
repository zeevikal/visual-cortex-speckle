#!/usr/bin/env python3
"""
Simple demo script to test synthetic shape dataset generation.

This script demonstrates the basic functionality of the shape dataset generator
and can be used to verify the implementation works correctly.
"""

import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demo_shape_generation():
    """Demonstrate shape dataset generation with minimal examples."""
    print("=" * 60)
    print("SYNTHETIC SHAPE DATASET GENERATION DEMO")
    print("=" * 60)
    
    try:
        from src.data.shape_dataset_generator import ShapeDatasetGenerator
        
        # Initialize generator with smaller parameters for demo
        generator = ShapeDatasetGenerator(
            frame_width=320,
            frame_height=240,
            frames_per_video=20,  # Shorter videos for demo
            fps=10,
            use_colorized_shapes=True,  # Enable colorized shapes
            motion_speed=0.4  # Faster motion for demo
        )
        
        print("Generator initialized successfully!")
        print(f"Frame dimensions: {generator.frame_width}x{generator.frame_height}")
        print(f"Frames per video: {generator.frames_per_video}")
        print(f"Colorized shapes: {generator.use_colorized_shapes}")
        print(f"Motion speed: {generator.motion_speed}")
        print(f"Video types available: {list(generator.video_types.keys())}")
        
        # Generate a small demo dataset
        output_dir = "data/demo_shapes"
        
        print(f"\\nGenerating demo dataset in: {output_dir}")
        
        stats = generator.generate_dataset(
            output_dir=output_dir,
            videos_per_type=2,  # Only 2 videos per type for demo
            seed=42
        )
        
        print("\\n" + "=" * 40)
        print("DEMO DATASET GENERATION COMPLETED")
        print("=" * 40)
        print(f"Total videos generated: {stats['total_videos']}")
        print(f"Output directory: {output_dir}")
        
        # Show directory structure
        print("\\nDataset structure:")
        for video_type, type_stats in stats['video_types'].items():
            print(f"  {video_type}/: {type_stats['count']} videos")
        
        # List some generated files
        print("\\nSample generated files:")
        for root, dirs, files in os.walk(output_dir):
            for file in files[:3]:  # Show first 3 files
                if file.endswith('.mp4'):
                    rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                    print(f"  {rel_path}")
        
        return True
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please install required dependencies:")
        print("  pip install opencv-python numpy tqdm")
        return False
    
    except Exception as e:
        print(f"Error during demo: {e}")
        return False


def demo_dataset_info():
    """Show information about dataset structure and video types."""
    print("\\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    print("Video Types (as described in the research paper):")
    print("1. Blank Background:")
    print("   - blank_white: Pure white background")
    print("   - blank_black: Pure black background")
    
    print("\\n2. Single Shapes:")
    print("   - single_circle: One circle on background")
    print("   - single_rectangle: One rectangle on background") 
    print("   - single_triangle: One triangle on background")
    
    print("\\n3. Multi Shapes (Same Type):")
    print("   - multi_circles: Multiple circles (3-8 objects)")
    print("   - multi_rectangles: Multiple rectangles (3-8 objects)")
    print("   - multi_triangles: Multiple triangles (3-8 objects)")
    
    print("\\n4. Mixed Multi Shapes:")
    print("   - mixed_multi_shapes: Multiple different shapes together (4-10 objects)")
    
    print("\\nVideo Properties:")
    print("- Duration: 100 frames (representing 100 seconds)")
    print("- Background: Randomly white or black")
    print("- Shape colors: Colorized (red, blue, green, etc.) or black/white based on settings")
    print("- Motion: Fast sinusoidal and circular movement patterns")
    print("- Shape sizes: Random between 20-80 pixels")
    print("- Overlap prevention: Shapes avoid overlapping when possible")
    
    print("\\nIntended Use:")
    print("- Visual cortex speckle pattern analysis")
    print("- Shape classification using distance metrics")
    print("- Machine learning model training and evaluation")
    print("- Testing with colorized vs monochrome stimuli")


def main():
    """Main demo function."""
    print("Synthetic Shape Dataset Generator Demo")
    print("=" * 60)
    
    # Show dataset information
    demo_dataset_info()
    
    # Ask user if they want to generate demo dataset
    print("\\n" + "=" * 60)
    response = input("Generate demo dataset? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        success = demo_shape_generation()
        
        if success:
            print("\\n" + "=" * 60)
            print("DEMO COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("Next steps:")
            print("1. Check the generated videos in data/demo_shapes/")
            print("2. Use the full pipeline script: python generate_shapes_dataset.py")
            print("3. Configure parameters in configs/synthetic_shapes.yaml")
        else:
            print("\\n" + "=" * 60)
            print("DEMO FAILED")
            print("=" * 60)
            print("Please install required dependencies and try again.")
    else:
        print("\\nDemo skipped. Use 'python demo_shapes.py' to run later.")


if __name__ == "__main__":
    main()
