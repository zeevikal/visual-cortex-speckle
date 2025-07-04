"""
Example usage of interpretability tools for visual cortex speckle imaging analysis.

This script demonstrates how to use the interpretability module to analyze
ConvLSTM models and understand which speckle features are most important
for shape classification.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.interpretability import SpeckleInterpretabilityAnalyzer
from models.convlstm import ConvLSTMClassifier
from data.dataset import create_data_loaders_from_pickle


def main():
    """Main function to demonstrate interpretability analysis."""
    print("=== Visual Cortex Speckle Imaging - Interpretability Analysis ===\n")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['Circle', 'Rectangle', 'Triangle']
    
    # Load trained model
    model_path = 'checkpoints/best_model.pth'  # Update with your model path
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first using: python src/main.py --mode train")
        return
    
    print(f"Loading model from {model_path}")
    model = ConvLSTMClassifier(
        num_classes=3,
        hidden_size=64,
        sequence_length=64,
        dropout_rate=0.1
    )
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load test data
    print("Loading test data...")
    try:
        _, _, test_loader, _ = create_data_loaders_from_pickle(
            pickle_path='data/vis_cortex_data.pickle',  # Update with your data path
            subjects=['ZeevKal'],  # Update with your subjects
            shape_filter=['Circle', 'Rectangle', 'Triangle'],
            feature_type='manhattan',
            batch_size=32,
            test_size=0.2
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure data is processed and available.")
        return
    
    # Get test data for analysis
    X_test, y_test = next(iter(test_loader))
    
    # Initialize interpretability analyzer
    print("Initializing interpretability analyzer...")
    analyzer = SpeckleInterpretabilityAnalyzer(
        model=model,
        device=device,
        class_names=class_names
    )
    
    # Create output directory
    output_dir = 'results/interpretability_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. SHAP Analysis
    print("\n1. Performing SHAP analysis...")
    try:
        shap_results = analyzer.analyze_feature_importance_shap(
            X_test=X_test,
            y_test=y_test,
            background_size=50,
            test_size=30,
            save_dir=os.path.join(output_dir, 'shap')
        )
        
        print("SHAP analysis completed successfully!")
        print(f"   - Analyzed {len(shap_results['test_labels'])} test samples")
        print(f"   - Feature importance calculated for {len(shap_results['feature_importance'])} classes")
        
        # Print top important features
        if 'overall' in shap_results['feature_importance']:
            overall_importance = shap_results['feature_importance']['overall']
            top_features = np.argsort(overall_importance)[-10:][::-1]
            print(f"   - Top 10 most important features: {top_features.tolist()}")
            
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("This might be due to missing SHAP library. Install with: pip install shap")
    
    # 2. Temporal Pattern Analysis
    print("\n2. Performing temporal pattern analysis...")
    try:
        temporal_results = analyzer.analyze_temporal_patterns(
            X_test=X_test,
            y_test=y_test,
            sequence_length=64,
            save_dir=os.path.join(output_dir, 'temporal')
        )
        
        print("Temporal analysis completed successfully!")
        if 'temporal_patterns' in temporal_results:
            patterns = temporal_results['temporal_patterns']
            print(f"   - Analyzed {len(patterns['labels'])} temporal patterns")
            print(f"   - PCA explained variance (top 5): {patterns['pca_explained_variance'][:5]}")
            
    except Exception as e:
        print(f"Temporal analysis failed: {e}")
    
    # 3. Spatial Cortical Mapping
    print("\n3. Performing spatial cortical mapping analysis...")
    try:
        # Define cortical regions based on speckle pattern spatial organization
        cortical_regions = {
            'V1_edge_detectors': (0, 1024),      # Early edge detection features
            'V1_orientation_columns': (1024, 2048),  # Orientation processing
            'V1_spatial_frequency': (2048, 3072),    # Spatial frequency analysis
            'V1_higher_order': (3072, 4096),         # Higher-order integration
        }
        
        spatial_results = analyzer.analyze_spatial_cortical_mapping(
            X_test=X_test,
            y_test=y_test,
            cortical_regions=cortical_regions,
            save_dir=os.path.join(output_dir, 'spatial')
        )
        
        print("Spatial cortical mapping completed successfully!")
        if 'region_importance' in spatial_results:
            print("   - Cortical region importance analysis:")
            for region, data in spatial_results['region_importance'].items():
                if 'overall' in data:
                    mean_imp = data['overall']['mean_importance']
                    print(f"     {region}: {mean_imp:.4f}")
                    
        if 'connectivity' in spatial_results:
            print("   - Cortical connectivity analysis:")
            for shape, conn_data in spatial_results['connectivity'].items():
                mean_conn = conn_data['mean_connectivity']
                print(f"     {shape}: mean connectivity = {mean_conn:.4f}")
                
    except Exception as e:
        print(f"Spatial cortical mapping failed: {e}")
    
    # 4. Generate Comprehensive Report
    print("\n4. Generating comprehensive interpretability report...")
    try:
        analyzer.create_interpretability_report(output_dir)
        print("Comprehensive report generated successfully!")
        print(f"   - Report saved to: {output_dir}/interpretability_report.md")
        print(f"   - Visualizations saved to: {output_dir}/")
        
    except Exception as e:
        print(f"Report generation failed: {e}")
    
    # 5. Research Insights Summary
    print("\n=== RESEARCH INSIGHTS SUMMARY ===")
    print("The interpretability analysis provides the following insights:")
    
    print("\n🧠 Neuroscientific Implications:")
    print("1. Feature importance maps reveal which speckle patterns correspond to")
    print("   different aspects of shape processing in visual cortex")
    print("2. Temporal analysis shows how shape recognition unfolds over time")
    print("3. Spatial mapping identifies potential cortical regions involved")
    print("4. Connectivity analysis reveals functional relationships between regions")
    
    print("\n📊 Expected Findings (based on your research):")
    print("• Rectangles: Strong edge detection features, high V1_edge_detectors importance")
    print("• Triangles: Moderate edge features, distributed across orientation columns")
    print("• Circles: Weak edge features, may show different spatial patterns")
    print("• Temporal patterns: Different integration timescales for different shapes")
    
    print("\n🔬 Clinical Applications:")
    print("1. Biomarker development for visual processing disorders")
    print("2. Diagnostic tools for cortical functionality assessment")
    print("3. Therapeutic target identification for visual rehabilitation")
    
    print("\n📈 Future Directions:")
    print("1. Validate findings with direct neural recordings")
    print("2. Test consistency across subjects and populations")
    print("3. Expand to more complex visual stimuli")
    print("4. Develop real-time interpretability tools")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print("\nFiles generated:")
    print("- interpretability_report.md: Comprehensive analysis report")
    print("- feature_importance.png: SHAP feature importance plots")
    print("- temporal_analysis.png: Temporal pattern visualizations")
    print("- cortical_mapping.png: Spatial cortical mapping")
    print("- cortical_connectivity.png: Regional connectivity analysis")


def analyze_shape_specific_patterns():
    """
    Additional analysis function to examine shape-specific patterns.
    This function provides detailed analysis of how different shapes
    activate different cortical regions and temporal patterns.
    """
    print("\n=== SHAPE-SPECIFIC PATTERN ANALYSIS ===")
    
    # This would be called after the main analysis
    # Implementation would include:
    # 1. Detailed per-shape feature importance
    # 2. Shape-specific temporal dynamics
    # 3. Differential cortical activation patterns
    # 4. Cross-shape similarity analysis
    
    print("Shape-specific analysis would include:")
    print("1. Rectangle-specific features (edge detection, corners)")
    print("2. Triangle-specific features (angular processing)")
    print("3. Circle-specific features (curved contours, if any)")
    print("4. Temporal dynamics comparison between shapes")
    print("5. Cortical region specialization for each shape")


def validate_interpretability_results():
    """
    Function to validate interpretability results against known neuroscience.
    This would compare the model's learned features with known properties
    of visual cortex processing.
    """
    print("\n=== INTERPRETABILITY VALIDATION ===")
    
    validation_checks = [
        "Edge detection features should be prominent for rectangles/triangles",
        "Orientation-specific features should show directional preferences",
        "Temporal dynamics should match known cortical processing timescales",
        "Spatial organization should reflect retinotopic mapping principles",
        "Connectivity patterns should match known cortical pathways"
    ]
    
    print("Validation checks to perform:")
    for i, check in enumerate(validation_checks, 1):
        print(f"{i}. {check}")


if __name__ == "__main__":
    main()
