#!/usr/bin/env python3
"""
Generate Publication-Quality Plots for Storage Simulation Thesis

This script uses the thesis-quality plotting library to generate
a complete set of publication-ready visualizations from simulation results.

Usage:
    python generate_thesis_plots.py

The script will create a 'thesis_plots' directory with all generated figures
in both PDF and PNG formats.
"""

import os
import sys
from thesis_quality_plotter import generate_all_plots

def main():
    # Check if results file exists
    results_file = "storage_results/storage_simulation_results.json"
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found.")
        print("Please ensure the simulation results JSON file is in the correct location.")
        sys.exit(1)
    
    # Generate all plots
    output_dir = "thesis_plots"
    print(f"Generating thesis-quality plots from {results_file}...")
    plot_dir = generate_all_plots(results_file, output_dir)
    
    # List all generated files
    print("\nGenerated the following plots:")
    for i, filename in enumerate(sorted(os.listdir(plot_dir))):
        if filename.endswith('.pdf') or filename.endswith('.png'):
            print(f"{i+1}. {filename}")
    
    print(f"\nAll plots have been saved to the '{plot_dir}' directory.")
    print("The plots are available in both PDF (vector) format for publication and PNG format for preview.")

if __name__ == "__main__":
    main()