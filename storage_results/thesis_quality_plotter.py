#!/usr/bin/env python3
"""
Academic Publication Quality Plotter for Storage Simulation Results

This script generates high-quality, publication-ready plots from storage simulation 
results for thesis or academic paper presentation.

Features:
- Clean, minimalist design following academic publication standards
- Consistent color schemes and styling
- Proper scaling, labels and annotations
- Multiple plot types for different analytical perspectives
- Vector and high-resolution outputs

Dependencies:
- matplotlib
- numpy
- json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os
from matplotlib import rcParams
from collections import defaultdict

# Set up matplotlib for publication quality
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'CMU Serif']
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 16
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 600
rcParams['savefig.format'] = 'pdf'
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05
rcParams['axes.grid'] = False
rcParams['axes.linewidth'] = 0.8
rcParams['lines.linewidth'] = 1.5
rcParams['grid.linewidth'] = 0.5
rcParams['lines.markersize'] = 6
rcParams['legend.framealpha'] = 0.8
rcParams['legend.edgecolor'] = '0.8'

# Define consistent colors for different workload types
COLORS = {
    'sequential_read': '#2060A0',  # Blue
    'sequential_write': '#C03020',  # Red
    'random_read': '#4090E0',  # Light blue
    'random_write': '#E05040',  # Light red
    'sequential_mixed': '#208040',  # Green
    'random_mixed': '#60B060',  # Light green
}

# Define color palettes for num_drives
DRIVE_COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, 10))

# Define hatches for additional visual encoding
HATCHES = ['', '//', '\\\\', 'xx', '..', 'oo', '++']

# Create output directory if it doesn't exist
def setup_output_dir(output_dir="thesis_plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Helper function to extract data in the right format
def extract_data_by_workload_and_drives(data):
    """Extract data organized by workload type and number of drives"""
    workloads = set()
    drive_counts = set()
    organized_data = defaultdict(dict)
    
    for key, value in data.items():
        workload_type = value['workload_type'].lower().replace(' ', '_').replace('-', '_')
        num_drives = value['num_drives']
        
        workloads.add(workload_type)
        drive_counts.add(num_drives)
        organized_data[workload_type][num_drives] = value
    
    return organized_data, sorted(list(workloads)), sorted(list(drive_counts))

def plot_throughput_comparison(data, output_dir="thesis_plots", filename="throughput_comparison"):
    """Create a throughput comparison plot showing scaling across drive counts"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.13
    index = np.arange(len(drive_counts))
    
    for i, workload in enumerate(workloads):
        throughputs = []
        for drive_count in drive_counts:
            if drive_count in organized_data[workload]:
                throughputs.append(organized_data[workload][drive_count]['system_stats']['total_throughput_workload'])
            else:
                throughputs.append(0)
        
        position = index + (i - len(workloads)/2 + 0.5) * bar_width
        bars = ax.bar(position, throughputs, bar_width, 
                     label=workload.replace('_', ' ').title(),
                     color=COLORS[workload], 
                     edgecolor='black', 
                     linewidth=0.8,
                     zorder=3)
    
    # Add labels, title and legend
    ax.set_xlabel('Number of Drives')
    ax.set_ylabel('Throughput (MB/s)')
    ax.set_title('Storage System Throughput by Workload Type and Drive Count')
    ax.set_xticks(index)
    ax.set_xticklabels(drive_counts)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), ncol=2, frameon=True)
    
    # Add grid on y-axis only
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, zorder=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_iops_comparison(data, output_dir="thesis_plots", filename="iops_comparison"):
    """Create an IOPS comparison plot showing scaling across drive counts"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.13
    index = np.arange(len(drive_counts))
    
    for i, workload in enumerate(workloads):
        iops_values = []
        for drive_count in drive_counts:
            if drive_count in organized_data[workload]:
                iops_values.append(organized_data[workload][drive_count]['system_stats']['total_iops_workload'])
            else:
                iops_values.append(0)
        
        position = index + (i - len(workloads)/2 + 0.5) * bar_width
        bars = ax.bar(position, iops_values, bar_width, 
                     label=workload.replace('_', ' ').title(),
                     color=COLORS[workload], 
                     edgecolor='black', 
                     linewidth=0.8,
                     zorder=3)
    
    # Add labels, title and legend
    ax.set_xlabel('Number of Drives')
    ax.set_ylabel('IOPS')
    ax.set_title('Storage System IOPS by Workload Type and Drive Count')
    ax.set_xticks(index)
    ax.set_xticklabels(drive_counts)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), ncol=2, frameon=True)
    
    # Add grid on y-axis only
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, zorder=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_latency_comparison(data, output_dir="thesis_plots", filename="latency_comparison"):
    """Create a latency comparison plot showing read and write latencies"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    # We'll focus on the 8-drive case for clarity
    target_drive_count = 8
    
    read_latencies = []
    write_latencies = []
    workload_labels = []
    
    for workload in workloads:
        if target_drive_count in organized_data[workload]:
            workload_data = organized_data[workload][target_drive_count]
            read_latencies.append(workload_data['system_stats']['avg_read_latency'] * 1000)  # Convert to ms
            write_latencies.append(workload_data['system_stats']['avg_write_latency'] * 1000)  # Convert to ms
            workload_labels.append(workload.replace('_', ' ').title())
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.35
    index = np.arange(len(workload_labels))
    
    # Plot read latencies
    ax.bar(index - bar_width/2, read_latencies, bar_width, color='#4090E0', 
           label='Read Latency', edgecolor='black', linewidth=0.8, zorder=3)
    
    # Plot write latencies
    ax.bar(index + bar_width/2, write_latencies, bar_width, color='#E05040', 
           label='Write Latency', edgecolor='black', linewidth=0.8, zorder=3)
    
    # Add labels and annotations
    ax.set_xlabel('Workload Type')
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title(f'Read and Write Latencies by Workload Type ({target_drive_count} Drives)')
    ax.set_xticks(index)
    ax.set_xticklabels(workload_labels, rotation=45, ha='right')
    ax.legend()
    
    # Add grid on y-axis only
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_scaling_efficiency(data, output_dir="thesis_plots", filename="scaling_efficiency"):
    """Plot the scaling efficiency (percentage of linear scaling) for each workload"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for workload in workloads:
        throughputs = []
        for drive_count in drive_counts:
            if drive_count in organized_data[workload]:
                throughputs.append(organized_data[workload][drive_count]['system_stats']['total_throughput_workload'])
            else:
                throughputs.append(0)
        
        # Calculate scaling efficiency (relative to perfect linear scaling)
        if throughputs[0] > 0:  # Avoid division by zero
            single_drive_throughput = throughputs[0]
            scaling_efficiency = [100 * throughput / (single_drive_throughput * drive_count) 
                                  for throughput, drive_count in zip(throughputs, drive_counts)]
            
            # Plot line
            ax.plot(drive_counts, scaling_efficiency, marker='o', label=workload.replace('_', ' ').title(),
                   color=COLORS[workload], linewidth=2)
    
    # Add ideal scaling reference line
    ax.axhline(y=100, color='k', linestyle='--', alpha=0.7, label='Ideal Scaling')
    
    # Add labels and annotations
    ax.set_xlabel('Number of Drives')
    ax.set_ylabel('Scaling Efficiency (%)')
    ax.set_title('Storage System Scaling Efficiency by Workload Type')
    ax.set_xticks(drive_counts)
    ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    # Set y-axis limits
    max_efficiency = 110
    ax.set_ylim(0, max_efficiency)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_backpressure_analysis(data, output_dir="thesis_plots", filename="backpressure_analysis"):
    """Create a plot showing backpressure rates across workloads and drive counts"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for workload in workloads:
        backpressure_rates = []
        for drive_count in drive_counts:
            if drive_count in organized_data[workload]:
                backpressure_rates.append(organized_data[workload][drive_count]['workload_stats']['backpressure_ratio'] * 100)
            else:
                backpressure_rates.append(0)
        
        # Plot line
        ax.plot(drive_counts, backpressure_rates, marker='o', label=workload.replace('_', ' ').title(),
               color=COLORS[workload], linewidth=2)
    
    # Add labels and annotations
    ax.set_xlabel('Number of Drives')
    ax.set_ylabel('Backpressure Rate (%)')
    ax.set_title('Backpressure Rate by Workload Type and Drive Count')
    ax.set_xticks(drive_counts)
    ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_throughput_vs_latency(data, output_dir="thesis_plots", filename="throughput_vs_latency"):
    """Create a scatter plot of throughput vs latency for different workloads"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for workload in workloads:
        throughputs = []
        avg_latencies = []
        marker_sizes = []
        
        for drive_count in drive_counts:
            if drive_count in organized_data[workload]:
                workload_data = organized_data[workload][drive_count]
                throughput = workload_data['system_stats']['total_throughput_workload']
                
                # For latency, we need to consider read/write mix
                read_ratio = workload_data['workload_stats'].get('read_ratio', 0)
                read_latency = workload_data['system_stats']['avg_read_latency']
                write_latency = workload_data['system_stats']['avg_write_latency']
                
                # Compute weighted average latency based on read/write ratio
                if read_ratio > 0 and write_latency > 0:
                    avg_latency = (read_latency * read_ratio + write_latency * (1-read_ratio)) * 1000  # in ms
                elif read_latency > 0:
                    avg_latency = read_latency * 1000
                elif write_latency > 0:
                    avg_latency = write_latency * 1000
                else:
                    avg_latency = 0
                
                throughputs.append(throughput)
                avg_latencies.append(avg_latency)
                marker_sizes.append(50 + drive_count * 20)  # Size represents drive count
        
        if throughputs:
            ax.scatter(avg_latencies, throughputs, s=marker_sizes, alpha=0.7, 
                      label=workload.replace('_', ' ').title(), color=COLORS[workload],
                      edgecolors='black', linewidth=0.8)
            
            # Connect points with a line to show progression
            sorted_indices = np.argsort(avg_latencies)
            sorted_latencies = [avg_latencies[i] for i in sorted_indices]
            sorted_throughputs = [throughputs[i] for i in sorted_indices]
            ax.plot(sorted_latencies, sorted_throughputs, color=COLORS[workload], 
                   alpha=0.5, linestyle='--')
    
    # Add labels and annotations
    ax.set_xlabel('Average Latency (ms)')
    ax.set_ylabel('Throughput (MB/s)')
    ax.set_title('Throughput vs. Latency by Workload Type')
    ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_queue_utilization(data, output_dir="thesis_plots", filename="queue_utilization"):
    """Create a plot showing average queue utilization by workload and drive count"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.13
    index = np.arange(len(drive_counts))
    
    for i, workload in enumerate(workloads):
        avg_utilizations = []
        for drive_count in drive_counts:
            if drive_count in organized_data[workload]:
                # Calculate average queue utilization across all drives
                workload_data = organized_data[workload][drive_count]
                drive_stats = workload_data['drive_stats']
                utilizations = [stats.get('avg_queue_utilization', 0) for drive_id, stats in drive_stats.items()]
                avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0
                avg_utilizations.append(avg_utilization)
            else:
                avg_utilizations.append(0)
        
        position = index + (i - len(workloads)/2 + 0.5) * bar_width
        bars = ax.bar(position, avg_utilizations, bar_width, 
                     label=workload.replace('_', ' ').title(),
                     color=COLORS[workload], 
                     edgecolor='black', 
                     linewidth=0.8,
                     zorder=3)
    
    # Add labels, title and legend
    ax.set_xlabel('Number of Drives')
    ax.set_ylabel('Average Queue Utilization (%)')
    ax.set_title('Average Queue Utilization by Workload Type and Drive Count')
    ax.set_xticks(index)
    ax.set_xticklabels(drive_counts)
    ax.legend(loc='upper right', ncol=2, frameon=True)
    
    # Add grid on y-axis only
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_detailed_workload_scaling(data, selected_workloads=None, output_dir="thesis_plots"):
    """Create detailed scaling plots for selected workloads (or all if None)"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    if selected_workloads is None:
        selected_workloads = workloads
    
    for workload in selected_workloads:
        if workload not in workloads:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Extract data for this workload
        throughputs = []
        read_throughputs = []
        write_throughputs = []
        iops_values = []
        read_iops = []
        write_iops = []
        backpressure_rates = []
        completion_rates = []
        
        for drive_count in drive_counts:
            if drive_count in organized_data[workload]:
                stats = organized_data[workload][drive_count]['system_stats']
                throughputs.append(stats['total_throughput_workload'])
                read_throughputs.append(stats['read_throughput_workload'])
                write_throughputs.append(stats['write_throughput_workload'])
                iops_values.append(stats['total_iops_workload'])
                read_iops.append(stats['read_iops_workload'])
                write_iops.append(stats['write_iops_workload'])
                
                workload_stats = organized_data[workload][drive_count]['workload_stats']
                backpressure_rates.append(workload_stats['backpressure_ratio'] * 100)
                completion_rates.append(workload_stats['successful_ratio'] * 100)
        
        # Plot throughput scaling on the left
        ax1.plot(drive_counts, throughputs, marker='o', linestyle='-', label='Total', linewidth=2, color='#333333')
        ax1.plot(drive_counts, read_throughputs, marker='s', linestyle='--', label='Read', linewidth=2, color='#4090E0')
        ax1.plot(drive_counts, write_throughputs, marker='^', linestyle='-.', label='Write', linewidth=2, color='#E05040')
        
        # Add perfect scaling reference line
        if throughputs and throughputs[0] > 0:
            perfect_scaling = [throughputs[0] * count for count in drive_counts]
            ax1.plot(drive_counts, perfect_scaling, linestyle=':', color='gray', alpha=0.7, label='Perfect Scaling')
        
        ax1.set_xlabel('Number of Drives')
        ax1.set_ylabel('Throughput (MB/s)')
        ax1.set_title(f'{workload.replace("_", " ").title()} Throughput Scaling')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='best')
        
        # Plot backpressure and completion rate on the right
        ax2.plot(drive_counts, backpressure_rates, marker='o', linestyle='-', label='Backpressure Rate', 
                color='#B03060', linewidth=2)
        ax2.plot(drive_counts, completion_rates, marker='s', linestyle='-', label='Completion Rate', 
                color='#40A070', linewidth=2)
        
        ax2.set_xlabel('Number of Drives')
        ax2.set_ylabel('Rate (%)')
        ax2.set_title(f'{workload.replace("_", " ").title()} Operation Rates')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='best')
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{workload}_detailed_scaling.pdf")
        plt.savefig(f"{output_dir}/{workload}_detailed_scaling.png", dpi=300)
        plt.close()

def plot_comparative_analysis(data, output_dir="thesis_plots", filename="comparative_analysis"):
    """Create a comprehensive comparative analysis plot for thesis presentation"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    # We'll focus on the largest drive configuration for comparison
    max_drive_count = max(drive_counts)
    
    # Set up the figure with a complex grid
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1])
    
    # Throughput comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # 1. Throughput by workload type
    workload_labels = []
    throughputs = []
    for workload in workloads:
        if max_drive_count in organized_data[workload]:
            workload_labels.append(workload.replace('_', ' ').title())
            throughputs.append(organized_data[workload][max_drive_count]['system_stats']['total_throughput_workload'])
    
    bars1 = ax1.bar(workload_labels, throughputs, color=[COLORS[w] for w in workloads], 
                   edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Workload Type')
    ax1.set_ylabel('Throughput (MB/s)')
    ax1.set_title(f'Throughput by Workload ({max_drive_count} Drives)')
    ax1.set_xticklabels(workload_labels, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. IOPS by workload type
    iops_values = []
    for workload in workloads:
        if max_drive_count in organized_data[workload]:
            iops_values.append(organized_data[workload][max_drive_count]['system_stats']['total_iops_workload'])
    
    bars2 = ax2.bar(workload_labels, iops_values, color=[COLORS[w] for w in workloads], 
                   edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Workload Type')
    ax2.set_ylabel('IOPS')
    ax2.set_title(f'IOPS by Workload ({max_drive_count} Drives)')
    ax2.set_xticklabels(workload_labels, rotation=45, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Latency comparison
    read_latencies = []
    write_latencies = []
    for workload in workloads:
        if max_drive_count in organized_data[workload]:
            workload_data = organized_data[workload][max_drive_count]
            read_latencies.append(workload_data['system_stats']['avg_read_latency'] * 1000)  # Convert to ms
            write_latencies.append(workload_data['system_stats']['avg_write_latency'] * 1000)  # Convert to ms
    
    x = np.arange(len(workload_labels))
    width = 0.35
    
    ax3.bar(x - width/2, read_latencies, width, label='Read Latency', color='#4090E0', 
           edgecolor='black', linewidth=0.8)
    ax3.bar(x + width/2, write_latencies, width, label='Write Latency', color='#E05040', 
           edgecolor='black', linewidth=0.8)
    
    ax3.set_xlabel('Workload Type')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title(f'Read/Write Latency by Workload ({max_drive_count} Drives)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(workload_labels, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Scaling comparison for sequential operations
    seq_workloads = ['sequential_read', 'sequential_write']
    for workload in seq_workloads:
        if workload in workloads:
            throughputs = []
            for drive_count in drive_counts:
                if drive_count in organized_data[workload]:
                    throughputs.append(organized_data[workload][drive_count]['system_stats']['total_throughput_workload'])
                else:
                    throughputs.append(0)
            
            # Plot line
            ax4.plot(drive_counts, throughputs, marker='o', 
                    label=workload.replace('_', ' ').title(),
                    color=COLORS[workload], linewidth=2)
            
            # Add perfect scaling reference if this is the first workload
            if workload == seq_workloads[0] and throughputs and throughputs[0] > 0:
                perfect_scaling = [throughputs[0] * count for count in drive_counts]
                ax4.plot(drive_counts, perfect_scaling, linestyle=':', 
                        color='gray', alpha=0.7, label='Perfect Scaling')
    
    ax4.set_xlabel('Number of Drives')
    ax4.set_ylabel('Throughput (MB/s)')
    ax4.set_title('Sequential Operations Scaling')
    ax4.set_xticks(drive_counts)
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # 5. Scaling comparison for random operations
    rand_workloads = ['random_read', 'random_write']
    for workload in rand_workloads:
        if workload in workloads:
            throughputs = []
            for drive_count in drive_counts:
                if drive_count in organized_data[workload]:
                    throughputs.append(organized_data[workload][drive_count]['system_stats']['total_throughput_workload'])
                else:
                    throughputs.append(0)
            
            # Plot line
            ax5.plot(drive_counts, throughputs, marker='o', 
                    label=workload.replace('_', ' ').title(),
                    color=COLORS[workload], linewidth=2)
            
            # Add perfect scaling reference if this is the first workload
            if workload == rand_workloads[0] and throughputs and throughputs[0] > 0:
                perfect_scaling = [throughputs[0] * count for count in drive_counts]
                ax5.plot(drive_counts, perfect_scaling, linestyle=':', 
                        color='gray', alpha=0.7, label='Perfect Scaling')
    
    ax5.set_xlabel('Number of Drives')
    ax5.set_ylabel('Throughput (MB/s)')
    ax5.set_title('Random Operations Scaling')
    ax5.set_xticks(drive_counts)
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # 6. Backpressure rates
    for workload in workloads:
        backpressure_rates = []
        for drive_count in drive_counts:
            if drive_count in organized_data[workload]:
                backpressure_rates.append(organized_data[workload][drive_count]['workload_stats']['backpressure_ratio'] * 100)
            else:
                backpressure_rates.append(0)
        
        # Plot line
        ax6.plot(drive_counts, backpressure_rates, marker='o', 
                label=workload.replace('_', ' ').title(),
                color=COLORS[workload], linewidth=2)
    
    ax6.set_xlabel('Number of Drives')
    ax6.set_ylabel('Backpressure Rate (%)')
    ax6.set_title('Backpressure by Workload and Drive Count')
    ax6.set_xticks(drive_counts)
    ax6.legend()
    ax6.grid(True, linestyle='--', alpha=0.7)
    ax6.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_throughput_to_latency_ratio(data, output_dir="thesis_plots", filename="throughput_to_latency_ratio"):
    """Plot the throughput-to-latency ratio, a measure of performance efficiency"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for workload in workloads:
        tl_ratios = []
        for drive_count in drive_counts:
            if drive_count in organized_data[workload]:
                workload_data = organized_data[workload][drive_count]
                throughput = workload_data['system_stats']['total_throughput_workload']
                
                # Calculate average latency based on operation mix
                read_ratio = workload_data['workload_stats'].get('read_ratio', 0)
                read_latency = workload_data['system_stats']['avg_read_latency']
                write_latency = workload_data['system_stats']['avg_write_latency']
                
                if read_ratio > 0 and write_latency > 0:
                    avg_latency = read_latency * read_ratio + write_latency * (1 - read_ratio)
                elif read_latency > 0:
                    avg_latency = read_latency
                elif write_latency > 0:
                    avg_latency = write_latency
                else:
                    avg_latency = 1  # Avoid division by zero
                
                # Calculate throughput-to-latency ratio
                tl_ratio = throughput / avg_latency
                tl_ratios.append(tl_ratio)
            else:
                tl_ratios.append(0)
        
        # Plot line
        ax.plot(drive_counts, tl_ratios, marker='o', 
               label=workload.replace('_', ' ').title(),
               color=COLORS[workload], linewidth=2)
    
    # Add labels and annotations
    ax.set_xlabel('Number of Drives')
    ax.set_ylabel('Throughput-to-Latency Ratio (MB/s/ms)')
    ax.set_title('Efficiency Ratio (Throughput per Unit Latency)')
    ax.set_xticks(drive_counts)
    ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_read_write_breakdown(data, output_dir="thesis_plots", filename="read_write_breakdown"):
    """Create a stacked bar chart showing read/write throughput breakdown by workload"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    # We'll focus on the largest drive configuration for comparison
    max_drive_count = max(drive_counts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    workload_labels = []
    read_throughputs = []
    write_throughputs = []
    
    for workload in workloads:
        if max_drive_count in organized_data[workload]:
            workload_data = organized_data[workload][max_drive_count]
            workload_labels.append(workload.replace('_', ' ').title())
            read_throughputs.append(workload_data['system_stats']['read_throughput_workload'])
            write_throughputs.append(workload_data['system_stats']['write_throughput_workload'])
    
    x = np.arange(len(workload_labels))
    
    # Create stacked bars
    ax.bar(x, read_throughputs, label='Read Throughput', color='#4090E0', 
          edgecolor='black', linewidth=0.8)
    ax.bar(x, write_throughputs, bottom=read_throughputs, label='Write Throughput', 
          color='#E05040', edgecolor='black', linewidth=0.8)
    
    # Add labels and annotations
    ax.set_xlabel('Workload Type')
    ax.set_ylabel('Throughput (MB/s)')
    ax.set_title(f'Read/Write Throughput Breakdown ({max_drive_count} Drives)')
    ax.set_xticks(x)
    ax.set_xticklabels(workload_labels, rotation=45, ha='right')
    ax.legend(loc='upper left')
    
    # Add grid on y-axis only
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    # Add throughput values as text on bars
    for i, (r, w) in enumerate(zip(read_throughputs, write_throughputs)):
        # Only show values if they're significant
        if r > 0.1:
            ax.text(i, r/2, f"{r:.1f}", ha='center', va='center', fontweight='bold', color='white')
        if w > 0.1:
            ax.text(i, r + w/2, f"{w:.1f}", ha='center', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_heatmap_comparison(data, output_dir="thesis_plots", filename="performance_heatmap"):
    """Create a heatmap of performance metrics across workloads and drive counts"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    # Normalize values for the heatmap
    throughput_matrix = np.zeros((len(workloads), len(drive_counts)))
    latency_matrix = np.zeros((len(workloads), len(drive_counts)))
    
    max_throughput = 0
    min_latency = float('inf')
    max_latency = 0
    
    # Collect data and find max values for normalization
    for i, workload in enumerate(workloads):
        for j, drive_count in enumerate(drive_counts):
            if drive_count in organized_data[workload]:
                workload_data = organized_data[workload][drive_count]
                throughput = workload_data['system_stats']['total_throughput_workload']
                throughput_matrix[i, j] = throughput
                
                # Calculate average latency
                read_ratio = workload_data['workload_stats'].get('read_ratio', 0)
                read_latency = workload_data['system_stats']['avg_read_latency'] * 1000  # ms
                write_latency = workload_data['system_stats']['avg_write_latency'] * 1000  # ms
                
                if read_ratio > 0 and write_latency > 0:
                    avg_latency = read_latency * read_ratio + write_latency * (1 - read_ratio)
                elif read_latency > 0:
                    avg_latency = read_latency
                elif write_latency > 0:
                    avg_latency = write_latency
                else:
                    avg_latency = 0
                
                latency_matrix[i, j] = avg_latency
                
                max_throughput = max(max_throughput, throughput)
                if avg_latency > 0:
                    min_latency = min(min_latency, avg_latency)
                    max_latency = max(max_latency, avg_latency)
    
    # Create figure with two heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Normalize throughput (higher is better)
    normalized_throughput = throughput_matrix / max_throughput if max_throughput > 0 else throughput_matrix
    
    # Normalize latency (lower is better) and invert (so higher values in heatmap = better performance)
    normalized_latency = 1 - ((latency_matrix - min_latency) / (max_latency - min_latency)) if max_latency > min_latency else np.zeros_like(latency_matrix)
    
    # Create custom colormaps
    throughput_cmap = plt.cm.YlGnBu
    latency_cmap = plt.cm.YlOrRd_r  # Reversed so higher values (lower latency) are better
    
    # Create heatmaps
    im1 = ax1.imshow(normalized_throughput, cmap=throughput_cmap, vmin=0, vmax=1, aspect='auto')
    im2 = ax2.imshow(normalized_latency, cmap=latency_cmap, vmin=0, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
    cbar1.set_label('Normalized Throughput')
    
    cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
    cbar2.set_label('Normalized Latency (Inverted)')
    
    # Set up axis ticks and labels
    ax1.set_xticks(np.arange(len(drive_counts)))
    ax1.set_yticks(np.arange(len(workloads)))
    ax1.set_xticklabels(drive_counts)
    ax1.set_yticklabels([w.replace('_', ' ').title() for w in workloads])
    
    ax2.set_xticks(np.arange(len(drive_counts)))
    ax2.set_yticks(np.arange(len(workloads)))
    ax2.set_xticklabels(drive_counts)
    ax2.set_yticklabels([w.replace('_', ' ').title() for w in workloads])
    
    # Rotate the x-axis labels
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title
    ax1.set_title('Throughput Heatmap')
    ax1.set_xlabel('Number of Drives')
    ax1.set_ylabel('Workload Type')
    
    ax2.set_title('Latency Heatmap (Lower is Better)')
    ax2.set_xlabel('Number of Drives')
    # ax2.set_ylabel('Workload Type')  # Don't need this on the second subplot
    
    # Add text annotations with actual values
    for i in range(len(workloads)):
        for j in range(len(drive_counts)):
            text1 = ax1.text(j, i, f"{throughput_matrix[i, j]:.1f}",
                         ha="center", va="center", color="black" if normalized_throughput[i, j] < 0.7 else "white",
                         fontsize=9)
            
            text2 = ax2.text(j, i, f"{latency_matrix[i, j]:.1f}",
                         ha="center", va="center", color="black" if normalized_latency[i, j] < 0.7 else "white",
                         fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_iops_comparison_by_operation(data, output_dir="thesis_plots", filename="iops_by_operation"):
    """Plot IOPS comparison by operation type (read/write) for maximum drive configuration"""
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    
    # We'll focus on the largest drive configuration for comparison
    max_drive_count = max(drive_counts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    workload_labels = []
    read_iops_values = []
    write_iops_values = []
    
    for workload in workloads:
        if max_drive_count in organized_data[workload]:
            workload_data = organized_data[workload][max_drive_count]
            workload_labels.append(workload.replace('_', ' ').title())
            read_iops_values.append(workload_data['system_stats']['read_iops_workload'])
            write_iops_values.append(workload_data['system_stats']['write_iops_workload'])
    
    x = np.arange(len(workload_labels))
    width = 0.35
    
    # Create grouped bars
    ax.bar(x - width/2, read_iops_values, width, label='Read IOPS', color='#4090E0', 
          edgecolor='black', linewidth=0.8)
    ax.bar(x + width/2, write_iops_values, width, label='Write IOPS', color='#E05040', 
          edgecolor='black', linewidth=0.8)
    
    # Add labels and annotations
    ax.set_xlabel('Workload Type')
    ax.set_ylabel('IOPS')
    ax.set_title(f'Read/Write IOPS Comparison ({max_drive_count} Drives)')
    ax.set_xticks(x)
    ax.set_xticklabels(workload_labels, rotation=45, ha='right')
    ax.legend()
    
    # Add grid on y-axis only
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.pdf")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def generate_all_plots(data_file, output_dir="thesis_plots"):
    """Generate all plots from the storage simulation results JSON file"""
    # Set up output directory
    output_dir = setup_output_dir(output_dir)
    
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"Generating plots in {output_dir}...")
    
    # Generate standard plots
    plot_throughput_comparison(data, output_dir)
    plot_iops_comparison(data, output_dir)
    plot_latency_comparison(data, output_dir)
    plot_scaling_efficiency(data, output_dir)
    plot_backpressure_analysis(data, output_dir)
    plot_throughput_vs_latency(data, output_dir)
    plot_queue_utilization(data, output_dir)
    
    # Generate detailed workload plots
    organized_data, workloads, drive_counts = extract_data_by_workload_and_drives(data)
    plot_detailed_workload_scaling(data, workloads, output_dir)
    
    # Generate advanced analysis plots
    plot_comparative_analysis(data, output_dir)
    plot_throughput_to_latency_ratio(data, output_dir)
    plot_read_write_breakdown(data, output_dir)
    plot_heatmap_comparison(data, output_dir)
    plot_iops_comparison_by_operation(data, output_dir)
    
    print(f"Successfully generated all plots in {output_dir}")
    return output_dir

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python thesis_plotter.py <simulation_results.json> [output_directory]")
        sys.exit(1)
    
    data_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "thesis_plots"
    
    generate_all_plots(data_file, output_dir)