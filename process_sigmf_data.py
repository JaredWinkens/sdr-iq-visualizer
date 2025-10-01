#!/usr/bin/env python3
"""
Simple Python script to read SigMF-data files using the official sigmf library.
This script uses the sigmf-python library from https://github.com/sigmf/sigmf-python
"""

import numpy as np
import argparse
from pathlib import Path

# Import the sigmf library
try:
    from sigmf import SigMFFile
    from sigmf.sigmffile import fromfile
except ImportError:
    print("Error: sigmf library not found!")
    print("Please install it with: pip install sigmf")
    exit(1)

def read_sigmf_with_library(sigmf_file_path):
    """
    Read a SigMF recording using the official sigmf library.
    
    Args:
        sigmf_file_path (str): Path to either .sigmf-data or .sigmf-meta file
    
    Returns:
        tuple: (sigmf_file_object, data_array)
    """
    
    # Convert to Path object
    file_path = Path(sigmf_file_path)
    
    # The sigmf library can work with either the .sigmf-data or .sigmf-meta file
    # It will automatically find the corresponding files
    if file_path.suffix == '.sigmf-data':
        # Remove the .sigmf-data extension to get the base name
        base_path = str(file_path).replace('.sigmf-data', '')
    elif file_path.suffix == '.sigmf-meta':
        # Remove the .sigmf-meta extension to get the base name
        base_path = str(file_path).replace('.sigmf-meta', '')
    else:
        # Assume it's a base path without extension
        base_path = str(file_path)
    
    try:
        # Load the SigMF file
        print(f"Loading SigMF recording from: {base_path}")
        sigmf_file = fromfile(base_path)
        
        # Read the data
        data = sigmf_file.read_samples()
        
        print(f"Successfully loaded SigMF recording")
        print(f"Number of samples: {len(data)}")
        print(f"Data type: {data.dtype}")
        
        return sigmf_file, data
        
    except Exception as e:
        print(f"Error loading SigMF file: {e}")
        return None, None

def print_sigmf_info(sigmf_file, data):
    """Print detailed information about the SigMF recording."""
    
    print("\n" + "="*60)
    print("SigMF RECORDING INFORMATION")
    print("="*60)
    
    if data is not None:
        print(f"Number of samples: {len(data):,}")
        print(f"Data type: {data.dtype}")
        print(f"Data shape: {data.shape}")
        
        if np.iscomplexobj(data):
            print(f"Complex data:")
            print(f"  I (Real) - Min: {np.real(data).min():.6f}, Max: {np.real(data).max():.6f}")
            print(f"  Q (Imag) - Min: {np.imag(data).min():.6f}, Max: {np.imag(data).max():.6f}")
            print(f"  Magnitude - Min: {np.abs(data).min():.6f}, Max: {np.abs(data).max():.6f}")
        else:
            print(f"Real data - Min: {data.min():.6f}, Max: {data.max():.6f}")
    
    if sigmf_file is not None:
        print("\nMETADATA:")
        print("-" * 30)
        
        # Global metadata
        global_meta = sigmf_file.get_global_info()
        print("Global Information:")
        for key, value in global_meta.items():
            print(f"  {key}: {value}")
        
        # Sample rate (if available)
        sample_rate = sigmf_file.get_global_field('core:sample_rate')
        if sample_rate:
            print(f"\nSample Rate: {sample_rate:,} Hz")
            if data is not None:
                duration = len(data) / sample_rate
                print(f"Recording Duration: {duration:.3f} seconds")
        
        # Data type
        datatype = sigmf_file.get_global_field('core:datatype')
        if datatype:
            print(f"Data Type: {datatype}")
        
        # Captures
        captures = sigmf_file.get_captures()
        if captures:
            print(f"\nCaptures ({len(captures)}):")
            for i, capture in enumerate(captures):
                print(f"  Capture {i}:")
                for key, value in capture.items():
                    print(f"    {key}: {value}")
        
        # Annotations
        annotations = sigmf_file.get_annotations()
        if annotations:
            print(f"\nAnnotations ({len(annotations)}):")
            for i, annotation in enumerate(annotations):
                print(f"  Annotation {i}:")
                for key, value in annotation.items():
                    print(f"    {key}: {value}")

def get_frequency_info(sigmf_file):
    """Extract frequency information from the SigMF file."""
    freq_info = {}
    
    # Get center frequency from captures
    captures = sigmf_file.get_captures()
    if captures and len(captures) > 0:
        center_freq = captures[0].get('core:frequency')
        if center_freq:
            freq_info['center_frequency'] = center_freq
    
    # Get sample rate
    sample_rate = sigmf_file.get_global_field('core:sample_rate')
    if sample_rate:
        freq_info['sample_rate'] = sample_rate
        freq_info['bandwidth'] = sample_rate
        
        if 'center_frequency' in freq_info:
            freq_info['start_frequency'] = center_freq - sample_rate / 2
            freq_info['end_frequency'] = center_freq + sample_rate / 2
    
    return freq_info

def plot_sigmf_data(sigmf_file, data, max_samples=10000):
    """Plot the SigMF data with proper frequency axis."""
    try:
        import matplotlib.pyplot as plt
        
        # Limit samples for plotting performance
        samples_to_plot = min(max_samples, len(data))
        plot_data = data[:samples_to_plot]
        
        # Get frequency information
        freq_info = get_frequency_info(sigmf_file)
        sample_rate = freq_info.get('sample_rate', 1.0)
        center_freq = freq_info.get('center_frequency', 0.0)
        
        # Create time axis
        time_axis = np.arange(samples_to_plot) / sample_rate
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SigMF Recording Visualization', fontsize=16)
        
        if np.iscomplexobj(plot_data):
            # Time domain - I and Q
            axes[0, 0].plot(time_axis, np.real(plot_data), label='I (Real)', alpha=0.7)
            axes[0, 0].plot(time_axis, np.imag(plot_data), label='Q (Imag)', alpha=0.7)
            axes[0, 0].set_title('Time Domain - I & Q Components')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Constellation plot
            axes[0, 1].scatter(np.real(plot_data), np.imag(plot_data), 
                              alpha=0.5, s=1, c='blue')
            axes[0, 1].set_title('I-Q Constellation')
            axes[0, 1].set_xlabel('I (Real)')
            axes[0, 1].set_ylabel('Q (Imaginary)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axis('equal')
            
            # Power spectrum
            freqs, psd = plt.psd(plot_data, NFFT=1024, Fs=sample_rate, 
                                Fc=center_freq, return_line=False)
            axes[1, 0].plot(freqs/1e6, psd)
            axes[1, 0].set_title('Power Spectral Density')
            axes[1, 0].set_xlabel('Frequency (MHz)')
            axes[1, 0].set_ylabel('Power (dB/Hz)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Magnitude over time
            magnitude = np.abs(plot_data)
            axes[1, 1].plot(time_axis, magnitude)
            axes[1, 1].set_title('Magnitude over Time')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Magnitude')
            axes[1, 1].grid(True, alpha=0.3)
            
        else:
            # Real data
            axes[0, 0].plot(time_axis, plot_data)
            axes[0, 0].set_title('Time Domain Signal')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Frequency domain
            freqs, psd = plt.psd(plot_data, NFFT=1024, Fs=sample_rate, 
                                Fc=center_freq, return_line=False)
            axes[0, 1].plot(freqs/1e6, psd)
            axes[0, 1].set_title('Power Spectral Density')
            axes[0, 1].set_xlabel('Frequency (MHz)')
            axes[0, 1].set_ylabel('Power (dB/Hz)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Hide unused subplots for real data
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print frequency information
        if freq_info:
            print(f"\nFrequency Information:")
            for key, value in freq_info.items():
                if 'frequency' in key:
                    print(f"  {key}: {value/1e6:.3f} MHz")
                else:
                    print(f"  {key}: {value:,}")
        
    except ImportError:
        print("matplotlib not available for plotting")
    except Exception as e:
        print(f"Error plotting data: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Read SigMF files using the official sigmf library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python read_sigmf_with_library.py recording.sigmf-data
            python read_sigmf_with_library.py recording.sigmf-meta --plot
            python read_sigmf_with_library.py recording --plot --samples 5000
        """
    )
    
    parser.add_argument('sigmf_file', 
                       help='Path to SigMF file (.sigmf-data, .sigmf-meta, or base name)')
    parser.add_argument('--plot', action='store_true', 
                       help='Plot the signal data')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Maximum number of samples to plot (default: 10000)')
    
    args = parser.parse_args()
    
    # Read the SigMF file
    sigmf_file, data = read_sigmf_with_library(args.sigmf_file)
    
    if sigmf_file is not None and data is not None:
        # Print information
        print_sigmf_info(sigmf_file, data)
        
        # Optional plotting
        if args.plot:
            plot_sigmf_data(sigmf_file, data, args.samples)
    
    return sigmf_file, data

if __name__ == "__main__":
    main()