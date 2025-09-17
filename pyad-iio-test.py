#!/usr/bin/env python3
"""
ADALM Pluto SDR Data Collection Example using pyadi-iio
This script demonstrates how to configure and collect data from the Pluto SDR
"""

import numpy as np
import matplotlib.pyplot as plt
import adi
import time

def basic_rx_example():
    """Basic receive example - capture IQ samples"""
    
    # Create radio object - replace URI with your Pluto's IP if using Ethernet
    # For USB connection, use default URI or "ip:192.168.2.1"
    # For Ethernet/WiFi, use "ip:your.pluto.ip.address"
    sdr = adi.Pluto("ip:192.168.2.1")
    
    # Configure Rx parameters
    sdr.sample_rate = int(2.4e6)  # 2.4 MHz sample rate
    sdr.rx_rf_bandwidth = int(2e6)  # 2 MHz RF bandwidth
    sdr.rx_lo = int(915e6)  # 915 MHz center frequency
    sdr.gain_control_mode_chan0 = 'manual'  # or 'slow_attack', 'fast_attack'
    sdr.rx_hardwaregain_chan0 = 50  # dB, range: -3 to 71
    
    # Configure buffer size
    sdr.rx_buffer_size = 2**12  # 4096 samples
    
    print(f"Sample Rate: {sdr.sample_rate/1e6:.1f} MHz")
    print(f"Center Frequency: {sdr.rx_lo/1e6:.1f} MHz")
    print(f"RF Bandwidth: {sdr.rx_rf_bandwidth/1e6:.1f} MHz")
    print(f"Hardware Gain: {sdr.rx_hardwaregain_chan0} dB")
    
    # Collect samples
    print("Collecting samples...")
    samples = sdr.rx()  # Returns complex IQ samples
    
    print(f"Collected {len(samples)} samples")
    print(f"Sample type: {type(samples[0])}")
    
    return samples, sdr

def spectrum_analysis_example():
    """Example showing spectrum analysis of received data"""
    
    sdr = adi.Pluto()
    
    # Configure for wider bandwidth capture
    sdr.sample_rate = int(20e6)  # 20 MHz sample rate
    sdr.rx_rf_bandwidth = int(18e6)  # 18 MHz RF bandwidth
    sdr.rx_lo = int(2.4e9)  # 2.4 GHz center frequency (WiFi band)
    sdr.gain_control_mode_chan0 = 'fast_attack'
    sdr.rx_buffer_size = 2**16  # 65536 samples for better frequency resolution
    
    print("Capturing spectrum data...")
    samples = sdr.rx()
    
    # Calculate power spectrum
    fft_samples = np.fft.fftshift(np.fft.fft(samples))
    fft_db = 20 * np.log10(np.abs(fft_samples))
    
    # Create frequency axis
    freq_axis = np.linspace(-sdr.sample_rate/2, sdr.sample_rate/2, len(fft_samples))
    freq_axis = (freq_axis + sdr.rx_lo) / 1e6  # Convert to MHz
    
    return freq_axis, fft_db, samples

def continuous_capture_example():
    """Example of continuous data capture"""
    
    sdr = adi.Pluto()
    
    # Configure parameters
    sdr.sample_rate = int(1e6)  # 1 MHz
    sdr.rx_lo = int(433e6)  # 433 MHz (ISM band)
    sdr.rx_rf_bandwidth = int(1e6)
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 30
    sdr.rx_buffer_size = 2**10  # 1024 samples per buffer
    
    print("Starting continuous capture (10 iterations)...")
    print("Press Ctrl+C to stop early")
    
    all_samples = []
    
    try:
        for i in range(10):
            samples = sdr.rx()
            all_samples.extend(samples)
            
            # Calculate some basic statistics
            power_db = 20 * np.log10(np.mean(np.abs(samples)))
            print(f"Buffer {i+1}: {len(samples)} samples, Avg Power: {power_db:.1f} dB")
            
            time.sleep(0.1)  # Small delay between captures
            
    except KeyboardInterrupt:
        print("Capture stopped by user")
    
    print(f"Total samples collected: {len(all_samples)}")
    return np.array(all_samples)

def transmit_receive_example():
    """Example showing both transmit and receive"""
    
    sdr = adi.Pluto()
    
    # Configure Tx parameters
    sdr.sample_rate = int(1e6)
    sdr.tx_rf_bandwidth = int(1e6)
    sdr.tx_lo = int(915e6)
    sdr.tx_hardwaregain_chan0 = -30  # dB, range: -89.75 to 0
    
    # Configure Rx parameters
    sdr.rx_rf_bandwidth = int(1e6)
    sdr.rx_lo = int(915e6)
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 20
    sdr.rx_buffer_size = 2**12
    
    # Generate a test signal (complex sinusoid)
    N = 2**12
    fs = int(sdr.sample_rate)
    fc = int(100e3)  # 100 kHz offset from center
    ts = 1 / float(fs)
    t = np.arange(0, N * ts, ts)
    i_signal = np.cos(2 * np.pi * fc * t) * 2**14
    q_signal = np.sin(2 * np.pi * fc * t) * 2**14
    tx_signal = i_signal + 1j * q_signal
    
    # Configure buffer sizes
    sdr.tx_buffer_size = len(tx_signal)
    
    print("Transmitting test signal...")
    sdr.tx(tx_signal)  # Start transmitting
    
    time.sleep(0.1)  # Small delay
    
    print("Receiving...")
    rx_samples = sdr.rx()  # Receive samples
    
    sdr.tx_destroy_buffer()  # Stop transmitting
    
    return tx_signal, rx_samples

def plot_results(samples, sdr_obj, title="IQ Samples"):
    """Helper function to plot IQ data"""
    
    plt.figure(figsize=(12, 8))
    
    # Time domain plot
    plt.subplot(2, 2, 1)
    plt.plot(np.real(samples[:1000]))
    plt.title('I (Real) - Time Domain')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(np.imag(samples[:1000]))
    plt.title('Q (Imaginary) - Time Domain')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Frequency domain plot
    plt.subplot(2, 2, 3)
    fft_samples = np.fft.fftshift(np.fft.fft(samples))
    fft_db = 20 * np.log10(np.abs(fft_samples))
    freq_axis = np.linspace(-sdr_obj.sample_rate/2, sdr_obj.sample_rate/2, len(fft_samples))
    plt.plot(freq_axis/1e6, fft_db)
    plt.title('Power Spectrum')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.grid(True)
    
    # Constellation plot
    plt.subplot(2, 2, 4)
    plt.scatter(np.real(samples[::10]), np.imag(samples[::10]), alpha=0.5, s=1)
    plt.title('IQ Constellation')
    plt.xlabel('I (Real)')
    plt.ylabel('Q (Imaginary)')
    plt.grid(True)
    plt.axis('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run examples"""
    
    print("ADALM Pluto SDR Examples using pyadi-iio")
    print("=" * 50)
    
    try:
        # Example 1: Basic receive
        print("\n1. Basic Receive Example:")
        samples, sdr = basic_rx_example()
        plot_results(samples, sdr, "Basic Receive - 915 MHz")
        
        # Example 2: Spectrum analysis
        print("\n2. Spectrum Analysis Example:")
        freq_axis, fft_db, spectrum_samples = spectrum_analysis_example()
        
        plt.figure(figsize=(12, 6))
        plt.plot(freq_axis, fft_db)
        plt.title('Spectrum Analysis - 2.4 GHz Band')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (dB)')
        plt.grid(True)
        plt.show()
        
        # Example 3: Continuous capture
        print("\n3. Continuous Capture Example:")
        continuous_samples = continuous_capture_example()
        
        # Example 4: Transmit and receive
        print("\n4. Transmit/Receive Example:")
        tx_sig, rx_sig = transmit_receive_example()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(np.real(tx_sig[:100]))
        plt.title('Transmitted Signal (I)')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(np.real(rx_sig[:100]))
        plt.title('Received Signal (I)')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the Pluto is connected and recognized")
        print("2. Check the URI (try 'ip:192.168.2.1' for USB or your IP for Ethernet)")
        print("3. Install required packages: pip install pyadi-iio numpy matplotlib")
        print("4. Make sure no other software is using the Pluto")

if __name__ == "__main__":
    main()