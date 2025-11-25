#!/usr/bin/env python3
"""
Example script demonstrating how to record live SDR stream to SigMF format.
"""

import sys
import time
from datetime import datetime
from sdr.streamer import sdr_streamer
from utils.sigmf_writer import SigMFWriter, StreamingSigMFWriter
from utils.logger import setup_logging

# Setup logging
setup_logging()

def record_basic_example():
    """Basic example: manually control recording"""
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"data/recording_{timestamp}"
    
    # Create writer
    writer = SigMFWriter(
        base_filename=base_filename,
        sample_rate=sdr_streamer.sample_rate,
        center_freq=sdr_streamer.center_freq
    )
    
    # Connect and start streaming
    if not sdr_streamer.connect():
        print("Failed to connect to SDR")
        return
    
    sdr_streamer.start_streaming()
    time.sleep(1)  # Let stream stabilize
    
    # Start recording
    writer.start_recording()
    
    try:
        print("Recording... Press Ctrl+C to stop")
        samples_written = 0
        
        while True:
            data = sdr_streamer.get_latest_data()
            if data and 'samples' in data:
                writer.write_samples(data['samples'])
                samples_written += len(data['samples'])
                
                # Print progress every second
                if samples_written % sdr_streamer.sample_rate < len(data['samples']):
                    duration = samples_written / sdr_streamer.sample_rate
                    print(f"Recorded {duration:.1f} seconds ({samples_written} samples)")
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        writer.stop_recording(description="Manual SDR recording")
        sdr_streamer.stop_streaming()
        print(f"Recording saved to {writer.data_filename}")


def record_streaming_example():
    """Simplified example using StreamingSigMFWriter"""
    # Connect to SDR
    if not sdr_streamer.connect():
        print("Failed to connect to SDR")
        return
    
    sdr_streamer.start_streaming()
    time.sleep(1)  # Let stream stabilize
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"data/recording_{timestamp}"
    
    # Create streaming writer
    writer = StreamingSigMFWriter(
        base_filename=base_filename,
        sdr_streamer=sdr_streamer
    )
    
    # Record for 10 seconds
    print("Recording 10 seconds of data...")
    writer.record_duration(duration_seconds=10)
    
    sdr_streamer.stop_streaming()
    print(f"Recording saved to {writer.data_filename}")


def record_with_context_manager():
    """Example using context manager for automatic cleanup"""
    if not sdr_streamer.connect():
        print("Failed to connect to SDR")
        return
    
    sdr_streamer.start_streaming()
    time.sleep(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"data/recording_{timestamp}"
    
    # Context manager automatically starts and stops recording
    with SigMFWriter(base_filename, sdr_streamer.sample_rate, sdr_streamer.center_freq) as writer:
        print("Recording 5 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 5.0:
            data = sdr_streamer.get_latest_data()
            if data and 'samples' in data:
                writer.write_samples(data['samples'])
            time.sleep(0.01)
    
    sdr_streamer.stop_streaming()
    print("Recording complete!")


if __name__ == "__main__":
    print("SDR to SigMF Recording Examples")
    print("=" * 50)
    print("1. Basic manual recording")
    print("2. Streaming writer (10 seconds)")
    print("3. Context manager example (5 seconds)")
    print()
    
    choice = input("Select example (1-3): ").strip()
    
    try:
        if choice == "1":
            record_basic_example()
        elif choice == "2":
            record_streaming_example()
        elif choice == "3":
            record_with_context_manager()
        else:
            print("Invalid choice")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
