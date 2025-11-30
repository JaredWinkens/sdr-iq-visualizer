import json
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SigMFWriter:
    """
    Write SDR live stream data to SigMF format.
    Creates both .sigmf-data (binary IQ samples) and .sigmf-meta (JSON metadata).
    """
    
    def __init__(self, base_filename, sample_rate, center_freq, datatype='cf32_le'):
        """
        Initialize SigMF writer.
        
        Args:
            base_filename: Base filename without extension (e.g., 'recording_2024')
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz
            datatype: Sample data type (cf32_le = complex float32 little-endian)
        """
        self.base_filename = base_filename
        self.data_filename = f"{base_filename}.sigmf-data"
        self.meta_filename = f"{base_filename}.sigmf-meta"
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.datatype = datatype
        self.sample_count = 0
        self.data_file = None
        self.start_time = None
        
    def start_recording(self):
        """Start recording to file"""
        try:
            self.data_file = open(self.data_filename, 'wb')
            self.start_time = datetime.utcnow().isoformat() + 'Z'
            self.sample_count = 0
            logger.info(f"Started SigMF recording: {self.data_filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    def write_samples(self, samples):
        """
        Write IQ samples to file.
        
        Args:
            samples: Complex numpy array of IQ samples
        """
        if self.data_file is None:
            raise RuntimeError("Recording not started. Call start_recording() first.")
        
        # Convert to complex64 (cf32) if not already
        if samples.dtype != np.complex64:
            samples = samples.astype(np.complex64)
        
        # Write binary data (little-endian)
        samples.tofile(self.data_file)
        self.sample_count += len(samples)
    
    def stop_recording(self, description="SDR Live Stream Recording"):
        """
        Stop recording and write metadata file.
        
        Args:
            description: Description for the recording
        """
        if self.data_file is None:
            logger.warning("No active recording to stop")
            return False
        
        # Close data file
        self.data_file.close()
        self.data_file = None
        
        # Create metadata
        metadata = self._create_metadata(description)
        
        # Write metadata file
        try:
            with open(self.meta_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Stopped recording. Wrote {self.sample_count} samples to {self.data_filename}")
            logger.info(f"Metadata saved to {self.meta_filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to write metadata: {e}")
            return False
    
    def _create_metadata(self, description):
        """Create SigMF metadata dictionary"""
        return {
            "global": {
                "core:datatype": self.datatype,
                "core:sample_rate": self.sample_rate,
                "core:version": "1.0.0",
                "core:description": description,
                "core:author": "SDR IQ Visualizer",
                "core:recorder": "PlutoSDR via pyadi-iio",
                "core:license": "CC0-1.0"
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:frequency": self.center_freq,
                    "core:datetime": self.start_time
                }
            ],
            "annotations": []
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_recording()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.data_file is not None:
            self.stop_recording()


class StreamingSigMFWriter(SigMFWriter):
    """
    Extended SigMF writer that integrates with SDRDataStreamer.
    Automatically records samples from the live stream.
    """
    
    def __init__(self, base_filename, sdr_streamer, max_samples=None):
        """
        Args:
            base_filename: Base filename without extension
            sdr_streamer: SDRDataStreamer instance
            max_samples: Maximum samples to record (None = unlimited)
        """
        super().__init__(
            base_filename=base_filename,
            sample_rate=sdr_streamer.sample_rate,
            center_freq=sdr_streamer.center_freq
        )
        self.sdr_streamer = sdr_streamer
        self.max_samples = max_samples
    
    def record_duration(self, duration_seconds):
        """
        Record for a specific duration.
        
        Args:
            duration_seconds: How long to record in seconds
        """
        target_samples = int(duration_seconds * self.sample_rate)
        return self.record_samples(target_samples)
    
    def record_samples(self, num_samples):
        """
        Record a specific number of samples from the live stream.
        
        Args:
            num_samples: Number of samples to record
        """
        if not self.sdr_streamer.is_connected():
            logger.error("SDR not connected")
            return False
        
        self.start_recording()
        
        try:
            while self.sample_count < num_samples:
                data = self.sdr_streamer.get_latest_data()
                if data and 'samples' in data:
                    samples_to_write = data['samples']
                    
                    # Trim if we'd exceed target
                    remaining = num_samples - self.sample_count
                    if len(samples_to_write) > remaining:
                        samples_to_write = samples_to_write[:remaining]
                    
                    self.write_samples(samples_to_write)
                    
                    if self.sample_count % (self.sample_rate) == 0:
                        logger.info(f"Recorded {self.sample_count}/{num_samples} samples "
                                  f"({self.sample_count/self.sample_rate:.1f}s)")
        except KeyboardInterrupt:
            logger.info("Recording interrupted by user")
        finally:
            self.stop_recording()
        
        return True
