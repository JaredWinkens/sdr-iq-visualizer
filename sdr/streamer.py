import adi
import numpy as np
import queue, threading, time, logging

logger = logging.getLogger(__name__)

class SDRDataStreamer:
    def __init__(self, uri="ip:192.168.2.1", sample_rate=1_000_000,
                 center_freq=2_400_000_000, rx_lo=2_400_000_000,
                 rx_rf_bandwidth=4_000_000, rx_buffer_size=2**12):
        self.uri = uri
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.rx_lo = rx_lo
        self.rx_rf_bandwidth = rx_rf_bandwidth
        self.rx_buffer_size = rx_buffer_size
        self.sdr = None
        self.data_queue = queue.Queue(maxsize=100)
        self.running = False
        self.thread = None
        self.connected = False

    def connect(self):
        """Connect to the SDR device"""
        try:
            # Try to connect to PlutoSDR first, fallback to other devices
            self.sdr = adi.Pluto(uri=self.uri)
            
            # Configure RX parameters
            self.sdr.sample_rate = int(self.sample_rate)
            self.sdr.rx_rf_bandwidth = int(self.rx_rf_bandwidth)
            self.sdr.rx_lo = int(self.rx_lo)
            self.sdr.rx_buffer_size = self.rx_buffer_size
            
            logger.info(f"Connected to SDR at {self.uri}")
            logger.info(f"Sample Rate: {self.sdr.sample_rate}")
            logger.info(f"Center Frequency: {self.sdr.rx_lo}")
            logger.info(f"Bandwidth: {self.sdr.rx_rf_bandwidth}")
            
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to SDR: {e}")
            self.connected = False
            return False
    
    def is_connected(self):
        """Check if SDR connection is still valid"""
        if not self.sdr or not self.connected:
            return False
        
        try:
            # Try to access a simple property to test connection
            _ = self.sdr.sample_rate
            return True
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            self.connected = False
            return False

    def start_streaming(self):
        if not self.sdr:
            logger.error("SDR not connected")
            return False
        self.running = True
        self.thread = threading.Thread(target=self._stream_data, daemon=True)
        self.thread.start()
        return True

    def stop_streaming(self):
        """Stop streaming data"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("Stopped SDR data streaming")
    
    def reconnect(self):
        """Attempt to reconnect to the SDR device"""
        logger.info("Attempting to reconnect to SDR...")
        if self.sdr:
            try:
                # Clean up existing connection
                del self.sdr
                self.sdr = None
            except:
                pass
        
        # Attempt to reconnect
        return self.connect()

    def _stream_data(self):
        """Internal method to continuously read SDR data"""
        while self.running:
            try:
                # Check if connection is still valid before attempting to read
                if not self.is_connected():
                    logger.error("SDR connection lost. Stopping data stream.")
                    self.running = False
                    break
                
                # Read samples from SDR
                samples = self.sdr.rx()
                
                # Calculate FFT for frequency domain representation
                fft_data = np.fft.fftshift(np.fft.fft(samples))
                freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/self.sample_rate))
                freqs = freqs + self.center_freq  # Shift to actual frequencies
                
                # Calculate power spectrum (dB)
                power_db = 20 * np.log10(np.abs(fft_data) + 1e-10)
                
                # Prepare data for plotting
                plot_data = {
                    'time': time.time(),
                    'samples': samples,
                    'freqs': freqs,
                    'power_db': power_db,
                    'sample_rate': self.sample_rate,
                    'center_freq': self.center_freq
                }
                
                self._push(plot_data)
                        
            except OSError as e:
                if e.errno == 9:  # Bad file descriptor
                    logger.error("Bad file descriptor error - SDR connection lost")
                    self.connected = False
                    self.running = False
                    break
                else:
                    logger.error(f"OS Error reading SDR data: {e}")
                    time.sleep(0.1)  # Brief pause before retrying
            except Exception as e:
                logger.error(f"Error reading SDR data: {e}")
                time.sleep(0.1)  # Brief pause before retrying

    def _push(self, data):
        try:
            self.data_queue.put_nowait(data)
        except queue.Full:
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(data)
            except queue.Empty:
                pass

    def get_latest_data(self):
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

# Shared global instance
sdr_streamer = SDRDataStreamer()
