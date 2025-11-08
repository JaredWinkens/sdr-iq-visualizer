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
        """Lightweight connection check (no property probing to avoid rx() interference)."""
        return self.sdr is not None and self.connected

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
        """Continuously read SDR data with backoff and minimal interference."""
        consecutive_errors = 0
        backoff = 0.1
        max_backoff = 1.6
        self.last_success_ts = None
        self.total_frames = 0

        while self.running:
            if not self.is_connected():
                logger.error("SDR connection lost before read; stopping stream.")
                self.running = False
                break
            try:
                samples = self.sdr.rx()  # Blocking hardware read
                consecutive_errors = 0
                backoff = 0.1

                # FFT & spectrum
                fft_data = np.fft.fftshift(np.fft.fft(samples))
                freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1 / self.sample_rate)) + self.center_freq
                power_db = 20 * np.log10(np.abs(fft_data) + 1e-12)

                plot_data = {
                    'time': time.time(),
                    'samples': samples,
                    'freqs': freqs,
                    'power_db': power_db,
                    'sample_rate': self.sample_rate,
                    'center_freq': self.center_freq
                }
                self._push(plot_data)
                self.last_success_ts = plot_data['time']
                self.total_frames += 1
            except OSError as e:
                consecutive_errors += 1
                # Fatal errors: 9 (bad descriptor), 10054 (connection reset)
                if e.errno in (9, 10054):
                    logger.error(f"Fatal OS error errno={e.errno}; terminating stream.")
                    self.connected = False
                    self.running = False
                    break
                logger.error(f"Non-fatal OS error reading SDR (errno={e.errno}): {e}")
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error reading SDR data: {e}")

            if consecutive_errors:
                backoff = min(backoff * 2, max_backoff)
                logger.warning(f"Read error #{consecutive_errors}; backoff {backoff:.2f}s")
                time.sleep(backoff)
                if consecutive_errors >= 3:
                    logger.error("Too many consecutive errors; stopping stream.")
                    self.running = False
                    self.connected = False
                    break

    def get_status(self):
        """Return a dict of streaming status metrics."""
        return {
            'connected': self.connected,
            'running': self.running,
            'queue_size': self.data_queue.qsize(),
            'last_success_age_ms': (time.time() - self.last_success_ts) * 1000 if getattr(self, 'last_success_ts', None) else None,
            'total_frames': getattr(self, 'total_frames', 0)
        }

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
