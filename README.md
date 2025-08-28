# SDR IQ Data Visualization Platform

> A web-based real-time visualization and analysis platform for Software-Defined Radio (SDR) IQ data

## 🎯 Overview

This platform provides an accessible, browser-based interface for visualizing and analyzing raw IQ (In-phase/Quadrature) data from Software-Defined Radios, specifically designed for the ADALM-Pluto SDR. Built to bridge the gap between complex SDR tools and educational/research needs, it offers real-time signal visualization without requiring specialized software installations.

## ✨ Features

### Core Visualizations
- **Time-domain plots** - Raw IQ sample visualization
- **Frequency spectrum analysis** - Real-time FFT processing and display
- **Constellation diagrams** - Interactive modulation characteristic visualization  
- **Waterfall plots** - Spectral activity over time with color-coded intensity

### Interactive Controls
- Real-time frequency tuning
- Adjustable bandwidth and gain settings
- Synchronized multi-view dashboard
- Device-agnostic web interface

### Extended Analysis (Planned)
- Basic demodulation modes (AM/FM)
- Machine learning-based modulation classification
- Data export for offline analysis
- Integration APIs for external tools

## 🏗️ Architecture

The platform consists of three main components:

1. **Backend Signal Processing** - Communicates with SDR hardware, captures IQ samples, and performs digital signal processing
2. **Web Frontend** - Browser-based visualization interface with real-time data streaming
3. **Analysis Engine** - Advanced signal processing and optional ML-based classification

## 🎓 Educational Impact

Designed specifically for:
- **Students** learning communication systems and DSP concepts
- **Researchers** needing flexible signal exploration tools  
- **Educators** demonstrating abstract RF concepts interactively
- **Hobbyists** exploring the radio frequency spectrum

## 🚀 Getting Started

### Prerequisites
- ADALM-Pluto SDR device
- Python 3.8+ (backend)
- Modern web browser (frontend)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/sdr-iq-visualizer.git
cd sdr-iq-visualizer

# Install dependencies
pip install -r requirements.txt

# Connect your ADALM-Pluto and run
python main.py

# Open browser to localhost:8080
```

## 🛠️ Technology Stack

- **Backend**: Python, NumPy, SciPy, PyADI-IIO
- **Frontend**: HTML5, JavaScript, WebGL/Canvas, WebSockets
- **Signal Processing**: FFT algorithms, digital filtering
- **Hardware**: ADALM-Pluto SDR

## 📊 Use Cases

- Visualizing Wi-Fi, Bluetooth, and cellular signals
- Educational demonstrations of modulation schemes
- Research into signal characteristics and interference
- Real-time spectrum monitoring and analysis

## 🤝 Contributing

This is a Fall 2025 capstone project by Sawyer Davis and Jared Winkens. Contributions, suggestions, and feedback are welcome!

---

*Making wireless signals visible and accessible through web-based visualization*
