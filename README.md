# SDR IQ Data Analysis Platform

> A web-based real-time visualization and analysis platform for Software-Defined Radio (SDR) IQ data

## 🎯 Overview

This platform provides an accessible, browser-based interface for visualizing and analyzing raw IQ (In-phase/Quadrature) data from Software-Defined Radios, specifically designed for the ADALM-Pluto SDR. Built to bridge the gap between complex SDR tools and educational/research needs, it offers real-time signal visualization and AI-powered analysis without requiring specialized software installations.

## ✨ Features

### Core Visualizations
- **Time-domain plots** - Raw IQ sample visualization
- **Frequency spectrum analysis** - Real-time FFT processing and display
- **Constellation diagrams** - Interactive modulation characteristic visualization  
- **Waterfall plots** - Spectral activity over time with color-coded intensity

### AI-Powered Analysis
- **Interactive Chatbot** - Ask questions about the signal, powered by Google Gemini
- **Signal Classification** - Automatic detection of signal properties (bandwidth, SNR, etc.)
- **Contextual Insights** - Get explanations for observed signal characteristics

### Data Export 
- Download captured IQ data for offline analysis

## 🏗️ Architecture

The platform consists of four main components:

1. **Backend Signal Processing** - Communicates with SDR hardware, captures IQ samples, and performs digital signal processing
2. **Web Frontend** - Browser-based visualization interface with real-time data streaming
3. **Analysis Engine** - Advanced signal processing and classification utilities
4. **AI Assistant** - Integrated chatbot using Google Gemini for signal interpretation and user assistance

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
- Google API Key (for AI features)

### Quick Start (Local Python)
```bash
# Clone the repository
git clone https://github.com/yourusername/sdr-iq-visualizer.git
cd sdr-iq-visualizer

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GOOGLE_API_KEY="your_api_key_here"

# Run the dashboard
python main.py

# Open browser to http://localhost:8050
```

### Quick Start (Docker Compose)
```bash
# 1. Copy example environment file
cp .env.example .env
# 2. Edit .env to add GOOGLE_API_KEY for chatbot features
# 3. Build and start
docker compose up --build
# 4. Open browser
open http://localhost:8050  # (Windows: start http://localhost:8050)
```

To stop:
```bash
docker compose down
```

### Docker Image Details
- Base image: python:3.11-slim
- System libs: libiio (PlutoSDR support), libusb, curl (healthcheck)
- Healthcheck: GET / on port 8050
- Environment overrides:
	- `DASH_HOST` (default 0.0.0.0)
	- `DASH_PORT` (default 8050)
	- `DASH_DEBUG` (default false)
	- `GOOGLE_API_KEY` (required for chatbot)

If you need USB passthrough (running container directly with attached PlutoSDR instead of network IP), uncomment the `devices` section in `docker-compose.yml` and start Docker with appropriate privileges. Otherwise, the default network URI `ip:192.168.2.1` is used.

## 🛠️ Technology Stack

- **Backend**: Python, Flask, PyADI-IIO
- **Frontend**: Dash, Plotly, Bootstrap
- **AI/ML**: Google Gemini, Pandas
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
