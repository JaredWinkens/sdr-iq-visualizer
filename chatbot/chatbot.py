from google import genai
from google.genai import types, chats
from typing import Callable, List, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
import plotly.io as pio
from config.settings import CHATBOT_CONFIG
from pathlib import Path
import numpy as np
import time
from sdr.streamer import sdr_streamer
from processing import classifier
gemini_client = genai.Client(api_key=CHATBOT_CONFIG['api_key'])

system_prompt = """
You are an interactive chatbot designed to analyze radio frequency data and provide useful insights based on that data.
You have access to various visualization graphs (time domain, frequency domain, waterfall, and constellation plots).
When the user asks what signal is present (e.g., "what signal am I seeing?"), first call the classify_signal tool to get the current classification and reasons, then summarize concisely.
When analyzing graphs, provide clear technical insights about the signal characteristics and reference the classifier result when available.
"""

class SignalAnalysis(BaseModel):
    stats: str = None
    include_graph: str | None = Field(description="which graph to include: 'td', 'fd', 'wf', 'con', or 'all'")

class Chatbot():

    def __init__(self, initial_history: list = [], model: str = CHATBOT_CONFIG['default_model']):
        
        self.history = initial_history
        
        self.model = model
        
        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.5,
            tools=[
                    self.analyze_signal,
                    self.classify_signal,
                    self.analyze_time_domain_graph,
                    self.analyze_freq_domain_graph,
                    self.analyze_waterfall_graph,
                    self.analyze_constellation_graph
                ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False
            ),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                mode="AUTO", #allowed_function_names=["get_current_temperature"]
                )
            )
        )

        self.chat = gemini_client.chats.create(
            model=self.model,
            history=self.history,
            config=self.config
        )

    def change_model(self, new_model: str = CHATBOT_CONFIG['default_model']):
        history = self.chat.get_history()
        self.chat = gemini_client.chats.create(
            model=new_model,
            history=history,
            config=self.config
        )
        self.model = new_model
    
    def clear_history(self):
        self.chat = gemini_client.chats.create(
            model=self.model,
            history=[],
            config=self.config
        )
    
    def send_message_with_context(self, message: str, include_graphs: list[tuple] = None):
        """
        Send a message with optional graph context.
        
        Args:
            message: The text message to send
            include_graphs: List of graph images in bytes to include
        """
        try:
            # Build the content parts
            content_parts = [message]
            
            # Add requested images
            if include_graphs:
                for name, image in include_graphs:
                    if image == None:
                        content_parts.append(f'The {name} is empty, make sure to connect the SDR and start streaming before analysis')
                    else:
                        content_parts.append(
                            types.Part.from_bytes(
                                data=image,
                                mime_type='image/png'
                            )
                        )
            
            response = self.chat.send_message(content_parts)
            return response.text

        except Exception as e:
            return f"An error occurred during interaction: {e}"
    
    def get_response(self, message: str, context: dict = {}):
        """
        Get response with automatic graph detection.
        Analyzes the message to determine if graphs should be included.
        """
        message_lower = message.lower()
        include_graphs: list[bytes] = []
        
        # Smart detection of which graphs to include
        if any(word in message_lower for word in ['time domain', 'time-domain', 'amplitude over time', 'waveform']):
            td_image = pio.to_image(context['td'], 'png') if context['td'] is not None else None
            include_graphs.append(('time domain plot', td_image))
        
        if any(word in message_lower for word in ['frequency', 'spectrum', 'freq domain', 'frequency-domain', 'spectral']):
            fd_image = pio.to_image(context['fd'], 'png') if context['fd'] is not None else None
            include_graphs.append(('frequency domain plot', fd_image))
        
        if any(word in message_lower for word in ['waterfall', 'spectrogram', 'time-frequency']):
            wf_image = pio.to_image(context['wf'], 'png') if context['wf'] is not None else None
            include_graphs.append(('waterfall plot', wf_image))
        
        if any(word in message_lower for word in ['constellation', 'iq plot', 'i/q', 'phase']):
            con_image = pio.to_image(context['con'], 'png') if context['con'] is not None else None
            include_graphs.append(('constellation plot', con_image))
        
        # Generic visualization requests include all graphs
        if any(word in message_lower for word in ['graph', 'plot', 'chart', 'visualization', 'all graphs', 'show me']):
            if not include_graphs:  # Only if no specific graph was mentioned
                td_image = pio.to_image(context['td'], 'png') if context['td'] is not None else None
                fd_image = pio.to_image(context['fd'], 'png') if context['fd'] is not None else None
                wf_image = pio.to_image(context['wf'], 'png') if context['wf'] is not None else None
                con_image = pio.to_image(context['con'], 'png') if context['con'] is not None else None

                include_graphs = [('time domain plot', td_image), ('frequency domain plot', fd_image), 
                                  ('waterfall plot', wf_image), ('constellation plot', con_image)]
        
        return self.send_message_with_context(message, include_graphs if include_graphs else None)
    
    def classify_signal(self) -> SignalAnalysis:
        """Classify the type of signal currently being received (advanced)."""
        print("TOOL CALL: classify_signal")
        data = sdr_streamer.get_latest_data()
        if data is None:
            return SignalAnalysis(
                stats="No SDR data available yet. Please start streaming.",
                include_graph=None
            )
        try:
            res = classifier.classify_signal_advanced(data['freqs'], data['power_db'])
            label = res.get('label', 'Unknown')
            conf = res.get('confidence', 0.0)
            feats = res.get('features', {})
            obw = feats.get('bandwidth_hz_20db', 0.0) / 1e6
            snr = feats.get('snr_db', 0.0)
            reasons = res.get('reasons', [])
            reason_text = "\n- " + "\n- ".join(reasons) if reasons else ""
            text = (
                f"Classification: {label} (conf {conf:.2f})\n"
                f"OBW20={obw:.2f} MHz, SNR={snr:.1f} dB{reason_text}"
            )
            return SignalAnalysis(
                stats=text,
                include_graph='fd'
            )
        except Exception as e:
            return SignalAnalysis(
                stats=f"Classification error: {e}",
                include_graph='fd'
            )

    def analyze_signal(self) -> SignalAnalysis:
        """Get detailed statistical information about the current signal sample."""
        print("TOOL CALL: analyze_signal")
        data = sdr_streamer.get_latest_data()
        
        return SignalAnalysis(
            stats=str(data),
            include_graph=None
        )
    
    def analyze_time_domain_graph(self) -> SignalAnalysis:
        """Analyze the time domain representation of the signal."""
        print("TOOL CALL: analyze_time_domain_graph")
        data = sdr_streamer.get_latest_data()

        return SignalAnalysis(
            stats=f"Time domain data - Sample rate: {data['sample_rate']} Hz, Center freq: {data['center_freq']} Hz",
            include_graph='td'
        )

    def analyze_freq_domain_graph(self) -> SignalAnalysis:
        """Analyze the frequency domain representation of the signal."""
        print("TOOL CALL: analyze_freq_domain_graph")
        data = sdr_streamer.get_latest_data()

        return SignalAnalysis(
            stats=f"Frequency domain data - Sample rate: {data['sample_rate']} Hz, Center freq: {data['center_freq']} Hz",
            include_graph='fd'
        )

    def analyze_waterfall_graph(self) -> SignalAnalysis:
        """Analyze the waterfall plot showing signal changes over time."""
        print("TOOL CALL: analyze_waterfall_graph")
        data = sdr_streamer.get_latest_data()

        return SignalAnalysis(
            stats=f"Waterfall data - Sample rate: {data['sample_rate']} Hz, Center freq: {data['center_freq']} Hz",
            include_graph='wf'
        )

    def analyze_constellation_graph(self) -> SignalAnalysis:
        """Analyze the constellation diagram of the signal."""
        print("TOOL CALL: analyze_constellation_graph")
        data = sdr_streamer.get_latest_data()

        return SignalAnalysis(
            stats=f"Constellation data - Sample rate: {data['sample_rate']} Hz, Center freq: {data['center_freq']} Hz",
            include_graph='con'
        )
