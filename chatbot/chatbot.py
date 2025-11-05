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

gemini_client = genai.Client(api_key=CHATBOT_CONFIG['api_key'])

system_prompt = """
You are an interactive chatbot design to analyze radio frequency data and provide useful insights based on that data.
"""

class PageState(BaseModel):
    td_image: bytes = None
    fd_image: bytes = None
    wf_image: bytes = None
    con_image: bytes = None

class SignalAnalysis(BaseModel):
    stats: str = None

class SignalAnalysisImage(BaseModel):
    stats: str = None
    image_desc: str = None

class Chatbot():

    def __init__(self):
        self.page_state: PageState = PageState()
        self.history = []

        self.chat = gemini_client.chats.create(
            model=CHATBOT_CONFIG['model'],
            history=self.history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5,
                tools=[
                        self.analyze_signal,
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
        )
    
    def update_state(
        self,
        time_domain_graph, 
        freq_domain_graph, 
        waterfall_graph,
        constellation_plot,
    ):
        print("State Updated")
        td_image = pio.to_image(time_domain_graph, 'png')
        fd_image = pio.to_image(freq_domain_graph, 'png')
        wf_image = pio.to_image(waterfall_graph, 'png')
        con_image = pio.to_image(constellation_plot, 'png')

        self.page_state.td_image = td_image
        self.page_state.fd_image = fd_image
        self.page_state.wf_image = wf_image
        self.page_state.con_image = con_image

    def get_response(self, message: str):
        
        try:
            response = self.chat.send_message(message)
            bot_response_text = response.text

        except Exception as e:
            bot_response_text = f"An error occurred during interaction: {e}"
            print(f"Error in get_response: {e}")
        
        return bot_response_text
    
    def analyze_signal(self) -> SignalAnalysis:
        """Get detailed information about a signal sample."""
        print("TOOL CALL: analyze_signal")
        data = sdr_streamer.get_latest_data()
        print(data)

        time = data['time']
        samples = data['samples']
        freqs = data['freqs']
        power_db = data['power_db']
        sample_rate = data['sample_rate']
        center_freq = data['center_freq']
        
        return SignalAnalysis(
            stats=str(data),
        )
    
    def analyze_time_domain_graph(self) -> SignalAnalysisImage:
        print("TOOL CALL: analyze_time_domain_graph")
        image = self.page_state.td_image
        data = sdr_streamer.get_latest_data()

        response = gemini_client.models.generate_content(
            model=CHATBOT_CONFIG['model'],
            contents=[
                types.Part.from_bytes(
                    data=image,
                    mime_type='image/png'
                ),
                "Describe this image in as much detail as possible"
            ]
        )
        image_desc = response.text
        print(image_desc)

        return SignalAnalysis(
            stats=str(data),
            image_desc=image_desc
        )

    def analyze_freq_domain_graph(self) -> SignalAnalysisImage:
        print("TOOL CALL: analyze_freq_domain_graph")
        image = self.page_state.fd_image
        data = sdr_streamer.get_latest_data()

        response = gemini_client.models.generate_content(
            model=CHATBOT_CONFIG['model'],
            contents=[
                types.Part.from_bytes(
                    data=image,
                    mime_type='image/png'
                ),
                "Describe this image in as much detail as possible"
            ]
        )
        image_desc = response.text
        print(image_desc)

        return SignalAnalysis(
            stats=str(data),
            image_desc=image_desc
        )

    def analyze_waterfall_graph(self) -> SignalAnalysisImage:
        print("TOOL CALL: analyze_waterfall_graph")
        image = self.page_state.wf_image
        data = sdr_streamer.get_latest_data()

        response = gemini_client.models.generate_content(
            model=CHATBOT_CONFIG['model'],
            contents=[
                types.Part.from_bytes(
                    data=image,
                    mime_type='image/png'
                ),
                "Describe this image in as much detail as possible"
            ]
        )
        image_desc = response.text
        print(image_desc)

        return SignalAnalysisImage(
            stats=str(data),
            image_desc=image_desc
        )

    def analyze_constellation_graph(self) -> SignalAnalysisImage:
        print("TOOL CALL: analyze_constellation_graph")
        image = self.page_state.con_image
        data = sdr_streamer.get_latest_data()

        response = gemini_client.models.generate_content(
            model=CHATBOT_CONFIG['model'],
            contents=[
                types.Part.from_bytes(
                    data=image,
                    mime_type='image/png'
                ),
                "Describe this image in as much detail as possible"
            ]
        )
        image_desc = response.text
        print(image_desc)

        return SignalAnalysisImage(
            stats=str(data),
            image_desc=image_desc
        )
