from google import genai
from google.genai import types, chats
from typing import Callable, List, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
from config.settings import CHATBOT_CONFIG
from pathlib import Path
import numpy as np

test_file = "/home/jared-winkens/sdr-iq-visualizer/data/1mhz-mcs0-chan43.sigmf-data"

# Import the sigmf library
try:
    from sigmf import SigMFFile
    from sigmf.sigmffile import fromfile
except ImportError:
    print("Error: sigmf library not found!")
    print("Please install it with: pip install sigmf")
    exit(1)

gemini_client = genai.Client(api_key=CHATBOT_CONFIG['api_key'])

system_prompt = """
You are an interactive chatbot design to analyze radio frequency data and provide useful insights based on that data.
"""

class SignalAnalysis(BaseModel):
    stats: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

class Chatbot():

    def __init__(self):
        self.page_state = None
        self.history = []

        self.chat = gemini_client.chats.create(
            model=CHATBOT_CONFIG['model'],
            history=self.history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5,
                tools=[self.analyze_signal],
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

    def get_response(self, message: str):
        
        try:
            response = self.chat.send_message(message)
            bot_response_text = response.text

        except Exception as e:
            bot_response_text = f"An error occurred during interaction: {e}"
            print(f"Error in get_response: {e}")
        
        return bot_response_text

    def _read_sigmf_with_library(self, sigmf_file_path=test_file):
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
    
    def analyze_signal(self) -> SignalAnalysis:
        """Get detailed information about a signal sample."""
        
        sigmf_file, data = self._read_sigmf_with_library()

        print("\n" + "="*60)
        print("SigMF RECORDING INFORMATION")
        print("="*60)
        
        stats = {}
        if data is not None:
            print(f"Number of samples: {len(data):,}")
            print(f"Data type: {data.dtype}")
            
            if np.iscomplexobj(data):
                print(f"Complex data:")
                print(f"  I (Real) - Min: {np.real(data).min():.6f}, Max: {np.real(data).max():.6f}")
                print(f"  Q (Imag) - Min: {np.imag(data).min():.6f}, Max: {np.imag(data).max():.6f}")
                print(f"  Magnitude - Min: {np.abs(data).min():.6f}, Max: {np.abs(data).max():.6f}")
                stats = {
                    "Number of samples": len(data),
                    "Data type": str(data.dtype),
                    "I (Real) - Min": str(np.real(data).min()), 
                    "I (Real) - Max": str(np.real(data).max()),
                    "Q (Imag) - Min": str(np.imag(data).min()), 
                    "Q (Imag) - Max": str(np.imag(data).max()),
                    "Magnitude - Min": str(np.abs(data).min()), 
                    "Magnitude - Max": str(np.abs(data).max()),
                }
            else:
                print(f"Real data - Min: {data.min():.6f}, Max: {data.max():.6f}")
                stats = {
                    "Number of samples": len(data),
                    "Data type": str(data.dtype),
                    "Real data - Min": str(np.abs(data).min()), 
                    "Real data - Max": str(np.abs(data).max()),
                }
        
        metadata = {}
        if sigmf_file is not None:
            print("\nMETADATA:")
            print("-" * 30)
            
            # Global metadata
            global_meta = sigmf_file.get_global_info()
            global_meta_list = []
            print("Global Information:")
            for key, value in global_meta.items():
                print(f"  {key}: {value}")
                global_meta_list.append(f"{key}: {value}")
            
            # Sample rate (if available)
            sample_rate = sigmf_file.get_global_field('core:sample_rate')
            sample_rate_clean = {}
            if sample_rate:
                print(f"\nSample Rate: {sample_rate:,} Hz")
                sample_rate_clean = f"{sample_rate:,} Hz"
                if data is not None:
                    duration = len(data) / sample_rate
                    print(f"Recording Duration: {duration:.3f} seconds")
            
            # Data type
            datatype = sigmf_file.get_global_field('core:datatype')
            datatype_clean = f'{datatype}'
            if datatype:
                print(f"Data Type: {datatype}")
            
            # Captures
            captures = sigmf_file.get_captures()
            capture_list = []
            if captures:
                print(f"\nCaptures ({len(captures)}):")
                for i, capture in enumerate(captures):
                    print(f"  Capture {i}:")
                    for key, value in capture.items():
                        print(f"    {key}: {value}")
                        capture_list.append({f"Capture {i}": f"{key}: {value}"})
            
            # Annotations
            annotations = sigmf_file.get_annotations()
            if annotations:
                print(f"\nAnnotations ({len(annotations)}):")
                for i, annotation in enumerate(annotations):
                    print(f"  Annotation {i}:")
                    for key, value in annotation.items():
                        print(f"    {key}: {value}")

            metadata = {
                'global_meta': global_meta_list,
                'sample_rate': sample_rate_clean,
                'datatype': datatype_clean,
                'captures': capture_list,
                'annotations': annotations
            }
        
        return SignalAnalysis(
            stats=stats,
            metadata=metadata
        )