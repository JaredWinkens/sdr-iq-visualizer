from google import genai
from google.genai import types, chats
from typing import Callable, List, Dict
from pydantic import BaseModel, Field
import pandas as pd
from config.settings import CHATBOT_CONFIG

gemini_client = genai.Client(api_key=CHATBOT_CONFIG['api_key'])

system_prompt = """
You are an interactive chatbot design to analyze radio frequency data and provide useful insights based on that data.
"""

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
                # tools=[],
                # automatic_function_calling=types.AutomaticFunctionCallingConfig(
                #     disable=False
                # ),
                # tool_config=types.ToolConfig(
                #     function_calling_config=types.FunctionCallingConfig(
                #     mode="AUTO", #allowed_function_names=["get_current_temperature"]
                #     )
                # )
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

