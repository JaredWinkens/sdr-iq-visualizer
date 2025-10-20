import dash
from dash import dcc, html, State, Input, Output, callback
import dash_bootstrap_components as dbc

def load_mini_chatbot_layout(initial_prompt: str):
    return html.Div([
        # Chat Widget
        dcc.Store(id='chat-mini-message-store', data=[]),
        dcc.Store(id='chat-mini-loading-store', data=False),
        dcc.Store(id='chat-mini-widget-minimized', data=False),
        
        html.Div([
            # Chat header with minimize button
            html.Div([
                html.Div([
                    html.Span("💬"),
                    html.Span("Chat Assistant")
                ], className="chat-mini-title"),
                html.Button("−", id="chat-mini-minimize-btn", className="chat-mini-minimize-btn")
            ], className="chat-mini-header", id="chat-mini-header"),
            
            # Chat content (hidden when minimized)
            html.Div([
                # Welcome message
                html.Div([
                    initial_prompt,
                ], className="chat-mini-welcome-message", id='chat-mini-welcome-message'),
                
                # Messages container
                html.Div(id="chat-mini-messages-container")
            ], className="chat-mini-messages", id="chat-mini-messages"),
            
            # Input container (hidden when minimized)
            html.Div([
                html.Div([
                    dcc.Input(
                        id="chat-mini-message-input",
                        placeholder="Type your message...",
                        type="text",
                        className="chat-mini-input",
                        autoComplete="off"
                    ),
                    html.Button("➤", id="chat-mini-send-button", className="chat-mini-send-button", n_clicks=0)
                ], className="chat-mini-input-group")
            ], className="chat-mini-input-container", id="chat-mini-input-container")
        ], className="chat-mini-widget", id="chat-mini-widget")
])