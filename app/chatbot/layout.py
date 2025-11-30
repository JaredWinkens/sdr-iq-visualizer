import dash
from dash import dcc, html, State, Input, Output, callback
import dash_bootstrap_components as dbc
from app.config.settings import CHATBOT_CONFIG

def load_mini_chatbot_layout(initial_prompt: str):
    return html.Div([
        # Chat Widget Stores
        dcc.Store(id='chat-mini-message-store', data=[]),
        dcc.Store(id='chat-mini-loading-store', data=False),
        dcc.Store(id='chat-mini-widget-minimized', data=False),
        dcc.Store(id='chat-mini-notification-count', data=0),
        
        # Minimized circular button (shown when minimized)
        html.Div([
            html.I(className="bi bi-chat-dots-fill"),
            html.Span(id='chat-notification-badge', className='chat-notification-badge', children='0')
        ], className="chat-mini-circle-btn", id="chat-mini-circle-btn", n_clicks=0),
        
        # Main Chat Widget (shown when expanded)
        html.Div([
            # Chat header
            html.Div([
                html.Div([
                    html.I(className="bi bi-robot me-2"),
                    html.Span("RF Data Assistant")
                ], className="chat-mini-title"),
                html.Div([
                    html.Button([html.I(className="bi bi-x-lg")], 
                               id="chat-mini-minimize-btn", 
                               className="chat-mini-header-btn",
                               title="Minimize")
                ], className="chat-mini-header-actions")
            ], className="chat-mini-header"),
            
            # Chat content
            html.Div([
                # Welcome message with suggested questions
                html.Div([
                    html.Div([
                        html.I(className="bi bi-lightbulb me-2"),
                        html.Span(initial_prompt, style={'fontWeight': 'normal'})
                    ], className="chat-welcome-text mb-3"),
                    
                    html.Div([
                        html.Div("Try asking:", className="chat-suggested-label mb-2"),
                        html.Div([
                            dbc.Button([
                                html.I(className="bi bi-wifi me-2"),
                                "What frequency bands are showing the most activity?"
                            ], id="suggested-q1", color="light", size="sm", 
                            className="chat-suggested-btn mb-2"),
                            
                            dbc.Button([
                                html.I(className="bi bi-graph-up me-2"),
                                "Analyze the signal strength patterns"
                            ], id="suggested-q2", color="light", size="sm", 
                            className="chat-suggested-btn mb-2"),
                            
                            dbc.Button([
                                html.I(className="bi bi-broadcast me-2"),
                                "What type of signal is currently detected?"
                            ], id="suggested-q3", color="light", size="sm", 
                            className="chat-suggested-btn")
                        ], className="chat-suggested-buttons")
                    ], id="chat-suggested-questions")
                ], className="chat-mini-welcome-message", id='chat-mini-welcome-message'),
                
                # Messages container
                html.Div(id="chat-mini-messages-container", className="chat-mini-messages-list")
            ], className="chat-mini-messages", id="chat-mini-messages"),
            
            # Quick actions bar
            html.Div([
                dbc.Button([html.I(className="bi bi-arrow-clockwise")], 
                          id="chat-clear-btn", 
                          color="link", 
                          size="sm",
                          className="chat-quick-action-btn",
                          title="Clear conversation"),
                html.Span("•", className="text-muted mx-1", style={'display': 'none'}),
                html.Span(id="chat-message-count", children="0 messages", 
                         className="text-muted small", style={'display': 'none'}),
                dbc.Select(
                    id="chatbot-select-model",
                    options=[{"label": l, "value": l} for l in CHATBOT_CONFIG['all_models']],
                    value=CHATBOT_CONFIG['default_model'],
                    size='sm'
                ),
            ], className="chat-quick-actions"),
            
            # Input container
            html.Div([
                html.Div([
                    dcc.Input(
                        id="chat-mini-message-input",
                        placeholder="Ask about your RF data...",
                        className="chat-mini-input",
                        #autoComplete="off",
                        maxLength=500,
                    ),
                    html.Button([html.I(className="bi bi-send-fill")], 
                               id="chat-mini-send-button", 
                               className="chat-mini-send-button", 
                               n_clicks=0)
                ], className="chat-mini-input-group")
            ], className="chat-mini-input-container", id="chat-mini-input-container")
        ], className="chat-mini-widget", id="chat-mini-widget")
    ])