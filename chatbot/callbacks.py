import dash
from dash import dcc, html, State, Input, Output, callback, callback_context
import dash_bootstrap_components as dbc
import plotly.io as pio
import pandas as pd
import datetime
from chatbot.chatbot import Chatbot

# Create chatbot instance
chatbot_sessions: dict[str, Chatbot] = {'default': Chatbot()}

def register_callbacks(app):

    # Callback to toggle minimize/maximize
    @app.callback(
        [Output('chat-mini-widget-minimized', 'data'),
        Output('chat-mini-minimize-btn', 'children')],
        [Input('chat-mini-minimize-btn', 'n_clicks'),
        Input('chat-mini-header', 'n_clicks')],
        [State('chat-mini-widget-minimized', 'data')]
    )
    def toggle_widget(minimize_clicks, header_clicks, is_minimized):
        ctx = callback_context
        if not ctx.triggered:
            return False, "−"
        
        # Toggle state
        new_state = not is_minimized
        button_text = "+" if new_state else "−"
        
        return new_state, button_text

    # Callback to update widget appearance based on minimized state
    app.clientside_callback(
        """
        function(is_minimized) {
            const widget = document.getElementById('chat-mini-widget');
            const messages = document.getElementById('chat-mini-messages');
            const input = document.getElementById('chat-mini-input-container');
            
            if (widget) {
                if (is_minimized) {
                    widget.classList.add('minimized');
                    if (messages) messages.style.display = 'none';
                    if (input) input.style.display = 'none';
                } else {
                    widget.classList.remove('minimized');
                    if (messages) messages.style.display = 'flex';
                    if (input) input.style.display = 'block';
                }
            }
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('chat-mini-widget', 'style'),
        [Input('chat-mini-widget-minimized', 'data')]
    )

    # Callback to handle sending messages
    @app.callback(
        [Output('chat-mini-message-store', 'data'),
        Output('chat-mini-loading-store', 'data'),
        Output('chat-mini-message-input', 'value')],
        [Input('chat-mini-send-button', 'n_clicks'),
        Input('chat-mini-message-input', 'n_submit')],
        [State('chat-mini-message-input', 'value'),
        State('chat-mini-message-store', 'data'),
        State('chat-mini-widget-minimized', 'data')]
    )
    def send_message(n_clicks, n_submit, message, messages, is_minimized):
        if not message or not message.strip() or is_minimized:
            return messages, False, ""
        
        # Add user message immediately
        messages.append({
            'type': 'user',
            'content': message.strip(),
            'timestamp': datetime.datetime.now().strftime('%H:%M')
        })
        
        return messages, True, ""

    # Callback to get a bot response
    @app.callback(
        [
            Output('chat-mini-message-store', 'data', allow_duplicate=True),
            Output('chat-mini-loading-store', 'data', allow_duplicate=True)
        ],
        [
            Input('chat-mini-loading-store', 'data')
        ],
        [
            State('chat-mini-message-store', 'data'),
            State('time-domain-graph', 'figure'),
            State('freq-domain-graph', 'figure'),
            State('waterfall-graph', 'figure'),
            State('constellation-plot', 'figure')
        ],
        prevent_initial_call=True
    )
    def bot_response(is_loading, messages, td_fig, fd_fig, wf_fig, con_fig):
        if not is_loading or not messages:
            return messages, False

        # Get the last user message
        last_message = messages[-1]
        if last_message['type'] != 'user':
            return messages, False

        # Generate bot response
        chatbot_sessions['default'].update_state(td_fig, fd_fig, wf_fig, con_fig)
        bot_reply = chatbot_sessions['default'].get_response(last_message['content'])
        
        # Add bot message
        messages.append({
            'type': 'bot',
            'content': bot_reply,
            'timestamp': datetime.datetime.now().strftime('%H:%M')
        })
        
        return messages, False

    # Callback to update messages display
    @app.callback(
        Output('chat-mini-messages-container', 'children'),
        Output('chat-mini-welcome-message', 'style'),
        [Input('chat-mini-message-store', 'data'),
        Input('chat-mini-loading-store', 'data')]
    )
    def update_messages(messages, is_loading):
        message_elements = []
        
        for msg in messages:
            if msg['type'] == 'user':
                message_elements.append(
                    html.Div([
                        html.Div([
                            html.Div(dcc.Markdown(msg['content']), className="chat-mini-message-bubble chat-mini-user-bubble"),
                            html.Div(msg['timestamp'], className="chat-mini-message-time")
                        ])
                    ], className="chat-mini-message chat-mini-user-message")
                )
            else:
                message_elements.append(
                    html.Div([
                        html.Div([
                            html.Div(dcc.Markdown(msg['content']), className="chat-mini-message-bubble chat-mini-bot-bubble"),
                            html.Div(msg['timestamp'], className="chat-mini-message-time")
                        ])
                    ], className="chat-mini-message chat-mini-bot-message")
                )
        
        # Add loading animation if bot is thinking
        if is_loading:
            message_elements.append(
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div(className="chat-mini-loading-dots", children=[
                                html.Div(),
                                html.Div(),
                                html.Div(),
                                html.Div()
                            ])
                        ], className="chat-mini-message-bubble chat-mini-bot-bubble")
                    ])
                ], className="chat-mini-message chat-mini-bot-message")
            )
        
        welcome_style = {"display": "none"}
        if not message_elements:
            welcome_style = {"display": "flex"}

        return message_elements, welcome_style

    # Auto-scroll functionality
    app.clientside_callback(
        """
        function(messages, loading) {
            setTimeout(function() {
                const container = document.querySelector('.chat-mini-messages');
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            }, 100);
            return window.dash_clientside.no_update;
        }
        """,
        Output('chat-mini-message-input', 'autoComplete'),
        [Input('chat-mini-message-store', 'data'),
        Input('chat-mini-loading-store', 'data')]
    )