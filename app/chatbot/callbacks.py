import dash
from dash import dcc, html, State, Input, Output, callback, callback_context, ALL
import dash_bootstrap_components as dbc
import plotly.io as pio
import pandas as pd
import datetime
from app.chatbot.chatbot import Chatbot

# Create chatbot instance
chatbot_sessions: dict[str, Chatbot] = {'default': Chatbot()}

def register_callbacks(app):

    # Callback to toggle minimize/maximize
    @app.callback(
        Output('chat-mini-widget-minimized', 'data'),
        [Input('chat-mini-minimize-btn', 'n_clicks'),
         Input('chat-mini-circle-btn', 'n_clicks')],
        [State('chat-mini-widget-minimized', 'data')],
        prevent_initial_call=True
    )
    def toggle_widget(minimize_clicks, circle_clicks, is_minimized):
        # Toggle state
        return not is_minimized

    # Callback to update widget and button visibility
    app.clientside_callback(
        """
        function(is_minimized) {
            const widget = document.getElementById('chat-mini-widget');
            const circleBtn = document.querySelector('.chat-mini-circle-btn');
            
            if (widget && circleBtn) {
                if (is_minimized) {
                    widget.classList.add('minimized');
                    circleBtn.style.display = 'flex';
                } else {
                    widget.classList.remove('minimized');
                    circleBtn.style.display = 'none';
                }
            }
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('chat-mini-widget', 'className'),
        [Input('chat-mini-widget-minimized', 'data')]
    )

    # Callback to handle clear chat
    @app.callback(
        Output('chat-mini-message-store', 'data', allow_duplicate=True),
        [Input('chat-clear-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def clear_chat(n_clicks):
        if n_clicks:
            chatbot_sessions['default'].clear_history()
            return []
        return dash.no_update

    # Callback to handle sending messages (including suggested questions)
    @app.callback(
        [Output('chat-mini-message-store', 'data'),
         Output('chat-mini-loading-store', 'data'),
         Output('chat-mini-message-input', 'value'),
         Output('chat-mini-notification-count', 'data')],
        [Input('chat-mini-send-button', 'n_clicks'),
         Input('chat-mini-message-input', 'n_submit'),
         Input('suggested-q1', 'n_clicks'),
         Input('suggested-q2', 'n_clicks'),
         Input('suggested-q3', 'n_clicks')],
        [State('chat-mini-message-input', 'value'),
         State('chat-mini-message-store', 'data'),
         State('chat-mini-widget-minimized', 'data'),
         State('chat-mini-notification-count', 'data'),
        State('chatbot-select-model', 'value')
]
    )
    def send_message(send_clicks, n_submit, q1_clicks, q2_clicks, q3_clicks, 
                     message, messages, is_minimized, notif_count, new_model):
        ctx = callback_context
        if not ctx.triggered:
            return messages, False, "", notif_count
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if new_model != chatbot_sessions['default'].model:
            print('Model Updated!')
            print('Old Model: ', chatbot_sessions['default'].model)
            chatbot_sessions['default'].change_model(new_model=new_model)
            print('New Model: ', chatbot_sessions['default'].model)
            
        # Handle suggested questions
        suggested_questions = {
            'suggested-q1': "What frequency bands are showing the most activity?",
            'suggested-q2': "Analyze the signal strength patterns",
            'suggested-q3': "What type of signal is currently detected?"
        }
        
        if trigger_id in suggested_questions:
            message = suggested_questions[trigger_id]
        elif not message or not message.strip():
            return messages, False, "", notif_count
        
        # Add user message
        messages.append({
            'type': 'user',
            'content': message.strip(),
            'timestamp': datetime.datetime.now().strftime('%H:%M')
        })
        
        return messages, True, "", notif_count

    # Callback to get bot response
    @app.callback(
        [Output('chat-mini-message-store', 'data', allow_duplicate=True),
         Output('chat-mini-loading-store', 'data', allow_duplicate=True),
         Output('chat-mini-notification-count', 'data', allow_duplicate=True)],
        [Input('chat-mini-loading-store', 'data')],
        [State('chat-mini-message-store', 'data'),
         State('time-domain-graph', 'figure'),
         State('freq-domain-graph', 'figure'),
         State('waterfall-graph', 'figure'),
         State('constellation-plot', 'figure'),
         State('chat-mini-widget-minimized', 'data'),
         State('chat-mini-notification-count', 'data')],
        prevent_initial_call=True
    )
    def bot_response(is_loading, messages, td_fig, fd_fig, wf_fig, con_fig, 
                     is_minimized, notif_count):
        if not is_loading or not messages:
            return messages, False, notif_count

        # Get the last user message
        last_message = messages[-1]
        if last_message['type'] != 'user':
            return messages, False, notif_count

        context = {
            'td': td_fig,
            'fd': fd_fig,
            'wf': wf_fig,
            'con': con_fig
        }

        # Generate bot response
        bot_reply = chatbot_sessions['default'].get_response(last_message['content'], context)
        
        # Add bot message
        messages.append({
            'type': 'bot',
            'content': bot_reply,
            'timestamp': datetime.datetime.now().strftime('%H:%M')
        })
        
        # Increment notification if minimized
        new_notif = notif_count + 1 if is_minimized else 0
        
        return messages, False, new_notif

    # Callback to update messages display
    @app.callback(
        [Output('chat-mini-messages-container', 'children'),
         Output('chat-mini-welcome-message', 'style'),
         Output('chat-message-count', 'children'),
         Output('chat-suggested-questions', 'style')],
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
                            html.Div(dcc.Markdown(msg['content']), 
                                    className="chat-mini-message-bubble chat-mini-user-bubble"),
                            html.Div(msg['timestamp'], className="chat-mini-message-time")
                        ])
                    ], className="chat-mini-message chat-mini-user-message")
                )
            else:
                message_elements.append(
                    html.Div([
                        html.Div([
                            html.Div(dcc.Markdown(msg['content']), 
                                    className="chat-mini-message-bubble chat-mini-bot-bubble"),
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
                                html.Div()
                            ])
                        ], className="chat-mini-message-bubble chat-mini-bot-bubble")
                    ])
                ], className="chat-mini-message chat-mini-bot-message")
            )
        
        # Hide welcome and suggested questions after first message
        welcome_style = {"display": "none"} if messages else {"display": "block"}
        suggested_style = {"display": "none"} if messages else {"display": "block"}
        
        # Update message count
        msg_count = f"{len(messages)} message{'s' if len(messages) != 1 else ''}"
        
        return message_elements, welcome_style, msg_count, suggested_style

    # Update notification badge
    app.clientside_callback(
        """
        function(count) {
            const badge = document.getElementById('chat-notification-badge');
            if (badge) {
                badge.textContent = count;
                if (count > 0) {
                    badge.classList.add('show');
                } else {
                    badge.classList.remove('show');
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('chat-notification-badge', 'className'),
        [Input('chat-mini-notification-count', 'data')]
    )

    # Reset notifications when opening chat
    @app.callback(
        Output('chat-mini-notification-count', 'data', allow_duplicate=True),
        [Input('chat-mini-widget-minimized', 'data')],
        prevent_initial_call=True
    )
    def reset_notifications(is_minimized):
        if not is_minimized:
            return 0
        return dash.no_update

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