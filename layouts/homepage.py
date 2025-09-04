import dash
from dash import html
import dash_bootstrap_components as dbc

# Define a function to create a basic card
def create_card(title, content):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4(title, className="card-title"),
                html.P(content, className="card-text"),
            ]
        ),
        class_name="home-page-card",  # Add margin-bottom for spacing between cards
    )

# Create the four cards
card1 = create_card("Time-domain plots", "This is the content for Card 1.")
card2 = create_card("Frequency spectrum analysis", "This is the content for Card 2.")
card3 = create_card("Constellation diagrams", "This is the content for Card 3.")
card4 = create_card("Waterfall plots", "This is the content for Card 4.")

def load_homepage_layout():
    return dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [html.H1("SDR IQ Data Visualization Platform", className="my-4")],
                className="d-flex justify-content-center"
            ),
        ),
        dbc.Row(
            [
                dbc.Col(card1, md=6),  # Takes 6 of 12 columns on medium screens and up
                dbc.Col(card2, md=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(card3, md=6),
                dbc.Col(card4, md=6),
            ]
        ),
    ],
    fluid=True,  # Makes the container fluid, filling the available width
)