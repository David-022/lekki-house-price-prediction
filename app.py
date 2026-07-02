import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import joblib

# Load dataset
pd.read_csv("data/filtered_lekki_price.csv")

# Extract unique values for dropdown menus
unique_neighborhood = data['Neighborhood'].unique()

# Load the trained model
model = joblib.load("models/lagos_house_price_model.pkl")

# File to store predictions
PREDICTIONS_FILE = 'lagos_predictions.csv'

# Initialize predictions DataFrame
try:
    predictions_df = pd.read_csv(PREDICTIONS_FILE)
except FileNotFoundError:
    predictions_df = pd.DataFrame(columns=['Furnished', 'bedrooms', 'bathrooms', 'toilets','neighborhood', 'city', 'PredictedPrice'])

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("House Price Prediction", style={'textAlign': 'center'}),
    html.H3("Lagos State", style={'textAlign': 'center'}),
    html.H6("Lekki City", style={'textAlign': 'center'}),

    # Input sections
    html.Div([
        
        html.Div([
            html.Label('Furnished:', style={'fontSize': '14px'}),
            dcc.Input(id='input-furnished', type='number', value=0, min=0,max=1, step=1,
                      style={'width': '100%', 'height': '25px', 'fontSize': '12px'}),
        ], style={'flex': '1', 'margin': '10px'}),
        
        html.Div([
            html.Label('Bedrooms:', style={'fontSize': '14px'}),
            dcc.Input(id='input-bedrooms', type='number', value=1, min=1, step=1,
                      style={'width': '100%', 'height': '25px', 'fontSize': '12px'}),
        ], style={'flex': '1', 'margin': '10px'}),
        
    ], style={'display': 'flex', 'justifyContent': 'center', 'width': '60%', 'margin': '0 auto'}),
    
    html.Div([
        
        html.Div([
            html.Label('Bathrooms:', style={'fontSize': '14px'}),
            dcc.Input(id='input-bathrooms', type='number', value=1, min=1, step=1,
                      style={'width': '100%', 'height': '25px', 'fontSize': '12px'}),
        ], style={'flex': '1', 'margin': '10px'}),
        
        html.Div([
            html.Label('Toilets:', style={'fontSize': '14px'}),
            dcc.Input(id='input-toilets', type='number', value=1, min=1, step=1,
                      style={'width': '100%', 'height': '25px', 'fontSize': '12px'}),
        ], style={'flex': '1', 'margin': '10px'}),
        
    ], style={'display': 'flex', 'justifyContent': 'center', 'width': '60%', 'margin': '0 auto'}),
    
    
    html.Div([
        
        html.Div([
            html.Label('N:', style={'fontSize': '14px'}),
            dcc.Dropdown(id='input-neighborhood', options=[{'label': n, 'value': n} for n in unique_neighborhood], value=unique_neighborhood[0],
                         style={'width': '100%', 'height': '25px', 'fontSize': '12px'}),
        ], style={'flex': '1', 'margin': '10px'}),
        
        html.Div([
            html.Label('City:', style={'fontSize': '14px'}),
            dcc.Input(id='input-state', type='text', value='Lekki', readOnly=True,
                      style={'width': '100%', 'height': '25px', 'fontSize': '12px'}),
        ], style={'flex': '1', 'margin': '10px'}),
        
    ], style={'display': 'flex', 'justifyContent': 'center', 'width': '60%', 'margin': '0 auto'}),
    
    

    # Buttons
    html.Div([
        html.Button('Predict Price', id='predict-button', n_clicks=0,
                    style={'width': '10%', 'height': '30px', 'fontSize': '14px', 'marginRight': '4%'}),
        html.Button('Reset', id='reset-button', n_clicks=0,
                    style={'width': '10%', 'height': '30px', 'fontSize': '14px'}),
    ], style={'textAlign': 'center', 'margin': '20px'}),

    # Output sections
    html.Div(id='output-prediction', style={'textAlign': 'center', 'fontSize': '20px', 'margin': '20px'}),
    
    html.Div([
        dcc.Graph(id='output-graph', style={'height': '50vh', 'width': '50vw', 'margin': '0 auto'})
    ], style={'textAlign': 'center'}),

    # Display predictions and summary
    html.Div([
        html.H3("Past Predictions", style={'textAlign': 'center'}),
        html.Div(id='predictions-table', style={'margin': '20px'}),
        html.H3("Summary", style={'textAlign': 'center'}),
        html.Div(id='summary', style={'textAlign': 'center', 'fontSize': '16px', 'margin': '20px'})
    ])
])

@app.callback(
    [Output('input-furnished', 'value'),
     Output('input-bedrooms', 'value'),
     Output('input-bathrooms', 'value'),
     Output('input-toilets', 'value'),
     Output('input-neighborhood', 'value'),
     Output('output-prediction', 'children'),
     Output('output-graph', 'figure'),
     Output('predictions-table', 'children'),
     Output('summary', 'children')],
    [Input('predict-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('input-furnished', 'value'),
     State('input-bedrooms', 'value'),
     State('input-bathrooms', 'value'),
     State('input-toilets', 'value'),
     State('input-neighborhood', 'value')]
)
def update_output(predict_n_clicks, reset_n_clicks, furnished, bedrooms, bathrooms, toilets, neighborhood):
    ctx = dash.callback_context

    if not ctx.triggered:
        return furnished, bedrooms, bathrooms, toilets, neighborhood, "", {}, "", ""

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'reset-button':
        return 0, 1, 1, 1, unique_neighborhood, "", {}, "", ""

    elif button_id == 'predict-button':
        try:
            # Validate inputs
            if None in [furnished, bedrooms, bathrooms, toilets, neighborhood]:
                return furnished, bedrooms, bathrooms, toilets, neighborhood, "Please provide all input values.", {}, "", ""

            total_rooms = bedrooms + bathrooms + toilets

            input_data = pd.DataFrame({
                'Furnished': [furnished],
                'Bedrooms': [bedrooms],
                'Bathrooms': [bathrooms],
                'Toilets': [toilets],
                'Neighborhood': [neighborhood],
            })

            predicted_price = model.predict(input_data)[0]
            predicted_price = (predicted_price * 100)/4

            # Create a figure showing the relationship between Total Rooms and Predicted Price
            fig = go.Figure(data=[
                go.Scatter(
                    x=[total_rooms],
                    y=[predicted_price],
                    mode='markers',
                    marker=dict(size=12, color='blue'),
                    name='Prediction'
                )
            ])
            fig.update_layout(
                title='Predicted Price based on Total Rooms',
                xaxis_title='Total Rooms',
                yaxis_title='Predicted Price',
                template='plotly_white'
            )

            prediction_text = f'Predicted House Price: N{predicted_price:,.2f}'

            # Append new prediction to the DataFrame
            global predictions_df
            new_row = pd.DataFrame({
                'Furnished': [furnished],
                'bedrooms': [bedrooms],
                'bathrooms': [bathrooms],
                'toilets': [toilets],
                'neighborhood': [neighborhood],
                'city': ['Lekki'],
                'total_rooms': [total_rooms],
                'PredictedPrice': [predicted_price]
            })

            predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)
            predictions_df.to_csv(PREDICTIONS_FILE, index=False)

            # Display predictions table
            table = html.Table([
                html.Thead(html.Tr([html.Th(col) for col in predictions_df.columns])),
                html.Tbody([
                    html.Tr([html.Td(predictions_df.iloc[i][col]) for col in predictions_df.columns])
                    for i in range(len(predictions_df))
                ])
            ])

            # Summary
            summary_text = f"Average Price: N{predictions_df['PredictedPrice'].mean():,.2f} | " \
                           f"Min Price: N{predictions_df['PredictedPrice'].min():,.2f} | " \
                           f"Max Price: N{predictions_df['PredictedPrice'].max():,.2f} | " \
                           f"Std Dev: N{predictions_df['PredictedPrice'].std():,.2f}"

            return furnished, bedrooms, bathrooms, toilets, neighborhood, prediction_text, fig, table, summary_text

        except Exception as e:
            return furnished, bedrooms, bathrooms, toilets, neighborhood, f"Error in prediction: {e}", {}, "", ""

    else:
        return furnished, bedrooms, bathrooms, toilets, neighborhood, "", {}, "", ""

if __name__ == '__main__':
    app.run_server(debug=True, port=8055)










#http://127.0.0.1:8055/