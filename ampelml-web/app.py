# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from pymongo import MongoClient
from bson.json_util import dumps
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

LIGHTCURVES = list()
CLASSES = {0:'pre-explosion', 1:'SN Ia', 2:'other'}

with open('assets/trained_model.json', 'r') as trained_model_file:
    trained_model = json.load(trained_model_file)
    # trained_model = pd.read_json(trained_model_file)

    trained_model_data = [{'id': key, 'property': key, 'value': val} for key, val in trained_model.items()]

app = dash.Dash(
    __name__, 
    # external_stylesheets=external_stylesheets,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    )
server = app.server

client = MongoClient('ampelml-mongo')
db = client.ampel_ml
classified_lightcurves = db.classified_lightcurves

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
)
fig.update_layout(title_text='Lightcurve Classification', title_x=0.5, title_font_size=20)
fig.update_layout(
    height=600,
)

hist = pd.read_pickle("assets/model_history.pickle")
fig_training = go.Figure()
for key in hist:
    fig_training.add_trace(go.Scatter(
        x=np.arange(len(hist[key])),
        y=hist[key],
        mode='lines+markers',
        name=key,
    ))
fig_training.update_layout(title_text='Model Training Metrics', title_x=0.5, title_font_size=20, hovermode="x")
fig_training.update_yaxes(title_text="Value")
fig_training.update_xaxes(title_text="Iteration")

app.layout = html.Div(children=[

    html.Div(
        [
            html.Div(
                [
                    html.H3(
                        "AMPEL ML Integration",
                        style={"margin-bottom": "0px"},
                    ),
                    html.H5(
                        "Testing ML classification integration into AMPEL",
                        style={"margin-top": "0px"},
                    ),
                ],
                className="twelve columns",
                id="title",
            ),
        ],
        id="header",
        className="row flex-display",
        style={"margin-bottom": "25px"},
    ),

    html.Div(
        [
            html.Button('Reload', id='reload-button'),
        ],
        id="button",
        className="row flex-display",
        style={"margin-bottom": "25px"},
    ),

    html.Div(
        [
            html.Div(
                [
                    dash_table.DataTable(
                        id='lc-table',
                        columns=[
                            {'name':'Timestamp', 'id':'timestamp'},
                            {'name':'Object ID', 'id':'obj-id'},
                            {'name':'First detection', 'id':'first-det'},
                            {'name':'Num. photopoints', 'id':'num-photopoints'},
                            {'name':'Overall class', 'id':'over-class'},
                            {'name':'Final class', 'id':'final-class'},
                        ],
                        row_selectable='single',
                    ),
                ],
                id='table-container',
                className="pretty_container six columns",
            ),
            html.Div(
                [
                    dcc.Graph(
                        id='lightcurve-graph',
                        figure=fig,
                    ),
                ],
                id="graph-container",
                className="pretty_container six columns",
            ),
        ],
        id='content',
        className="row flex-display",
    ),

    html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H4(
                                "Model: ASTRORAPID",
                            )
                        ],
                        id="model-title",
                        className="row"
                    ),
                    html.Div(
                        [
                            html.Img(
                                src="https://i.imgur.com/WvMp5De.png",
                                id='model-graph',
                            )
                        ],
                        id="model-graph-container",
                        className="row"
                    ),
                    html.Div(
                        [
                            dash_table.DataTable(
                                id='model-table',
                                columns=[
                                    {'name':'Property', 'id':'property'},
                                    {'name':'Value', 'id':'value'},
                                ],
                                data=trained_model_data,
                            ),
                        ],
                        id="model-table-container",
                        className="row"
                    ),
                    
                ],
                id='model-vis-container',
                className="pretty_container six columns"
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                id='training-graph',
                                figure=fig_training,
                            ),
                        ],
                        id='model-training-graph-container',
                        className="row"
                    )
                ],
                id='model-training-container',
                className="pretty_container six columns"
            )
        ],
        id='model-container',
        className="row flex-display",
    )

    # html.Div(
    #     id='log-div',
    #     style=dict(height='300px',overflow='auto'),
    #     className="row flex-display",
    #     ),

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
    )

def get_classif_lc_info(classif_lc):

    pred_y = pd.DataFrame(classif_lc['predicted_y'])

    return_dict = {
        'timestamp': classif_lc['time'].strftime("%d.%m.%Y %H:%M:%S.%f")[:-3],
        'obj-id': classif_lc['obj_id'],
        'first-det': classif_lc['mjd'][0],
        'num-photopoints': len(classif_lc['mjd']),
        'over-class': pred_y.columns[np.argmax(np.trapz(pred_y, axis=0))],
        'final-class': pred_y.columns[np.argmax(pred_y.iloc[-1])],
    }

    return return_dict

@app.callback(
    Output(component_id='lc-table', component_property='data'),
    [Input(component_id='reload-button', component_property='n_clicks')]
)
def update_table(n_clicks):

    global LIGHTCURVES
    classif_lcs = classified_lightcurves.find().sort([('time', -1)])
    LIGHTCURVES = list(classif_lcs)
    return [get_classif_lc_info(classif_lc) for classif_lc in LIGHTCURVES]

@app.callback(
    Output(component_id='lightcurve-graph', component_property='figure'),
    # Output(component_id='log-div', component_property='children'),
    [Input(component_id='lc-table', component_property='selected_rows')]
)
def update_graph(selected_rows):

    if selected_rows is None:
        raise PreventUpdate

    # return dumps(LIGHTCURVES[selected_rows[0]])

    # try:
    #     last_class_lc = classified_lightcurves.find().sort([('time', -1)]).limit(1)[0]
    # except IndexError as e:
    #     raise PreventUpdate

    last_class_lc = LIGHTCURVES[selected_rows[0]]

    # last_class_lc = classified_lightcurves.find().sort([('time', -1)]).limit(1)[0]
    this_lc = pd.DataFrame({key: last_class_lc[key] for key in ['mjd', 'flux', 'fluxerr', 'passband']})
    this_classif = pd.DataFrame(last_class_lc['predicted_y'])
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
    )
    
    figure.update_layout(title_text=f'Object Id: {last_class_lc["obj_id"]}', title_x=0.5, title_font_size=20, hovermode="x")
    figure.update_yaxes(title_text="flux", row=1, col=1)
    figure.update_yaxes(title_text="class prob.", row=2, col=1)
    figure.update_xaxes(title_text="jd", row=2, col=1)

    for band in this_lc.passband.unique():
        lc_band = this_lc[this_lc.passband == band]
        figure.add_trace(go.Scatter(
            x=lc_band['mjd'],
            y=lc_band['flux'],
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=lc_band['fluxerr'],
                visible=True),
            mode='markers',
            name=str(band),
            ),
            row=1,
            col=1,
        )

    for class_ in this_classif.columns:
        figure.add_trace(go.Scatter(
            x=last_class_lc['timesteps'],
            y=this_classif[class_],
            name=CLASSES[class_],
            ),
            row=2,
            col=1,
        )

    return figure

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=True, port=8080)