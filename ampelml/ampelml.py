# -*- coding: utf-8 -*-
import zerorpc
from astrorapid.classify import Classify
import numpy as np
from pymongo import MongoClient
import datetime
from bson.json_util import dumps
import sys

sys.stdout = sys.stderr

class Classifier(object):

    def classify(self, lc):
        
        mjd, flux, fluxerr, passband, ra, dec, objid, redshift, mwebv = lc[0]

        photflag = [4096] * len(flux)
        photflag[np.argmax(flux)] = 6144
        photflag = np.array(photflag)

        light_curve_info1 = (mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv)
        light_curve_list = [light_curve_info1,]

        classification = Classify(known_redshift=True, model_filepath='keras_model.hdf5')
        predictions = classification.get_predictions(light_curve_list)
        
        y_predict = predictions[0][0]
        time_steps = predictions[1][0]

        classified_lightcurve = {
            'time': datetime.datetime.utcnow(),
            'obj_id': objid,
            'mjd': mjd,
            'flux': flux,
            'fluxerr': fluxerr,
            'passband': passband,
            'ra': ra,
            'dec': dec,
            'redshift': redshift,
            'mwebv': mwebv,
            'photflag': photflag.tolist(),
            'predicted_y': y_predict.tolist(),
            'timesteps': time_steps.tolist(),
        }

        print(classified_lightcurve['timesteps'])

        client = MongoClient('ampelml-mongo')
        db = client.ampel_ml
        classified_lightcurves = db.classified_lightcurves

        class_lc_id = classified_lightcurves.insert_one(classified_lightcurve).inserted_id

        last_class_lc = classified_lightcurves.find().sort([('time', -1)]).limit(1)[0]

        return dumps(last_class_lc)
        #return dumps(classified_lightcurve)

s = zerorpc.Server(Classifier())
s.bind("tcp://0.0.0.0:4242")
s.run()

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),

#     html.Div(children='''
#         Dash: A web application framework for Python.
#     '''),

#     dcc.Graph(
#         id='example-graph',
#         figure={
#             'data': [
#                 {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
#                 {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
#             ],
#             'layout': {
#                 'title': 'Dash Data Visualization'
#             }
#         }
#     )
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0')