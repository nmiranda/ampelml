FROM tensorflow/tensorflow:2.0.0-py3

RUN pip3 install zerorpc pymongo

RUN \
  apt-get update && apt-get upgrade -y && \
  apt-get install -y git

RUN \
  git clone https://github.com/daniel-muthukrishna/astrorapid.git  && \
  cd astrorapid && \
  pip3 install .

CMD ["python3","ampelml.py"]

EXPOSE 4242

COPY ampelml.py keras_model.hdf5 ./