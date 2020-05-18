from python:3.7

RUN pip install dash==1.9.1 pymongo

RUN pip install pandas

COPY app.py .

COPY s1.css styles.css trained_model.json model_history.pickle assets/

EXPOSE 8080

CMD ["python", "app.py"]