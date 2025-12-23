FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy only code; data comes from a runtime mount
COPY main_train_TCN.py main_train_TCN.py

ENV MPLBACKEND=Agg
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

CMD ["python", "main_train_TCN.py"]
