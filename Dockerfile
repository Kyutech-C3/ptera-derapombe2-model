FROM tensorflow/serving
COPY assets/model/RasSignModelCNN_9-2.h5 /models/
ENV MODEL_NAME=RasSignModelCNN_9-2.h5
EXPOSE 8501