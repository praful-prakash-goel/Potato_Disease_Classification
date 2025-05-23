FROM tensorflow/serving:latest

# Copy your saved model into the container's model path
COPY saved_models /models/potatoes_model

# Set environment variables for TensorFlow Serving
ENV MODEL_NAME=potatoes_model

# Expose TensorFlow Serving's REST API port
EXPOSE 8501

# Run TensorFlow Serving
ENTRYPOINT ["/usr/bin/tensorflow_model_server", "--rest_api_port=8501", "--model_name=potatoes_model", "--model_base_path=/models/potatoes_model"]


