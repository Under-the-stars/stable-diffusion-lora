FROM pytorch/pytorch:latest

WORKDIR /app

# Install necessary libraries
RUN pip install git+https://github.com/huggingface/diffusers transformers streamlit

# Copy the Streamlit app and model weights
# COPY app.py /app/
# COPY pytorch_lora_weights (1).safetensors /app/
COPY . .
# Streamlit uses this port
EXPOSE 8501

# Run the app
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

#SHELL [ "executable" ]
