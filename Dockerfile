FROM pytorch/pytorch:latest

WORKDIR /app

# Install necessary libraries
RUN pip install git+https://github.com/huggingface/diffusers transformers streamlit

# Copy the Streamlit app and model weights
COPY app.py /app/
COPY pytorch_lora_weights (1).safetensors /app/

# Streamlit uses this port
EXPOSE 8501

# Run the app
CMD streamlit run app.py
