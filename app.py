# streamlit_app.py

import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np

# # Load model
# model_path = "model/pytorch_lora_weights.safetensors"
# pipe_base_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
# trained_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
# trained_model.unet.load_attn_procs(model_path)
# #trained_model.to("cuda")
# #pipe_base_model.to("cuda")

@st.cache
def model_load():
    # Load model
    model_path = "model/pytorch_lora_weights.safetensors"
    # pipe_base_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    trained_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    trained_model.unet.load_attn_procs(model_path)
    # trained_model.to("cuda")
    # pipe_base_model.to("cuda")
    return trained_model
trained_model=model_load()

def process_image(img):
    # Add code to preprocess the image as required by your model
    # Convert the PIL Image to a PyTorch tensor, process it, etc.
    # Then, use your model to predict and return the result.
    # This is a placeholder, so you'll need to fill in the details.
    tensor_image = torch.Tensor(np.array(img)).unsqueeze(0).to("cuda")
    with torch.no_grad():
        output = trained_model(tensor_image)
    return output


st.title("Stable Diffusion Model Inference")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    st.write("")
    st.write("Predicting...")

    output = process_image(img)

    st.image(output, caption="Processed Image.", use_column_width=True)

#if __name__ == '__main__':
#    st.run()
