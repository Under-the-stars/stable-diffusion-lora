import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.title("Image Generator with LoRA")

model_path = "/app/pytorch_lora_weights (1).safetensors"

pipe_base_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

trained_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
trained_model.unet.load_attn_procs(model_path)
trained_model.to("cuda")
pipe_base_model.to("cuda")

prompt = st.text_input("Enter your prompt:")

if prompt:
    generated_img = img_gen(prompt)
    st.image(generated_img, caption='Generated Image.', use_column_width=True)

def img_gen(prompt):
    train = trained_model(prompt * 1, num_inference_steps=50, guidance_scale=8.0).images
    return train[0]
    
if _name_ == '_main_':
    st.run()
