import replicate
import os
from dotenv import load_dotenv

import streamlit as st
import replicate

REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
client = replicate.Client(api_token=REPLICATE_API_TOKEN)


def get_sam2_mask(image_path, boxes):
    output = client.run(
        "meta/sam-2:9dcd6d78e7c6560c340d916fe32e9f24aabfa331e5cce95fe31f77fb03121426",
        input={
            "image": open(image_path, "rb"),
            "boxes": boxes,
            "mask_format": "rgba"
        }
    )
    # 'masks' is a list of URLs to output RGBA masks
    return output["masks"]
