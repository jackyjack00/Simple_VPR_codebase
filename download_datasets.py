
their_URLS = {
    "tokyo_xs": "https://drive.google.com/file/d/15QB3VNKj93027UAQWv7pzFQO1JDCdZj2/view?usp=share_link",
    "sf_xs": "https://drive.google.com/file/d/1tQqEyt3go3vMh4fj_LZrRcahoTbzzH-y/view?usp=share_link",
    "gsv_xs": "https://drive.google.com/file/d/1Lp_FX-5fSV2jH0DoH-LLUuMvH1jlRD2d/view?usp=share_link"
}

URLS = {
    "tokyo_xs": "https://drive.google.com/file/d/1nOJ_uHB1DqxL6rEDKYyZO6xJiZCy2pbX/view?usp=share_link",
    "sf_xs": "https://drive.google.com/file/d/1CbfHbZzu43rXwlaKQ3bANQtbIjSjzopl/view?usp=share_link",
    "gsv_xs": "https://drive.google.com/file/d/1nOJ_uHB1DqxL6rEDKYyZO6xJiZCy2pbX/view?usp=share_link"
}

import os
import gdown
import shutil

os.makedirs("data", exist_ok=True)
for dataset_name, url in URLS.items():
    print(f"Downloading {dataset_name}")
    zip_filepath = f"data/{dataset_name}.zip"
    gdown.download(url, zip_filepath, fuzzy=True)
    shutil.unpack_archive(zip_filepath, extract_dir="data")
    os.remove(zip_filepath)

