import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

competition = "rsna-knee-abnormality-detection"
target_dir = "./sample_dcm"
os.makedirs(target_dir, exist_ok=True)

print("Listing files from Kaggle...")
response = api.competition_list_files(competition)

# Access the files attribute on the response object
files = getattr(response, "files", response)

dcm_files = [f.name for f in files if f.name.endswith(".dcm")][:5]

if not dcm_files:
    print("No .dcm files found in the current page.")
else:
    for file_name in dcm_files:
        print(f"Downloading: {file_name}")
        api.competition_download_file(competition, file_name, path=target_dir, force=True)
    print(f"Successfully downloaded {len(dcm_files)} sample slices to {target_dir}")
