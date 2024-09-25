from pathlib import Path
import pandas as pd
from datasets.image_gathering import get_all_images_with_parent_folder_name


name_map = {
    "melanoma": "MEL",
    "basal_cell_carcinoma": "BCC",
    "squamous_cell_carcinoma": "SCC",
}


def get_name(name: str) -> str:
    if name in name_map:
        return name_map[name]
    return "OTHER"


image_location = Path(input("Write the path: "))

csv_dict = {
    "image_path": [],
    "classification": [],
    "metaclass": [],
}

for parent_name, image_path in get_all_images_with_parent_folder_name(image_location):
    metaclass = get_name(parent_name)
    csv_dict["image_path"].append(image_path)
    csv_dict["classification"].append(parent_name)
    csv_dict["metaclass"].append(metaclass)

df = pd.DataFrame(csv_dict)
df.to_csv("classification_dataset.csv", index=False)
