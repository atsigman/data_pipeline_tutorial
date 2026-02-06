import sys
import json

from typing import Dict
from tqdm import tqdm

from music_data_pipeline.audio_dataset import AudioDataset


def test_dataset(input_data: Dict) -> Dict:
    """
    Iterates through dataset, making successive __getitem__() calls.
    Logs errors in returned error_dict.
    """
    error_dict = {}
    audio_ds = AudioDataset(input_data)

    for i in tqdm(range(len(audio_ds)), desc="Dataset entries"):
        try:
            audio_ds[i]

        except Exception as e:
            error_dict[i] = str(e)

    print(f"{len(error_dict)} errors detected.")
    return error_dict


if __name__ == "__main__":
    json_path = sys.argv[1]

    with open(json_path, "r") as f:
        input_data = json.load(f)

    err_dict = test_dataset(input_data)

    with open("dataset_errors.json", "w") as f:
        json.dump(err_dict, f, indent=4)
