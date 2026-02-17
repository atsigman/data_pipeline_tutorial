"""
Script for running data processing pipeline end-to-end.
Input should be in JSON or CSV format.
Example usage:

python run_pipeline.py -i {PATH/TO/INPUT_DATA} -o {PATH/TO/OUTPUT_JSON}

(Be sure to that the indicated audio paths in the input data are either absolute or
 relative to the directory from which this script is executed.)
"""

import ast
import argparse
import json
import uuid

from typing import Dict, List

import pandas as pd

from music_data_pipeline.util.pipeline_utils import (
    validate_prune_data,
    find_similar_audio,
    tokenize_metadata,
    extract_blacklisted_genres,
    chunk_audio,
    add_silent_regions,
)


class MusicDataPipeline:
    """
    Extendable class for end-to-end music data pipeline execution.
    """

    def __init__(self, entries: List[Dict], output_path: str):

        self.entries = entries
        self.output_path = output_path

    def run(self) -> None:
        """
        Runs pipeline stages
        """
        print("Step 1. Validating entries...")
        self.entries = validate_prune_data(self.entries)

        print("Step 2: Checking for audio path/content duplicates...")
        self.entries = find_similar_audio(self.entries)

        print("Step 3: Tokenizing text metadata,,,")
        self.entries = tokenize_metadata(self.entries)

        print("Step 4: Blacklisting genres...")
        self.entries = extract_blacklisted_genres(self.entries)

        print("Step 5: Segmenting long audio duration tracks/entries...")
        self.entries = chunk_audio(self.entries)

        print("Step 6: Detecting silent regions...")
        self.entries = add_silent_regions(self.entries)

        print("Pipeline stages complete!")

    def save(self) -> None:
        """
        Saves entries to JSON
        """
        with open(self.output_path, "w") as f:
            json.dump(self.entries, f, indent=4)

        print(f"Processed data saved to {self.output_path}")


def df_to_entries(df: pd.DataFrame) -> List[Dict]:
    """
    Makes necessary conversions to dataframe, and converts
    to a list of dictionaries.
    """
    df = df.drop_duplicates(subset="track_id", keep="first")
    df["blacklist_flags"] = [[] for _ in range(len(df))]

    # Convert NaN audio paths to "":
    df["audio_path"] = df["audio_path"].apply(
        lambda x: "" if isinstance(x, float) else x
    )

    # Convert genre lists from "[]" to []:
    if df["genres"].dtype == "string":
        df["genres"] = df["genres"].apply(lambda x: ast.literal_eval(x))

    # Add _ids if absent:
    if "_id" not in df.columns:
        df["_id"] = [str(uuid.uuid1()) for _ in range(len(df))]

    # Convert to list of dictionaries:
    entries = df.to_dict(orient="records")

    return entries


def main(input_path: str, output_path: str) -> None:
    """
    Instantiates a MusicDataPipeline class. Runs pipeline and saves data to JSON.
    """
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
        entries = df_to_entries(df)

    elif input_path.endswith(".json"):
        with open(input_path, "r") as f:
            entries = json.load(f)

    else:
        raise ValueError("Input data file format must be in JSON format")

    pipeline = MusicDataPipeline(entries, output_path)
    pipeline.run()
    pipeline.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, help="Path to input JSON or CSV.")
    parser.add_argument("-o", required=True, help="Output JSON path")
    args = vars(parser.parse_args())

    main(args["i"], args["o"])
