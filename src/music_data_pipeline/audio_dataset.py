import torch
import torchaudio

from math import ceil
from typing import Dict, List

from torch.utils.data import Dataset
from torchaudio.functional import resample

from music_data_pipeline.conditioners import TextCondition

from music_data_pipeline.constants import (
    DEFAULT_FILTER_QUERY,
    DEFAULT_SR,
    DEFAULT_CROP_DUR,
    AUGMENTATION_HM,
)

from music_data_pipeline.util.dataset_utils import (
    apply_filter_query,
    crop_pad_audio,
    apply_augmentations,
    generate_description,
)


class AudioDataset(Dataset):
    """
    Dataset class for handling audio with text conditions.
    A filter query is applied to the given input data/collection.
    Crops (or pads) audio, optionally applies augmentations and/or
    an audio transform (e.g., MelSpectrogram or MFCC), and creates
    a TextCondition based upon entry (text) metadata.

    Returns a processed audio tensor and a TextCondition.
    """

    def __init__(
        self,
        input_data: List[Dict],
        filter_query: Dict = DEFAULT_FILTER_QUERY,
        target_sr: int = DEFAULT_SR,
        crop_dur: int = DEFAULT_CROP_DUR,
        random_crop: bool = True,
        augmentations: Dict = AUGMENTATION_HM,
        transform=None,
    ):
        print(f"Pre-filter: {len(input_data)} entries.")

        if filter_query is not None:
            self.data = apply_filter_query(input_data, filter_query)
            print(f"Post-filter: {len(self.data)} entries.")

        else:
            self.data = input_data

        self.target_sr = target_sr
        self.crop_dur = crop_dur
        self.random_crop = random_crop
        self.augmentations = augmentations
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # load audio:
        backend = "ffmpeg" if entry["audio_path"].endswith(".mp3") else "soundfile"

        audio, sr = torchaudio.load(entry["audio_path"], backend=backend)

        # Convert to mono (as dataset may contain a mix of stereo and mono files):
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample, if source sr != target_sr
        if sr != self.target_sr:
            audio = resample(audio, orig_freq=sr, new_freq=self.target_sr)

        audio_dur = entry["duration"]
        # Crop audio:
        if audio_dur > self.crop_dur:
            if self.random_crop:
                trimmed_audio = crop_pad_audio(
                    audio, self.target_sr, audio_dur, self.crop_dur
                )

            # By default, take the first self.crop_dur seconds:
            else:
                trimmed_audio = audio[:, : self.crop_dur * self.target_sr]

        # Pad if necessary:
        elif audio_dur < self.crop_dur:
            delta = ceil((self.crop_dur - entry["duration"]) * self.target_sr)
            trimmed_audio = torch.nn.functional.pad(audio, (0, delta))

        else:
            trimmed_audio = audio

        # No need to keep in memory:
        del audio

        trimmed_audio = self._adjust_trimmed_duration(trimmed_audio)

        # Apply random augmentation(s):
        if self.augmentations is not None:
            trimmed_audio = apply_augmentations(
                trimmed_audio, self.target_sr, self.augmentations
            )

        # Apply audio transform, if a transform is provided:
        if self.transform is not None:
            trimmed_audio = self.transform(trimmed_audio)

        # Add a template-based description:
        if "description" not in entry:
            description = generate_description(entry)
            entry["description"] = description

        # Construct TextCondition:
        text_condition = TextCondition(
            artist=entry.get("artist", None),
            album_title=entry.get("album_title", None),
            track_title=entry.get("track_title", None),
            genres=entry.get("genres", None),
            tempo=entry.get("tempo", None),
            description=entry["description"],
        )

        return trimmed_audio, text_condition

    def _adjust_trimmed_duration(self, trimmed_audio: torch.Tensor) -> torch.Tensor:
        """
        Verify that trimmed audio is self.crop_dur duration.
        Pad and/or crop accordingly if there are discrepancies.
        """
        trimmed_audio_dur = trimmed_audio.shape[1] / self.target_sr

        # If trimmed_audio_dur is < the crop dur, pad
        # (There may be slight numerical accuracy issues):
        if trimmed_audio_dur < self.crop_dur:
            delta = ceil((self.crop_dur - trimmed_audio_dur) * self.target_sr)
            trimmed_audio = torch.nn.functional.pad(trimmed_audio, (0, delta))

        # Assure that the audio duration is exactly self.crop_dur seconds:
        trimmed_audio = trimmed_audio[:, : self.crop_dur * self.target_sr]
        trimmed_audio_dur = trimmed_audio.shape[1] / self.target_sr

        # Ensure duration parity:
        fail_msg = (
            f"Audio duration is {trimmed_audio_dur} seconds; expected {self.crop_dur}"
        )
        assert trimmed_audio_dur == self.crop_dur, fail_msg

        return trimmed_audio
