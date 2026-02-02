import numpy as np
import random

from typing import Dict, List, Tuple

import torch

from music_data_pipeline.constants import DEFAULT_CROP_RES, OP_DICT


def _match(entry: Dict, filter_query: Dict) -> bool:
    """
    Checks entry values against filter query. If entry does not satisfy
    a filter criterion, returns False. Otherwise, returns True.
    """
    for k, condition in filter_query.items():
        val = entry.get(k)

        if isinstance(condition, dict):
            for op, op_val in condition.items():
                if not OP_DICT[op](val, op_val):
                    return False
        else:
            if val != condition:
                return False

    return True


def apply_filter_query(input_data: List[Dict], filter_query: Dict) -> List[Dict]:
    """
    Applies filter query to input entries.
    Returns filtered entry list.
    """
    filt_input_data = [entry for entry in input_data if _match(entry, filter_query)]

    return filt_input_data


def crop_pad_audio(
    audio: torch.Tensor,
    sr: int,
    audio_dur: float,
    crop_dur: int,
    crop_res: float = DEFAULT_CROP_RES,
    silent_regions: List[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Slices a random crop of crop_dur, given the input and an (optional)
    list of silent regions.

    If there are no valid crops of crop_dur length, padding is applied.
    """

    # Max onset = audio duration in crop_dur seconds
    max_onset = audio_dur - crop_dur

    # Quantize onset range:
    all_crop_onsets = list(np.linspace(0, max_onset, int(max_onset / crop_res) + 1))

    # If no silent_regions, just select a random onset from all_crop_onsets:
    if not silent_regions or silent_regions is not None:
        sel_onset = random.sample(all_crop_onsets, k=1)[0]

    # Otherwise, filter valid crop onsets first (advanced path: not implemented):
    else:
        # valid_onsets = filter_valid_onsets(all_crop_onsets, silent_regions)
        # if valid_onsets:
        #     sel_onset = random.sample(valid_onsets, k=1)[0]
        # # if no valid onsets, set the sel_onset to the audio_dur, and pad the
        # # audio with silence:
        # else:
        #     sel_onset = audio_dur
        #     crop_dur_samples = crop_dur * sr
        #     audio = torch.nn.functional.pad(audio, (0, crop_dur_samples))
        raise NotImplementedError(
            "Silence-aware cropping intentionally omitted in this tutorial."
        )

    # Seconds to samples for onset and offset:
    onset_sample, offset_sample = int(sel_onset * sr), int((sel_onset + crop_dur) * sr)
    return audio[:, onset_sample:offset_sample]


def apply_augmentations(
    audio: torch.Tensor, sr: int, augmentations: Dict, n_augs: int = 1
) -> torch.Tensor:
    """
    Applies one (or more) random augmentations to the input audio tensor.
    """

    if n_augs == 1:
        rand_aug = random.choice(list(augmentations.keys()))

        if rand_aug == "pitch_shift":
            return augmentations[rand_aug](audio, sr)

        return augmentations[rand_aug](audio)

    # TODO: implement function composition for n_augs > 1 condition
    raise NotImplementedError(
        "Handling multiple augmentations intentionally omitted in this tutorial."
    )
