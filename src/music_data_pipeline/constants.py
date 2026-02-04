import random
import torch
import torchaudio.functional as F

"""
Audio data preprocessing
"""
# The similarity threshold between 2 audio files:
DEFAULT_SIM_THRES = 0.995

# Duration difference threshold between 2 audio files, in seconds:
DEFAULT_DUR_DELTA_THRES = 2.0

# Silence RMS threshold:
DEFAULT_SILENCE_THRES = 0.005

# Minimum inter-onset interval (in seconds) between 2 onsets in
# the silent region detection context:
DEFAULT_SILENT_REGION_THRES = 2

# Maximum audio chunk duration (in seconds):
DEFAULT_MAX_CHUNK_DUR = 180

"""
Metadata (text) preprocessing
"""
DEFAULT_METADATA_TAGS = {"artist", "album_title", "track_title", "genres"}

CHARS_TO_STRIP = "_/-"

BLACKLIST_GENRES = {
    "podcast",
    "audiobook",
    "audio book",
    "spoken word",
    "documentary",
    "field recording",
}


"""
Audio Dataset
"""
DEFAULT_SR = 16000
DEFAULT_CROP_DUR = 10
# Default resolution for gradating/quantizing onset crop ranges
# (in seconds):
DEFAULT_CROP_RES = 0.5

# Filter query to apply to input data:
DEFAULT_FILTER_QUERY = {
    "duration": {"$gt": 10},
    "blacklist_flags": [],
    "silent_regions": {"$exists": True},
}

# Filter query operator dispatch table
# (for applying filter query to pseudo-Mongo collection dictionary):
OP_DICT = {
    "$eq": lambda val, op_val: val == op_val,
    "$gt": lambda val, op_val: val > op_val,
    "lt": lambda val, op_val: val < op_val,
    "$gte": lambda val, op_val: val >= op_val,
    "$lte": lambda val, op_val: val <= op_val,
    "$in": lambda val, op_val: val in op_val,
    "$nin": lambda val, op_val: val not in op_val,
    "$exists": lambda val, op_val: (val is not None) == op_val,
}


def _apply_gain(audio, low=-0.8, high=1.2):
    """
    Randomly scales the waveform amplitude and clamps to the [-1, 1] range.
    """
    gain = torch.empty(1).uniform_(low, high)
    return torch.clamp(audio * gain, -1.0, 1.0)


# Hashmap of audio augmentation operations:
AUGMENTATION_HM = {
    "none": lambda audio: audio,
    "invert": lambda audio: -audio,
    "gain": lambda audio, low=0.8, high=1.2: _apply_gain(audio, low, high),
    "noise": lambda audio: audio + 0.01 * torch.rand_like(audio),
    "pitch_shift": lambda audio, sr, min_shift=-2.0, max_shift=2.0: F.pitch_shift(
        audio,
        sample_rate=sr,
        n_steps=torch.empty(1).uniform_(min_shift, max_shift).item(),
    ),
}


# Template for generating text descriptions,
# based upon available tag(s):
DESCRIPTION_TEMPLATE = {
    "artist": lambda x: f"by the artist {x.title()}",
    "album_title": lambda x: f"from the album {x.title()}",
    "tempo": lambda x: [f"at {x} BPM", f"at tempo {x}"],
    "genres": lambda x: [
        f"in the {random.choice(x)} genre",
        f"best categorized as {random.choice(x)}",
    ],
}
