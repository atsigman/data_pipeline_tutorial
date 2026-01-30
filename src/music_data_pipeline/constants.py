import torch
import torchaudio.functional as F


"""
Audio data preprocessing
"""
# The similarity threshold between 2 audio files:
DEFAULT_SIM_THRES = 0.9

# Silence RMS threshold:
DEFAULT_SILENCE_THRES = 0.005

# Minimum inter-onset interval (in seconds) between 2 onsets in
# the silent region detection context:
DEFAULT_SILENT_REGION_THRES = 2


# Maximum audio chunk duration (in seconds):
DEFAULT_MAX_CHUNK_DUR = 600

"""
Metadata (text) preprocessing
"""
BLACKLIST_GENRES = {
    "podcast",
    "audiobook",
    "spoken word",
    "documentary",
    "field recording",
}


"""
Audio Dataset
"""
DEFAULT_SR = 44100
DEFAULT_CROP_DUR = 10
# Default resolution for gradating/quantizing onset crop ranges
# (in seconds):
DEFAULT_CROP_RES = 0.5

# Filter query to apply to input data:
DEFAULT_FILTER_QUERY = {
    "duration": {"$gt": 10},
    "blacklist_flags": [],
    "silent_regions": {"$exists": True}
}

# Filter query operator dispatch table
# (for applying filter query to pseudo-Mongo collection dictionary):
OP_DICT = {"$eq": lambda val, op_val: val == op_val,
           "$gt": lambda val, op_val: val > op_val,
           "lt": lambda val, op_val: val < op_val,
           "$gte": lambda val, op_val: val >= op_val,
           "$lte": lambda val, op_val: val <= op_val,
           "$in": lambda val, op_val: val in op_val,
           "$nin": lambda val, op_val: val not in op_val,
           "$exists": lambda val, op_val: (val is not None) == op_val,
}

# Hashmap of audio augmentation operations:
AUGMENTATION_HM = {
        "none": lambda audio: audio,
        "invert": lambda audio: -audio,
        "flip_chan": lambda audio: torch.flip(audio, dims=[0]),
        "gain": lambda audio, low=0.8, high=1.1: audio * torch.empty(1).uniform_(low, high),
        "noise": lambda audio: audio + 0.01 * torch.rand_like(audio),
        "pitch_shift": lambda audio, sr, min_shift=-2.0, max_shift=2.0: F.pitch_shift(
            audio,
            sample_rate=sr,
            n_steps=torch.empty(1).uniform_(min_shift, max_shift).item()
        ),
    }
