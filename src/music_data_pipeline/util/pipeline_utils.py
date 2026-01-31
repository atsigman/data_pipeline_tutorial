import torch
import torchaudio

from copy import deepcopy
from pathlib import Path, PurePath
from typing import Dict, List, Set, Tuple, Union

from tqdm import tqdm

from music_data_pipeline.constants import (
    DEFAULT_SIM_THRES,
    DEFAULT_SILENCE_THRES,
    DEFAULT_SILENT_REGION_THRES,
    DEFAULT_MAX_CHUNK_DUR,
    DEFAULT_CROP_DUR,
    DEFAULT_METADATA_TAGS,
    BLACKLIST_GENRES,
)


def validate_prune_data(
    entries: List[Dict], metadata_tags: DEFAULT_METADATA_TAGS
) -> List[Dict]:
    """
    Removes entries lacking audio files or at least 1 metadata tag.
    """

    # No audio path:
    no_audio_indices = [i for i, _ in enumerate(entries) if not entries["audio_path"]]

    # No relevant metadata tags:
    no_metadata_indices = [
        i for i, e in enumerate(entries) if not any(t in e for t in metadata_tags)
    ]

    return [
        e
        for i, e in enumerate(entries)
        if i not in no_audio_indices and i not in no_metadata_indices
    ]


"""
Audio data
"""


def _get_audio_duration(audio: torch.Tensor, sr: int) -> float:
    """
    Extracts audio duration, given audio tensor.
    """
    return round(audio.shape[1] / sr, 3)


def _compute_audio_similarity(
    entries: List[Dict],
    audio_1: torch.Tensor,
    audio_2: torch.Tensor,
    sim_thres: float,
    i: int,
    j: int,
) -> List[Dict]:
    """
    Computes mean absolute difference between waveforms.
    If the similarity is > sim_thres, "duplicate" is appended to
    the audio_2 entry's blacklist_flags list.

    Returns (possibly modified) entries.
    """
    audio_sim = torch.abs(audio_1 - audio_2).mean()
    if audio_sim > sim_thres:
        sim_perc = round(audio_sim * 100, 1)
        print(f"{sim_perc}% similarity detected between entries {i} and {i + j}...")

        entries[i + j]["blacklist_flags"].append("duplicate")

    return entries


def find_similar_audio(
    entries: List[Dict], sim_thres: float = DEFAULT_SIM_THRES
) -> List[Dict]:
    """
    Compares each pair of audio files, and flags (potential) duplicates.
    Returns (potentially modified) entry list.

    If 2 audio *paths* are identical, the second pair member is removed.

    Also computes and logs audio duration.

    N.b.: by default, the pair member closer to the end of the entry list
    is flagged as a duplicate.
    """
    remove_idxs = []
    for i, e_1 in enumerate(tqdm(entries[:-1], desc="Duplicate detection")):
        audio_1, sr_1 = torchaudio.load(e_1["audio_path"])
        e_1_duration = _get_audio_duration(audio_1, sr_1)
        entries[i]["duration"] = e_1_duration

        # Compare against all other audio:
        for j, e_2 in enumerate(entries[i + 1 :]):

            # If audio path the same as for e_1, mark e_2 for deletion:
            if e_1["audio_path"] == e_2["audio_path"]:
                remove_idxs.append(i + j)

            # Continue if already flagged as a duplicate:
            if "duplicate" in e_2["blacklist_flags"]:
                continue

            # Load and extract duration:
            audio_2, sr_2 = torchaudio.load(e_2["audio_path"])
            e_2_duration = _get_audio_duration(audio_2, sr_2)
            entries[i + j]["duration"] = e_2_duration

            # If differing sample rates or durations, continue:
            if sr_1 != sr_2 or e_1_duration != e_2_duration:
                continue

            # Compute audio similarity and update entries:
            entries = _compute_audio_similarity(
                entries, audio_1, audio_2, sim_thres, i, j
            )

    # If there are any entries to remove,filter entries:
    if remove_idxs:
        print(f"Deleting {len(remove_idxs)} redundant entries...")
        entries = [e for i, e in enumerate(entries) if i not in remove_idxs]

    return entries


def _detect_silent_regions(
    audio: torch.Tensor,
    sr: int,
    total_dur: float,
    silence_thres: float,
    silent_region_thres: float,
) -> List[Tuple[float, float]]:
    """
    Detects silent regions, given a silence threshold and a minimum inter-onset interval.
    Returns a list of (onset, offset) tuples.
    """
    silent_regions = []

    # Convert waveform samples to frames, and compute energy (rms):
    frames = audio.unfold(-1, size=2048, step=512)
    energy = torch.sqrt(torch.mean(frames**2, dim=-1))

    # Valid onsets have energy > the silence_thres. Filter these onsets:
    onset_frames = torch.where(energy > silence_thres)[1]

    # Convert frames to seconds:
    # frames[i] = seconds[i] * sr / hop_size
    onset_seconds = onset_frames.float() * 512 / sr

    # If no detected onsets, the entire waveform is a silent region:
    if onset_seconds.shape[0] == 0:
        return [(0, total_dur)]

    start = onset_seconds[0]

    # If the first onset is > silent_region_thres seconds,
    # the first silent region = (0, start)
    if start > silent_region_thres:
        silent_regions.append((0, start))

    for sec in onset_seconds:
        delta = sec - start

        if delta > silent_region_thres:
            silent_regions.append((start, sec))

        # Update start to current onset sec:
        start = sec

    # If the final onset < total_dur by > silent_region_thres,
    # the final silent region = (sec, total_dur):
    if total_dur - start > silent_region_thres:
        silent_regions.append((start, total_dur))

    return silent_regions


def add_silent_regions(
    entries: List[Dict],
    silence_thres: float = DEFAULT_SILENCE_THRES,
    silent_region_thres: float = DEFAULT_SILENCE_THRES,
) -> List[Dict]:
    """
    Collects silent regions for all entries. Stores as (onset, offset) tuples.
    """
    for i, e in enumerate(tqdm(entries, desc="Silent region detection")):
        audio, sr = torchaudio.load(e["audio_path"])
        total_dur = _get_audio_duration(audio, sr)
        silent_regions = _detect_silent_regions(
            audio, sr, total_dur, silence_thres, silent_region_thres
        )
        entries[i]["silent_regions"] = silent_regions

    return entries


def chunk_audio(
    entries: List[Dict],
    min_chunk_dur=DEFAULT_CROP_DUR,
    max_chunk_dur=DEFAULT_MAX_CHUNK_DUR,
) -> List[Dict]:
    """
    Partitions audio of entries of duration > max_chunk_dur into
    multiple audio files. Updates source entry audio paths, and appends new entries
    (replicating source entry metadata). Saves new audio segments to
    the existing audio directory.
    """
    long_dur_entries = [
        (i, e) for i, e in enumerate(entries) if e["duration"] > max_chunk_dur
    ]

    if not long_dur_entries:
        return entries

    for i, e in long_dur_entries:
        audio, sr = torchaudio.load(e["audio_path"])
        total_dur = e["duration"]
        max_chunk_samples = max_chunk_dur * sr
        n_chunks = total_dur // max_chunk_dur
        # If the remainder >= min_chunk_dur, add 1 chunk
        # (of some duration < max_chunk_dur):
        if total_dur % max_chunk_dur >= min_chunk_dur:
            n_chunks += 1

        for j in range(n_chunks):
            if j == n_chunks - 1:
                audio_seg = audio[:, max_chunk_samples * j :]
            else:
                audio_seg = audio[
                    :, max_chunk_samples * j : max_chunk_samples * (j + 1)
                ]

            # Define segment audio path
            # audio directory + {basename_partition}.wav:
            pure_path = PurePath(e["audio_path"])
            new_filename = pure_path.name[:-4] + f"_{j}" + ".wav"
            new_path = Path(pure_path.parent, new_filename)

            # Write audio segment to wav file:
            torchaudio.save(new_path, audio_seg, format="WAV")

            # Update or create entry. Include partition index:
            if j == 0:
                entries[i]["audio_path"] = str(new_path)
                entries[i]["partition"] = j
            else:
                new_entry = deepcopy(e)
                new_entry["_id"] = entries[-1]["_id"] + 1
                new_entry["audio_path"] = str(new_path)
                new_entry["partition"] = j
                entries.append(new_entry)

    return entries


"""
Text preprocessing
"""


def _tokenize(text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Tokenizes individual strings or lists of strings.
    """

    def tidy_text(text):
        # Lower case, remove any hyphenation, and trim whitespace:
        return (
            text.lower().replace("-", " ").replace("/", " ").replace("_", " ").strip()
        )

    if isinstance(text, str):
        return tidy_text(text)

    if isinstance(text, list):
        for i, el in enumerate(text):
            text[i] = tidy_text(el)
        return text

    raise TypeError("Text must be either a string or a list of strings.")


def tokenize_metadata(
    entries: List[Dict], metadata_tags: List[str] = DEFAULT_METADATA_TAGS
):
    """
    Tokenizes text metadata for all entries.
    Returns modified entry list.
    """
    for i, e in enumerate(tqdm(entries, desc="Tokenizing metadata")):
        for tag in metadata_tags:
            if tag in e:
                tokenized_tag = _tokenize(e[tag])
                entries[i][tag] = tokenized_tag

    return entries


def _contains_blacklist_genre(entry: Dict, blacklist: Set[str]) -> bool:
    """
    Returns True if the list of genres for an entry contains a
    blacklisted item.
    """
    if any(genre in entry["genres"] for genre in blacklist):
        return True

    return False


def extract_blacklisted_genres(
    entries: List[Dict], blacklist_genres: Set[str] = BLACKLIST_GENRES
) -> List[Dict]:
    """
    Adds "bad_genre" blacklist flag for entries containing blacklisted genres.
    """
    for i, e in enumerate(entries):
        if "genre" in e and _contains_blacklist_genre(e, blacklist_genres):
            entries[i]["blacklist_flags"].append("bad_genre")

    return entries
