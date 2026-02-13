import uuid
import torch
import torchaudio

from copy import deepcopy
from pathlib import Path, PurePath
from typing import Dict, List, Set, Tuple, Union

from tqdm import tqdm

import music_data_pipeline.constants as C


def validate_prune_data(
    entries: List[Dict], metadata_tags: Set = C.DEFAULT_METADATA_TAGS
) -> List[Dict]:
    """
    Removes entries lacking audio files or at least 1 metadata tag.
    """

    # No audio path:
    no_audio_indices = [
        i
        for i, e in enumerate(entries)
        if not e["audio_path"] or e["audio_path"] is None
    ]

    if no_audio_indices:
        print(f"{len(no_audio_indices)} entries with no audio path.")

    # No relevant metadata tags:
    no_metadata_indices = [
        i for i, e in enumerate(entries) if not any(t in e for t in metadata_tags)
    ]

    if no_metadata_indices:
        print(f"{len(no_metadata_indices)} entries with no audio path.")


    entries = [
        e
        for i, e in enumerate(entries)
        if i not in no_audio_indices and i not in no_metadata_indices
    ]

    print(f"{len(entries)} remaining entries.")

    return entries


"""
Audio data
"""


def _get_audio_duration(audio: torch.Tensor, sr: int) -> float:
    """
    Extracts audio duration, given audio tensor.
    """
    return round(audio.shape[1] / sr, 3)


def _compute_embedding(audio: torch.Tensor, sr: int) -> torch.Tensor:
    """
    Returns a fixed-length normalized log mel embedding for duplicate detection.
    """
    # Stereo -> mono
    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=64,
        n_fft=1024,
        hop_length=512,
    )(audio)

    log_mel = torch.log(mel + 1e-6)
    embedding = log_mel.mean(dim=-1).squeeze(0)  # shape: [64]

    return torch.nn.functional.normalize(embedding, dim=0)


def compute_all_embeddings(entries: List[Dict]) -> Tuple[List[Dict], List[int]]:
    """
    Computes fixed-length log mel spectrogram embeddings for all entries.
    Logs embedding, sample rate, audio path, and duration to a dictionary.

    If the audio filepath or content for a given entry is corrupted, the entry's id is
    appended to a remove_ids list.

    Returns the embeddings and remove_ids lists.
    """
    embeddings, remove_ids = [], []

    for i, e in enumerate(tqdm(entries, desc="Computing embeddings")):
        try:
            audio, sr = torchaudio.load(e["audio_path"])
            duration = _get_audio_duration(audio, sr)
            embedding = _compute_embedding(audio, sr)
            embedding_entry = {
                "_id": e["_id"],
                "track_id": e["track_id"],
                "embedding": embedding,
                "sample_rate": sr,
                "audio_path": e["audio_path"],
                "duration": duration,
            }
            embeddings.append(embedding_entry)

        # If file or filepath is corrupted, append the _id to
        # remove_ids, and a dictionary containing only the _id
        # to embeddings (to ensure length parity between entries and embeddings):
        except Exception as e:
            print(e)
            remove_ids.append(e["_id"])
            embeddings.append({"_id": e["_id"]})

    return embeddings, remove_ids


def find_similar_audio(
    entries: List[Dict],
    sim_thres: float = C.DEFAULT_SIM_THRES,
    dur_delta_thres: float = C.DEFAULT_DUR_DELTA_THRES,
) -> List[Dict]:
    """
    Compares each pair of audio files, and flags (potential) duplicates.
    Returns (potentially modified) entry list.

    If 2 audio *paths* are identical, flags both entries as "duplicate_audio_paths".
    If the *content* of 2 audio files is highly similar, flags both entries as
    "duplicate_audio_content."

    Also computes and logs audio duration.
    """

    # Compute audio embeddings and compile ids of corrupted audio files/paths.
    embeddings, remove_ids = compute_all_embeddings(entries)

    # If remove_ids is nonempty, prune the entries list accordingly.
    # This will align embeddings with entries.
    if remove_ids:
        entries = [e for e in entries if e["_id"] not in remove_ids]
        print(f"{len(remove_ids)} entries removed, ",
              f"{len(entries)} entries remaining.")

    # Sanity check that embeddings and entries lists are aligned:
    assert len(embeddings) == len(entries), "Embeddings and entries lists should the same length."

    # Compare each embedding against every other embedding:
    for i, e_1 in enumerate(tqdm(embeddings, desc="Duplicate detection")):

        # Copy duration from embeddings list to entries list:
        if "duration" not in entries[i]:
            entries[i]["duration"] = e_1["duration"]

        # Compare against all other audio:
        for j in range(i + 1, len(embeddings)):
            e_2 = embeddings[j]

            def add_blacklist_flags(flag: str, sim_score: float = None) -> None:
                """
                Add given flag to 2 similar entries, and cross reference
                track_ids. If audio sim_score is provided, adds to both entries.
                """
                cross_ref_key = flag + "_of"
                for idx, track_id in zip([i, j], [e_2["track_id"], e_1["track_id"]]):
                    if flag not in entries[idx]["blacklist_flags"]:
                        entries[idx]["blacklist_flags"].append(flag)
                    entries[idx][cross_ref_key] = track_id

                    # Add similarity score, if given:
                    if sim_score is not None:
                        entries[idx]["audio_similarity_score"] = sim_score

            # If audio path the same as for e_1, add "duplicate_audio_path" flags to both:
            if e_1["audio_path"] == e_2["audio_path"]:
                add_blacklist_flags("duplicate_audio_path")
                continue

            # If differing sample rates or durations, continue:
            if (
                e_1["sample_rate"] != e_2["sample_rate"]
                or abs(e_1["duration"] - e_2["duration"]) > dur_delta_thres
            ):
                continue

            # Compute cosine similarity between embeddings:
            sim = torch.dot(e_1["embedding"], e_2["embedding"]).item()
            print(sim)

            if sim > sim_thres:
                add_blacklist_flags("duplicate_audio_content",
                                    sim_score=sim)

    return entries


def chunk_audio(
    entries: List[Dict],
    min_chunk_dur: int = C.DEFAULT_CROP_DUR,
    max_chunk_dur: int = C.DEFAULT_MAX_CHUNK_DUR,
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

    for i, e in tqdm(long_dur_entries, desc="Audio chunking"):
        audio, sr = torchaudio.load(e["audio_path"])
        total_dur = e["duration"]
        max_chunk_samples = max_chunk_dur * sr
        n_chunks = int(total_dur // max_chunk_dur)
        # If the remainder >= min_chunk_dur, add 1 chunk
        # (of some duration < max_chunk_dur):
        if total_dur % max_chunk_dur >= min_chunk_dur:
            n_chunks += 1

        print(f"Segmenting entry {i} into {n_chunks} chunks...")

        pure_path = PurePath(e["audio_path"])

        for j in range(n_chunks):
            if j == n_chunks - 1:
                audio_seg = audio[:, max_chunk_samples * j :]
            else:
                audio_seg = audio[
                    :, max_chunk_samples * j : max_chunk_samples * (j + 1)
                ]

            # Compute chunk duration:
            chunk_dur = round(audio_seg.shape[1] / sr, 3)

            # Define segment audio path
            # audio directory + {basename_partition}.wav:
            new_filename = pure_path.name[:-4] + f"_{j}" + ".wav"
            new_path = str(Path(pure_path.parent, new_filename))

            # Write audio segment to wav file:
            torchaudio.save(new_path, audio_seg, sr, format="WAV")

            # Update or create entry. Include partition index.
            # Nb: track_id is shared among partitions.
            if j == 0:
                entries[i]["audio_path"] = new_path
                entries[i]["partition"] = j
                entries[i]["duration"] = chunk_dur
                entries[i]["start_sec"] = 0
            else:
                new_entry = deepcopy(e)
                new_entry["_id"] = str(uuid.uuid1())  # assign a unique ID
                new_entry["audio_path"] = new_path
                new_entry["partition"] = j
                new_entry["duration"] = chunk_dur
                new_entry["start_sec"] = max_chunk_dur * j

                entries.append(new_entry)

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

    start = round(onset_seconds[0].item(), 3)

    # If the first onset is > silent_region_thres seconds,
    # the first silent region = (0, start)
    if start > silent_region_thres:
        silent_regions.append((0, start))

    for sec in onset_seconds:
        sec = round(sec.item(), 3)
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
    silence_thres: float = C.DEFAULT_SILENCE_THRES,
    silent_region_thres: float = C.DEFAULT_SILENT_REGION_THRES,
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


"""
Text preprocessing
"""


def _tokenize(text: Union[str, List[str]], strip_chars: str) -> Union[str, List[str]]:
    """
    Tokenizes individual strings or lists of strings.
    """

    def tidy_text(text, strip_chars):
        # Lower case, remove any hyphenation, and trim whitespace:
        table = str.maketrans({c: " " for c in strip_chars})
        text = text.translate(table)
        return text.lower().strip()

    if isinstance(text, str):
        return tidy_text(text, strip_chars)

    if isinstance(text, list):
        for i, el in enumerate(text):
            text[i] = tidy_text(el, strip_chars)
        return text

    raise TypeError("Text must be either a string or a list of strings.")


def tokenize_metadata(
    entries: List[Dict],
    metadata_tags: List[str] = C.DEFAULT_METADATA_TAGS,
    strip_chars: str = C.CHARS_TO_STRIP,
):
    """
    Tokenizes text metadata for all entries.
    Returns modified entry list.
    """
    for i, e in enumerate(tqdm(entries, desc="Tokenizing metadata")):
        for tag in metadata_tags:
            if tag in e:
                tokenized_tag = _tokenize(e[tag], strip_chars)
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
    entries: List[Dict], blacklist_genres: Set[str] = C.BLACKLIST_GENRES
) -> List[Dict]:
    """
    Adds "bad_genre" blacklist flag for entries containing blacklisted genres.
    """
    blacklist_tally = 0
    for i, e in enumerate(entries):
        if "genres" in e and _contains_blacklist_genre(e, blacklist_genres):
            entries[i]["blacklist_flags"].append("bad_genre")
            blacklist_tally += 1

    print(f"{blacklist_tally} entries with at least one blacklisted genre.")
    return entries
