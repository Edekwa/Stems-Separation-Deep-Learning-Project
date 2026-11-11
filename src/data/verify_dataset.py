from pathlib import Path

import soundfile as sf

from src.config import AUDIO_FILES, N_CHANNELS, SAMPLE_RATE, TEST_DIR, TRAIN_DIR


def verify_split(directory: Path, expected_tracks: int) -> None:
    """Verify the structure and audio metadata of one dataset split."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")

    tracks = sorted(path for path in directory.iterdir() if path.is_dir())

    print(f"\n{directory.name.upper()}: {len(tracks)} tracks")

    if len(tracks) != expected_tracks:
        raise ValueError(f"Expected {expected_tracks} tracks, found {len(tracks)}")

    for track in tracks:
        reference_frames = None

        for stem in AUDIO_FILES:
            audio_path = track / f"{stem}.wav"

            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)

            info = sf.info(audio_path)

            if info.samplerate != SAMPLE_RATE:
                raise ValueError(f"Incorrect sample rate: {audio_path}")

            if info.channels != N_CHANNELS:
                raise ValueError(f"Incorrect channel count: {audio_path}")

            if reference_frames is None:
                reference_frames = info.frames
            elif info.frames != reference_frames:
                raise ValueError(f"Inconsistent audio duration: {audio_path}")

    print("All tracks verified successfully.")


if __name__ == "__main__":
    verify_split(TRAIN_DIR, expected_tracks=100)
    verify_split(TEST_DIR, expected_tracks=50)

    print("\nMUSDB18-HQ verification completed!")
