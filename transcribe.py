#!/usr/bin/env python3
"""
Transcribe video and audio files into structured JSON files.

Supported inputs:
- video: .mp4, .mov, .m4v
- audio: .mp3, .wav, .m4a, .flac, .aac, .ogg, .opus, .wma, .aiff

Inputs can be:
- one media file
- one folder containing media files
- one folder plus an explicit list of media filenames to process
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma", ".aiff"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


class TranscriptionError(RuntimeError):
    """Raised when one media file cannot be processed."""


@dataclass(frozen=True)
class MediaJob:
    source_file: Path
    output_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe video (.mp4/.mov/.m4v) and audio (.mp3/.wav/.m4a/...) files to structured JSON."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to one media file (video or audio) or a folder containing media files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Folder where JSON transcript files will be written.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help=(
            "Optional list of media filenames to process from the input folder. "
            "Example: --input ./audio --files clip1.mp3 clip2.mp3"
        ),
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        help=(
            "Optional text file containing one media filename per line, relative "
            "to the input folder. Blank lines and lines starting with # are ignored."
        ),
    )
    parser.add_argument(
        "--model-size",
        default="small",
        help=(
            "Whisper model size for faster-whisper. Common values: tiny, base, "
            "small, medium, large-v3. Default: small."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used by faster-whisper. Default: auto.",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help=(
            "faster-whisper compute type. Examples: int8, float16, float32. "
            "Use 'default' to let faster-whisper choose."
        ),
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code such as 'en'. Omit to auto-detect.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a folder, scan subfolders recursively.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON output files.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size used for decoding. Default: 5.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg was not found on PATH. Install ffmpeg first, then rerun this script."
        )
    if shutil.which("ffprobe") is None:
        raise SystemExit(
            "ffprobe was not found on PATH. Install ffmpeg first, then rerun this script."
        )


def validate_media_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Listed media file does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"Listed media path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise SystemExit(
            f"Unsupported listed media extension: {path.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path


def read_file_list(file_list_path: Path) -> list[str]:
    file_list_path = file_list_path.expanduser().resolve()
    if not file_list_path.exists():
        raise SystemExit(f"File list does not exist: {file_list_path}")
    try:
        lines = file_list_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"Could not read file list {file_list_path}: {exc}") from exc
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def discover_listed_media(input_folder: Path, names: Iterable[str]) -> list[Path]:
    input_folder = input_folder.expanduser().resolve()
    if not input_folder.exists():
        raise SystemExit(f"Input folder does not exist: {input_folder}")
    if not input_folder.is_dir():
        raise SystemExit("--files and --file-list require --input to be a folder.")

    media_files: list[Path] = []
    for name in names:
        listed_path = Path(name)
        if listed_path.is_absolute():
            raise SystemExit(
                f"Use filenames relative to the input folder, not absolute paths: {name}"
            )
        media_files.append(validate_media_file(input_folder / listed_path))

    if not media_files:
        raise SystemExit("No media filenames were provided in --files or --file-list.")
    return media_files


def discover_media(
    input_path: Path,
    recursive: bool,
    filenames: list[str] | None,
    file_list_path: Path | None,
) -> list[Path]:
    listed_names: list[str] = []
    if filenames:
        listed_names.extend(filenames)
    if file_list_path:
        listed_names.extend(read_file_list(file_list_path))
    if listed_names:
        return discover_listed_media(input_path, listed_names)

    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return [validate_media_file(input_path)]

    pattern = "**/*" if recursive else "*"
    media_files = sorted(
        path.resolve()
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not media_files:
        raise SystemExit(f"No supported media files found in: {input_path}")
    return media_files


def make_jobs(media_files: Iterable[Path], output_dir: Path, overwrite: bool) -> list[MediaJob]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[MediaJob] = []
    seen_outputs: dict[Path, int] = {}
    for media in media_files:
        base_output = output_dir / f"{media.stem}.json"
        output_file = base_output

        if output_file in seen_outputs:
            seen_outputs[base_output] += 1
            output_file = output_dir / f"{media.stem}_{seen_outputs[base_output]}.json"
        else:
            seen_outputs[base_output] = 1

        if output_file.exists() and not overwrite:
            logging.info("Skipping existing output: %s", output_file)
            continue
        jobs.append(MediaJob(source_file=media, output_file=output_file))
    return jobs


def run_command(command: list[str], error_message: str) -> subprocess.CompletedProcess[str]:
    logging.debug("Running command: %s", " ".join(command))
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise TranscriptionError(f"{error_message}: command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise TranscriptionError(f"{error_message}: {details}") from exc


def probe_duration_seconds(media_path: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        f"Could not read media duration for {media_path}",
    )
    try:
        return round(float(result.stdout.strip()), 3)
    except ValueError as exc:
        raise TranscriptionError(
            f"ffprobe returned an invalid duration for {media_path}: {result.stdout!r}"
        ) from exc


def extract_audio(media_path: Path, audio_path: Path) -> None:
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ],
        f"Audio extraction failed for {media_path}",
    )
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise TranscriptionError(f"Audio extraction produced an empty file for {media_path}")


def create_model(model_size: str, device: str, compute_type: str) -> "WhisperModel":
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "Python package 'faster-whisper' is not installed. "
            "Run: pip install -r requirements-transcribe.txt"
        ) from exc

    kwargs: dict[str, str] = {}
    if compute_type != "default":
        kwargs["compute_type"] = compute_type
    return WhisperModel(model_size, device=device, **kwargs)


def transcribe_audio(
    model: Any,
    audio_path: Path,
    language: str | None,
    beam_size: int,
) -> tuple[str | None, list[dict[str, float | str]]]:
    try:
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            word_timestamps=False,
        )
        segments = [
            {
                "start_time": round(segment.start, 3),
                "end_time": round(segment.end, 3),
                "text": segment.text.strip(),
            }
            for segment in segments_iter
            if segment.text.strip()
        ]
        return info.language, segments
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed for {audio_path}: {exc}") from exc


def build_transcript_json(
    source_file: Path,
    duration_seconds: float,
    language_detected: str | None,
    segments: list[dict[str, float | str]],
) -> dict[str, object]:
    full_transcript = " ".join(str(segment["text"]) for segment in segments).strip()
    return {
        "source_file": str(source_file),
        "duration_seconds": duration_seconds,
        "language_detected": language_detected,
        "full_transcript": full_transcript,
        "segments": segments,
    }


def write_json(output_file: Path, payload: dict[str, object]) -> None:
    try:
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except OSError as exc:
        raise TranscriptionError(f"Could not write JSON output {output_file}: {exc}") from exc


def process_media(
    job: MediaJob,
    model: Any,
    language: str | None,
    beam_size: int,
) -> None:
    logging.info("Processing: %s", job.source_file)
    with tempfile.TemporaryDirectory(prefix="media_transcribe_") as temp_dir:
        audio_path = Path(temp_dir) / f"{job.source_file.stem}.wav"
        duration_seconds = probe_duration_seconds(job.source_file)
        extract_audio(job.source_file, audio_path)
        language_detected, segments = transcribe_audio(
            model=model,
            audio_path=audio_path,
            language=language,
            beam_size=beam_size,
        )
        payload = build_transcript_json(
            source_file=job.source_file,
            duration_seconds=duration_seconds,
            language_detected=language_detected,
            segments=segments,
        )
        write_json(job.output_file, payload)
    logging.info("Wrote: %s", job.output_file)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    media_files = discover_media(args.input, args.recursive, args.files, args.file_list)
    jobs = make_jobs(media_files, args.output, args.overwrite)
    if not jobs:
        logging.info("No media files need processing.")
        return 0

    require_ffmpeg()
    logging.info("Loading faster-whisper model: %s", args.model_size)
    try:
        model = create_model(args.model_size, args.device, args.compute_type)
    except TranscriptionError as exc:
        logging.error("%s", exc)
        return 1
    except Exception as exc:
        logging.error("Could not load Whisper model: %s", exc)
        return 1

    failures = 0
    for job in jobs:
        try:
            process_media(job, model, args.language, args.beam_size)
        except TranscriptionError as exc:
            failures += 1
            logging.error("%s", exc)
        except KeyboardInterrupt:
            logging.warning("Interrupted by user.")
            return 130

    if failures:
        logging.error("Completed with %s failed file(s).", failures)
        return 1
    logging.info("Completed %s transcript file(s).", len(jobs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
