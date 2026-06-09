# Media Transcription to JSON

`transcribe.py` extracts audio from video (`.mp4`, `.mov`, `.m4v`) and audio (`.mp3`, `.wav`, `.m4a`, `.flac`, `.aac`, `.ogg`, `.opus`, `.wma`, `.aiff`) files with ffmpeg, transcribes speech with faster-whisper, and writes one UTF-8 JSON file per input.

## Installation

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv-transcribe
source .venv-transcribe/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements-transcribe.txt
```

3. Install ffmpeg:

macOS with Homebrew:

```bash
brew install ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

Windows:

Install ffmpeg from https://ffmpeg.org/download.html, then add the `bin` folder containing `ffmpeg.exe` and `ffprobe.exe` to your `PATH`.

## Usage

Transcribe one file (video or audio):

```bash
python transcribe.py --input "/path/to/video.mp4" --output "/path/to/output_folder"
```

Transcribe every supported media file in a folder:

```bash
python transcribe.py --input "./audio" --output "./transcripts"
```

Scan subfolders too:

```bash
python transcribe.py --input "/path/to/videos" --output "/path/to/output_folder" --recursive
```

Use a larger model:

```bash
python transcribe.py --input "/path/to/video.mp4" --output "/path/to/output_folder" --model-size medium

```

CPU-friendly mode:

```bash
python transcribe.py --input "/path/to/video.mp4" --output "/path/to/output_folder" --device cpu --compute-type int8
```

## CLI Options

- `--input`: Required. A single media file (video or audio) or a folder of media files.
- `--output`: Required. Folder for JSON transcript files.
- `--model-size`: Whisper model size. Examples: `tiny`, `base`, `small`, `medium`, `large-v3`. Default: `small`.
- `--device`: `auto`, `cpu`, or `cuda`. Default: `auto`.
- `--compute-type`: faster-whisper compute type such as `int8`, `float16`, or `float32`. Default: `default`.
- `--language`: Optional language code such as `en`. Omit for automatic language detection.
- `--recursive`: Search subfolders when the input is a folder.
- `--overwrite`: Replace existing JSON files.
- `--beam-size`: Decoding beam size. Default: `5`.
- `--verbose`: Print detailed logs.

## Long Video Handling

The script writes extracted audio to a temporary WAV file, then streams segments from faster-whisper instead of building custom in-memory chunks. `vad_filter=True` is enabled to reduce work on long silent sections. For very long videos or limited hardware, use a smaller model such as `base` or `small`, and use `--device cpu --compute-type int8`.

## Example JSON Output

```json
{
  "source_file": "/path/to/video.mp4",
  "duration_seconds": 124.738,
  "language_detected": "en",
  "full_transcript": "Welcome to the demo. Today we will review the quarterly results.",
  "segments": [
    {
      "start_time": 0.48,
      "end_time": 2.92,
      "text": "Welcome to the demo."
    },
    {
      "start_time": 3.14,
      "end_time": 6.81,
      "text": "Today we will review the quarterly results."
    }
  ]
}
```

## Notes

- The first run downloads the selected Whisper model, so it can take longer and requires network access.
- Output files are named after the source file stem, for example `meeting.mp4` and `Pres #1.mp3` become `meeting.json` and `Pres #1.json`.
- Files with duplicate stems in different folders get numeric suffixes to avoid overwriting in the same output folder.
- Existing JSON files are skipped unless `--overwrite` is passed.
