# Sample Audio Files

This directory contains sample audio files for testing beat and downbeat detection.

## Files

| File | BPM | Time Signature | Description |
|------|-----|----------------|-------------|
| (add your files here) | | | |

## Usage

```python
import madmom

# Beat detection
proc = madmom.features.beats.RNNBeatProcessor()
act = proc('samples/your_file.mp3')
beats = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)(act)
print(f"Beats: {beats}")

# Downbeat detection
proc = madmom.features.downbeats.RNNDownBeatProcessor()
act = proc('samples/your_file.mp3')
downbeats = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
    beats_per_bar=[4], fps=100
)(act)
print(f"Downbeats: {downbeats}")
```

## Web Interface

You can also test files using the web interface:

```bash
cd webapp
python app.py
# Open http://localhost:5000 and upload a sample file
```
