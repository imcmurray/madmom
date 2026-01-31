# Sample Audio Files

This directory contains sample audio files for testing beat and downbeat detection.

## Files

| File | BPM | Time Signature | Description |
|------|-----|----------------|-------------|
| (add your files here) | | | |

## Usage

First, set up and activate a virtual environment:

```bash
cd /path/to/madmom-modern

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install madmom-modern
pip install -e .
```

Then you can use the sample files in Python:

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

You can also test files using the web interface. Sample files will appear automatically on the main page.

```bash
# Make sure venv is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

cd webapp
pip install -r requirements.txt
python app.py

# Open http://localhost:5000 - sample files will be shown on the page
```
