# YOFLO

Real-time object detection and visual Q&A using Microsoft Florence-2. Like YOLO, but with language understanding.

## Install

```bash
pip install yoflo
```

## Quick Start

```bash
# Download model and run object detection on webcam
python -m yoflo -dm -od

# Or use the daemon for repeated fast queries
python -m yoflo.daemon start --model large
python -m yoflo.daemon query --image photo.jpg --action detect
python -m yoflo.daemon query --image photo.jpg --question "Is there a cat?" --action ask
```

## Features

### Object Detection (`-od`)

Detect objects in video streams or images.

```bash
# Detect all objects
python -m yoflo -dm -od

# Filter to specific classes
python -m yoflo -dm -od person car dog

# From RTSP stream
python -m yoflo -dm -od -rtsp rtsp://192.168.1.100:554/stream

# From YouTube live stream
python -m yoflo -dm -od -yt "https://www.youtube.com/watch?v=..."
```

### Yes/No Questions (`-ph`)

Ask binary questions about what the model sees.

```bash
python -m yoflo -dm -ph "Is the door open?"
python -m yoflo -dm -ph "Is anyone wearing a hard hat?"
```

### Inference Chains (`-ic`)

Ask multiple questions per frame. Returns PASS if 2+ are true.

```bash
python -m yoflo -dm -ic "Is there a person?" "Are they seated?" "Are they at a desk?"
```

### Headless Mode (`-hl`)

No GUI window. Use on servers or for better performance (~20% faster).

```bash
python -m yoflo -dm -od -hl
```

### Screenshot on Detection (`-ss`)

Auto-save timestamped screenshots when objects are detected.

```bash
python -m yoflo -dm -od cat -ss
```

### Logging (`-lf`)

Write detections to `alerts.log`.

```bash
python -m yoflo -dm -od -lf
```

### Video Recording (`-r`)

Record video based on triggers.

```bash
# Record when objects detected
python -m yoflo -dm -od -r od

# Record when inference is "yes", stop on "no"
python -m yoflo -dm -ph "Is there motion?" -r infy

# Record when inference is "no", stop on "yes"
python -m yoflo -dm -ph "Is the area clear?" -r infn
```

### Multi-Camera (`-wi`, `-rtsp`)

Monitor multiple feeds simultaneously.

```bash
# Multiple webcams
python -m yoflo -dm -od -wi 0 1 2

# Multiple RTSP streams
python -m yoflo -dm -od -rtsp rtsp://cam1/stream rtsp://cam2/stream
```

### Inference Rate Control (`-il`, `-ir`)

```bash
# Limit to 5 inferences per second
python -m yoflo -dm -od -il 5

# Display inference rate
python -m yoflo -dm -od -ir
```

### Pretty Print (`-pp`)

Formatted detection output instead of raw JSON.

```bash
python -m yoflo -dm -od -pp
```

### 4-Bit Quantization (`-4bit`)

Reduce VRAM usage with quantized model.

```bash
python -m yoflo -dm -od -4bit
```

### PTZ Camera Control

Control pan/tilt/zoom cameras via HID (USB) or ONVIF (IP network).

**HID PTZ** (Logitech PTZ Pro, etc.):
```bash
pip install hid

# Manual PTZ control with keyboard
python -m yoflo -dm -od -ptz

# Auto-track an object class
python -m yoflo -dm -od person -ptz track -to person
```

**ONVIF PTZ** (Hikvision, Dahua, Axis, most IP cameras):
```bash
pip install onvif-zeep
# or: pip install yoflo[onvif]

# Connect to ONVIF camera and auto-track
python -m yoflo -dm -od person -onvif 192.168.1.100 -onvif-user admin -onvif-pass password -to person

# ONVIF will auto-detect the camera's RTSP stream if no other source specified
```

| Flag | Description |
|------|-------------|
| `-ptz` | Enable HID PTZ (USB cameras) |
| `-onvif HOST` | Enable ONVIF PTZ (IP cameras) |
| `-onvif-port N` | ONVIF port (default: 80) |
| `-onvif-user USER` | ONVIF username (default: admin) |
| `-onvif-pass PASS` | ONVIF password |
| `-to CLASS` | Object class to auto-track |

## Daemon Mode

Keep the model loaded in memory for fast repeated queries. No 90-second reload each time.

```bash
# Start daemon (loads model once)
python -m yoflo.daemon start --model large

# Query instantly
python -m yoflo.daemon query --image photo.jpg --action detect
python -m yoflo.daemon query --image "https://youtube.com/watch?v=..." --action detect
python -m yoflo.daemon query --image photo.jpg --question "Do you see a car?" --action ask

# Check status
python -m yoflo.daemon status

# Stop daemon
python -m yoflo.daemon stop
```

Python client:

```python
from yoflo.client import YofloClient

client = YofloClient()
detections = client.detect("image.jpg")
answer = client.ask("image.jpg", "Is there a person?")
```

## Model Options

| Model | Size | Flag |
|-------|------|------|
| Florence-2-base-ft | 233M params | `--model base` |
| Florence-2-large-ft | 770M params | `--model large` |

Use `-dm` to auto-download, or `-mp /path/to/model` for local models.

## All Flags

| Flag | Description |
|------|-------------|
| `-dm` | Download model from HuggingFace |
| `-mp PATH` | Use local model directory |
| `-od [CLASSES]` | Object detection (optional class filter) |
| `-ph QUESTION` | Yes/no question |
| `-ic Q1 Q2 ...` | Inference chain (multiple questions) |
| `-hl` | Headless mode (no display) |
| `-ss` | Screenshot on detection |
| `-lf` | Log to alerts.log |
| `-ir` | Display inference rate |
| `-il N` | Limit to N inferences/second |
| `-pp` | Pretty print output |
| `-wi 0 1 2` | Webcam indices |
| `-rtsp URL ...` | RTSP stream URLs |
| `-yt URL` | YouTube live stream URL |
| `-r od\|infy\|infn` | Recording trigger mode |
| `-4bit` | 4-bit quantization |
| `-ptz [track]` | HID PTZ control (USB cameras) |
| `-onvif HOST` | ONVIF PTZ control (IP cameras) |
| `-onvif-port N` | ONVIF port (default: 80) |
| `-onvif-user USER` | ONVIF username |
| `-onvif-pass PASS` | ONVIF password |
| `-to CLASS` | Object class to track |

## Requirements

- Python 3.8+
- CUDA GPU with 8GB+ VRAM (16GB recommended for large model)
- Works on Windows, Linux, macOS

## License

MIT
