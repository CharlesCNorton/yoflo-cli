# YOFLO-CLI

YOFLO-CLI is a command-line interface for the YO-FLO package, providing advanced object detection and binary inference capabilities using the Florence-2 vision-language model in real-time. This tool leverages state-of-the-art vision models to perform tasks such as object detection and binary inference based on referring expression comprehension.

## Features

### Object Detection

Identify and classify objects within a video feed. You can specify class names or display all detections if no class names are provided. Unlike traditional models limited to a fixed set of classes (e.g., COCO dataset), Florence-2 can detect an expansive array of objects due to its advanced vision-language capabilities. While it may be slower than models like YOLO for pure object detection, it excels when natural language processing (NLP) logic is required, allowing it to handle tasks that require understanding and reasoning about the visual input.

### Binary Inference

Answer yes/no questions based on the visual input. This feature leverages Florence-2’s ability to understand complex visual scenes and provide binary responses, making it ideal for simple decision-making tasks. For example, it can answer questions like "Is the person smiling?" by analyzing the visual input in real-time.

### Inference Chain

Evaluate multiple inferences and determine overall results based on a sequence of phrases. This allows for a more comprehensive context analysis within individual frames by examining multiple aspects of the scene. For example, to determine if a person is working, you might check if their eyes are open, their hands are on the keyboard, and they are facing the computer. This feature addresses the limitation that newer and smaller vision-language models are capable of answering simple questions, but not compound ones.

### Inference Rate Calculation

Measure the rate of inferences per second. This feature helps monitor the performance of the system and optimize for real-time processing, providing insights into how efficiently the model processes visual data.

### Real-time Processing

Process video feeds from a webcam in real-time. This enables immediate analysis and response to visual input, which is crucial for applications such as surveillance, live monitoring, and interactive systems.

### Screenshot on Detection

Automatically capture and save a screenshot when a target object is detected. This feature is useful for logging and reviewing detections, providing a visual record of the events.

### Logging Detections

Log detection events to a file. This creates a persistent record of detection events, which is useful for auditing, analysis, and troubleshooting. Detection events will be logged to a file named `alerts.log`.

### Headless Mode

Run the tool without displaying the video feed, suitable for server environments or automated systems where a display is not necessary. This mode is useful for running on servers or in background processes. Enabling this mode should result in an ~20% increase in inference speed.

### Pretty Print

Enable formatted output of detections for better readability. This makes it easier to interpret the results, especially when monitoring the output in real-time.

### Model Download

Option to download the Florence-2 model directly from the Hugging Face Hub. This simplifies the setup process by automating the model download and initialization.

### Multi-Webcam Support

Support for multiple webcams, allowing concurrent processing and inference on multiple video feeds. This is useful for surveillance systems, multi-view analysis, and other applications requiring inputs from several cameras.

### Video Recording

Added video recording functionality with conditions to trigger recording based on object detection and inference results. This feature allows capturing video segments of interest for further analysis or documentation.

## Model Information

This tool uses Microsoft's Florence-2, a powerful vision-language model designed to understand and generate detailed descriptions of visual inputs. Florence-2 combines advanced image processing with natural language understanding, making it ideal for complex tasks that require both visual and textual analysis. Florence-2 uses a unified sequence-to-sequence architecture to handle tasks from image-level understanding to fine-grained visual-semantic alignment. The model is trained on a large-scale, high-quality multitask dataset FLD-5B, which includes 126M images and billions of text annotations.

## Installation

### From Source

1. Clone the repository:
    ```sh
    git clone https://github.com/CharlesCNorton/yoflo-cli.git
    cd yoflo-cli
    ```

2. Install the package:
    ```sh
    pip install .
    ```

### From PyPI

You can also install YOFLO-CLI directly from PyPI:
```sh
pip install yoflo
```

## Usage

Run the script with the desired arguments. Below are the available flags and their descriptions:

### Flags

- `-mp`, `--model_path`: Path to the pre-trained model directory. Specify the directory where the Florence-2 model is located. This flag is mandatory if the model is not being downloaded.
- `-od`, `--object_detection`: Enable object detection. Optionally, you can specify class names to detect (e.g., `"cat"`, `"dog"`). If no class names are provided, all detections will be displayed.
- `-ph`, `--phrase`: Enable binary inference with a yes/no question based on the visual input. For example, "Is the person smiling?". This flag initiates the expression comprehension feature of the tool.
- `-hl`, `--headless`: Run in headless mode without displaying the video feed. This mode is useful for server environments or situations where a display is not available or necessary.
- `-ss`, `--screenshot`: Enable screenshot on detection. When a target object is detected, a screenshot will be automatically captured and saved with a timestamped filename.
- `-lf`, `--log_to_file`: Enable logging of detection events to a file. Detection events will be logged to a file named `alerts.log`, creating a persistent record of detection events.
- `-is`, `--display_inference_speed`: Display inference speed. This flag logs the rate of inferences per second, providing insight into the performance of the detection process.
- `-dm`, `--download_model`: Download the Florence-2 model from the Hugging Face Hub. This option can be used to download and initialize the model if it is not already available locally.
- `-pp`, `--pretty_print`: Enable pretty print for detections. This flag formats the output of detections for better readability, making it easier to interpret the results.
- `-il`, `--inference_limit`: Limit the inference rate to a specified number of inferences per second. This can help manage performance and ensure the system is not overloaded, providing a smoother operation.
- `-ic`, `--inference_chain`: Enable inference chain with specified phrases. Provide phrases in quotes, separated by spaces (e.g., `"Is it sunny?" "Is it raining?"`).
- `-wi`, `--webcam_indices`: Specify the indices of the webcams to use (e.g., `0 1 2`). If not provided, the first webcam (index 0) will be used by default.
- `-rtsp`, `--rtsp_urls`: Specify the RTSP URLs for the video streams.
- `-r`, `--record`: Enable video recording and specify the recording mode: 'od' to start/stop based on object detection, 'infy' to start on 'yes' inference and stop on 'no', and 'infn' to start on 'no' inference and stop on 'yes'.

## Inference Chain Feature

### Overview

The inference chain feature allows you to evaluate multiple inferences and determine an overall result based on a sequence of phrases. This capability leverages the power of the Florence-2 model to handle more complex logic that requires analyzing multiple aspects of the visual input.

### Importance

Vision-language models like Florence-2 examine each frame in isolation and can only answer simple, specific questions. By using an inference chain, you can string together multiple simple questions to examine more features and derive a more comprehensive understanding of the context. For example, asking the model if someone is sleeping isn't reliable just by the state of their eyelids; other context such as their posture or presence of a pillow is necessary.

### Example Use Case

Imagine you want to determine if a person is working. This might involve checking several conditions:

1. Is the person sitting?
2. Is the person typing?
3. Is the person awake?

The inference chain feature allows you to evaluate multiple inferences and determine an overall result based on a sequence of phrases. By setting up an inference chain with these phrases, the system can evaluate each condition separately and provide an overall result based on the combined outcomes. This capability leverages the power of YO-FLO to process more complex logic than Florence-2 could on its own.

### How to Use

To use the inference chain feature, specify the `--inference_chain` flag followed by the phrases you want to evaluate. Each phrase should be enclosed in quotes and separated by spaces.

#### Command Example:
```sh
python yoflo.py --model_path /path/to/Florence-2-base-ft --inference_chain "Is the person wearing glasses?" "Is the person wearing headphones?" "Is the person smiling?" --headless --display_inference_speed
```

## Example Commands

### Object Detection for Specific Classes
To perform object detection and only display detections for specific classes such as `"person"`:
```sh
python yoflo.py --model_path /path/to/Florence-2-base-ft --object_detection "person"
```

### Binary Inference (e.g., "Is the person smiling?")
To perform binary inference based on a yes/no question related to the visual input:
```sh
python yoflo.py --model_path /path/to/Florence-2-base-ft --phrase "Is the person smiling?"
```

### Inference Chain
To perform a sequence of inferences and determine overall results:
```sh
python yoflo.py --model_path /path/to/Florence-2-base-ft --inference_chain "Is the person wearing glasses?" "Is the

 person wearing headphones?" "Is the person smiling?" --headless --display_inference_speed
```

### Headless Mode
To run the tool in headless mode without displaying the video feed:
```sh
python yoflo.py --model_path /path/to/Florence-2-base-ft --object_detection "person" --headless
```

### Enable Screenshot on Detection
To enable screenshot capture whenever a target object is detected:
```sh
python yoflo.py --model_path /path/to/Florence-2-base-ft --object_detection "person" --screenshot
```

### Enable Logging of Detection Events to File
To log detection events to a file named `alerts.log`:
```sh
python yoflo.py --model_path /path/to/Florence-2-base-ft --object_detection "person" --log_to_file
```

### Display Inference Speed
To log and display the inference speed (inferences per second):
```sh
python yoflo.py --model_path /path/to/Florence-2-base-ft --object_detection "person" --display_inference_speed
```

### Download Model from Hugging Face
To download the Florence-2 model from the Hugging Face Hub:
```sh
python yoflo.py --download_model
```

### Pretty Print Detections
To enable formatted output of detections for better readability:
```sh
python yoflo.py --model_path /path/to/Florence-2-base-ft --object_detection "person" --pretty_print
```

### Limit Inference Rate
To limit the inference rate to a specified number of inferences per second, for example, 5 inferences per second:
```sh
python yoflo.py --model_path /path/to/Florence-2-base-ft --object_detection "person" --inference_limit 5
```

### Use Multiple Webcams
To use multiple webcams for object detection or inference:
```sh
python yoflo.py --model_path /path/to/Florence-2-base-ft --object_detection "person" --webcam_indices 0 1 --inference_limit 3
```

### Video Recording
To enable video recording based on object detection or inference results:
```sh
python yoflo.py --model_path /path/to/Florence-2-base-ft --object_detection "person" --record od
```

## Minimum Requirements for Running YOFLO

1. **Operating System**:
   - **Ubuntu 22.04** (or compatible Linux distribution)
   - **Windows 11** (Takes a lot of work, but it can be done!)

2. **Minimum Hardware**:
   - **CPU**: Intel Core i7
   - **GPU**:  16 GB VRAM
   - **RAM**:  32 GB RAM
   - **Camera**: USB camera connected

3. **Python Version**:
   - **Python 3.10**

4. **CUDA Version**:
   - **CUDA 12.1** 

5. **Environment Variables**:
   - Set the following in your `~/.bashrc` or equivalent:
     ```bash
     export CUDA_HOME=/usr/local/cuda
     export PATH=$CUDA_HOME/bin:$PATH
     export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
     source ~/.bashrc
     ```

6. **Required Python Packages**:
   - `torch`
   - `transformers>=4.38.0`
   - `Pillow`
   - `numpy`
   - `opencv-python`
   - `huggingface-hub`
   - `datasets`
   - `flash-attn`

## Development Status

YOFLO-CLI has been successfully converted into a full Python package and is available on PyPI. The package currently supports object detection, binary inference based on referring expression comprehension, as well as inference trees consisting of multiple phrases. Future updates will focus on optimizations and adding new features as the project evolves.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Make sure to follow the existing code style and add tests for any new features or bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The Florence-2 model is developed by Microsoft and is available on the Hugging Face Model Hub.
- This project uses several open-source libraries, including PyTorch, Transformers, OpenCV, Pillow, and NumPy.