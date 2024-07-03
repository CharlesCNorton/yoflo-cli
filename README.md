
# YO-FLO-CLI

YO-FLO-CLI is a command-line interface for the YO-FLO package, providing advanced object detection and binary inference capabilities using the Florence-2 vision-language model in real-time. This tool leverages state-of-the-art vision models to perform tasks such as object detection and binary inference based on referring expression comprehension.

## Features

- **Object Detection**: Identify and classify objects within a video feed.
- **Binary Inference**: Answer yes/no questions based on the visual input.
- **Inference Rate Calculation**: Measure the rate of inferences per second.
- **Real-time Processing**: Process video feeds from a webcam in real-time.
- **Screenshot on Detection**: Automatically capture and save a screenshot when a target object is detected.
- **Sound Alert on Detection**: Play a sound alert when a target object is detected.
- **Headless Mode**: Run the tool without displaying the video feed, suitable for server environments.

## Model Information

This tool uses Microsoft's Florence-2, a powerful vision-language model designed to understand and generate detailed descriptions of visual inputs. Florence-2 combines advanced image processing with natural language understanding, making it ideal for complex tasks that require both visual and textual analysis.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/yoflo-cli.git
    cd yoflo-cli
    ```

2. Install the package:
    ```sh
    pip install .
    ```

## Usage

```sh
yoflo --model_path /path/to/model --class_name "cat" --phrase "Is the person smiling?" --debug --screenshot --beep
```

### Command-line Arguments

- `--model_path`: Path to the pre-trained model directory.
- `--class_name`: Class name to detect (e.g., 'cat', 'dog').
- `--phrase`: Yes/No question for expression comprehension (e.g., 'Is the person smiling?').
- `--debug`: Enable debug mode.
- `--headless`: Run in headless mode without displaying the video feed.
- `--screenshot`: Enable screenshot on detection.
- `--beep`: Enable beep on detection.

## Example

```sh
yoflo --model_path ./models/florence-2 --class_name "dog" --phrase "Is the dog sitting?" --debug --screenshot --beep
```

This command starts the YO-FLO-CLI tool using the Florence-2 model located in `./models/florence-2`, sets the target class to "dog", and the binary inference phrase to "Is the dog sitting?". The `--debug` flag enables debug mode for detailed output. The `--screenshot` and `--beep` flags enable screenshot capture and sound alerts upon detection, respectively.

## Script Overview

The core functionality of YO-FLO-CLI is encapsulated in the `yoflo.py` script. Here’s a high-level overview of its capabilities:

- **Model Initialization**: Loads the Florence-2 model and processor.
- **Object Detection**: Uses the model to detect objects in images and annotate them with bounding boxes and labels.
- **Expression Comprehension**: Answers yes/no questions based on the visual content of the image.
- **Real-time Processing**: Captures video frames from a webcam, processes them for object detection or binary inference, and displays the results.
- **Screenshot and Sound Alert**: Captures screenshots and plays sound alerts upon detection of target objects.

### Main Functions

- `init_model()`: Initializes the model and processor from the specified path.
- `run_object_detection()`: Performs object detection on a given image.
- `run_expression_comprehension()`: Performs binary inference based on a given phrase.
- `start_webcam_detection()`: Starts real-time processing of webcam feed.
- `stop_webcam_detection()`: Stops the real-time processing of the webcam feed.
- `save_screenshot()`: Saves a screenshot when a target object is detected.
- `beep_sound()`: Plays a sound alert when a target object is detected.
- `toggle_screenshot()`: Toggles the screenshot feature on or off.
- `toggle_beep()`: Toggles the beep feature on or off.

## Development Status

YO-FLO-CLI is currently in the process of being converted into a full Python package. We are starting small with just object detection and binary inference based on referring expression comprehension. Future updates will include optimizations and additional features as the project evolves.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Florence-2 model is developed by Microsoft and is available on the HuggingFace Model Hub.
- This project uses several open-source libraries, including PyTorch, Transformers, OpenCV, Pillow, and NumPy.
