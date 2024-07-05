# YOFLO-CLI

YOFLO-CLI is a command-line interface for the YO-FLO package, providing advanced object detection and binary inference capabilities using the Florence-2 vision-language model in real-time. This tool leverages state-of-the-art vision models to perform tasks such as object detection and binary inference based on referring expression comprehension.

## Features

- **Object Detection**: Identify and classify objects within a video feed.
- **Binary Inference**: Answer yes/no questions based on the visual input.
- **Inference Rate Calculation**: Measure the rate of inferences per second.
- **Real-time Processing**: Process video feeds from a webcam in real-time.
- **Screenshot on Detection**: Automatically capture and save a screenshot when a target object is detected.
- **Logging Alerts**: Log detection alerts to a file.
- **Headless Mode**: Run the tool without displaying the video feed, suitable for server environments.
- **Pretty Print**: Enable formatted output of detections for better readability.
- **Model Download**: Option to download the Florence-2 model directly from the Hugging Face Hub.

## Model Information

This tool uses Microsoft's Florence-2, a powerful vision-language model designed to understand and generate detailed descriptions of visual inputs. Florence-2 combines advanced image processing with natural language understanding, making it ideal for complex tasks that require both visual and textual analysis.

## Installation

### From Source

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/yoflo-cli.git
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

Run the script with the desired arguments:
```sh
python yoflo.py --model_path /path/to/model --class_name "person" --phrase "Is the person smiling?" --debug --screenshot --log_to_file --headless --pretty_print
```

### Command-line Arguments

- `-mp`, `--model_path`: Path to the pre-trained model directory.
- `-cn`, `--class_name`: Class name to detect (e.g., 'cat', 'dog').
- `-ph`, `--phrase`: Yes/No question for expression comprehension (e.g., 'Is the person smiling?').
- `-d`, `--debug`: Enable debug mode.
- `-hl`, `--headless`: Run in headless mode without displaying video.
- `-ss`, `--screenshot`: Enable screenshot on detection.
- `-lf`, `--log_to_file`: Enable logging alerts to file.
- `-is`, `--display_inference_speed`: Display inference speed.
- `-od`, `--object_detection`: Enable object detection.
- `-dm`, `--download_model`: Download model from Hugging Face.
- `-pp`, `--pretty_print`: Enable pretty print for detections.
- `-ao`, `--alert_on`: Trigger alert on 'yes' or 'no' result for expression comprehension or 'class' for object detection.
- `-il`, `--inference_limit`: Limit the inference rate to X inferences per second.

## Overview of Capabilities

YOFLO-CLI provides a comprehensive set of features for real-time object detection and expression comprehension. Here's an overview of its capabilities:

- **Model Initialization**: Loads the Florence-2 model and processor.
- **Object Detection**: Uses the model to detect objects in images and annotate them with bounding boxes and labels.
- **Expression Comprehension**: Answers yes/no questions based on the visual content of the image.
- **Real-time Processing**: Captures video frames from a webcam, processes them for object detection or binary inference, and displays the results.
- **Screenshot and Logging Alerts**: Captures screenshots and logs alerts to a file upon detection of target objects.

### Main Functions

- `init_model()`: Initializes the model and processor from the specified path.
- `run_object_detection()`: Performs object detection on a given image.
- `run_expression_comprehension()`: Performs binary inference based on a given phrase.
- `start_webcam_detection()`: Starts real-time processing of webcam feed.
- `stop_webcam_detection()`: Stops the real-time processing of the webcam feed.
- `save_screenshot()`: Saves a screenshot when a target object is detected.
- `log_alert()`: Logs detection alerts to a file.
- `toggle_screenshot()`: Toggles the screenshot feature on or off.
- `toggle_log_to_file()`: Toggles the log to file feature on or off.
- `pretty_print_detections()`: Formats and prints detection results with datetime for readability.

## Development Status

YOFLO-CLI has been successfully converted into a full Python package and is available on PyPI. The package currently supports object detection and binary inference based on referring expression comprehension. Future updates will focus on optimizations and adding new features as the project evolves.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Make sure to follow the existing code style and add tests for any new features or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Florence-2 model is developed by Microsoft and is available on the Hugging Face Model Hub.
- This project uses several open-source libraries, including PyTorch, Transformers, OpenCV, Pillow, and NumPy.
