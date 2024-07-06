# YOFLO-CLI

YOFLO-CLI is a command-line interface for the YO-FLO package, providing advanced object detection and binary inference capabilities using the Florence-2 vision-language model in real-time. This tool leverages state-of-the-art vision models to perform tasks such as object detection and binary inference based on referring expression comprehension.

## Features

- **Object Detection**: Identify and classify objects within a video feed. You can specify class names or display all detections if no class names are provided.
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

Run the script with the desired arguments. Below are some example commands:

### Object Detection for Specific Classes
```sh
python yoflo.py -mp /path/to/model -od cat dog
```

### Object Detection Displaying All Detections
```sh
python yoflo.py -mp /path/to/model -od
```

### Binary Inference (e.g., "Is the person smiling?")
```sh
python yoflo.py -mp /path/to/model -ph "Is the person smiling?"
```

### Additional Options

- **Debug Mode**:
    ```sh
    python yoflo.py -mp /path/to/model -od cat dog -d
    ```

- **Headless Mode**:
    ```sh
    python yoflo.py -mp /path/to/model -od cat dog -hl
    ```

- **Enable Screenshot on Detection**:
    ```sh
    python yoflo.py -mp /path/to/model -od cat dog -ss
    ```

- **Enable Logging Alerts to File**:
    ```sh
    python yoflo.py -mp /path/to/model -od cat dog -lf
    ```

- **Display Inference Speed**:
    ```sh
    python yoflo.py -mp /path/to/model -od cat dog -is
    ```

- **Enable Pretty Print for Detections**:
    ```sh
    python yoflo.py -mp /path/to/model -od cat dog -pp
    ```

- **Limit Inference Rate (e.g., 5 inferences per second)**:
    ```sh
    python yoflo.py -mp /path/to/model -od cat dog -il 5
    ```

- **Alert on Specific Results for Binary Inference**:
    ```sh
    python yoflo.py -mp /path/to/model -od cat dog -ao yes
    ```

## Main Functions

- `init_model()`: Initializes the model and processor from the specified path.
- `run_object_detection()`: Performs object detection on a given image.
- `run_expression_comprehension()`: Performs binary inference based on a given phrase.
- `start_webcam_detection()`: Starts real-time processing of webcam feed.
- `stop_webcam_detection()`: Stops the real-time processing of the webcam feed.
- `save_screenshot()`: Saves a screenshot when a target object is detected.
- `log_alert()`: Logs detection alerts to a file.
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