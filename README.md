# YOFLO-CLI

YOFLO-CLI is a command-line interface for the YO-FLO package, providing advanced object detection and binary inference capabilities using the Florence-2 vision-language model in real-time. This tool leverages state-of-the-art vision models to perform tasks such as object detection and binary inference based on referring expression comprehension.

## Features

- **Object Detection**: Identify and classify objects within a video feed. You can specify class names or display all detections if no class names are provided.
- **Binary Inference**: Answer yes/no questions based on the visual input.
- **Inference Rate Calculation**: Measure the rate of inferences per second.
- **Real-time Processing**: Process video feeds from a webcam in real-time.
- **Screenshot on Detection**: Automatically capture and save a screenshot when a target object is detected.
- **Logging Detections**: Log detection events to a file.
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

## Example Commands

### Object Detection for Specific Classes
To perform object detection and only display detections for specific classes such as `"cat"` and `"dog"`:
```sh
python "D:\GitHub\yoflo-cli\yoflo\yoflo.py" --model_path "D:\text-generation-webui\models\Florence-2-base-ft" --object_detection "cat" "dog"
```

### Object Detection Displaying All Detections
To perform object detection and display all detected objects without filtering by class:
```sh
python "D:\GitHub\yoflo-cli\yoflo\yoflo.py" --model_path "D:\text-generation-webui\models\Florence-2-base-ft" --object_detection
```

### Binary Inference (e.g., "Is the person smiling?")
To perform binary inference based on a yes/no question related to the visual input:
```sh
python "D:\GitHub\yoflo-cli\yoflo\yoflo.py" --model_path "D:\text-generation-webui\models\Florence-2-base-ft" --phrase "Is the person smiling?"
```

### Headless Mode
To run the tool in headless mode without displaying the video feed:
```sh
python "D:\GitHub\yoflo-cli\yoflo\yoflo.py" --model_path "D:\text-generation-webui\models\Florence-2-base-ft" --object_detection "cat" "dog" --headless
```

### Enable Screenshot on Detection
To enable screenshot capture whenever a target object is detected:
```sh
python "D:\GitHub\yoflo-cli\yoflo\yoflo.py" --model_path "D:\text-generation-webui\models\Florence-2-base-ft" --object_detection "cat" "dog" --screenshot
```

### Enable Logging of Detection Events to File
To log detection events to a file named `alerts.log`:
```sh
python "D:\GitHub\yoflo-cli\yoflo\yoflo.py" --model_path "D:\text-generation-webui\models\Florence-2-base-ft" --object_detection "cat" "dog" --log_to_file
```

### Display Inference Speed
To log and display the inference speed (inferences per second):
```sh
python "D:\GitHub\yoflo-cli\yoflo\yoflo.py" --model_path "D:\text-generation-webui\models\Florence-2-base-ft" --object_detection "cat" "dog" --display_inference_speed
```

### Download Model from Hugging Face
To download the Florence-2 model from the Hugging Face Hub:
```sh
python "D:\GitHub\yoflo-cli\yoflo\yoflo.py" --download_model
```

### Pretty Print Detections
To enable formatted output of detections for better readability:
```sh
python "D:\GitHub\yoflo-cli\yoflo\yoflo.py" --model_path "D:\text-generation-webui\models\Florence-2-base-ft" --object_detection "cat" "dog" --pretty_print
```

### Limit Inference Rate
To limit the inference rate to a specified number of inferences per second, for example, 5 inferences per second:
```sh
python "D:\GitHub\yoflo-cli\yoflo\yoflo.py" --model_path "D:\text-generation-webui\models\Florence-2-base-ft" --object_detection "cat" "dog" --inference_limit 5
```

## Development Status

YOFLO-CLI has been successfully converted into a full Python package and is available on PyPI. The package currently supports object detection and binary inference based on referring expression comprehension. Future updates will focus on optimizations and adding new features as the project evolves.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Make sure to follow the existing code style and add tests for any new features or bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The Florence-2 model is developed by Microsoft and is available on the Hugging Face Model Hub.
- This project uses several open-source libraries, including PyTorch, Transformers, OpenCV, Pillow, and NumPy.