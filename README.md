YOFLO-CLI (v1.0.0)

By: Charles C. Norton

INTRODUCTION
------------
YOFLO-CLI is a robust command-line interface for the YO-FLO package. Version 1.0.0 marks our official transition out of beta, representing a major milestone in stability, feature completeness, and extended functionality. Building on the powerful Microsoft Florence-2 vision-language model, YOFLO-CLI provides a flexible platform that unites real-time object detection, yes/no inference, multi-step inference chaining, automated screenshot capture, logging, and video recording. This comprehensive solution is suitable for security applications, visual monitoring, content creation, and any scenario that benefits from advanced vision-language understanding.


TABLE OF CONTENTS
-----------------
1. INTRODUCTION
2. WHAT IS YOFLO-CLI?
3. OFFICIAL 1.0.0 RELEASE OVERVIEW
4. CORE FUNCTIONALITY
5. KEY FEATURES
6. DIFFERENCES FROM PREVIOUS VERSIONS
7. TABLE OF CONTENTS EXPLANATION
8. EXTENDED DETAILED FEATURES
   8.1 OBJECT DETECTION
   8.2 BINARY YES/NO INFERENCE
   8.3 INFERENCE CHAINS
   8.4 INFERENCE RATE & PERFORMANCE MONITORING
   8.5 SCREENSHOT ON DETECTION
   8.6 DETECTION LOGGING
   8.7 HEADLESS MODE
   8.8 PRETTY PRINT
   8.9 MULTI-WEBCAM SUPPORT
   8.10 VIDEO RECORDING
   8.11 MODEL DOWNLOAD
9. HOW IT WORKS UNDER THE HOOD
10. USE CASES & APPLICATION SCENARIOS
11. FULL INSTALLATION GUIDE
   11.1 INSTALLATION FROM SOURCE
   11.2 INSTALLATION FROM PYPI
12. SYSTEM REQUIREMENTS & ENVIRONMENT SETUP
   12.1 OPERATING SYSTEM
   12.2 GPU / CUDA
   12.3 PYTHON VERSION
   12.4 DEPENDENCIES & ENVIRONMENT VARIABLES
13. COMPLETE USAGE INSTRUCTIONS
   13.1 PRIMARY COMMAND-LINE FLAGS
   13.2 ADVANCED COMMAND-LINE FLAGS
   13.3 EXAMPLE COMMANDS
14. INFERENCE CHAIN & COMPLEX LOGIC
   14.1 WHY IT MATTERS
   14.2 EXAMPLES
15. PERFORMANCE TUNING & OPTIMIZATION
   15.1 INFERENCE LIMIT
   15.2 HEADLESS MODE
   15.3 MULTI-THREADED OPERATIONS
   15.4 GPU OPTIMIZATIONS
16. TROUBLESHOOTING
   16.1 COMMON ERRORS
   16.2 MODEL LOADING ISSUES
   16.3 CAMERA / VIDEO FEED PROBLEMS
   16.4 SLOW PERFORMANCE
17. FREQUENTLY ASKED QUESTIONS (FAQ)
   17.1 CAN I USE A DIFFERENT MODEL?
   17.2 WILL THIS WORK WITHOUT A GPU?
   17.3 HOW DO I RUN MULTIPLE INSTANCES AT ONCE?
   17.4 DOES HEADLESS MODE AFFECT ACCURACY?
18. CONTRIBUTING & COMMUNITY
19. LICENSE
20. ACKNOWLEDGMENTS
21. FUTURE DIRECTIONS
22. CONCLUSION


1. INTRODUCTION
---------------
YOFLO-CLI is a command-line interface designed to unify advanced computer vision and natural language understanding tasks in one cohesive toolkit. Version 1.0.0 marks the milestone release out of beta, focusing on stability, comprehensive user documentation, and fully-tested functionality. This software builds on Microsoft’s Florence-2 model to achieve robust object detection and detailed linguistic reasoning, enabling real-time insights far beyond typical bounding box detection.


2. WHAT IS YOFLO-CLI?
---------------------
YOFLO-CLI (YOFLO stands for "Your Object-Finding Language Operator") is a versatile command-line tool facilitating object detection, binary yes/no inference, multi-phrase inference chaining, screenshot capture, detection logging, real-time inference speed measurement, and selective video recording. Its remarkable feature is the ability to parse everyday language queries against live video feeds, harnessing the strength of Florence-2’s vision-language alignment to detect objects, answer questions, and perform elaborate checks on visual scenes.


3. OFFICIAL 1.0.0 RELEASE OVERVIEW
----------------------------------
With this first official release:
- We have exited the beta stage.
- Introduced refined inference chain logic and improved performance for multi-webcam scenarios.
- Enhanced the user experience with verbose logging, extended code comments, and a structured codebase for easier debugging and future expansion.
- Thoroughly tested on Ubuntu 22.04, verified partial Windows 11 support, and built in-house GPU optimizations for real-time usage.


4. CORE FUNCTIONALITY
---------------------
- Object Detection: Identify objects in the camera feed or RTSP stream. By specifying class names, the system can filter out irrelevant detections, or it can display everything recognized.
- Binary Inference: Pose yes/no questions such as “Is the table empty?” or “Is the person wearing sunglasses?” to get a direct answer, leveraging Florence-2’s text-image alignment.
- Inference Chains: Allows multiple yes/no checks to be performed in a single pass, returning an aggregated “pass” or “fail” verdict. This effectively simulates multi-step reasoning.
- Screenshot on Detection: Whenever a target object is detected, the system can automatically save a timestamped image file for reference.
- Logging: All detection events can be appended to a dedicated file (alerts.log) for auditing, record-keeping, or post-event analysis.
- Video Recording: By specifying triggers (object detection or certain inference outcomes), video recording can be started and stopped automatically, saving only the intervals of interest.


5. KEY FEATURES
---------------
- Real-Time Operation: Processes frames in real-time, making it suitable for live surveillance or interactive display setups.
- Headless Mode: Ideal for server or cloud environments where no GUI is available. Gains a typical 20% improvement in throughput when no display is rendered.
- Inference Speed Logging: Monitors inferences per second, enabling optimization and hardware scaling decisions.
- Multi-Webcam and RTSP: Users can pass multiple webcam indices or RTSP URIs for parallel analysis of various angles or streams.


6. DIFFERENCES FROM PREVIOUS VERSIONS
-------------------------------------
Compared to version 0.9.5 (beta), 1.0.0 introduces:
- Official production readiness, with more stringent testing under high-load conditions.
- Deeper inference chain logic, supporting complex multi-step scenario checks.
- Expanded video recording triggers, including “od” (detect objects), “infy” (start on yes, end on no), and “infn” (start on no, end on yes).
- Significantly improved error handling, logging clarity, and stable concurrency management across multiple threads.


7. TABLE OF CONTENTS EXPLANATION
--------------------------------
This document is structured for maximum clarity. Sections detail every aspect of YOFLO-CLI, from the simplest usage instructions (to help new users get started quickly) through extensive customization options. Follow each section for deeper insights into the tool’s architecture, deployment in varied environments, and solutions for common issues.


8. EXTENDED DETAILED FEATURES
-----------------------------
8.1 OBJECT DETECTION:
YOFLO-CLI leverages Florence-2’s advanced vision-language alignment to detect an extensive range of object types. Unlike standard object detectors that rely on fixed class sets, Florence-2 can interpret textual labels dynamically. If no class names are specified, YOFLO-CLI prints out everything the model recognizes in the frame.

8.2 BINARY YES/NO INFERENCE:
YOFLO-CLI’s yes/no queries rely on Florence-2’s capacity for visual question answering. Users submit a question like “Is the car parked?” and the system processes the frame to determine a yes or no answer. This feature is ideal for quick gating conditions or real-time decision-making scenarios where a binary outcome is sufficient.

8.3 INFERENCE CHAINS:
Arguably the biggest advantage for complex tasks. By sequentially testing multiple yes/no queries, the system compiles a final pass/fail. For instance, you might check “Is there a person?” “Are they awake?” “Are they seated at a desk?” If all conditions are “yes,” the final result is a pass.

8.4 INFERENCE RATE & PERFORMANCE MONITORING:
YOFLO-CLI can measure inferences per second in real time, assisting in diagnosing bottlenecks. By adjusting the inference_limit, advanced users can balance system load with responsiveness, suitable for resource-constrained hardware or multi-feed environments.

8.5 SCREENSHOT ON DETECTION:
Whenever an object of interest is detected, YOFLO-CLI can optionally capture the current frame. These screenshots are automatically timestamped, simplifying traceability or dataset creation for future model training.

8.6 DETECTION LOGGING:
All detections (and optionally inferences) may be logged in an alerts.log file. Each entry is date- and time-stamped for subsequent auditing or historical records. Perfect for analyzing system performance or investigating anomalies after the fact.

8.7 HEADLESS MODE:
Running “headless” means no OpenCV display windows. This mode is highly beneficial for servers, cloud VMs, or HPC clusters, where no display is available. It also typically speeds up frame processing by eliminating rendering overhead.

8.8 PRETTY PRINT:
A user-friendly textual output that organizes detections neatly. Instead of raw bounding box coordinates, you’ll get a well-formatted listing of labels, bounding boxes, and confidence values. Good for quick debugging or demonstration.

8.9 MULTI-WEBCAM SUPPORT:
Spin up multiple threads to handle multiple cameras simultaneously, each with independent detection. This proves invaluable in surveillance contexts (monitoring multiple areas at once) or advanced research setups requiring multiple viewpoints.

8.10 VIDEO RECORDING:
Specify conditions under which YOFLO-CLI starts/stops recording. If “record=od,” the system records whenever an object is detected; if “record=infy,” it records when inference is yes and stops on no, etc. This feature conserves space by only saving relevant footage.

8.11 MODEL DOWNLOAD:
Rather than manually acquiring the Florence-2 model, you can instruct YOFLO-CLI to pull it straight from the Hugging Face Hub with -dm, drastically simplifying the setup process.


9. HOW IT WORKS UNDER THE HOOD
------------------------------
YOFLO-CLI orchestrates concurrency across multiple camera streams, feeding frames to the Florence-2 model. Florence-2, being a large vision-language transformer, is invoked for either bounding box generation (object detection) or textual question answering (yes/no inferences). The tool aggregates these outputs, logs relevant events, and optionally triggers recording or screenshots. Communication between threads is carefully managed, ensuring no frame backlog or concurrency conflicts degrade performance.


10. USE CASES & APPLICATION SCENARIOS
-------------------------------------
- Security & Surveillance: Monitor multiple cameras in real time, automatically record or capture screenshots when suspicious objects or behaviors are detected.
- Research & Development: Perform quick experiments on vision-language tasks, create custom inference chains for more advanced scenario testing, or gather data for training other models.
- Content Creation: Live streaming with an overlay of recognized objects or yes/no Q&A prompts.
- Industrial Automation: Check production lines for anomalies or worker compliance with safety protocols, triggered by real-time detection and multi-step inference.


11. FULL INSTALLATION GUIDE
---------------------------
11.1 INSTALLATION FROM SOURCE:
1) Clone the repository to your local machine.
2) Navigate into the project directory.
3) Run pip install . from within that directory.
4) Confirm installation by typing yoflo.py --help.

11.2 INSTALLATION FROM PYPI:
1) Run pip install yoflo in your terminal or command prompt.
2) Check success with yoflo.py --help or simply python -m yoflo --help, if using a module-based invocation.


12. SYSTEM REQUIREMENTS & ENVIRONMENT SETUP
-------------------------------------------
12.1 OPERATING SYSTEM:
- Ubuntu 22.04 recommended for maximum compatibility.
- Windows 11 support exists but can demand additional steps (drivers, environment variables, etc.).

12.2 GPU / CUDA:
- GPU recommended for real-time operation. The official minimum is 16 GB VRAM for stable performance at reasonable inference rates.
- CUDA 12.1 or later is preferred. Make sure your GPU driver is up to date.

12.3 PYTHON VERSION:
- Python 3.10 is tested and required. Older versions risk missing language or library features.

12.4 DEPENDENCIES & ENVIRONMENT VARIABLES:
- Key libraries: torch, transformers>=4.38.0, Pillow, numpy, opencv-python, huggingface-hub, datasets, flash-attn.
- For GPU usage, ensure:
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


13. COMPLETE USAGE INSTRUCTIONS
-------------------------------
13.1 PRIMARY COMMAND-LINE FLAGS:
- -mp: Path to the locally saved Florence-2 model.
- -od: Activates object detection; optional class filters.
- -ph: Single yes/no question for real-time evaluation.
- -hl: Runs in console-only mode with no display windows.
- -ss: Captures frame screenshots on detection.
- -lf: Enables appending detection events to alerts.log.
- -is: Logs inferences/sec.
- -dm: Automates model download from Hugging Face.

13.2 ADVANCED COMMAND-LINE FLAGS:
- -pp: Produces visually pleasing detection logs.
- -ic: Evaluates multiple yes/no queries in one pass.
- -il: Restricts max inferences per second.
- -wi: Allows specifying multiple local webcam indices.
- -rtsp: Use RTSP streams instead of local cameras.
- -r: “od” for object detection triggers, “infy” or “infn” for yes/no-based triggers.

13.3 EXAMPLE COMMANDS:
1) Basic Object Detection (all classes):
   python yoflo.py -mp /path/to/Florence-2 -od

2) Binary Inference:
   python yoflo.py -mp /path/to/Florence-2 -ph "Is the person smiling?"

3) Inference Chain:
   python yoflo.py -mp /path/to/Florence-2 -ic "Is there a computer?" "Is the monitor on?"

4) Headless Operation:
   python yoflo.py -mp /path/to/Florence-2 -od "person" -hl

5) Screenshot on Detection:
   python yoflo.py -mp /path/to/Florence-2 -od "cat" -ss

6) Logging to File:
   python yoflo.py -mp /path/to/Florence-2 -od "dog" -lf

7) Video Recording Triggered by Inference:
   python yoflo.py -mp /path/to/Florence-2 -ph "Is the door open?" -r infy

8) Multi-Webcam:
   python yoflo.py -mp /path/to/Florence-2 -wi 0 1 -od "car"


14. INFERENCE CHAIN & COMPLEX LOGIC
-----------------------------------
14.1 WHY IT MATTERS:
Single yes/no queries can be too simplistic for many real-world scenarios. Inference chains break down complex logic into a series of simpler checks. Users can easily craft multi-step constraints and combine them for a final pass/fail verdict.

14.2 EXAMPLES:
- “Is there a person?” “Is the person awake?” “Is the person wearing a security badge?” => If all are yes, we conclude the scenario is authorized.
- “Is it raining?” “Is there an umbrella in use?” => Could indicate preparedness for adverse weather conditions.


15. PERFORMANCE TUNING & OPTIMIZATION
-------------------------------------
15.1 INFERENCE LIMIT:
Useful on slower systems or when CPU resources are shared among multiple processes. Setting an inference limit of 3 means YOFLO-CLI will only attempt up to 3 inferences per second, preventing spiking CPU/GPU usage.

15.2 HEADLESS MODE:
Eliminates GUI overhead, speeding up frame handling by 20% or more in many environments.

15.3 MULTI-THREADED OPERATIONS:
Each camera feed has its own thread, maximizing concurrency. However, be mindful of the GPU load if using many streams concurrently.

15.4 GPU OPTIMIZATIONS:
Ensure you are using a modern GPU driver and a compatible CUDA installation. For best results, close other GPU-intensive tasks during operation.


16. TROUBLESHOOTING
-------------------
16.1 COMMON ERRORS:
- “Model not found” => Check the path or confirm successful download.
- “No suitable webcam device” => Ensure you have the correct webcam index or RTSP URL.

16.2 MODEL LOADING ISSUES:
- OSError or permission errors may indicate no read permissions on the model folder. Verify your file paths and user privileges.

16.3 CAMERA / VIDEO FEED PROBLEMS:
- Confirm the index is valid (e.g. on Linux, /dev/video0 is typically index 0).
- For RTSP, verify the network connection or correct URL format (rtsp://...).

16.4 SLOW PERFORMANCE:
- Switch to headless mode, reduce inference_limit, or close other GPU processes. Check that you have enough VRAM free.


17. FREQUENTLY ASKED QUESTIONS (FAQ)
------------------------------------
17.1 CAN I USE A DIFFERENT MODEL?
Currently, YOFLO-CLI is tightly integrated with Florence-2. Future versions may add support for custom HF Transformers.

17.2 WILL THIS WORK WITHOUT A GPU?
Yes, but real-time performance will be severely limited. CPU-only mode is viable for testing, not recommended for production.

17.3 HOW DO I RUN MULTIPLE INSTANCES AT ONCE?
Each instance must target separate cameras or streams. Be mindful of system resource constraints if running them in parallel on the same GPU.

17.4 DOES HEADLESS MODE AFFECT ACCURACY?
No. The removal of the GUI output has zero impact on detection or inference accuracy. It purely saves CPU/GPU cycles that would have been spent rendering.


18. CONTRIBUTING & COMMUNITY
----------------------------
We welcome contributions and user feedback. Submit issues, pull requests, or general suggestions on our GitHub repository. All code changes should include relevant tests or usage examples where possible. Our community fosters collaboration, encouraging open dialogue, bug reporting, and performance benchmarking.


19. LICENSE
----------
YOFLO-CLI is released under the MIT License, granting you broad rights to modify and distribute the software. Refer to the LICENSE file for the entire legal text.


20. ACKNOWLEDGMENTS
-------------------
- Microsoft for developing Florence-2 and releasing it on the Hugging Face Model Hub.
- Contributors to open-source libraries such as PyTorch, Transformers, OpenCV, Pillow, and NumPy, which collectively enable YOFLO-CLI’s functionality.
- Everyone who tested earlier beta builds, providing invaluable feedback and bug reports.


21. FUTURE DIRECTIONS
---------------------
- Expanded PTZ Camera Control: Possibly integrate pan/tilt/zoom logic into YOFLO-CLI for automated object tracking.
- Advanced Alerting: Beyond logging, implement email or webhook alerts on custom triggers.
- Hybrid Models: Allow seamlessly switching between Florence-2 and other HF Transformers to handle specialized tasks.
- Deployment Tooling: Docker containers, Helm charts, or serverless wrappers for enterprise-scale rollouts.


22. CONCLUSION
-------------
YOFLO-CLI v1.0.0 ushers in a new era of unified computer vision and language reasoning. By pairing real-time object detection, yes/no inference, multi-step logic, logging, screenshots, and conditional video recording, it operates as a universal toolkit. We encourage users to explore the wide range of functionalities, experiment with advanced chain logic, and contribute new features or improvements. Thank you for choosing YOFLO-CLI as your vision-language command-line solution out of beta. We hope it meets all your demanding real-time analysis needs.