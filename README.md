
# Tennis Analysis

## Introduction
This project provides an in-depth analysis of tennis matches by leveraging computer vision techniques. It tracks players and the tennis ball in real-time, analyzing key metrics such as player speed, ball shot speed, and the number of shots exchanged during a rally.
The project employs state-of-the-art machine learning models such as YOLO (You Only Look Once) for object detection and Convolutional Neural Networks (CNNs) to extract tennis court keypoints. This hands-on project is ideal for honing your skills in machine learning, deep learning, and computer vision, while also gaining experience in sports analytics. 

## Key Features
1. Player Detection: Detects and tracks players in a tennis match using YOLOv8.
2. Tennis Ball Detection: Fine-tuned YOLO detects the tennis ball with high precision, even during fast movements.
3. Court Keypoint Extraction: A CNN model extracts the boundaries and key points of the tennis court to provide accurate spatial context for analysis.
4. Metrics Extraction:
    * Player Speed: Calculates player movement speeds in real-time.
    * Ball Shot Speed: Measures the velocity of the tennis ball during gameplay.
    * Shot Count: Tracks the number of shots in each rally.

## Output Examples
The final output consists of annotated videos showcasing:
* Player positions and speeds.
* Ball trajectory and shot speed.
* Court overlay with keypoints highlighted.

## Output Videos
Here is a screenshot from one of the output videos:

![Screenshot](output_videos/screenshot.jpeg)


## Models Utilized
1. YOLOv8 for Player Detection:
    * Pre-trained and fine-tuned for robust and real-time player identification.
2. Fine-Tuned YOLO for Tennis Ball Detection:
    * Optimized for high accuracy in detecting small, fast-moving objects like tennis balls.
3. CNN for Court Keypoint Extraction:
    * Identifies key landmarks on the tennis court to provide spatial awareness for accurate calculations.

4. Trained YOLOV5 model: https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing
5. Trained tennis court key point model: https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing


## Training
* Tennis ball detetcor with YOLO: training/tennis_ball_detector_training.ipynb
* Tennis court keypoint with Pytorch: training/tennis_court_keypoints_training.ipynb


## System Requirements
To run this project, ensure your system meets the following prerequisites:

### Environment Setup:
1. Python: python3.8 or higher
2. Deep Learning Frameworks:
    * ultralytics (for YOLO)
    * pytorch (for CNN-based models)
3. Supporting Libraries:
    * pandas (data manipulation)
    * numpy (numerical computations)
    * opencv-python (image and video processing)

### Installation:
Install the required Python libraries using the command below:
    ```bash
    pip install ultralytics torch pandas numpy opencv-python
    ```
### How To Run
    ```bash
    python main.py
    ```

## How It Works
1. Video Input: The input video of a tennis match is processed frame by frame.
2. Detection:
    * Players and the tennis ball are detected using YOLO models.
    * Court keypoints are extracted using the CNN model.

3. Analysis:
    * The movement of players is tracked to calculate their speed.
    * The trajectory and velocity of the tennis ball are analyzed.
    * Shot counts are updated based on ball transitions.

4. Output Generation:
    * Annotated video showing all calculated metrics and overlays.

## Applications
This project has several practical applications:
* Sports Analytics: Analyze player performance and match statistics.
* Training Tools: Provide actionable insights for athletes and coaches.
* Broadcast Enhancements: Add real-time visualizations for viewers.
