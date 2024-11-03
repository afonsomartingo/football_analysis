# Football Analysis System

## Overview
This project focuses on the detection and tracking of players, referees, and footballs within video footage using the YOLO (You Only Look Once) object detection model. By training the model, we aim to enhance its accuracy. Additionally, players are grouped into teams based on their jersey colors through pixel segmentation and clustering using KMeans. This enables the assessment of a team's ball possession percentage during matches. We employ optical flow techniques to analyze camera movements between frames, providing precise measurements of player actions. With perspective transformation, we can better visualize the scene's depth, converting measurements from pixels to meters. Furthermore, we calculate each player's speed and total distance covered. This project integrates various concepts and addresses real-world challenges.

[output_video (1).webm](https://github.com/user-attachments/assets/409687ef-da3e-4987-a6aa-8caa42fcdefd)

## Technologies Used
The following technologies and libraries are essential for this project:
- **YOLO**: For object detection
- **KMeans**: For color segmentation and clustering of jersey colors
- **Optical Flow**: For analyzing camera motion
- **Perspective Transformation**: To depict depth and spatial relations
- **Speed and Distance Metrics**: To evaluate player performance

## Pre-trained Models
- [Download Trained YOLO v5 Model](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Example Video
- [View Sample Input Video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Installation Requirements
To execute this project, make sure to have the following libraries installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

## Credits
This project draws inspiration from the original [Football Analysis Project](https://github.com/abdullahtarek/football_analysis/tree/main), used for educational purposes only.
