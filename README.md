# Facial Recognition Emotion Detection

This project uses facial recognition to detect emotions in real-time using a webcam. It leverages OpenCV for facial detection and a pre-trained deep learning model to classify emotions.

## Project Structure

- `data/`: Contains sample images and instructions for data.
- `models/`: Stores pre-trained models.
- `notebooks/`: Jupyter notebooks for experimentation.
- `scripts/`: Python scripts for emotion detection and training.
- `requirements.txt`: List of dependencies.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/facial-recognition-emotion-detection.git
    ```

2. Navigate to the project directory:
    ```bash
    cd facial-recognition-emotion-detection
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Emotion Detection Script

To detect emotions in real-time using your webcam, run:

```bash
python scripts/detect_emotion.py
