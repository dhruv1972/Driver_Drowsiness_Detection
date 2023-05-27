# Driver Drowsiness Detection

This repository contains the code for a web application that uses computer vision techniques and a convolutional neural network (CNN) to detect when a driver is feeling drowsy.

## Project Description

Driver drowsiness detection is a safety technology that can prevent accidents that are caused by drivers who have fallen asleep at the wheel. This tech must detect drowsiness in a driver and take immediate action. This repository provides a simple solution to this problem by creating a Flask-based web application that uses OpenCV and a trained Keras model to detect when a driver is falling asleep.

## Project Structure

The project is structured as follows:

1. `app.py`: The main Python file of the application which handles the server-side operations.
2. `models/`: Directory contains the trained Keras model for eye state (Open/Closed) detection.
3. `haar cascade files/`: Directory contains the Haar cascade XML files used by OpenCV for facial and eye detection.
4. `Templates/`: Directory contains the HTML templates used for the web application.
5. `Static/`: Directory contains any static files needed by the web application.

## How it Works

1. The application starts by initializing a video capture object and reading frames from it in real time.

2. Each frame is first converted to grayscale and then used to detect faces and eyes using the Haar cascade files.

3. The regions of the image containing the eyes are then passed through a trained Keras model which predicts whether the eyes are open or closed.

4. If the model predicts that the eyes have been closed for a certain period of time, an alarm sound is played to alert the driver.

5. The application also includes a Flask server which streams the video capture feed to a webpage in real time.

## Requirements

- Python 3.6 or later
- Flask
- OpenCV
- Keras
- Pygame
- Numpy

## Usage

1. Clone the repository:

```
git clone https://github.com/dhruv1972/Driver_Drowsiness_Detection.git
cd Driver_Drowsiness_Detection
```

2. Install the necessary packages:

```
pip install -r requirements.txt
```

3. Run the Flask app:

```
python app.py
```

4. Open a web browser and go to `http://localhost:5000` to view the live feed.

## Disclaimer

This application is intended for educational purposes only and should not be relied upon for real-world driver drowsiness detection. It is always the responsibility of the driver to ensure they are awake and alert while operating a vehicle.

## Contributing

Feel free to contribute to this project by submitting issues and/or pull requests. Your valuable feedback is much appreciated!

## License

This project is licensed under the [GNU General Public License](https://github.com/dhruv1972/Driver_Drowsiness_Detection/LICENSE.md)
