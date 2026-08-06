# Exercise Detection from Wrist Motion

A system that identifies which strength exercise is being performed from wrist-worn IMU data (accelerometer, gyroscope, orientation), then displays the recognized exercise as a 3D reference animation.

## What it does
Given a recording of a set from a wrist sensor, the system extracts motion features, classifies which exercise it is, and reports accuracy along with which exercises are most often confused. The recognized exercise is shown as a canonical 3D animation (a reference demonstration of the detected movement, not a reconstruction of the user's body).

## Data
Wrist IMU recordings at 100Hz, one file per set, each labeled with its exercise. Filenames encode the activity code. Data is not committed (see .gitignore).

## Stack
Python (pandas, NumPy, SciPy, scikit-learn) for the classifier. Matplotlib for evaluation plots. Optional web layer (Next.js, Three.js) for the 3D visualization.

## Approach
Whole-set classification: each recording becomes a fixed set of descriptive motion features, a simple explainable model is trained and validated on held-out data, and performance is reported with a confusion matrix to show where the wrist signal can and cannot distinguish similar movements.
