# Face Detection Using Haar Cascades with OpenCV

## Aim

To implement face detection using Haar Cascade classifiers with Python and OpenCV by performing the following operations:

1. Extract ROI (Region of Interest) from an image.
2. Perform face detection in static images.
3. Perform eye detection in images.
4. Perform face detection with labels in real-time webcam video using Matplotlib.

## Learning Objectives

- Understand Haar Cascade classifiers.
- Load and display multiple images.
- Detect faces in single-person and group photographs.
- Perform eye detection.
- Extract a Region of Interest from an image.
- Implement real-time face detection using a webcam.
- Understand the accuracy and limitations of Haar Cascade detection.

## Software Required

- Anaconda – Python
- Jupyter Notebook / VS Code
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- IPython

## Input Images Required

Place these three images inside your existing images folder:

- `person.jpg` – Single-person photograph.
- `glasses.jpg` – Person wearing glasses.
- `group.jpg` – Group photograph.

**Image folder location:** `C:\Users\acer\images`

## Algorithm

1. Import OpenCV, NumPy, Matplotlib, and required libraries.
2. Load the input images and display them.
3. Load the Haar Cascade face and eye classifiers.
4. Convert the images into grayscale.
5. Detect faces using `detectMultiScale()`.
6. Draw bounding boxes around detected faces.
7. Extract the face ROI from the original image.
8. Detect eyes using the Haar Cascade eye classifier.
9. Display the face and eye detection results.
10. Capture webcam frames and detect faces in real time.
11. Display detected faces with labels using Matplotlib.
12. Analyze the accuracy and limitations of Haar Cascade detection.

## Program

### Step 1: Import Required Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### Step 2: Load Haar Cascade Classifiers

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("Haar Cascade files could not be loaded.")

print("Haar Cascade Classifiers Loaded Successfully")
```

### Step 3: Read and Display Multiple Images

```python
image_folder = r"C:\Users\acer\images"

image_paths = {
    "Single Person": image_folder + r"\person.jpg",
    "With Glasses": image_folder + r"\glasses.jpg",
    "Group Photo": image_folder + r"\group.jpg"
}

images = {}

for name, path in image_paths.items():
    image = cv2.imread(path)

    if image is None:
        print("Image not found:", path)
        continue

    images[name] = image

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(7, 5))
    plt.imshow(image_rgb)
    plt.title(name)
    plt.axis("off")
    plt.show()
```

### Step 4: Define Face Detection Function

```python
def detect_face(image):
    result = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            result,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return result, faces
```

### Step 5: Perform Face Detection on Multiple Images

```python
for name, image in images.items():

    result, faces = detect_face(image)

    print(name, "- Faces Detected:", len(faces))

    result_rgb = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(8, 5))
    plt.imshow(result_rgb)
    plt.title("Face Detection - " + name)
    plt.axis("off")
    plt.show()
```

### Step 6: Extract ROI (Region of Interest)

```python
if "Single Person" in images:

    image = images["Single Person"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) > 0:

        x, y, w, h = faces[0]

        roi = image[y:y+h, x:x+w]

        roi_rgb = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        )

        plt.figure(figsize=(6, 5))
        plt.imshow(roi_rgb)
        plt.title("Extracted Face ROI")
        plt.axis("off")
        plt.show()

    else:
        print("No face detected for ROI extraction.")
```

### Step 7: Define Eye Detection Function

```python
def detect_eyes(image):

    result = image.copy()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face_roi = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                result,
                (x + ex, y + ey),
                (x + ex + ew, y + ey + eh),
                (255, 0, 0),
                2
            )

    return result
```

### Step 8: Perform Eye Detection on Images

```python
for name, image in images.items():

    result = detect_eyes(image)

    result_rgb = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(8, 5))
    plt.imshow(result_rgb)
    plt.title("Eye Detection - " + name)
    plt.axis("off")
    plt.show()
```

### Step 9: Real-Time Face Detection Using Webcam with Matplotlib

```python
from IPython.display import display
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam could not be opened.")

fig, ax = plt.subplots(figsize=(8, 5))

display_handle = display(fig, display_id=True)

try:
    for frame_number in range(150):

        ret, frame = cap.read()

        if not ret:
            print("Unable to read webcam frame.")
            break

        result, faces = detect_face(frame)

        cv2.putText(
            result,
            "Faces Detected: " + str(len(faces)),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        frame_rgb = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )

        ax.clear()
        ax.imshow(frame_rgb)
        ax.set_title("Real-Time Face Detection")
        ax.axis("off")

        display_handle.update(fig)
        time.sleep(0.03)

except KeyboardInterrupt:
    print("Webcam detection stopped.")

finally:
    cap.release()
    plt.close(fig)
    print("Webcam released successfully.")
```

### Step 10: Analyze Detection Results

```python
for name, image in images.items():

    result, faces = detect_face(image)

    print("--------------------------------")
    print("Image:", name)
    print("Number of Faces Detected:", len(faces))

    if len(faces) == 0:
        print("No faces detected.")
    else:
        print("Face detection completed.")

    print("--------------------------------")
```

## Developed By

- **Name:** Shaajini N
- **Register No:** 212225040395

## Output

### 1. Original Images

- Single-person image is displayed.
- Image with glasses is displayed.
- Group photograph is displayed.

### 2. Face Detection

- Haar Cascade classifier is applied.
- Green bounding boxes are drawn around detected faces.
- Face detection results are displayed for all three images.

### 3. ROI Extraction

- A detected face region is extracted from the original image.
- The extracted face region is displayed separately.

### 4. Eye Detection

- Eye Cascade classifier is applied to detected face regions.
- Blue bounding boxes are drawn around detected eyes.
- Eye detection results are displayed.

### 5. Real-Time Webcam Detection

- Webcam captures video frames.
- Haar Cascade detects faces in each frame.
- Bounding boxes and face labels are displayed using Matplotlib.

## Accuracy and Limitations

### Accuracy

- Haar Cascade can detect frontal faces under suitable lighting conditions.
- It can detect multiple faces in a group photograph.
- It can identify some faces even when glasses are present.

### Limitations

- Detection accuracy may decrease under poor lighting.
- Side-facing and partially covered faces may not be detected.
- Glasses and reflections may affect eye detection.
- False detections may occur in complex backgrounds.
- Detection performance depends on image quality and classifier settings.

## Result

Thus, face detection using Haar Cascade classifiers was successfully implemented using Python and OpenCV. Face detection, eye detection, ROI extraction, and real-time webcam face detection were performed and visualized using Matplotlib.

## Output Screenshots

Add your Jupyter Notebook screenshots below:

- **Output 1:** Original Single-Person Image
- **Output 2:** Original Image with Glasses
- **Output 3:** Original Group Photograph
- **Output 4:** Face Detection Results
- **Output 5:** Extracted Face ROI
- **Output 6:** Eye Detection Results
- **Output 7:** Real-Time Webcam Face Detection
- **Output 8:** Detection Analysis

## Instructions

1. Open Anaconda Navigator and launch Jupyter Notebook.
2. Install OpenCV and Matplotlib if they are not already installed.
3. Place the three input images inside `C:\Users\acer\images`.
4. Run each program separately in the given order.
5. Allow webcam access when prompted.
6. Verify the face and eye detection outputs.
7. Capture screenshots of all outputs.
8. Save the completed notebook as `Face-Detection.ipynb`.
9. Upload both `README.md` and `Face-Detection.ipynb` to your GitHub repository.
10. Attach the lab report PDF separately in LMS.
11. Mention the GitHub repository URL in the submission text.
