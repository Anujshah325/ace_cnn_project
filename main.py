import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("cnn_mnist.h5")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame (grayscale + resize)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(reshaped, verbose=0)
    digit = np.argmax(prediction)

    # Display prediction
    cv2.putText(frame, f"Prediction: {digit}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Digit Recognition", frame)

    # ✅ Quit on 'q' or 'ESC'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 = ESC key
        break


cap.release()
cv2.destroyAllWindows()
