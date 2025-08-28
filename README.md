# 🧠 ACE CNN Project – Handwritten Digit Recognition (MNIST)


This project demonstrates a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset** to recognize handwritten digits (0–9).  
It includes:
-  Model Training (`train.py`)  
-  Real-time Inference via Webcam (`main.py`)  
-  Interactive Web App using Streamlit (`app_streamlit.py`)  

---

## 📂 Project Structure

ace_cnn_project/
│── model.py # CNN model architecture
│── train.py # Training script
│── main.py # Real-time webcam inference
│── app_streamlit.py # Streamlit web interface
│── requirements.txt # Python dependencies
│── README.md # Documentation
│── cnn_mnist.h5 # Saved trained model (after training)


---

##  Setup Instructions

1. Create Virtual Environment

Using Conda:

conda create -n ace_env python=3.10 -y
conda activate ace_env

Or using venv:

python -m venv ace_env
.\ace_env\Scripts\activate  # On Windows

2. Install Dependencies

pip install -r requirements.txt

If requirements.txt is missing:

pip install tensorflow numpy pillow matplotlib seaborn scikit-learn opencv-python streamlit

🚀 Training the Model

Run:

python train.py

✔️ This will:

    Train the CNN on the MNIST dataset

    Save the model as cnn_mnist.h5

    Plot Accuracy & Loss graphs

    Display a Confusion Matrix

📊 Sample Training Curves:

    Accuracy: ~99%

    Loss converges quickly with minimal overfitting

🎥 Real-Time Webcam Inference

To predict digits in real-time using your webcam:

python main.py

🎯 Features:

    Opens a webcam feed

    Detects handwritten digits

    Press q to quit

🌐 Web Interface (Optional)

Run the Streamlit App:

streamlit run app_streamlit.py

🌟 Features:

    Upload digit images

    Get instant predictions

    Clean UI powered by Streamlit

📊 Results

    ✅ Test Accuracy: ~99% on MNIST

    ✅ Robust confusion matrix (minimal misclassifications)

    ⚡ Extendable to custom datasets

📌 Requirements

    Python 3.10+

    TensorFlow 2.x

    NumPy, Pillow, Matplotlib, Seaborn

    Scikit-learn, OpenCV

    Streamlit




## 📌 Video link
 Using Web interface: https://1drv.ms/v/c/470a9832c1a989ec/EQOtoU0nYblBpHFKeq2ztO0Bc-ozPv_UQEIA01Fgi0jOOw?e=89Epl0
Using Web cam: https://1drv.ms/v/c/470a9832c1a989ec/ETIkJzwQwsxDq0n1FJ6pGRwBYHFCS_Ak3TUVkeksxf4r6w?e=omUIK1






