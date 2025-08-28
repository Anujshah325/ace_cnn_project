# Real-Time Object Classification (MNIST Digits)

## 📌 Project Structure
- `model.py` → CNN architecture
- `train.py` → Training & evaluation (run in Colab)
- `main.py` → Real-time webcam inference (local PC)
- `app_streamlit.py` → Optional web interface
- `cnn_mnist.h5` → Trained model

---
## 📌 Features
- Train a CNN on MNIST or custom dataset
- Real-time prediction using webcam (OpenCV)
- Optional web interface using Streamlit


## 🚀 How to Run
```bash
pip install -r requirements.txt
python model.py
python main.py
streamlit run app.py



