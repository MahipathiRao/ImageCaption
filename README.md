# 🖼️ ImageCaptionGenerator

> An interactive deep learning tool that generates context-aware captions for images in real time — powered by LLaMA and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LLaMA](https://img.shields.io/badge/LLaMA_Model-6D28D9?style=flat&logo=meta&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📌 Overview

**ImageCaptionGenerator** is an AI-powered image captioning application that uses the **LLaMA language model** combined with deep learning vision techniques to generate accurate, descriptive, and context-aware captions for any image. With a clean **Streamlit** interface, users can upload images and receive human-quality captions instantly — no ML expertise required.

---

## ✨ Features

- 🧠 Context-aware caption generation using the **LLaMA model**
- ⚡ Real-time image processing with instant caption output
- 🖥️ Intuitive, responsive UI built with **Streamlit**
- 📷 Supports common image formats: JPG, PNG, WEBP
- 🔍 Deep learning vision pipeline for accurate image feature extraction
- 🔄 Process multiple images in a single session

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI / Frontend | Streamlit |
| Language Model | LLaMA (Meta) |
| Deep Learning | PyTorch / TensorFlow |
| Image Processing | PIL / OpenCV |
| Backend | Python |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip
- LLaMA model weights (see setup below)
- GPU recommended for faster inference (CPU supported)

### Installation

```bash
# Clone the repository
git clone https://github.com/MahipathiRao/image-caption-generator.git
cd image-caption-generator

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Model Setup

Download the LLaMA model weights and place them in the `models/` directory:

```bash
mkdir models
# Place your LLaMA model weights here
# e.g., models/llama-model.bin
```

> You can obtain LLaMA model weights from [Meta's official page](https://ai.meta.com/llama/) after accepting their usage terms.

### Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Requirements

```
streamlit
torch
transformers
Pillow
opencv-python
numpy
```

---

## 📸 Screenshots

> _Add screenshots of the app UI here_

---

## 📂 Project Structure

```
image-caption-generator/
├── app.py                      # Main Streamlit application
├── models/
│   └── README.md               # Instructions for placing model weights
├── src/
│   ├── captioner.py            # LLaMA caption generation logic
│   ├── image_processor.py      # Image preprocessing pipeline
│   └── utils.py                # Helper functions
├── sample_images/
│   └── test.jpg                # Sample image for testing
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

1. **Image Upload**: User uploads an image via the Streamlit interface
2. **Preprocessing**: The image is resized, normalized, and passed through a deep learning vision encoder to extract visual features
3. **Caption Generation**: Extracted features are fed into the LLaMA model, which generates a descriptive natural language caption
4. **Display**: The generated caption is rendered in real time alongside the uploaded image

```
Image Input → Vision Encoder → Feature Vector → LLaMA Model → Caption Output
```

---

## ⚙️ Configuration

You can adjust the following settings in `app.py` or via a `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 100 | Max caption length |
| `temperature` | 0.7 | Creativity of output |
| `image_size` | 224x224 | Input resolution for encoder |

---

## 🔮 Future Improvements

- [ ] Support for batch image captioning
- [ ] Add multilingual caption output
- [ ] Fine-tune LLaMA on domain-specific image datasets
- [ ] Deploy as a REST API with FastAPI
- [ ] Export captions to CSV / JSON

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Venkata Mahipathi Rao Topella**  
📧 mahitthopella2004@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/mahi-topella) | [GitHub](https://github.com/MahipathiRao)
