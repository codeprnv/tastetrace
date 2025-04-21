# TasteTrace: Object Detection for Food Ingredients in Google Colab 🍎🥕

Welcome to **TasteTrace**, a powerful object detection pipeline that identifies food ingredients in images with precision and ease, optimized for Google Colab! Powered by **GroundingDINO**, **Segment Anything Model (SAM)**, and **YOLO**, TasteTrace automates dataset annotation, trains custom models, performs inference, and exports to **TFLite** for deployment on edge devices. Whether you're a student, researcher, or food tech enthusiast, TasteTrace makes ingredient detection accessible and fun in the cloud! 🚀

This project is perfect for building smart kitchen assistants, dietary tracking apps, or automated grocery inventory systems. Run it entirely in Google Colab with GPU acceleration and start detecting ingredients like `carrot`, `bittergourd`, `brinjal`, and more in no time!

## 🌟 Features

- **Automated Annotation**: Use GroundingDINO and SAM to generate bounding boxes and masks for food ingredients in images.
- **Custom Dataset Training**: Train a YOLO model on your dataset for tailored ingredient detection.
- **Flexible Inference**: Detect ingredients in single images (Colab supports image inference; real-time webcam detection is not supported due to lack of webcam access).
- **TFLite Export**: Convert models to TFLite for mobile or edge device deployment.
- **Colab-Optimized**: Seamless setup with GPU support, file uploads, and cloud-based execution.
- **GitHub-Ready**: Well-documented code for sharing and collaboration.

## 📂 Project Structure

```
tastetrace/
├── data.yaml              # YOLO dataset configuration
├── tastetrace_dataset/    # Directory for input images (upload to Colab)
├── weights/               # Pre-trained and trained model weights
├── GroundingDINO/         # GroundingDINO repository (cloned in notebook)
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation (you're here!)
└── TasteTrace.ipynb       # Main Jupyter Notebook for Colab
```

## 🛠️ Prerequisites

To run TasteTrace in Google Colab, you need:

- A **Google account** to access Colab.
- A **dataset of food images** (e.g., JPG, PNG) for annotation and training.
- **Internet access** for cloning repositories and downloading weights.
- Basic familiarity with uploading files to Colab.

## 🚀 Setup Instructions for Google Colab

Follow these steps to run TasteTrace in Google Colab:

1. **Open Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/).
   - Create a new notebook or upload `TasteTrace.ipynb` (see step 2).

2. **Clone the Repository or Upload Files**:
   - **Option 1: Clone the Repository**: Run the following command in a Colab cell to clone the TasteTrace repository:
     ```bash
     !git clone https://github.com/codeprnv/tastetrace.git
     %cd tastetrace
     ```
   - **Option 2: Upload Files**:
     - Download `TasteTrace.ipynb`, `data.yaml`, and `requirements.txt` from the [TasteTrace repository](https://github.com/codeprnv/tastetrace).
     - In Colab, click the **Files** tab (left sidebar) and upload these files to `/content/`.
     - Upload your dataset images to `/content/tastetrace_dataset/images/` (create the folder if needed):
       ```bash
       !mkdir -p /content/tastetrace_dataset/images
       ```

3. **Enable GPU Acceleration**:
   - Go to `Runtime > Change runtime type`.
   - Select **GPU** (e.g., T4 GPU) as the hardware accelerator and click **Save**.
   - Verify GPU availability by running:
     ```python
     import torch
     print(torch.cuda.is_available())
     ```

4. **Install Dependencies**:
   - Ensure `requirements.txt` is in `/content/tastetrace/` (or `/content/` if not cloned).
   - Run the following in a Colab cell to install dependencies:
     ```bash
     !pip install -r requirements.txt
     ```
   - Note: `GroundingDINO` and `segment-anything` are installed via notebook cells (not in `requirements.txt`). The notebook will handle their installation automatically.

5. **Prepare Your Dataset**:
   - Ensure your food images (e.g., `carrot.jpg`, `brinjal.jpg`) are in `/content/tastetrace_dataset/images/`.
   - Upload or create `data.yaml` in `/content/tastetrace/` (or `/content/`) with the following content:
     ```yaml
     names:
       - carrot
       - bittergourd
       - bottlegourd
       - brinjal
       - broccoli
       - potato
       - tomato
     nc: 7
     test: ./tastetrace_dataset/test/images
     train: ./tastetrace_dataset/train/images
     val: ./tastetrace_dataset/valid/images
     ```
   - Adjust paths in `data.yaml` if your dataset is elsewhere (e.g., `/content/tastetrace_dataset/`).

6. **Download Model Weights**:
   - The notebook includes cells to download weights for GroundingDINO and SAM. Run these cells to download:
     - GroundingDINO weights: `groundingdino_swint_ogc.pth`
     - SAM weights: `sam_vit_h_4b8939.pth`
   - Weights are saved to `/content/tastetrace/weights/`.

7. **Run the Notebook**:
   - Open `TasteTrace.ipynb` in Colab:
     - If cloned, navigate to `/content/tastetrace/TasteTrace.ipynb`.
     - If uploaded, open `/content/TasteTrace.ipynb`.
   - Execute the cells sequentially (`Shift + Enter`) to:
     - Install `GroundingDINO` and `segment-anything`.
     - Download model weights.
     - Annotate your dataset.
     - Train a YOLO model.
     - Perform inference.
     - Export to TFLite.

8. **Download Outputs**:
   - After training, the trained model is saved in `/content/tastetrace/saved_model/trained_model.pt`.
   - The TFLite model is saved as `/content/tastetrace/tastetrace.tflite`.
   - Download these files from the Colab **Files** tab (right-click > Download) or use:
     ```python
     from google.colab import files
     files.download('/content/tastetrace/tastetrace.tflite')
     ```

## 📖 Usage in Colab

TasteTrace is designed for ease of use in Colab. Here’s how to use the notebook:

### 1. **Annotate Your Dataset**
- Run the annotation cells to input class names (e.g., `carrot, bittergourd, brinjal`).
- GroundingDINO and SAM generate bounding boxes and masks for images in `/content/tastetrace_dataset/images/`.
- Annotations are saved in YOLO format under `/content/tastetrace_dataset/` (train, valid, test splits).

### 2. **Train a YOLO Model**
- Specify the path to `data.yaml` (e.g., `/content/tastetrace/data.yaml`).
- Train a YOLO model with GPU acceleration.
- Adjust parameters (e.g., epochs, batch size) in the training cell if needed.
- The trained model is saved as `/content/tastetrace/saved_model/trained_model.pt`.

### 3. **Perform Inference**
- **Single Image**: Upload an image (e.g., `/content/tastetrace_dataset/images/carrot.jpg`) and run the inference cell to detect ingredients with bounding boxes and labels.
- **Real-Time**: Real-time webcam detection is not supported in Colab due to lack of webcam access. Use image inference instead.
- Example:
  ```python
  main(mode="image", yaml_path="/content/tastetrace/data.yaml", image_path="/content/tastetrace_dataset/images/carrot.jpg", save=True)
  ```

### 4. **Export to TFLite**
- Convert the trained YOLO model to TFLite for edge deployment.
- The output is saved as `/content/tastetrace/tastetrace.tflite`.
- Download the TFLite model for use in mobile or IoT applications.

## 🎨 Example Output

Here’s what you’ll see in Colab:

- **Annotated Image**: *Carrot detected with bounding box and mask, labeled with confidence score.*
- **Training Logs**: Metrics and progress for YOLO model training, displayed in Colab’s output.
- **Inference Results**: Visualized images with detected ingredients (e.g., `tomato: 0.95`).
- **TFLite Model**: A lightweight `tastetrace.tflite` file for deployment.

## 🧑‍💻 Contributing

We welcome contributions to enhance TasteTrace! Here’s how to contribute via GitHub:

1. **Fork the Repository**:
   ```bash
   git fork https://github.com/codeprnv/tastetrace.git
   ```
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature
   ```
3. **Commit Changes**:
   ```bash
   git commit -m "Add your feature"
   ```
4. **Push to GitHub**:
   ```bash
   git push origin feature/your-feature
   ```
5. **Open a Pull Request**: Describe your changes on GitHub.

### Ideas for Contributions
- Add support for additional models (e.g., Faster R-CNN).
- Optimize TFLite conversion for specific devices.
- Create a Streamlit app for interactive inference in Colab.
- Enhance dataset annotation with more class names.

## 📜 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute TasteTrace as you see fit!

## 🙏 Acknowledgments

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO): For zero-shot object detection.
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything): For precise segmentation.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics): For fast and robust detection.
- [TensorFlow](https://www.tensorflow.org/): For TFLite conversion.
- [Google Colab](https://colab.research.google.com/): For free GPU access and cloud computing.
- The open-source community for inspiration and support.

Happy detecting, and let TasteTrace bring your food recognition projects to life in Google Colab! 🍽️