# TumorScope: Brain MRI Classification

This project uses deep learning to classify MRI images of the brain into four categories:
1. Glioma
2. Meningioma
3. Pituitary
4. No Tumor

The goal is to assist medical professionals by providing an automated tool for brain tumor detection and categorization.

## Features
- Pretrained convolutional neural network (CNN) for image classification
- Web interface built using Streamlit for user-friendly interaction
- Supports real-time image uploads for predictions

## Dataset
The dataset used for training and testing the model is available on Kaggle:
[Brain Tumor MRI Dataset ](https://www.kaggle.com/datasets/darshandalvi12/brain-tumor-dataset)

### Download Instructions
1. Visit the dataset page: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/darshandalvi12/brain-tumor-dataset)
2. Click on the **Download** button.
3. Extract the downloaded file to your local machine.
4. Place the extracted folder in your project directory.

   For example, if your project structure is:
   ```
   TumorScope--Brain-MRI-Classification/
   |-- model.ipynb
   |-- app.py
   |-- brain_tumor_data/  # Place dataset here
   |-- brain_tumor_model.h5
   ```

## Getting Started

### Prerequisites
Ensure you have Python and the required libraries installed. Use the following command to install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Web App
1. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
2. Open the provided URL in your browser.

### Using the App
- Upload an MRI image using the interface.
- The app will classify the image and display the predicted tumor type.

## Model Training
If you wish to retrain the model, open `model.ipynb` and follow the instructions. After training, save the model:
```python
classifier.save('brain_tumor_model.h5')
```
Ensure the `brain_tumor_model.h5` file is in the project directory for the web app to function.

## Directory Structure
```
TumorScope--Brain-MRI-Classification/
|-- app.py                # Streamlit app script
|-- model.ipynb           # Jupyter notebook for model training
|-- brain_tumor_model.h5              # Pretrained model file
|-- brain_tumor_data/     # Dataset folder
|-- README.md             # Project documentation
|-- requirements.txt      # Python dependencies
```

## License
This project is licensed under the MIT License. See `LICENSE` for details.

