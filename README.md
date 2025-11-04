# Medical Text Classification

## Project Summary

This project implements a medical text classification system for categorizing medical transcriptions into appropriate medical specialties. The system uses natural language processing and machine learning techniques to analyze medical text data and predict the corresponding medical specialty.

## Features

- Medical text preprocessing and feature extraction
- Multiple classification models support
- Interactive web interface for inference
- Jupyter notebooks for experimentation and analysis
- Model training and evaluation pipeline

## Data

**Important:** Please download and place MTSamples.csv into data/raw/

The dataset should contain medical transcription samples with their corresponding medical specialty labels.

## Project Structure

```
medical-text-classification/
├── data/
│   ├── raw/              # Place MTSamples.csv here
│   └── processed/        # Processed data files
├── models/
│   ├── saved_models/     # Trained model files
│   └── config.yaml       # Model configuration
├── notebooks/            # Jupyter notebooks for exploration
├── frontend/
│   └── index.html        # Web interface
├── inference.py          # Inference script
├── app.py               # Main application
├── requirements.txt      # Dependencies
├── quickstart.md        # Quick start guide
├── notebooks-guide.md   # Notebooks documentation
└── .gitignore           # Git ignore rules
```

## Requirements

### Dependencies

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/PyTorch
- NLTK
- Flask
- Jupyter
- PyYAML
- Matplotlib
- Seaborn

See `requirements.txt` for complete list with versions.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/KaplenoTace/medical-text-classification.git
cd medical-text-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and place MTSamples.csv into `data/raw/`

4. Run the application:
```bash
python app.py
```

For detailed instructions, see `quickstart.md`

## Notebooks

This project includes several Jupyter notebooks for data exploration, model training, and evaluation. See `notebooks-guide.md` for detailed information about each notebook.

## Modules

- **inference.py**: Standalone inference module for making predictions on new text
- **app.py**: Main Flask application for web-based interaction
- **models/config.yaml**: Configuration file for model parameters

## Usage

### Web Interface

Run `python app.py` and navigate to `http://localhost:5000` to use the web interface.

### Command Line Inference

```python
from inference import MedicalTextClassifier

classifier = MedicalTextClassifier(model_path='models/saved_models/best_model.pkl')
prediction = classifier.predict("Your medical text here")
print(f"Predicted Specialty: {prediction}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
