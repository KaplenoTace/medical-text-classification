# Quick Start Guide

## Medical Text Classification Project

This guide will help you get started with the Medical Text Classification project quickly.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- 4GB RAM minimum (8GB recommended)
- Internet connection for downloading dependencies

## Step 1: Clone the Repository

```bash
git clone https://github.com/KaplenoTace/medical-text-classification.git
cd medical-text-classification
```

## Step 2: Create a Virtual Environment

It's recommended to use a virtual environment to avoid dependency conflicts.

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Note: Installation may take 5-10 minutes depending on your internet connection.

## Step 4: Download NLTK Data (First Time Only)

Run Python and execute the following:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

Or run:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Step 5: Prepare the Data

1. **Download the MTSamples dataset**
   - Visit the MTSamples source
   - Download the `MTSamples.csv` file

2. **Place the dataset in the correct location:**
   ```bash
   # Make sure the data/raw/ directory exists
   mkdir -p data/raw
   
   # Move your downloaded file
   mv ~/Downloads/MTSamples.csv data/raw/
   ```

## Step 6: Explore the Notebooks

Start Jupyter Notebook to explore the data and models:

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and open:
- `01_data_exploration.ipynb` - Explore the dataset
- `02_preprocessing.ipynb` - Data preprocessing
- `03_model_training.ipynb` - Train models
- `04_evaluation.ipynb` - Evaluate model performance

Refer to `notebooks-guide.md` for detailed information about each notebook.

## Step 7: Run the Web Application

Start the Flask web server:

```bash
python app.py
```

The application will start on `http://localhost:5000`

Open your browser and navigate to:
```
http://localhost:5000
```

## Step 8: Make Predictions

### Using the Web Interface:
1. Open `http://localhost:5000` in your browser
2. Enter medical text in the text area
3. Click "Classify"
4. View the predicted medical specialty

### Using the Command Line:

```python
from inference import MedicalTextClassifier

# Initialize the classifier
classifier = MedicalTextClassifier(model_path='models/saved_models/best_model.pkl')

# Example medical text
text = """The patient presents with severe headache and dizziness. 
Neurological examination shows normal reflexes."""

# Make prediction
result = classifier.predict(text)
print(f"Predicted Specialty: {result['specialty']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Data file not found
**Solution:** Verify MTSamples.csv is in `data/raw/` directory:
```bash
ls data/raw/MTSamples.csv
```

### Issue: Port 5000 already in use
**Solution:** Use a different port:
```bash
python app.py --port 5001
```

### Issue: Out of memory during model training
**Solution:** Reduce batch size in `models/config.yaml`

## Next Steps

1. **Explore the Notebooks**: Learn about data preprocessing and model training
2. **Train Your Own Model**: Use the provided notebooks to train custom models
3. **Customize the Web Interface**: Modify `frontend/index.html` to suit your needs
4. **Experiment with Hyperparameters**: Edit `models/config.yaml`
5. **Add New Features**: Extend the codebase with additional functionality

## Quick Commands Reference

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Start Jupyter
jupyter notebook

# Run web app
python app.py

# Run inference script
python inference.py --text "Your medical text here"

# Deactivate virtual environment
deactivate
```

## Additional Resources

- See `README.md` for project overview
- See `notebooks-guide.md` for notebook documentation
- Check `models/config.yaml` for configuration options

## Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review the documentation in the notebooks
3. Open an issue on GitHub

Happy classifying! 🏥📊
