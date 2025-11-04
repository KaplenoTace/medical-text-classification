# Notebooks Guide

## Medical Text Classification - Jupyter Notebooks Documentation

This guide provides detailed information about the Jupyter notebooks included in this project. Each notebook is designed for a specific stage of the medical text classification pipeline.

## Overview

The notebooks in this project follow a sequential workflow:

1. Data Exploration → 2. Preprocessing → 3. Model Training → 4. Evaluation

## Notebook Descriptions

### 1. `01_data_exploration.ipynb`

**Purpose:** Initial exploration and analysis of the MTSamples medical transcription dataset.

**What you'll learn:**
- Dataset structure and size
- Distribution of medical specialties
- Text length statistics
- Common terms and phrases
- Data quality issues

**Key Sections:**
- Loading and inspecting the dataset
- Specialty distribution visualization
- Text length analysis
- Word frequency analysis
- Sample transcription review
- Missing data analysis

**Prerequisites:** None - This is the starting point

**Estimated Time:** 15-20 minutes

**Expected Outputs:**
- Distribution plots
- Summary statistics
- Word clouds for different specialties

---

### 2. `02_preprocessing.ipynb`

**Purpose:** Clean and prepare the medical text data for machine learning models.

**What you'll learn:**
- Text cleaning techniques
- Tokenization methods
- Stop word removal
- Lemmatization/Stemming
- Feature extraction

**Key Sections:**
- Text cleaning (removing special characters, HTML tags)
- Lowercasing and normalization
- Tokenization using NLTK
- Stop word removal
- Lemmatization
- TF-IDF vectorization
- Creating train/test splits
- Saving processed data

**Prerequisites:** Completed `01_data_exploration.ipynb`

**Estimated Time:** 20-30 minutes

**Expected Outputs:**
- Processed text data in `data/processed/`
- Vectorizer objects saved for inference
- Train/test split files

---

### 3. `03_model_training.ipynb`

**Purpose:** Train multiple classification models and compare their performance.

**What you'll learn:**
- Training different ML algorithms
- Hyperparameter tuning
- Cross-validation techniques
- Model comparison

**Key Sections:**
- Loading preprocessed data
- Baseline model (Naive Bayes)
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Neural Network (if applicable)
- Hyperparameter tuning with GridSearchCV
- Saving best models

**Prerequisites:** Completed `02_preprocessing.ipynb`

**Estimated Time:** 30-45 minutes (longer if training deep learning models)

**Expected Outputs:**
- Trained models saved in `models/saved_models/`
- Training metrics and logs
- Model comparison table

---

### 4. `04_evaluation.ipynb`

**Purpose:** Evaluate model performance using various metrics and visualizations.

**What you'll learn:**
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrix analysis
- ROC curves and AUC scores
- Error analysis

**Key Sections:**
- Loading trained models
- Predictions on test set
- Confusion matrix visualization
- Per-class performance metrics
- Misclassification analysis
- ROC curves for multi-class classification
- Model strengths and weaknesses
- Recommendations for improvement

**Prerequisites:** Completed `03_model_training.ipynb`

**Estimated Time:** 20-30 minutes

**Expected Outputs:**
- Evaluation metrics report
- Confusion matrix plots
- ROC curve visualizations
- Error analysis document

---

## How to Use These Notebooks

### Sequential Approach (Recommended for Beginners)

1. Start with `01_data_exploration.ipynb` to understand the data
2. Proceed to `02_preprocessing.ipynb` to prepare the data
3. Train models using `03_model_training.ipynb`
4. Evaluate results with `04_evaluation.ipynb`

### Modular Approach (For Experienced Users)

If you're familiar with the dataset or have already completed some steps:
- Jump to any notebook based on your needs
- Use saved intermediate files (preprocessed data, trained models)
- Experiment with specific sections

## Running the Notebooks

### Method 1: Jupyter Notebook

```bash
jupyter notebook
```

Navigate to `notebooks/` and open the desired notebook.

### Method 2: JupyterLab

```bash
jupyter lab
```

Provides a more modern interface with additional features.

### Method 3: VS Code

If you have the Jupyter extension installed in VS Code:
1. Open the notebook file
2. Select your Python kernel
3. Run cells interactively

## Notebook Best Practices

### Before Running

1. **Ensure data is available:**
   ```bash
   ls data/raw/MTSamples.csv
   ```

2. **Check kernel:** Make sure you're using the correct Python environment with all dependencies installed

3. **Clear previous outputs:** Kernel → Restart & Clear Output (for a fresh start)

### During Execution

1. **Run cells sequentially:** Don't skip cells unless you know what you're doing
2. **Read markdown cells:** They contain important context and instructions
3. **Monitor resource usage:** Some operations (especially model training) can be memory-intensive
4. **Save frequently:** File → Save and Checkpoint

### After Completion

1. **Review outputs:** Check that all expected files were created
2. **Save results:** Export important plots and metrics
3. **Document changes:** If you modify notebooks, add comments

## Troubleshooting

### Issue: Kernel keeps dying
**Cause:** Out of memory
**Solution:** 
- Reduce batch size
- Process data in chunks
- Use a machine with more RAM

### Issue: ModuleNotFoundError
**Cause:** Missing dependencies
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: File not found errors
**Cause:** Incorrect file paths or missing data
**Solution:**
- Check that MTSamples.csv is in `data/raw/`
- Ensure previous notebooks have been run
- Verify directory structure

### Issue: NLTK data not found
**Cause:** NLTK resources not downloaded
**Solution:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Customization Ideas

### For Learning
- Add your own exploratory visualizations
- Experiment with different preprocessing techniques
- Try additional machine learning algorithms
- Implement feature engineering methods

### For Research
- Compare transformer-based models (BERT, BioBERT)
- Test different tokenization approaches
- Implement ensemble methods
- Add cross-validation strategies

### For Production
- Add data validation checks
- Implement logging
- Create model versioning
- Add performance benchmarks

## Expected Outputs Summary

| Notebook | Output Location | Files Created |
|----------|----------------|---------------|
| 01_data_exploration | `notebooks/figures/` | Distribution plots, word clouds |
| 02_preprocessing | `data/processed/` | Cleaned data, vectorizers, splits |
| 03_model_training | `models/saved_models/` | Model files (.pkl), training logs |
| 04_evaluation | `notebooks/results/` | Metrics reports, confusion matrices |

## Additional Resources

- **Python Libraries Documentation:**
  - [Pandas](https://pandas.pydata.org/docs/)
  - [Scikit-learn](https://scikit-learn.org/stable/)
  - [NLTK](https://www.nltk.org/)
  - [Matplotlib](https://matplotlib.org/)

- **Medical NLP Resources:**
  - [MTSamples Dataset](https://www.mtsamples.com/)
  - [Medical Text Classification Papers](https://scholar.google.com/)

- **Jupyter Tips:**
  - [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)
  - [Keyboard Shortcuts](https://towardsdatascience.com/jypyter-notebook-shortcuts-bf0101a98330)

## Contributing

If you create additional notebooks or improve existing ones:
1. Follow the naming convention: `##_description.ipynb`
2. Include markdown documentation
3. Test on a fresh environment
4. Update this guide

## Questions?

If you have questions about the notebooks:
1. Check this guide first
2. Review the notebook's markdown cells
3. Consult the README.md
4. Open an issue on GitHub

Happy exploring! 📊💻
