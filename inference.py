#!/usr/bin/env python3
"""
Medical Text Classification - Inference Module

This module provides the MedicalTextClassifier class for making
predictions on medical transcription text.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicalTextClassifier:
    """
    Medical Text Classification System
    
    This class handles loading trained models and making predictions
    on medical transcription text to classify them into appropriate
    medical specialties.
    """
    
    def __init__(self, 
                 model_path: str = 'models/saved_models/best_model.pkl',
                 vectorizer_path: str = 'models/saved_models/vectorizer.pkl'):
        """
        Initialize the Medical Text Classifier
        
        Args:
            model_path: Path to the trained model file
            vectorizer_path: Path to the vectorizer file
        """
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.specialties = None
        
        logger.info(f"Initializing MedicalTextClassifier")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Vectorizer path: {self.vectorizer_path}")
        
        self._load_model()
        self._load_vectorizer()
    
    def _load_model(self):
        """
        Load the trained classification model
        """
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info("Model loaded successfully")
            
            # Try to extract label information if available
            if hasattr(self.model, 'classes_'):
                self.specialties = list(self.model.classes_)
                logger.info(f"Found {len(self.specialties)} specialty classes")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_vectorizer(self):
        """
        Load the text vectorizer
        """
        try:
            if not self.vectorizer_path.exists():
                raise FileNotFoundError(f"Vectorizer file not found: {self.vectorizer_path}")
            
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            logger.info("Vectorizer loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load vectorizer: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text
        
        Args:
            text: Raw medical text
        
        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        text = text.strip()
        text = text.lower()
        
        # Additional preprocessing can be added here
        # (lemmatization, stop word removal, etc.)
        
        return text
    
    def predict(self, text: str, top_k: int = 3) -> Dict:
        """
        Predict the medical specialty for given text
        
        Args:
            text: Medical transcription text
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Validate input
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")
            
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            # Vectorize the text
            X = self.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                confidence = float(np.max(probabilities))
                
                # Get top k predictions
                top_indices = np.argsort(probabilities)[-top_k:][::-1]
                top_predictions = [
                    {
                        'specialty': self.specialties[idx] if self.specialties else f"Class_{idx}",
                        'confidence': float(probabilities[idx])
                    }
                    for idx in top_indices
                ]
            else:
                confidence = 1.0
                top_predictions = [{
                    'specialty': prediction,
                    'confidence': 1.0
                }]
            
            result = {
                'specialty': prediction,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'text_length': len(text),
                'processed_length': len(processed_text)
            }
            
            logger.info(f"Prediction: {prediction} (confidence: {confidence:.3f})")
            
            return result
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict medical specialties for multiple texts
        
        Args:
            texts: List of medical transcription texts
        
        Returns:
            List of prediction results
        """
        results = []
        
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict text: {e}")
                results.append({
                    'error': str(e),
                    'specialty': None,
                    'confidence': 0.0
                })
        
        return results
    
    def get_specialties(self) -> List[str]:
        """
        Get list of available medical specialties
        
        Returns:
            List of specialty names
        """
        if self.specialties:
            return self.specialties
        elif hasattr(self.model, 'classes_'):
            return list(self.model.classes_)
        else:
            return []
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_type': type(self.model).__name__,
            'model_path': str(self.model_path),
            'vectorizer_type': type(self.vectorizer).__name__,
            'vectorizer_path': str(self.vectorizer_path),
            'num_specialties': len(self.specialties) if self.specialties else 0,
            'specialties': self.specialties
        }
        
        return info


def main():
    """
    Command-line interface for the classifier
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Medical Text Classification - Inference'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Medical text to classify'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Path to text file containing medical text'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/saved_models/best_model.pkl',
        help='Path to model file'
    )
    parser.add_argument(
        '--vectorizer',
        type=str,
        default='models/saved_models/vectorizer.pkl',
        help='Path to vectorizer file'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top predictions to show'
    )
    
    args = parser.parse_args()
    
    # Initialize classifier
    try:
        classifier = MedicalTextClassifier(
            model_path=args.model,
            vectorizer_path=args.vectorizer
        )
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        return
    
    # Get text from arguments or file
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return
    else:
        logger.error("Please provide either --text or --file argument")
        return
    
    # Make prediction
    try:
        result = classifier.predict(text, top_k=args.top_k)
        
        print("\n" + "="*60)
        print("Medical Text Classification Result")
        print("="*60)
        print(f"\nPredicted Specialty: {result['specialty']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nTop {args.top_k} Predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {pred['specialty']}: {pred['confidence']:.2%}")
        print(f"\nText Length: {result['text_length']} characters")
        print("="*60 + "\n")
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")


if __name__ == '__main__':
    main()
