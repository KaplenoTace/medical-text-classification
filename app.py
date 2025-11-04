#!/usr/bin/env python3
"""
Medical Text Classification - Flask Web Application

This is the main Flask application that serves the web interface
for the medical text classification system.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='frontend',
            template_folder='frontend')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['MODEL_PATH'] = 'models/saved_models/best_model.pkl'
app.config['VECTORIZER_PATH'] = 'models/saved_models/vectorizer.pkl'

# Try to import inference module
try:
    from inference import MedicalTextClassifier
    model_available = True
    logger.info("Inference module loaded successfully")
except ImportError as e:
    logger.warning(f"Could not load inference module: {e}")
    logger.warning("Running in demo mode without actual predictions")
    model_available = False
    MedicalTextClassifier = None

# Initialize classifier (if available)
classifier = None
if model_available and os.path.exists(app.config['MODEL_PATH']):
    try:
        classifier = MedicalTextClassifier(
            model_path=app.config['MODEL_PATH'],
            vectorizer_path=app.config['VECTORIZER_PATH']
        )
        logger.info("Classifier loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        logger.info("Running in demo mode")
else:
    logger.info("Model not found. Running in demo mode.")
    logger.info("Please train a model first using the notebooks.")


@app.route('/')
def index():
    """
    Serve the main application page
    """
    return send_from_directory('frontend', 'index.html')


@app.route('/health')
def health():
    """
    Health check endpoint
    """
    status = {
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'version': '1.0.0'
    }
    return jsonify(status)


@app.route('/api/classify', methods=['POST'])
def classify_text():
    """
    Classify medical text
    
    Expects JSON payload with 'text' field
    Returns JSON with prediction results
    """
    try:
        # Get the text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing "text" field in request'
            }), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({
                'error': 'Text cannot be empty'
            }), 400
        
        # Check if model is available
        if classifier is None:
            # Demo mode - return mock prediction
            return jsonify({
                'specialty': 'Demo Mode',
                'confidence': 0.0,
                'message': 'Model not loaded. Please train a model first.',
                'demo_mode': True,
                'top_predictions': [
                    {'specialty': 'Demo Mode', 'confidence': 0.0}
                ]
            })
        
        # Make prediction
        result = classifier.predict(text)
        
        logger.info(f"Classified text successfully: {result['specialty']}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        return jsonify({
            'error': f'Classification failed: {str(e)}'
        }), 500


@app.route('/api/specialties')
def get_specialties():
    """
    Get list of available medical specialties
    """
    if classifier and hasattr(classifier, 'get_specialties'):
        specialties = classifier.get_specialties()
    else:
        # Demo specialties
        specialties = [
            'Allergy / Immunology',
            'Cardiovascular / Pulmonary',
            'Dentistry',
            'Dermatology',
            'ENT - Otolaryngology',
            'Gastroenterology',
            'General Medicine',
            'Neurology',
            'Obstetrics / Gynecology',
            'Orthopedic',
            'Psychiatry / Psychology',
            'Radiology',
            'Surgery',
            'Urology'
        ]
    
    return jsonify({
        'specialties': specialties,
        'count': len(specialties)
    })


@app.route('/api/stats')
def get_stats():
    """
    Get application statistics
    """
    stats = {
        'model_loaded': classifier is not None,
        'model_path': app.config['MODEL_PATH'],
        'specialties_count': 14,  # Approximate
        'version': '1.0.0'
    }
    
    return jsonify(stats)


@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors
    """
    return jsonify({
        'error': 'Endpoint not found',
        'status': 404
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 errors
    """
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'status': 500
    }), 500


if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Check if running in debug mode
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Medical Text Classification App")
    logger.info(f"Port: {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Model loaded: {classifier is not None}")
    
    if classifier is None:
        logger.warning("\n" + "="*60)
        logger.warning("WARNING: Running in DEMO MODE")
        logger.warning("No trained model found. Please train a model first.")
        logger.warning("See notebooks/ directory for training instructions.")
        logger.warning("="*60 + "\n")
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
