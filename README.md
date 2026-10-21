# SkinSakhi — AI-Assisted Skin Condition Screening

**An AI-powered web application for preliminary skin-condition screening and awareness.**

## Overview

SkinSakhi is an AI-assisted web application that analyzes uploaded skin images and predicts one of eight skin-condition categories. It uses a trained MobileNetV3-Small image-classification model to provide preliminary results through a simple web interface.

The project explores how deep learning can support skin-health awareness and encourage timely consultation with healthcare professionals.

## Problem Statement

Recognizing skin conditions can be challenging without medical expertise. SkinSakhi explores an accessible, web-based approach to analyzing skin images and presenting preliminary classification results to support awareness and further medical consultation.

## Key Features

- **Image-based screening:** Upload a skin image for analysis.
- **AI-powered classification:** Uses a trained MobileNetV3-Small model.
- **Eight skin-condition categories:** Provides a predicted category from the supported classes.
- **Uncertainty handling:** Uses prediction-confidence checks to flag uncertain results.
- **Web-based interface:** Access the application through a browser.
- **Medical disclaimer:** Communicates that predictions are not a substitute for professional diagnosis.

## Supported Categories

The model is trained to classify images into these eight dataset categories:

1. Cellulitis
2. Impetigo
3. Athlete's foot
4. Nail fungus
5. Ringworm
6. Cutaneous larva migrans
7. Chickenpox
8. Shingles

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning framework | TensorFlow / Keras |
| Classification model | MobileNetV3-Small |
| Dataset | Public Kaggle skin-disease image dataset |
| Web interface | Gradio |
| Deployment | Render |

## Model Performance

The model achieved approximately **87.98% accuracy on the held-out test dataset**.

This result reflects evaluation on the available dataset only. It does not establish clinical accuracy or guarantee reliable predictions on images from real-world users.

## Live Demo

**Try SkinSakhi:** https://skinsakhi.onrender.com

Open the website, upload a skin image, and select the analysis option to view the model's preliminary prediction.

*Note: The free hosting service may take longer to respond after a period of inactivity.*

## Limitations and Medical Disclaimer

SkinSakhi is an experimental educational prototype and has not been clinically validated. It is not a medical device and cannot confirm or rule out a skin disease. Predictions may be incorrect, particularly for images or conditions that differ from the training dataset.

The application is intended for preliminary screening and awareness only. Users should consult a qualified healthcare professional for diagnosis and treatment.

## Future Scope

- Evaluate the model using more diverse skin-image datasets.
- Improve performance on categories with lower test recall.
- Seek review and validation from qualified medical professionals.
- Explore support for additional conditions and languages.
- Improve accessibility and usability for community health settings.

---

**SkinSakhi — Exploring AI for accessible skin-health awareness.**
