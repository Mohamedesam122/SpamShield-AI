# Spam Detector App

A modern Spam Detection web application built using Machine Learning, NLP, and Streamlit.

The application can classify SMS or email messages as Spam or Ham in real time using a Multinomial Naive Bayes model.

---

<img width="1879" height="915" alt="image" src="https://github.com/user-attachments/assets/fe4b258e-ec4f-43a0-a045-99d09382acb6" />
<img width="1400" height="748" alt="image" src="https://github.com/user-attachments/assets/ba2e9ac1-dad0-49d0-b341-70855751042c" />
<img width="1857" height="657" alt="image" src="https://github.com/user-attachments/assets/6c6e586d-c95c-4edd-be93-fcf2c983ebb7" />




## Features

- Spam / Ham message classification
- Real-time prediction
- English and Arabic language support
- Dark and Light themes
- Interactive dashboard with visualizations
- Prediction confidence scores
- Analysis history tracking
- CSV export for prediction history
- Responsive and modern UI

---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Plotly
- NLP
- CountVectorizer
- Multinomial Naive Bayes

---

## Machine Learning Workflow

1. Data Cleaning
2. Removing Duplicate Messages
3. Train/Test Split
4. Text Vectorization using CountVectorizer
5. Model Training using Multinomial Naive Bayes
6. Model Evaluation
7. Streamlit Deployment

---

## Imbalanced Dataset

The dataset was not balanced, so different techniques were tested including:

- Oversampling
- Undersampling
- SMOTE
- Class Weight

However, the original dataset achieved better and more stable results.

---

## Model Performance

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Run The App

### Install Requirements

```bash
pip install -r requirements.txt
