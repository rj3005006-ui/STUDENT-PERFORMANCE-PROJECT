# Student Performance Prediction System

## Project Overview
The Student Performance Prediction System is a machine learning web application that predicts a student's academic performance category based on study time, past class failures, absences, and previous-period grades.

This project was developed as part of the Week 4 AI/ML Virtual Internship Capstone.

## Objectives
- Build an end-to-end machine learning application.
- Preprocess and analyze a student performance dataset.
- Train and evaluate a Random Forest classification model.
- Save the trained model using Joblib.
- Develop a web interface using Flask.
- Accept student inputs and display predictions.

## Technologies Used
- Python 3.10
- Pandas
- Scikit-learn
- Flask
- Joblib
- HTML and CSS

## Dataset
Source: UCI Machine Learning Repository — Student Performance Dataset

https://archive.ics.uci.edu/dataset/320/student%2Bperformance

Dataset file used: `student-mat.csv`

## Machine Learning Workflow
1. Load the student dataset.
2. Select the input features.
3. Create three performance categories from the final grade.
4. Split the data into training and testing sets.
5. Train a Random Forest classifier.
6. Evaluate the model using accuracy, precision, recall, and F1-score.
7. Save the trained model using Joblib.
8. Load the model into a Flask application.
9. Display predictions through an HTML form.

## Model Results
Test accuracy: 86.08%

Evaluation metrics include precision, recall, F1-score, and classification accuracy.

## Application Features
- Student information input form
- Performance category prediction
- Browser-based interface
- Saved machine learning model
- Flask-powered prediction application

## Project Files
- `train_model.py` — trains and evaluates the model
- `app.py` — Flask application
- `student_model.joblib` — saved model and feature information
- `student-mat.csv` — dataset used for training
- `requirements.txt` — Python dependencies
- `templates/index.html` — web interface

## Limitations
The predictions are estimates based on a small educational dataset. The model uses previous-period grades, which are strongly associated with final grades. The application is a learning demonstration and should not be used to make important academic decisions.

## Conclusion
This project demonstrates an end-to-end machine learning workflow, including data handling, model training, evaluation, serialization, and web application integration using Flask.
