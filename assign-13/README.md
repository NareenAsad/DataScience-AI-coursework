# Assignment 13 – Model Deployment using Flask

## Course
Applied Data Science with AI  
BS Software Engineering – 7th Semester

## Project Title
House Price Prediction

---

## Objective
The objective of this assignment is to deploy a trained machine learning model using a Flask-based REST API. This deployment enables real-time house price predictions through HTTP requests, completing an end-to-end machine learning pipeline.

---

## Tools & Technologies
- Python
- Flask
- Scikit-learn
- NumPy
- Joblib
- Postman (for API testing)

---

## Files in this Folder
- `app.py` – Flask application for model deployment  
- `house_price_model.pkl` – Trained machine learning model  
- `train_cleaned.csv` – Cleaned dataset used for training  

---

## API Endpoints

### Home Endpoint
**URL:** `/`  
**Method:** `GET`  

Returns a confirmation message indicating that the API is running.

---

### Prediction Endpoint
**URL:** `/predict`  
**Method:** `POST`  

Accepts house features in JSON format and returns the predicted sale price.

#### Sample Request Body
```json
{
  "GrLivArea": 1800,
  "OverallQual": 7,
  "GarageCars": 2
}
````

#### Sample Response

```json
{
  "Predicted_SalePrice": 235000.47
}
```

---

## How to Run the Application

1. Navigate to the assignment folder:

```bash
cd assign-13
```

2. Start the Flask server:

```bash
python app.py
```

3. Open a browser and visit:

```
http://127.0.0.1:5000/
```

---

## API Testing

The `/predict` endpoint was tested using **Postman** by sending a POST request with JSON input. The API successfully returned house price predictions, confirming correct deployment and model integration.

---

## Learning Outcome

Through this assignment, I learned how to:

* Deploy a machine learning model using Flask
* Create RESTful API endpoints
* Handle JSON input and output
* Test APIs using Postman
* Build an end-to-end machine learning workflow

---

## Project Milestone

✅ End-to-end House Price Prediction pipeline successfully deployed and tested on localhost.
