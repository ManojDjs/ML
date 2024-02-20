# Scikit-learn model for Titanic Dataset

This model is created with Scikit-learn.

## Run the server  

```
python app.py
```

## API  

A RESTful API is used to simulate deployment of a model.  

#### Request

- In order to make a prediction, call a PUT request with a dictionary with all these attributes: ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']  
For an example,  
```json
{
	"PassengerId": [1987, 1123], 
	"Pclass": [3, 2], 
	"Name": ["Carey, Ms. Jenna", "Smith, Mr. Steven"], 
	"Sex": ["female", "male"], 
	"Age": [24, 23], 
	"SibSp": [0, 0], 
	"Parch": [0, 2], 
	"Ticket": ["", ""], 
	"Fare": [112.0, 67.0], 
	"Cabin": ["", ""], 
	"Embarked": ["S", "Q"]
}
```

#### Response

- A valid request will result in a dictionary including only one attribute, "prediction" and its value will be the prediction from the values you parsed with the PUT request. If the prediction is 1, the person survived and if it was 0, the person did not survive.  
This is an example response from the model.  
```json
{
    "predictions": {
        "1": 1,
        "2": 0
    }
}
```

#### Example  

Create a PUT request using __curl__:
```
curl -XPUT -H "Content-type: application/json" -d '{"PassengerId": 1987, "Pclass": 3, "Name": "Sharapova, Ms. Maria", "Sex": "female", "Age": 24, "SibSp": 0, "Parch": 0, "Ticket": "", "Fare": 112.0, "Cabin": "", "Embarked": "S"}' 'http://127.0.0.1:5000/predict'
```  

Then you'll recieve something like the following:  
```json
{
    "predictions": {
        "1": 1
    }
}
```

## Files

1. *create_model.py*: Python script that is responisble for transforming the training dataset (using a data transformation pipeline), training the model, saving both pipeline and model as binary files.
2. *app.py*: Python script which contains the Flask server. A basic deployment of the model.
3. *model.pkl*: Binary file containing the trained SVM model.
4. *pipeline.pkl*: Binary file containing data transformation pipeline