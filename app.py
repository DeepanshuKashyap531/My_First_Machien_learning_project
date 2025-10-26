from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load your ML model
with open("CPP.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    predicted_price = None
    
    if request.method == "POST":
        # Get form values
        make = request.form["make"]
        colour = request.form["colour"]
        doors = float(request.form["doors"])
        odometer = float(request.form["odometer"])
        
        # Create a DataFrame for prediction
        data = pd.DataFrame({
            "Make": [make],
            "Colour": [colour],
            "Doors": [doors],
            "Odometer (KM)": [odometer]
        })
        
        # Make prediction
        predicted_price = model.predict(data)[0]
    
    return render_template("index.html", predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
