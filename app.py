from fileinput import filename
from flask import Flask, render_template, request, jsonify
from predict import get_prediction, load_models
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            #filename = file.filename
            filename = 'input.csv'
            save_location = os.path.join('Input', filename)
            file.save(save_location)
            output = get_prediction(save_location)
            return jsonify({'output': output})
        else:
            return jsonify({'output': "Unable to read the file. Please try again."})

if __name__ == '__main__':
    load_models()
    app.run(debug=False)