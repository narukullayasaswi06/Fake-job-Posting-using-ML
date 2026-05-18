from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
CORS(app)

# Sample dataset
data = {
    "text": [
        "urgent hiring no experience high salary",
        "pay registration fee to apply",
        "software engineer job with good salary",
        "data analyst position no fee required",
        "earn money quickly pay fee"
    ],
    "label": [1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

model = MultinomialNB()
model.fit(X, df["label"])

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.json["text"]
    transformed = vectorizer.transform([user_input])
    prediction = model.predict(transformed)[0]

    if prediction == 1:
        return jsonify({"result": "FAKE"})
    else:
        return jsonify({"result": "SAFE"})

if __name__ == "__main__":
    app.run(debug=True)