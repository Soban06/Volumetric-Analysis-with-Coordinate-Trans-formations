from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import math
import joblib

from flask import Flask, request, render_template

app = Flask(__name__)

# Load models
coord_model = joblib.load("coord_system_model.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence_scores = {}

    if request.method == "POST":
        question = request.form["question"]
        prediction = coord_model.predict([question])[0]
        probs = coord_model.predict_proba([question])[0]
        confidence_scores = dict(zip(coord_model.classes_, probs))

    return render_template("index.html",
                           prediction=prediction,
                           confidence_scores=confidence_scores)


def load_dataset(file_path="volume_questions.json"):
    with open(file_path, "r") as file:
        return json.load(file)


def display_dataset(dataset):
    print("📚 Loaded Examples:\n")
    for idx, example in enumerate(dataset, 1):
        print(f"Example {idx}:")
        print(f"Question: {example['question_text']}")
        print(f"Coordinate System: {example['coordinate_system']}")
        print("Integral Setup:")
        for var, bound in example['integral_setup']['bounds'].items():
            print(f"  {var}: {bound}")
        print(f"Integrand: {example['integral_setup']['integrand']}")
        print(f"Volume: {example['volume']}")
        print("-" * 40)


def validate_dataset(path="volume_questions.json"):
    with open(path, "r") as file:
        data = json.load(file)

    valid_coords = {"rectangular", "cylindrical", "spherical"}
    errors = []

    for i, entry in enumerate(data, 1):
        if entry["coordinate_system"] not in valid_coords:
            errors.append(
                f"❌ Entry {i} has invalid coordinate system: {entry['coordinate_system']}"
            )
        if not entry["integral_setup"]["integrand"]:
            errors.append(f"⚠️ Entry {i} missing integrand.")
        # Add more checks as needed...

    if errors:
        print("🛑 Found issues in the dataset:\n")
        for e in errors:
            print(e)
    else:
        print("✅ Dataset is valid!")


def main():

    validate_dataset()
    dataset = load_dataset()
    # display_dataset(dataset)

    texts = [item["question_text"] for item in dataset]
    labels = [item["coordinate_system"] for item in dataset]

    # print(texts, labels)

    X_train, X_test, y_train, y_test = train_test_split(texts,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42)

    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(X_train, y_train)

    # Evaluate

    # predictions = model.predict(X_test)
    # print(classification_report(y_test, predictions))

    question = "Find the area of the region that lies inside the circle r = 2costheta and outside the circle r = 1."
    probs = model.predict_proba([question])[0]
    prediction = model.predict([question])[0]

    # Map class labels to their probabilities
    label_probs = dict(zip(model.classes_, probs))

    print(f"Predicted coordinate system: {prediction}")
    print("Confidence scores:")
    for label, prob in sorted(label_probs.items(),
                              key=lambda x: x[1],
                              reverse=True):
        print(f"  {label}: {prob:.4f}")

    joblib.dump(model, "coord_system_model.pkl")

    app.run(host="0.0.0.0", port=81)


if __name__ == "__main__":
    main()
