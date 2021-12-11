from flask import Flask, render_template, request, url_for, redirect
import json

app = Flask(__name__)

@app.route("/health")
def health():
    return "Ok"

@app.route("/")
def translator():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def get_translation():
    data = json.loads(request.data.decode())
    sentence = data["sentence"]
    # translate sentence

    return sentence

if __name__ == "__main__":
    app.run(port=8000, debug=True)
