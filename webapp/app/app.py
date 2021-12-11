from flask import Flask, render_template, request, url_for, redirect
import json
import subprocess

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

    result = subprocess.run(f"anlp_project inference --checkpoint checkpoint-340500 --sentence \"{sentence}\"", shell=True, capture_output=True).stdout.decode()

    sentence = result.strip().split('\n')[-1]

    return sentence

if __name__ == "__main__":
    app.run(port=8000, debug=True)
