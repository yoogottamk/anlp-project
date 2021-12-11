from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__)

@app.route("/health")
def health():
    return "Ok"

@app.route("/")
def translator():
    return render_template("index.html")

@app.route("/translate", methods=["GET"])
def translation_page():
    return render_template("translate.html")

@app.route("/translate", methods=["POST"])
def get_translation():
    sentence = request.form["sentence"]
    # translate sentence

    return redirect(url_for("translate.html"))

if __name__ == "__main__":
    app.run(port=8000, debug=True)
