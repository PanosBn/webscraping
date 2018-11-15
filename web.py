import naive_svm
import pipeline_csv

from flask import Flask, render_template
app = Flask(__name__)

# @app.route("/")
# def hello():
#     x = "123"

@app.route("/")
def pipeline():
    headlines = pipeline_csv.prepare_data()
    headlines = pipeline_csv.find_polarity(headlines)
    models = pipeline_csv.create_models(headlines)
    #return str(models)

    pinakas = {"a": 1, "b": 2}
    return render_template("landingpage.html", models=models)

app.run()