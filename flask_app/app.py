import os
import pandas as pd
from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc

#  ← Aquí configuras el tracking URI
mlflow.set_tracking_uri("http://localhost:9090")

# Resuelve la ruta a best_run_id.txt...
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
run_id_path = os.path.join(ROOT_DIR, "best_run_id.txt")
with open(run_id_path, "r") as f:
    run_id = f.read().strip()

model_uri = f"runs:/{run_id}/model_rf_100"
model = mlflow.pyfunc.load_model(model_uri)

app = Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df   = pd.DataFrame(data)
    preds = model.predict(df)
    return jsonify({"predictions": preds.tolist()})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)

