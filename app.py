from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# === Cargar modelo y scaler ===
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener valores del formulario
        embarazos = float(request.form['embarazos'])
        glucosa = float(request.form['glucosa'])
        presion = float(request.form['presion'])
        piel = float(request.form['piel'])
        insulina = float(request.form['insulina'])
        imc = float(request.form['imc'])
        historial = float(request.form['historial'])
        edad = float(request.form['edad'])

        # Vector de entrada
        features = np.array([[embarazos, glucosa, presion, piel, insulina, imc, historial, edad]])

        # Escalado
        features_scaled = scaler.transform(features)

        # Predicción y probabilidad
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1] * 100

        if pred == 1:
            resultado = "RIESGO ALTO de diabetes"
            color = "danger"
        else:
            resultado = "RIESGO BAJO de diabetes"
            color = "success"

        return render_template('result.html', resultado=resultado, probabilidad=round(prob, 2), color=color)
    except Exception as e:
        return f"Ocurrió un error: {e}"


if __name__ == '__main__':
    app.run(debug=True)
