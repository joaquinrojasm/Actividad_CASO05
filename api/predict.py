# api/predict.py
from typing import Dict, Any
import json
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- construir sistema difuso (idéntico a tu código original) ---
temperatura = ctrl.Antecedent(np.arange(0, 41, 1), 'temperatura')
ventilador = ctrl.Consequent(np.arange(0, 101, 1), 'ventilador')

temperatura['baja'] = fuzz.trimf(temperatura.universe, [0, 0, 20])
temperatura['media'] = fuzz.trimf(temperatura.universe, [10, 20, 30])
temperatura['alta'] = fuzz.trimf(temperatura.universe, [20, 40, 40])

ventilador['lento'] = fuzz.trimf(ventilador.universe, [0, 0, 50])
ventilador['medio'] = fuzz.trimf(ventilador.universe, [25, 50, 75])
ventilador['rápido'] = fuzz.trimf(ventilador.universe, [50, 100, 100])

rule1 = ctrl.Rule(temperatura['baja'], ventilador['lento'])
rule2 = ctrl.Rule(temperatura['media'], ventilador['medio'])
rule3 = ctrl.Rule(temperatura['alta'], ventilador['rápido'])

sistema = ctrl.ControlSystem([rule1, rule2, rule3])

# --- handler esperado por Vercel Python runtime ---
def handler(request):
    try:
        data = None
        
        try:
            data = request.json
        except Exception:
            try:
                data = json.loads(request.body.decode())
            except Exception:
                data = {}

        temp = float(data.get("temperatura", 0))
        sim = ctrl.ControlSystemSimulation(sistema)
        sim.input['temperatura'] = temp
        sim.compute()
        velocidad = float(sim.output['ventilador'])

        body = json.dumps({"velocidad": velocidad, "temperatura": temp})
        return {
            "statusCode": 200,
            "headers": {"content-type": "application/json"},
            "body": body
        }
    except Exception as e:
        err = {"error": str(e)}
        return {
            "statusCode": 500,
            "headers": {"content-type": "application/json"},
            "body": json.dumps(err)
        }