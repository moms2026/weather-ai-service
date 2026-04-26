from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

print("Chargement IA...")
model  = joblib.load('weather_model.pkl')
scaler = joblib.load('scaler.pkl')
print("IA prête !")

CLASSES = {0: 'Mauvais', 1: 'Moyen', 2: 'Bon'}
LEDS    = {0: 'Rouge 🔴', 1: 'Bleue 🔵', 2: 'Verte 🟢'}

# ── Page web intégrée ──────────────────────────────────────
PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Météo IA — STM32</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
    .card { background: #1e293b; border-radius: 16px; padding: 2.5rem; width: 100%; max-width: 440px; box-shadow: 0 25px 50px rgba(0,0,0,0.5); }
    h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.4rem; }
    .sub { color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }
    label { display: block; font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.4rem; margin-top: 1.2rem; }
    input[type=range] { width: 100%; accent-color: #6366f1; }
    .val { font-size: 1.4rem; font-weight: 700; color: #818cf8; margin-left: 0.5rem; }
    .row { display: flex; align-items: center; justify-content: space-between; }
    button { margin-top: 2rem; width: 100%; padding: 0.9rem; background: #6366f1; color: white; border: none; border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    button:hover { background: #4f46e5; }
    .result { margin-top: 1.5rem; padding: 1.2rem; border-radius: 10px; text-align: center; display: none; }
    .result.show { display: block; }
    .result.bad  { background: #450a0a; border: 1px solid #ef4444; }
    .result.mid  { background: #1e1b4b; border: 1px solid #6366f1; }
    .result.good { background: #052e16; border: 1px solid #22c55e; }
    .result .emoji { font-size: 3rem; }
    .result .label { font-size: 1.3rem; font-weight: 700; margin: 0.5rem 0; }
    .result .conf  { font-size: 0.85rem; color: #94a3b8; }
    .led-row { display: flex; gap: 0.6rem; justify-content: center; margin-top: 0.8rem; }
    .led { width: 18px; height: 18px; border-radius: 50%; opacity: 0.2; transition: opacity 0.3s, box-shadow 0.3s; }
    .led.on { opacity: 1; }
    .led.red  { background: #ef4444; box-shadow: 0 0 12px #ef4444; }
    .led.blue { background: #6366f1; box-shadow: 0 0 12px #6366f1; }
    .led.green{ background: #22c55e; box-shadow: 0 0 12px #22c55e; }
  </style>
</head>
<body>
<div class="card">
  <h1>🌤️ Météo IA</h1>
  <p class="sub">Simulation capteurs STM32 — NUCLEO-N657X0-Q</p>

  <label>🌡️ Température <span class="val" id="vTemp">20°C</span></label>
  <input type="range" id="temp" min="-10" max="45" value="20"
         oninput="document.getElementById('vTemp').textContent=this.value+'°C'">

  <label>🔵 Pression atmosphérique <span class="val" id="vPres">1013 hPa</span></label>
  <input type="range" id="pres" min="970" max="1050" value="1013"
         oninput="document.getElementById('vPres').textContent=this.value+' hPa'">

  <label>💨 Vent (estimé) <span class="val" id="vWind">10 km/h</span></label>
  <input type="range" id="wind" min="0" max="80" value="10"
         oninput="document.getElementById('vWind').textContent=this.value+' km/h'">

  <label>🌧️ Précipitations <span class="val" id="vRain">0 mm</span></label>
  <input type="range" id="rain" min="0" max="30" value="0"
         oninput="document.getElementById('vRain').textContent=this.value+' mm'">

  <button onclick="predict()">⚡ Prédire le temps</button>

  <div class="result" id="result">
    <div class="emoji" id="rEmoji"></div>
    <div class="label" id="rLabel"></div>
    <div class="conf"  id="rConf"></div>
    <div class="led-row">
      <div class="led red"   id="ledR"></div>
      <div class="led blue"  id="ledB"></div>
      <div class="led green" id="ledG"></div>
    </div>
  </div>
</div>

<script>
async function predict() {
  const btn = document.querySelector('button');
  btn.textContent = '⏳ Analyse...';
  btn.disabled = true;

  const body = {
    temp: document.getElementById('temp').value,
    pres: document.getElementById('pres').value,
    wind: document.getElementById('wind').value,
    rain: document.getElementById('rain').value
  };

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();

    const emojis = {0:'⛈️', 1:'🌥️', 2:'☀️'};
    const themes = {0:'bad', 1:'mid', 2:'good'};

    document.getElementById('rEmoji').textContent = emojis[data.prediction];
    document.getElementById('rLabel').textContent = data.nom + ' — LED ' + data.led;
    document.getElementById('rConf').textContent  = 'Confiance : ' + Math.round(data.confiance*100) + '%';

    const r = document.getElementById('result');
    r.className = 'result show ' + themes[data.prediction];

    document.getElementById('ledR').classList.toggle('on', data.prediction === 0);
    document.getElementById('ledB').classList.toggle('on', data.prediction === 1);
    document.getElementById('ledG').classList.toggle('on', data.prediction === 2);
  } catch(e) {
    alert('Erreur : ' + e.message);
  }
  btn.textContent = '⚡ Prédire le temps';
  btn.disabled = false;
}
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    temp = float(data.get('temp', 20))
    pres = float(data.get('pres', 1013))
    wind = float(data.get('wind', 0))
    rain = float(data.get('rain', 0))

    X  = np.array([[temp, pres, wind, rain]])
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[0]
    pred  = int(np.argmax(proba))

    return jsonify({
        'prediction': pred,
        'nom':        CLASSES[pred],
        'led':        LEDS[pred],
        'confiance':  round(float(max(proba)), 3)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)