from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import joblib
import urllib.request
import json
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

print("Chargement IA...")
model  = joblib.load('weather_model.pkl')
scaler = joblib.load('scaler.pkl')
print("IA prête !")

CLASSES = {0: 'Mauvais', 1: 'Moyen', 2: 'Bon'}
LEDS    = {0: 'Rouge 🔴', 1: 'Bleue 🔵', 2: 'Verte 🟢'}
EMOJIS  = {0: '⛈️', 1: '🌥️', 2: '☀️'}

def get_forecast(lat=48.8566, lon=2.3522):
    """Récupère les vraies prévisions Open-Meteo pour aujourd'hui + 2 jours"""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"surface_pressure_mean,wind_speed_10m_max,precipitation_sum"
        f"&timezone=Europe%2FParis&forecast_days=3"
    )
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        daily = data['daily']
        results = []
        for i in range(3):
            tavg = (daily['temperature_2m_max'][i] + daily['temperature_2m_min'][i]) / 2
            pres = daily['surface_pressure_mean'][i]
            wspd = daily['wind_speed_10m_max'][i]
            prcp = daily['precipitation_sum'][i]
            X  = np.array([[tavg, pres, wspd, prcp]])
            Xs = scaler.transform(X)
            proba = model.predict_proba(Xs)[0]
            pred  = int(np.argmax(proba))
            results.append({
                'date':       daily['time'][i],
                'temp_max':   round(daily['temperature_2m_max'][i], 1),
                'temp_min':   round(daily['temperature_2m_min'][i], 1),
                'pres':       round(pres, 1),
                'wspd':       round(wspd, 1),
                'prcp':       round(prcp, 1),
                'prediction': pred,
                'nom':        CLASSES[pred],
                'led':        LEDS[pred],
                'emoji':      EMOJIS[pred],
                'confiance':  round(float(max(proba)), 3)
            })
        return results
    except Exception as e:
        return None

PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Météo IA — STM32</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 2rem 1rem; }
    .container { max-width: 500px; margin: 0 auto; }
    h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
    .sub { color: #94a3b8; font-size: 0.85rem; margin-bottom: 2rem; }

    /* Carte simulation manuelle */
    .card { background: #1e293b; border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem; box-shadow: 0 25px 50px rgba(0,0,0,0.5); }
    .card h2 { font-size: 1rem; color: #94a3b8; margin-bottom: 1.2rem; text-transform: uppercase; letter-spacing: 0.05em; }
    label { display: block; font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.4rem; margin-top: 1rem; }
    input[type=range] { width: 100%; accent-color: #6366f1; }
    .val { font-size: 1.2rem; font-weight: 700; color: #818cf8; margin-left: 0.5rem; }
    button { margin-top: 1.5rem; width: 100%; padding: 0.9rem; background: #6366f1; color: white; border: none; border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    button:hover { background: #4f46e5; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }

    /* Résultat simulation */
    .result { margin-top: 1.2rem; padding: 1.2rem; border-radius: 10px; text-align: center; display: none; }
    .result.show { display: block; }
    .result.bad  { background: #450a0a; border: 1px solid #ef4444; }
    .result.mid  { background: #1e1b4b; border: 1px solid #6366f1; }
    .result.good { background: #052e16; border: 1px solid #22c55e; }
    .result .emoji { font-size: 2.5rem; }
    .result .rlabel { font-size: 1.2rem; font-weight: 700; margin: 0.4rem 0; }
    .result .conf   { font-size: 0.85rem; color: #94a3b8; }
    .led-row { display: flex; gap: 0.6rem; justify-content: center; margin-top: 0.8rem; }
    .led { width: 16px; height: 16px; border-radius: 50%; opacity: 0.2; transition: all 0.3s; }
    .led.on { opacity: 1; }
    .led.red   { background: #ef4444; box-shadow: 0 0 12px #ef4444; }
    .led.blue  { background: #6366f1; box-shadow: 0 0 12px #6366f1; }
    .led.green { background: #22c55e; box-shadow: 0 0 12px #22c55e; }

    /* Prévisions */
    .forecast-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; }
    .fcard { background: #1e293b; border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid #334155; }
    .fcard.today { border-color: #6366f1; }
    .fcard .fday  { font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.4rem; text-transform: uppercase; }
    .fcard .femoji { font-size: 2rem; margin: 0.4rem 0; }
    .fcard .fnom  { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.3rem; }
    .fcard .ftemp { font-size: 0.8rem; color: #94a3b8; }
    .fcard .fconf { font-size: 0.7rem; color: #475569; margin-top: 0.3rem; }
    .fcard .fwind { font-size: 0.7rem; color: #64748b; }
    .bad-fc  { border-color: #ef444488 !important; }
    .mid-fc  { border-color: #6366f188 !important; }
    .good-fc { border-color: #22c55e88 !important; }

    .loading { text-align: center; color: #64748b; padding: 2rem; font-size: 0.9rem; }
    .error-msg { color: #ef4444; text-align: center; font-size: 0.85rem; padding: 1rem; }
    .refresh-btn { background: transparent; border: 1px solid #334155; color: #94a3b8; font-size: 0.8rem; padding: 0.4rem 0.8rem; border-radius: 6px; cursor: pointer; margin-top: 0.8rem; width: auto; }
    .refresh-btn:hover { border-color: #6366f1; color: #818cf8; }
    .forecast-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
  </style>
</head>
<body>
<div class="container">
  <h1>🌤️ Météo IA</h1>
  <p class="sub">Simulation capteurs STM32 — NUCLEO-N657X0-Q</p>

  <!-- PRÉVISIONS RÉELLES -->
  <div class="card">
    <div class="forecast-header">
      <h2>📡 Prévisions réelles (Paris)</h2>
      <button class="refresh-btn" onclick="loadForecast()">🔄 Actualiser</button>
    </div>
    <div id="forecast"><div class="loading">⏳ Chargement des prévisions...</div></div>
  </div>

  <!-- SIMULATION MANUELLE -->
  <div class="card">
    <h2>🎛️ Simulation manuelle capteurs</h2>

    <label>🌡️ Température <span class="val" id="vTemp">20°C</span></label>
    <input type="range" id="temp" min="-10" max="45" value="20"
           oninput="document.getElementById('vTemp').textContent=this.value+'°C'">

    <label>🔵 Pression <span class="val" id="vPres">1013 hPa</span></label>
    <input type="range" id="pres" min="970" max="1050" value="1013"
           oninput="document.getElementById('vPres').textContent=this.value+' hPa'">

    <label>💨 Vent <span class="val" id="vWind">10 km/h</span></label>
    <input type="range" id="wind" min="0" max="80" value="10"
           oninput="document.getElementById('vWind').textContent=this.value+' km/h'">

    <label>🌧️ Précipitations <span class="val" id="vRain">0 mm</span></label>
    <input type="range" id="rain" min="0" max="30" value="0"
           oninput="document.getElementById('vRain').textContent=this.value+' mm'">

    <button onclick="predict()">⚡ Prédire le temps</button>

    <div class="result" id="result">
      <div class="emoji" id="rEmoji"></div>
      <div class="rlabel" id="rLabel"></div>
      <div class="conf"   id="rConf"></div>
      <div class="led-row">
        <div class="led red"   id="ledR"></div>
        <div class="led blue"  id="ledB"></div>
        <div class="led green" id="ledG"></div>
      </div>
    </div>
  </div>
</div>

<script>
// ── Prévisions réelles ──────────────────────────────────────
async function loadForecast() {
  document.getElementById('forecast').innerHTML =
    '<div class="loading">⏳ Chargement des prévisions...</div>';
  try {
    const res  = await fetch('/forecast');
    const days = await res.json();
    if (days.error) throw new Error(days.error);

    const labels = ['Aujourd\\'hui', 'Demain', 'Après-demain'];
    const themes = {0:'bad-fc', 1:'mid-fc', 2:'good-fc'};

    let html = '<div class="forecast-grid">';
    days.forEach((d, i) => {
      const todayClass = i === 0 ? 'today' : '';
      html += `
        <div class="fcard ${todayClass} ${themes[d.prediction]}">
          <div class="fday">${labels[i]}</div>
          <div class="femoji">${d.emoji}</div>
          <div class="fnom">${d.nom}</div>
          <div class="ftemp">🌡️ ${d.temp_min}° / ${d.temp_max}°C</div>
          <div class="fwind">💨 ${d.wspd} km/h · 🌧️ ${d.prcp}mm</div>
          <div class="fconf">Confiance ${Math.round(d.confiance*100)}%</div>
        </div>`;
    });
    html += '</div>';
    document.getElementById('forecast').innerHTML = html;
  } catch(e) {
    document.getElementById('forecast').innerHTML =
      `<div class="error-msg">❌ Impossible de charger les prévisions<br><small>${e.message}</small></div>`;
  }
}

// ── Simulation manuelle ─────────────────────────────────────
async function predict() {
  const btn = document.querySelector('.card:last-child button');
  btn.textContent = '⏳ Analyse...';
  btn.disabled = true;
  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        temp: document.getElementById('temp').value,
        pres: document.getElementById('pres').value,
        wind: document.getElementById('wind').value,
        rain: document.getElementById('rain').value
      })
    });
    const data = await res.json();
    const themes = {0:'bad', 1:'mid', 2:'good'};
    document.getElementById('rEmoji').textContent = data.emoji;
    document.getElementById('rLabel').textContent = data.nom + ' — ' + data.led;
    document.getElementById('rConf').textContent  = 'Confiance : ' + Math.round(data.confiance*100) + '%';
    const r = document.getElementById('result');
    r.className = 'result show ' + themes[data.prediction];
    document.getElementById('ledR').classList.toggle('on', data.prediction === 0);
    document.getElementById('ledB').classList.toggle('on', data.prediction === 1);
    document.getElementById('ledG').classList.toggle('on', data.prediction === 2);
  } catch(e) { alert('Erreur : ' + e.message); }
  btn.textContent = '⚡ Prédire le temps';
  btn.disabled = false;
}

// Charger les prévisions au démarrage
loadForecast();
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(PAGE)

@app.route('/forecast')
def forecast():
    data = get_forecast()
    if data is None:
        return jsonify({'error': 'API météo indisponible'}), 503
    return jsonify(data)

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
        'emoji':      EMOJIS[pred],
        'confiance':  round(float(max(proba)), 3)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)