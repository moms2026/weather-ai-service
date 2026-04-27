from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import joblib
import urllib.request
import json

app = Flask(__name__)
CORS(app)

print("Chargement IA...")
model  = joblib.load('weather_model.pkl')
scaler = joblib.load('scaler.pkl')
print("IA prête !")

CLASSES = {0: 'Mauvais', 1: 'Moyen', 2: 'Bon'}
LEDS    = {0: 'Rouge 🔴', 1: 'Bleue 🔵', 2: 'Verte 🟢'}
EMOJIS  = {0: '⛈️', 1: '🌥️', 2: '☀️'}

LAT  = 45.6442
LON  = 5.8728
CITY = "Le Bourget-du-Lac, FR"

TS_CHANNEL_ID = "3349807"
TS_API_KEY    = "0HUN3VWRIKPZZNRA"

# ───────────── ThingSpeak ─────────────
def get_thingspeak():
    url = f"https://api.thingspeak.com/channels/{TS_CHANNEL_ID}/feeds/last.json?api_key={TS_API_KEY}"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())

        temp  = float(data.get('field1') or 0)
        pres  = float(data.get('field2') or 1013)
        acc_x = float(data.get('field3') or 0)
        acc_y = float(data.get('field4') or 0)
        acc_z = float(data.get('field5') or 0)
        acc_mag = round((acc_x**2 + acc_y**2 + acc_z**2)**0.5, 2)

        X  = np.array([[temp, pres, 0.0, 0.0]])
        Xs = scaler.transform(X)
        proba = model.predict_proba(Xs)[0]
        pred  = int(np.argmax(proba))

        return {
            'ok': True,
            'timestamp': data.get('created_at', ''),
            'temp': round(temp, 2),
            'pres': round(pres, 2),
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z,
            'acc_mag': acc_mag,
            'prediction': pred,
            'nom': CLASSES[pred],
            'led': LEDS[pred],
            'emoji': EMOJIS[pred],
            'confiance': round(float(max(proba)), 3)
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}

# ───────────── Forecast ─────────────
def get_forecast(lat, lon):
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
            pres = daily['surface_pressure_mean'][i] or 1013
            wspd = daily['wind_speed_10m_max'][i] or 0
            prcp = daily['precipitation_sum'][i] or 0

            X  = np.array([[tavg, pres, wspd, prcp]])
            Xs = scaler.transform(X)
            proba = model.predict_proba(Xs)[0]
            pred  = int(np.argmax(proba))

            results.append({
                'date': daily['time'][i],
                'temp_max': round(daily['temperature_2m_max'][i], 1),
                'temp_min': round(daily['temperature_2m_min'][i], 1),
                'pres': round(pres, 1),
                'wspd': round(wspd, 1),
                'prcp': round(prcp, 1),
                'prediction': pred,
                'nom': CLASSES[pred],
                'led': LEDS[pred],
                'emoji': EMOJIS[pred],
                'confiance': round(float(max(proba)), 3)
            })

        return results
    except:
        return None

# ───────────── PAGE HTML ─────────────
PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Météo IA STM32</title>

<style>
body {
  font-family: 'Segoe UI';
  background: #0f172a;
  color: white;
  margin: 0;
}

.container {
  max-width: 1200px;
  margin: auto;
  padding: 20px;
}

.main {
  display: grid;
  grid-template-columns: 1fr 1.4fr;
  gap: 20px;
}

.card {
  background: #1e293b;
  padding: 20px;
  border-radius: 12px;
}

h1 {
  margin-bottom: 10px;
}

.live-box {
  text-align: center;
}

.led {
  font-size: 20px;
}

.accel {
  display: flex;
  justify-content: space-around;
  margin-top: 10px;
}

</style>
</head>

<body>

<div class="container">
<h1>🌤️ Dashboard STM32 + IA</h1>

<div class="main">

<!-- GAUCHE -->
<div class="card">
<h2>📡 Capteurs STM32</h2>
<div id="live">Chargement...</div>
</div>

<!-- DROITE -->
<div class="card">
<h2>🌦️ Météo + IA</h2>
<div id="forecast">Chargement...</div>
</div>

</div>
</div>

<script>

async function loadLive() {
  const res = await fetch('/live');
  const d = await res.json();

  if (!d.ok) {
    document.getElementById('live').innerHTML = "Erreur capteurs";
    return;
  }

  document.getElementById('live').innerHTML = `
    <div class="live-box">
      <h3>${d.emoji} ${d.nom}</h3>
      <p>Température: ${d.temp} °C</p>
      <p>Pression: ${d.pres} hPa</p>

      <div class="accel">
        <div>X: ${d.acc_x}</div>
        <div>Y: ${d.acc_y}</div>
        <div>Z: ${d.acc_z}</div>
      </div>

      <p>Magnitude: ${d.acc_mag}</p>
      <p>Confiance: ${Math.round(d.confiance*100)}%</p>
      <div class="led">${d.led}</div>
    </div>
  `;
}

async function loadForecast() {
  const res = await fetch('/forecast');
  const data = await res.json();

  let html = "";
  data.days.forEach(d => {
    html += `
      <div style="margin-bottom:10px">
        <b>${d.date}</b> — ${d.emoji} ${d.nom}<br>
        ${d.temp_min}°C / ${d.temp_max}°C<br>
        Vent: ${d.wspd} km/h — Pluie: ${d.prcp} mm
      </div>
    `;
  });

  document.getElementById('forecast').innerHTML = html;
}

loadLive();
loadForecast();
setInterval(loadLive, 15000);

</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(PAGE)

@app.route('/live')
def live():
    return jsonify(get_thingspeak())

@app.route('/forecast')
def forecast():
    data = get_forecast(LAT, LON)
    return jsonify({'days': data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)