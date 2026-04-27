from flask import Flask, request, jsonify, render_template_string
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

# ── Position fixe ───────────────────────────────────────────
LAT  = 45.6442
LON  = 5.8728
CITY = "Le Bourget-du-Lac, FR"

# ── ThingSpeak ──────────────────────────────────────────────
TS_CHANNEL_ID = "3349807"
TS_API_KEY    = "0HUN3VWRIKPZZNRA"

def get_thingspeak():
    url = (
        f"https://api.thingspeak.com/channels/{TS_CHANNEL_ID}"
        f"/feeds/last.json?api_key={TS_API_KEY}"
    )
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
            'ok':         True,
            'timestamp':  data.get('created_at', ''),
            'temp':       round(temp, 2),
            'pres':       round(pres, 2),
            'acc_x':      round(acc_x, 2),
            'acc_y':      round(acc_y, 2),
            'acc_z':      round(acc_z, 2),
            'acc_mag':    acc_mag,
            'prediction': pred,
            'nom':        CLASSES[pred],
            'led':        LEDS[pred],
            'emoji':      EMOJIS[pred],
            'confiance':  round(float(max(proba)), 3)
        }
    except Exception as e:
        print(f"Erreur ThingSpeak : {e}")
        return {'ok': False, 'error': str(e)}

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
            pres = daily['surface_pressure_mean'][i] or 1013.0
            wspd = daily['wind_speed_10m_max'][i]    or 0.0
            prcp = daily['precipitation_sum'][i]     or 0.0
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
        print(f"Erreur forecast : {e}")
        return None

PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Dashboard STM32 + IA</title>

<style>

body {
  margin:0;
  font-family: 'Segoe UI';
  background: linear-gradient(135deg,#0f172a,#020617);
  color:white;
}

.main {
  display:grid;
  grid-template-columns: 350px 1fr;
  gap:20px;
  padding:20px;
}

/* LEFT */
.left {
  background:#1e293b;
  border-radius:20px;
  padding:20px;
}

/* RIGHT */
.right {
  background:#1e293b;
  border-radius:20px;
  padding:25px;
}

/* STM32 */
.sensor {
  background:#0f172a;
  padding:15px;
  border-radius:12px;
  margin-bottom:15px;
}

.big {
  font-size:28px;
  font-weight:bold;
}

/* WEATHER */
.top {
  display:flex;
  justify-content:space-between;
}

.temp {
  font-size:70px;
  font-weight:300;
}

.desc {
  color:#94a3b8;
}

.days {
  display:flex;
  margin-top:30px;
  gap:10px;
}

.day {
  flex:1;
  background:#0f172a;
  padding:10px;
  border-radius:12px;
  text-align:center;
}

.day:hover {
  background:#1e293b;
}

.chart {
  margin-top:20px;
}

</style>
</head>

<body>

<div class="main">

<!-- STM32 -->
<div class="left">
<h2>📡 STM32</h2>

<div id="live"></div>
</div>

<!-- METEO -->
<div class="right">

<div class="top">
  <div>
    <div class="temp" id="temp">--°</div>
    <div class="desc" id="desc">Chargement...</div>
  </div>
  <div id="emoji" style="font-size:60px;"></div>
</div>

<div class="chart">
<canvas id="chart" height="80"></canvas>
</div>

<div class="days" id="days"></div>

</div>

</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<script>

// STM32
async function loadLive(){
  let r = await fetch('/live');
  let d = await r.json();

  if(!d.ok){
    document.getElementById('live').innerHTML="Erreur STM32";
    return;
  }

  document.getElementById('live').innerHTML = `
    <div class="sensor">
      Température<br>
      <div class="big">${d.temp} °C</div>
    </div>

    <div class="sensor">
      Pression<br>
      <div class="big">${d.pres} hPa</div>
    </div>

    <div class="sensor" style="text-align:center">
      <div style="font-size:30px">${d.emoji}</div>
      <b>${d.nom}</b><br>
      Confiance ${d.conf}%<br>
      ${d.led}
    </div>
  `;
}

// WEATHER
async function loadForecast(){
  let r = await fetch('/forecast');
  let d = await r.json();

  let temps = [];
  let labels = [];

  let html="";

  d.days.forEach(day=>{
    temps.push(day.max);
    labels.push(day.date.slice(5));

    html += `
      <div class="day">
        ${day.date.slice(5)}<br>
        ☀️<br>
        ${day.max}°<br>
        <span style="color:#94a3b8">${day.min}°</span>
      </div>
    `;
  });

  document.getElementById('days').innerHTML = html;
  document.getElementById('temp').innerHTML = temps[0]+"°";
  document.getElementById('desc').innerHTML = "Prévision IA";
  document.getElementById('emoji').innerHTML = "☀️";

  new Chart(document.getElementById('chart'), {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        data: temps,
        borderColor: '#facc15',
        backgroundColor: 'rgba(250,204,21,0.2)',
        fill:true,
        tension:0.4
      }]
    },
    options:{
      plugins:{legend:{display:false}},
      scales:{
        x:{display:false},
        y:{display:false}
      }
    }
  });
}

loadLive();
loadForecast();
setInterval(loadLive,15000);

</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(PAGE)

@app.route('/live')
def live():
    data = get_thingspeak()
    response = jsonify(data)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    return response

@app.route('/forecast')
def forecast():
    data = get_forecast(LAT, LON)
    if data is None:
        return jsonify({'error': 'API météo indisponible'}), 503
    response = jsonify({'city': CITY, 'days': data})
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response

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