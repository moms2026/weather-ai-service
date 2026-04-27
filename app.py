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
<title>STM32 + IA</title>

<style>
body {
    margin:0;
    font-family: 'Segoe UI';
    background:#0f172a;
    color:white;
}

.main {
    display:flex;
    gap:20px;
    padding:20px;
}

/* LEFT PANEL */
.left {
    width:30%;
    background:#1e293b;
    border-radius:20px;
    padding:20px;
}

.sensor {
    background:#0f172a;
    padding:15px;
    border-radius:12px;
    margin-bottom:15px;
}

.sensor h3 {
    margin:0;
    font-size:14px;
    color:#94a3b8;
}

.sensor p {
    font-size:22px;
    font-weight:bold;
}

/* RIGHT PANEL */
.right {
    width:70%;
    background:#1e293b;
    border-radius:20px;
    padding:20px;
}

.big {
    font-size:60px;
    font-weight:300;
}

.days {
    display:flex;
    justify-content:space-between;
    margin-top:20px;
}

.day {
    background:#0f172a;
    padding:15px;
    border-radius:12px;
    text-align:center;
    width:30%;
}
</style>
</head>

<body>

<div class="main">

<!-- LEFT -->
<div class="left">
<h2>📡 STM32</h2>

<div id="live"></div>

</div>

<!-- RIGHT -->
<div class="right">

<div id="weather"></div>

</div>

</div>

<script>

// ================= LIVE STM32 =================
async function loadLive(){
    const res = await fetch('/live');
    const d = await res.json();

    if(!d.ok){
        document.getElementById("live").innerHTML="Erreur capteurs";
        return;
    }

    document.getElementById("live").innerHTML = `
    <div class="sensor">
        <h3>Température</h3>
        <p>${d.temp} °C</p>
    </div>

    <div class="sensor">
        <h3>Pression</h3>
        <p>${d.pres} hPa</p>
    </div>

    <div class="sensor">
        <h3>Accélération</h3>
        <p>X:${d.acc_x} Y:${d.acc_y} Z:${d.acc_z}</p>
    </div>

    <div class="sensor">
        <h3>IA</h3>
        <p>${d.emoji} ${d.nom}</p>
        <small>Confiance ${Math.round(d.confiance*100)}%</small>
    </div>
    `;
}

// ================= WEATHER =================
async function loadWeather(){
    const res = await fetch('/forecast');
    const data = await res.json();

    const days = data.days;

    if(!days){
        document.getElementById("weather").innerHTML="Erreur météo";
        return;
    }

    const today = days[0];

    document.getElementById("weather").innerHTML = `
        <div>
            <div class="big">${today.temp_max}°</div>
            <div>${today.nom}</div>
        </div>

        <div class="days">
            ${days.map(d=>`
                <div class="day">
                    <div>${d.date}</div>
                    <div>${d.emoji}</div>
                    <div>${d.temp_max}° / ${d.temp_min}°</div>
                </div>
            `).join("")}
        </div>
    `;
}

// INIT
loadLive();
loadWeather();
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