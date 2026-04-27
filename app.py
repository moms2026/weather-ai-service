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
    """Lit la dernière mesure des capteurs STM32 depuis ThingSpeak"""
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

        # Magnitude totale de l'accélération
        acc_mag = round((acc_x**2 + acc_y**2 + acc_z**2)**0.5, 2)

        # Prédiction IA (temp + pres + vent=0 + pluie=0)
        # L'accélération ne rentre pas dans le modèle météo
        X  = np.array([[temp, pres, 0.0, 0.0]])
        Xs = scaler.transform(X)
        proba = model.predict_proba(Xs)[0]
        pred  = int(np.argmax(proba))

        return {
            'ok':        True,
            'timestamp': data.get('created_at', ''),
            'temp':      round(temp, 2),
            'pres':      round(pres, 2),
            'acc_x':     round(acc_x, 2),
            'acc_y':     round(acc_y, 2),
            'acc_z':     round(acc_z, 2),
            'acc_mag':   acc_mag,
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
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Météo IA — STM32</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 2rem 1rem; }
    .container { max-width: 520px; margin: 0 auto; }
    h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
    .sub { color: #94a3b8; font-size: 0.85rem; margin-bottom: 2rem; }
    .card { background: #1e293b; border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem; box-shadow: 0 25px 50px rgba(0,0,0,0.4); }
    .card-title { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 1rem; }

    /* ── Capteurs live ── */
    .live-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem; }
    .live-title  { font-size: 1rem; font-weight: 700; }
    .live-status { font-size: 0.72rem; color: #475569; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #22c55e; margin-right: 5px; animation: pulse 2s infinite; }
    .dot.offline { background: #ef4444; animation: none; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

    .sensors-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.8rem; margin-bottom: 1.2rem; }
    .sensor-box { background: #0f172a; border-radius: 10px; padding: 0.9rem; border: 1px solid #334155; }
    .sensor-box.accent { border-color: #6366f166; }
    .s-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; margin-bottom: 0.3rem; }
    .s-value { font-size: 1.3rem; font-weight: 700; color: #818cf8; }
    .s-unit  { font-size: 0.75rem; color: #475569; margin-left: 0.2rem; }

    .accel-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin-bottom: 1.2rem; }
    .accel-box { background: #0f172a; border-radius: 8px; padding: 0.7rem; border: 1px solid #1e293b; text-align: center; }
    .a-label { font-size: 0.65rem; color: #64748b; }
    .a-value { font-size: 1rem; font-weight: 700; color: #94a3b8; }

    .live-result { border-radius: 10px; padding: 1rem; text-align: center; border: 1px solid #334155; }
    .live-result.bad  { background: #450a0a; border-color: #ef4444; }
    .live-result.mid  { background: #1e1b4b; border-color: #6366f1; }
    .live-result.good { background: #052e16; border-color: #22c55e; }
    .lr-emoji { font-size: 2rem; }
    .lr-label { font-size: 1rem; font-weight: 700; margin: 0.3rem 0; }
    .lr-conf  { font-size: 0.78rem; color: #94a3b8; }
    .led-row { display: flex; gap: 0.6rem; justify-content: center; margin-top: 0.7rem; }
    .led { width: 14px; height: 14px; border-radius: 50%; opacity: 0.15; transition: all 0.3s; }
    .led.on { opacity: 1; }
    .led.red   { background: #ef4444; box-shadow: 0 0 12px #ef4444; }
    .led.blue  { background: #6366f1; box-shadow: 0 0 12px #6366f1; }
    .led.green { background: #22c55e; box-shadow: 0 0 12px #22c55e; }

    .refresh-btn { background: transparent; border: 1px solid #334155; color: #94a3b8; font-size: 0.75rem; padding: 0.35rem 0.75rem; border-radius: 6px; cursor: pointer; transition: all 0.2s; }
    .refresh-btn:hover { border-color: #6366f1; color: #818cf8; }
    .ts-time { font-size: 0.7rem; color: #475569; text-align: right; margin-top: 0.6rem; }

    /* ── Prévisions ── */
    .forecast-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.2rem; }
    .city-name { font-size: 1.05rem; font-weight: 700; }
    .city-sub  { font-size: 0.7rem; color: #475569; margin-top: 0.15rem; }
    .forecast-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; }
    .fcard { background: #0f172a; border-radius: 10px; padding: 0.9rem 0.6rem; text-align: center; border: 1px solid #334155; transition: transform 0.2s; }
    .fcard:hover { transform: translateY(-2px); }
    .fcard.today { border-color: #6366f1; }
    .fday  { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; }
    .fdate { font-size: 0.62rem; color: #475569; margin-bottom: 0.3rem; }
    .femoji{ font-size: 1.8rem; margin: 0.3rem 0; }
    .fnom  { font-size: 0.8rem; font-weight: 700; margin-bottom: 0.25rem; }
    .ftemp { font-size: 0.72rem; color: #94a3b8; }
    .fwind { font-size: 0.65rem; color: #64748b; margin-top: 0.2rem; }
    .fconf { font-size: 0.62rem; color: #475569; margin-top: 0.3rem; }
    .bad-fc  { border-color: #ef444466 !important; background: #1c0606 !important; }
    .mid-fc  { border-color: #6366f166 !important; background: #0d0f1e !important; }
    .good-fc { border-color: #22c55e66 !important; background: #031209 !important; }

    /* ── Simulation manuelle ── */
    label { display: block; font-size: 0.85rem; color: #94a3b8; margin-top: 1.1rem; margin-bottom: 0.3rem; }
    input[type=range] { width: 100%; accent-color: #6366f1; cursor: pointer; }
    .val { font-size: 1.1rem; font-weight: 700; color: #818cf8; margin-left: 0.4rem; }
    .predict-btn { margin-top: 1.5rem; width: 100%; padding: 0.9rem; background: #6366f1; color: white; border: none; border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    .predict-btn:hover { background: #4f46e5; }
    .predict-btn:disabled { opacity: 0.6; cursor: not-allowed; }
    .result { margin-top: 1.2rem; padding: 1.2rem; border-radius: 10px; text-align: center; display: none; }
    .result.show { display: block; }
    .result.bad  { background: #450a0a; border: 1px solid #ef4444; }
    .result.mid  { background: #1e1b4b; border: 1px solid #6366f1; }
    .result.good { background: #052e16; border: 1px solid #22c55e; }
    .remoji { font-size: 2.5rem; }
    .rlabel { font-size: 1.1rem; font-weight: 700; margin: 0.5rem 0 0.2rem; }
    .rconf  { font-size: 0.82rem; color: #94a3b8; }
    .loading   { text-align: center; color: #64748b; padding: 2rem; font-size: 0.88rem; }
    .error-msg { color: #ef4444; font-size: 0.82rem; padding: 0.5rem; text-align: center; }
  </style>
</head>
<body>
<div class="container">
  <h1>🌤️ Météo IA</h1>
  <p class="sub">Système embarqué STM32 — NUCLEO-N657X0-Q</p>

  <!-- ── CAPTEURS LIVE STM32 ── -->
  <div class="card">
    <div class="live-header">
      <div>
        <div class="live-title">
          <span class="dot" id="dot"></span>Capteurs STM32 Live
        </div>
        <div class="live-status" id="liveStatus">Connexion ThingSpeak...</div>
      </div>
      <button class="refresh-btn" onclick="loadLive()">🔄 Rafraîchir</button>
    </div>

    <div id="liveContent">
      <div class="loading">⏳ Lecture des capteurs...</div>
    </div>
  </div>

  <!-- ── PRÉVISIONS 3 JOURS ── -->
  <div class="card">
    <div class="forecast-header">
      <div>
        <div class="city-name">📍 Le Bourget-du-Lac, FR</div>
        <div class="city-sub">Lac du Bourget · Savoie (73)</div>
      </div>
      <button class="refresh-btn" onclick="loadForecast()">🔄 Actualiser</button>
    </div>
    <div id="forecast">
      <div class="loading">⏳ Chargement des prévisions...</div>
    </div>
  </div>

  <!-- ── SIMULATION MANUELLE ── -->
  <div class="card">
    <div class="card-title">🎛️ Simulation manuelle capteurs</div>

    <label>🌡️ Température <span class="val" id="vTemp">20°C</span></label>
    <input type="range" id="temp" min="-10" max="45" value="20"
           oninput="document.getElementById('vTemp').textContent=this.value+'°C'">

    <label>🔵 Pression <span class="val" id="vPres">1013 hPa</span></label>
    <input type="range" id="pres" min="970" max="1050" value="1013"
           oninput="document.getElementById('vPres').textContent=this.value+' hPa'">

    <label>💨 Vent estimé <span class="val" id="vWind">10 km/h</span></label>
    <input type="range" id="wind" min="0" max="80" value="10"
           oninput="document.getElementById('vWind').textContent=this.value+' km/h'">

    <label>🌧️ Précipitations <span class="val" id="vRain">0 mm</span></label>
    <input type="range" id="rain" min="0" max="30" value="0"
           oninput="document.getElementById('vRain').textContent=this.value+' mm'">

    <button class="predict-btn" onclick="predict()">⚡ Prédire le temps</button>

    <div class="result" id="result">
      <div class="remoji" id="rEmoji"></div>
      <div class="rlabel" id="rLabel"></div>
      <div class="rconf"  id="rConf"></div>
      <div class="led-row">
        <div class="led red"   id="ledR"></div>
        <div class="led blue"  id="ledB"></div>
        <div class="led green" id="ledG"></div>
      </div>
    </div>
  </div>

</div>

<script>
// ── Capteurs STM32 Live ──────────────────────────────────────
async function loadLive() {
  try {
    const res  = await fetch('/live?_t=' + Date.now());
    const data = await res.json();

    if (!data.ok) throw new Error(data.error || 'Erreur ThingSpeak');

    // Statut connecté
    document.getElementById('dot').classList.remove('offline');
    document.getElementById('liveStatus').textContent =
      'Connecté · mise à jour automatique toutes les 15s';

    const themes = {0: 'bad', 1: 'mid', 2: 'good'};

    // Formater l'heure
    const d = new Date(data.timestamp);
    const timeStr = isNaN(d) ? data.timestamp :
      d.toLocaleString('fr-FR', {day:'2-digit', month:'short',
                                  hour:'2-digit', minute:'2-digit'});

    document.getElementById('liveContent').innerHTML = `
      <div class="sensors-grid">
        <div class="sensor-box accent">
          <div class="s-label">🌡️ Température</div>
          <div class="s-value">${data.temp}<span class="s-unit">°C</span></div>
        </div>
        <div class="sensor-box accent">
          <div class="s-label">🔵 Pression</div>
          <div class="s-value">${data.pres}<span class="s-unit">hPa</span></div>
        </div>
      </div>

      <div class="s-label" style="margin-bottom:0.5rem">📐 Accélération (mg)</div>
      <div class="accel-row">
        <div class="accel-box">
          <div class="a-label">X</div>
          <div class="a-value">${data.acc_x}</div>
        </div>
        <div class="accel-box">
          <div class="a-label">Y</div>
          <div class="a-value">${data.acc_y}</div>
        </div>
        <div class="accel-box">
          <div class="a-label">Z</div>
          <div class="a-value">${data.acc_z}</div>
        </div>
      </div>

      <div class="live-result ${themes[data.prediction]}">
        <div class="lr-emoji">${data.emoji}</div>
        <div class="lr-label">${data.nom} — ${data.led}</div>
        <div class="lr-conf">Confiance : ${Math.round(data.confiance * 100)}%</div>
        <div class="led-row">
          <div class="led red  ${data.prediction === 0 ? 'on' : ''}"></div>
          <div class="led blue ${data.prediction === 1 ? 'on' : ''}"></div>
          <div class="led green${data.prediction === 2 ? 'on' : ''}"></div>
        </div>
      </div>
      <div class="ts-time">📡 Dernière mesure : ${timeStr}</div>
    `;

  } catch(e) {
    document.getElementById('dot').classList.add('offline');
    document.getElementById('liveStatus').textContent = 'Hors ligne';
    document.getElementById('liveContent').innerHTML =
      `<div class="error-msg">❌ ${e.message}<br>
       <small>Vérifie que la STM32 envoie des données sur ThingSpeak</small></div>`;
  }
}

// Rafraîchissement automatique toutes les 15 secondes
setInterval(loadLive, 15000);

// ── Prévisions ───────────────────────────────────────────────
async function loadForecast() {
  document.getElementById('forecast').innerHTML =
    '<div class="loading">⏳ Chargement...</div>';
  try {
    const res  = await fetch('/forecast?_t=' + Date.now());
    const json = await res.json();
    if (json.error) throw new Error(json.error);

    const days   = json.days;
    const labels = ["Aujourd'hui", 'Demain', 'Après-demain'];
    const themes = {0: 'bad-fc', 1: 'mid-fc', 2: 'good-fc'};
    let html = '<div class="forecast-grid">';
    days.forEach((d, i) => {
      const todayClass = i === 0 ? 'today' : '';
      const dateObj = new Date(d.date + 'T12:00:00');
      const dateStr = dateObj.toLocaleDateString('fr-FR',
        {weekday:'short', day:'numeric', month:'short'});
      html += `
        <div class="fcard ${todayClass} ${themes[d.prediction]}">
          <div class="fday">${labels[i]}</div>
          <div class="fdate">${dateStr}</div>
          <div class="femoji">${d.emoji}</div>
          <div class="fnom">${d.nom}</div>
          <div class="ftemp">🌡️ ${d.temp_min}° / ${d.temp_max}°C</div>
          <div class="fwind">💨 ${d.wspd} km/h · 🌧️ ${d.prcp} mm</div>
          <div class="fconf">Confiance ${Math.round(d.confiance * 100)}%</div>
        </div>`;
    });
    html += '</div>';
    document.getElementById('forecast').innerHTML = html;
  } catch(e) {
    document.getElementById('forecast').innerHTML =
      `<div class="error-msg">❌ ${e.message}</div>`;
  }
}

// ── Simulation manuelle ──────────────────────────────────────
async function predict() {
  const btn = document.querySelector('.predict-btn');
  btn.textContent = '⏳ Analyse...';
  btn.disabled = true;
  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        temp: document.getElementById('temp').value,
        pres: document.getElementById('pres').value,
        wind: document.getElementById('wind').value,
        rain: document.getElementById('rain').value
      })
    });
    const data = await res.json();
    const themes = {0: 'bad', 1: 'mid', 2: 'good'};
    document.getElementById('rEmoji').textContent = data.emoji;
    document.getElementById('rLabel').textContent = data.nom + ' — ' + data.led;
    document.getElementById('rConf').textContent  =
      'Confiance : ' + Math.round(data.confiance * 100) + '%';
    const r = document.getElementById('result');
    r.className = 'result show ' + themes[data.prediction];
    document.getElementById('ledR').classList.toggle('on', data.prediction === 0);
    document.getElementById('ledB').classList.toggle('on', data.prediction === 1);
    document.getElementById('ledG').classList.toggle('on', data.prediction === 2);
  } catch(e) { alert('Erreur : ' + e.message); }
  btn.textContent = '⚡ Prédire le temps';
  btn.disabled = false;
}

// Démarrage
loadLive();
loadForecast();
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