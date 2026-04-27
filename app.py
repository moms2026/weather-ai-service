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
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Météo IA — STM32</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 2rem 1rem; }
    .container { max-width: 560px; margin: 0 auto; }
    h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
    .sub { color: #94a3b8; font-size: 0.85rem; margin-bottom: 2rem; }
    .card { background: #1e293b; border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; box-shadow: 0 25px 50px rgba(0,0,0,0.4); }
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
    .ts-time { font-size: 0.7rem; color: #475569; text-align: right; margin-top: 0.6rem; }
    .refresh-btn { background: transparent; border: 1px solid #334155; color: #94a3b8; font-size: 0.75rem; padding: 0.35rem 0.75rem; border-radius: 6px; cursor: pointer; transition: all 0.2s; }
    .refresh-btn:hover { border-color: #6366f1; color: #818cf8; }

    /* ── Widget météo ── */
    .weather-widget { background: #1e293b; border-radius: 16px; margin-bottom: 1.5rem; overflow: hidden; box-shadow: 0 25px 50px rgba(0,0,0,0.4); }
    .today-header { padding: 1.5rem 2rem 1rem; display: flex; justify-content: space-between; align-items: flex-start; }
    .today-city { font-size: 1rem; color: #94a3b8; margin-bottom: 0.2rem; }
    .today-day  { font-size: 0.8rem; color: #64748b; }
    .today-temp { font-size: 4rem; font-weight: 300; line-height: 1; }
    .today-temp sup { font-size: 1.5rem; vertical-align: super; }
    .today-desc { font-size: 0.9rem; color: #94a3b8; margin-top: 0.4rem; }
    .today-details { font-size: 0.78rem; color: #64748b; margin-top: 0.3rem; line-height: 1.7; }
    .today-right { text-align: right; }
    .today-emoji { font-size: 4rem; }
    .today-led-row { display: flex; gap: 6px; justify-content: flex-end; margin-top: 0.5rem; }
    .tabs { display: flex; gap: 0; border-bottom: 1px solid #334155; padding: 0 2rem; }
    .tab { font-size: 0.82rem; padding: 0.5rem 1rem 0.4rem; cursor: pointer; color: #64748b; border-bottom: 2px solid transparent; transition: all 0.2s; }
    .tab.active { color: #f59e0b; border-bottom-color: #f59e0b; }
    .chart-area { padding: 1rem 1.5rem 0.5rem; }
    .chart-svg { width: 100%; overflow: visible; }
    .days-row { display: grid; grid-template-columns: repeat(3, 1fr); border-top: 1px solid #334155; }
    .day-cell { padding: 0.9rem 0.5rem; text-align: center; border-right: 1px solid #1e293b; cursor: pointer; transition: background 0.2s; }
    .day-cell:last-child { border-right: none; }
    .day-cell:hover { background: #ffffff08; }
    .day-cell.active-day { background: #ffffff0d; }
    .day-name  { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 0.3rem; }
    .day-emoji { font-size: 1.6rem; margin: 0.2rem 0; }
    .day-temps { font-size: 0.8rem; }
    .day-max   { color: #e2e8f0; font-weight: 600; }
    .day-min   { color: #64748b; margin-left: 0.3rem; }
    .day-badge { display: inline-block; font-size: 0.6rem; padding: 0.1rem 0.4rem; border-radius: 4px; margin-top: 0.3rem; }
    .badge-bad  { background: #ef444422; color: #ef4444; }
    .badge-mid  { background: #6366f122; color: #818cf8; }
    .badge-good { background: #22c55e22; color: #22c55e; }
    .widget-footer { display: flex; justify-content: space-between; align-items: center; padding: 0.8rem 2rem; border-top: 1px solid #1e293b; }
    .widget-source { font-size: 0.68rem; color: #475569; }
    .loading   { text-align: center; color: #64748b; padding: 2rem; font-size: 0.88rem; }
    .error-msg { color: #ef4444; font-size: 0.82rem; padding: 0.5rem; text-align: center; }
  </style>
</head>
<body>
<div class="container">
  <h1>🌤️ Météo IA</h1>
  <p class="sub">Système embarqué STM32 — NUCLEO-N657X0-Q</p>

  <div class="card">
    <div class="live-header">
      <div>
        <div class="live-title"><span class="dot" id="dot"></span>Capteurs STM32 Live</div>
        <div class="live-status" id="liveStatus">Connexion ThingSpeak...</div>
      </div>
      <button class="refresh-btn" onclick="loadLive()">🔄 Rafraîchir</button>
    </div>
    <div id="liveContent"><div class="loading">⏳ Lecture des capteurs...</div></div>
  </div>

  <div class="weather-widget" id="weatherWidget">
    <div class="loading" style="padding:3rem">⏳ Chargement météo...</div>
  </div>
</div>

<script>
// ── Capteurs STM32 Live ──────────────────────────────────────
async function loadLive() {
  try {
    const res  = await fetch('/live?_t=' + Date.now());
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'Erreur ThingSpeak');
    document.getElementById('dot').classList.remove('offline');
    document.getElementById('liveStatus').textContent = 'Connecté · auto-refresh 15s';
    const themes = {0:'bad', 1:'mid', 2:'good'};
    const d = new Date(data.timestamp);
    const timeStr = isNaN(d) ? data.timestamp :
      d.toLocaleString('fr-FR',{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'});
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
        <div class="accel-box"><div class="a-label">X</div><div class="a-value">${data.acc_x}</div></div>
        <div class="accel-box"><div class="a-label">Y</div><div class="a-value">${data.acc_y}</div></div>
        <div class="accel-box"><div class="a-label">Z</div><div class="a-value">${data.acc_z}</div></div>
      </div>
      <div class="live-result ${themes[data.prediction]}">
        <div class="lr-emoji">${data.emoji}</div>
        <div class="lr-label">${data.nom} — ${data.led}</div>
        <div class="lr-conf">Confiance : ${Math.round(data.confiance*100)}%</div>
        <div class="led-row">
          <div class="led red   ${data.prediction===0?'on':''}"></div>
          <div class="led blue  ${data.prediction===1?'on':''}"></div>
          <div class="led green ${data.prediction===2?'on':''}"></div>
        </div>
      </div>
      <div class="ts-time">📡 Dernière mesure : ${timeStr}</div>`;
  } catch(e) {
    document.getElementById('dot').classList.add('offline');
    document.getElementById('liveStatus').textContent = 'Hors ligne';
    document.getElementById('liveContent').innerHTML =
      `<div class="error-msg">❌ ${e.message}<br><small>Vérifie que la STM32 envoie des données</small></div>`;
  }
}
setInterval(loadLive, 15000);

// ── Widget météo ─────────────────────────────────────────────
let forecastData = [];

function drawChart(days, mode) {
  const W = 500, H = 80;
  const n = days.length;
  let values, color, unit;
  if (mode === 'temp') {
    values = days.map(d => (d.temp_max + d.temp_min) / 2);
    color = '#f59e0b'; unit = '°';
  } else if (mode === 'prcp') {
    values = days.map(d => d.prcp);
    color = '#6366f1'; unit = 'mm';
  } else {
    values = days.map(d => d.wspd);
    color = '#22c55e'; unit = 'km/h';
  }
  const minV = Math.min(...values) - 2;
  const maxV = Math.max(...values) + 2;
  const xs = days.map((_, i) => 40 + i * ((W - 80) / (n - 1)));
  const ys = values.map(v => H - 10 - ((v - minV) / (maxV - minV)) * (H - 20));
  let path = `M ${xs[0]} ${ys[0]}`;
  for (let i = 1; i < n; i++) {
    const mx = (xs[i-1] + xs[i]) / 2;
    path += ` C ${mx} ${ys[i-1]}, ${mx} ${ys[i]}, ${xs[i]} ${ys[i]}`;
  }
  let fill = path + ` L ${xs[n-1]} ${H} L ${xs[0]} ${H} Z`;
  let labels = '';
  xs.forEach((x, i) => {
    labels += `<text x="${x}" y="${ys[i]-8}" text-anchor="middle"
      fill="${color}" font-size="11" font-weight="600">${values[i].toFixed(1)}${unit}</text>`;
    labels += `<circle cx="${x}" cy="${ys[i]}" r="4" fill="${color}"/>`;
  });
  return `
    <svg class="chart-svg" viewBox="0 0 ${W} ${H+10}" preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="${color}" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <path d="${fill}" fill="url(#grad)"/>
      <path d="${path}" fill="none" stroke="${color}" stroke-width="2.5"
            stroke-linecap="round" stroke-linejoin="round"/>
      ${labels}
    </svg>`;
}

function renderWidget(days, activeTab) {
  const today = days[0];
  const dayNames = ["Dim.","Lun.","Mar.","Mer.","Jeu.","Ven.","Sam."];
  const badgeClass = {0:'badge-bad', 1:'badge-mid', 2:'badge-good'};
  const badgeLabel = {0:'Mauvais', 1:'Moyen', 2:'Bon'};
  const ledColors  = {0:'#ef4444', 1:'#6366f1', 2:'#22c55e'};
  const now = new Date();
  const dayStr = now.toLocaleDateString('fr-FR',
    {weekday:'long', day:'numeric', month:'long'});
  const todayLeds = [0,1,2].map(i =>
    `<div style="width:10px;height:10px;border-radius:50%;
      background:${ledColors[i]};
      opacity:${today.prediction===i?1:0.15};
      box-shadow:${today.prediction===i?`0 0 8px ${ledColors[i]}`:'none'}"></div>`
  ).join('');
  const daysHTML = days.map((d, i) => {
    const dateObj = new Date(d.date + 'T12:00:00');
    const dayName = dayNames[dateObj.getDay()];
    return `
      <div class="day-cell ${i===0?'active-day':''}" onclick="switchDay(${i})">
        <div class="day-name">${i===0?'Auj.':dayName}</div>
        <div class="day-emoji">${d.emoji}</div>
        <div class="day-temps">
          <span class="day-max">${d.temp_max}°</span>
          <span class="day-min">${d.temp_min}°</span>
        </div>
        <div><span class="day-badge ${badgeClass[d.prediction]}">${badgeLabel[d.prediction]}</span></div>
      </div>`;
  }).join('');
  const chart = drawChart(days, activeTab);
  document.getElementById('weatherWidget').innerHTML = `
    <div class="today-header">
      <div class="today-left">
        <div class="today-city">📍 Le Bourget-du-Lac, FR</div>
        <div class="today-day">${dayStr}</div>
        <div class="today-temp">${today.temp_max}<sup>°C</sup></div>
        <div class="today-desc">${today.nom}</div>
        <div class="today-details">
          💨 Vent : ${today.wspd} km/h<br>
          🌧️ Précip. : ${today.prcp} mm<br>
          🔵 Pression : ${today.pres} hPa
        </div>
      </div>
      <div class="today-right">
        <div class="today-emoji">${today.emoji}</div>
        <div class="today-led-row">${todayLeds}</div>
        <div style="font-size:0.7rem;color:#475569;margin-top:0.3rem">
          Confiance ${Math.round(today.confiance*100)}%
        </div>
        <button class="refresh-btn" style="margin-top:0.8rem" onclick="loadForecast()">🔄</button>
      </div>
    </div>
    <div class="tabs">
      <div class="tab ${activeTab==='temp'?'active':''}"  onclick="switchTab('temp')">Température</div>
      <div class="tab ${activeTab==='prcp'?'active':''}"  onclick="switchTab('prcp')">Précipitations</div>
      <div class="tab ${activeTab==='wind'?'active':''}"  onclick="switchTab('wind')">Vent</div>
    </div>
    <div class="chart-area">${chart}</div>
    <div class="days-row">${daysHTML}</div>
    <div class="widget-footer">
      <span class="widget-source">IA entraînée sur données réelles 2018-2023</span>
      <span class="widget-source">Open-Meteo · Scikit-learn</span>
    </div>`;
}

function switchTab(tab) {
  if (forecastData.length) renderWidget(forecastData, tab);
}

function switchDay(i) {
  document.querySelectorAll('.day-cell').forEach((el, j) => {
    el.classList.toggle('active-day', i === j);
  });
}

async function loadForecast() {
  document.getElementById('weatherWidget').innerHTML =
    '<div class="loading" style="padding:3rem">⏳ Chargement météo...</div>';
  try {
    const res  = await fetch('/forecast?_t=' + Date.now());
    const json = await res.json();
    if (json.error) throw new Error(json.error);
    forecastData = json.days;
    renderWidget(forecastData, 'temp');
  } catch(e) {
    document.getElementById('weatherWidget').innerHTML =
      `<div class="error-msg" style="padding:2rem">❌ ${e.message}</div>`;
  }
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