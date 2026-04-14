/**
 * ElevatorAI — Dashboard JavaScript
 * IoT Predictive Elevator Maintenance System
 * Real-time sensor polling, Chart.js visualizations, alerts
 */

// ─────────────────────────────────────────
// Chart Configuration
// ─────────────────────────────────────────
const MAX_POINTS = 60;
const CHART_DEFAULTS = {
  type: 'line',
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 200 },
    interaction: { intersect: false, mode: 'index' },
    plugins: { legend: { display: false }, tooltip: { enabled: true } },
    scales: {
      x: {
        ticks: { color: '#475569', maxTicksLimit: 6, font: { size: 10 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
      },
      y: {
        ticks: { color: '#475569', font: { size: 10 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
      }
    },
    elements: { point: { radius: 0 }, line: { tension: 0.4, borderWidth: 2 } }
  }
};

function makeDataset(color, label) {
  return {
    label,
    data: [],
    borderColor: color,
    backgroundColor: color.replace(')', ',0.08)').replace('rgb', 'rgba'),
    fill: true,
  };
}

function initChart(canvasId, color, label, yMin, yMax) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  const cfg = JSON.parse(JSON.stringify(CHART_DEFAULTS));
  cfg.data = { labels: [], datasets: [makeDataset(color, label)] };
  if (yMin !== undefined) cfg.options.scales.y.suggestedMin = yMin;
  if (yMax !== undefined) cfg.options.scales.y.suggestedMax = yMax;
  return new Chart(ctx, cfg);
}

const charts = {
  vibration: initChart('chartVibration', 'rgb(239,68,68)',   'Vibration RMS', 0, 5),
  temp:      initChart('chartTemp',      'rgb(249,115,22)',  'Temperature °C', 20, 120),
  current:   initChart('chartCurrent',   'rgb(99,102,241)',  'Motor Current A', 0, 40),
  door:      initChart('chartDoor',      'rgb(59,130,246)',  'Door Variance', 0, 12),
};

function pushChartPoint(chart, label, value) {
  const d = chart.data;
  d.labels.push(label);
  d.datasets[0].data.push(value);
  if (d.labels.length > MAX_POINTS) {
    d.labels.shift();
    d.datasets[0].data.shift();
  }
  chart.update('none');
}

// ─────────────────────────────────────────
// Gauge Updater
// ─────────────────────────────────────────
const GAUGE_CONFIG = {
  vibration: { id: 'vibration', max: 15, warn: 4, crit: 8 },
  temp:      { id: 'temp',      max: 120, warn: 65, crit: 85 },
  current:   { id: 'current',   max: 40,  warn: 18, crit: 28 },
  door:      { id: 'door',      max: 12,  warn: 2.5, crit: 5 },
  accel:     { id: 'accel',     max: 10,  warn: 2,  crit: 5 },
};

const GAUGE_COLORS = {
  normal:   '#22c55e',
  warning:  '#f97316',
  critical: '#ef4444',
};

function updateGauge(key, value, decimals = 2) {
  const cfg = GAUGE_CONFIG[key];
  const pct = Math.min((value / cfg.max) * 100, 100);
  const level = value >= cfg.crit ? 'critical' : value >= cfg.warn ? 'warning' : 'normal';
  const color = GAUGE_COLORS[level];

  const card = document.getElementById(`g-${cfg.id}`);
  card.className = `gauge-card ${level !== 'normal' ? level : ''}`;

  document.getElementById(`gv-${cfg.id}`).textContent = value.toFixed(decimals);
  document.getElementById(`gv-${cfg.id}`).style.color = color;

  const bar = document.getElementById(`gb-${cfg.id}`);
  bar.style.width = pct + '%';
  bar.style.background = color;

  const statusEl = document.getElementById(`gs-${cfg.id}`);
  statusEl.textContent = level.charAt(0).toUpperCase() + level.slice(1);
  statusEl.className = `gauge-status ${level !== 'normal' ? level : ''}`;
}

// ─────────────────────────────────────────
// Prediction Panel
// ─────────────────────────────────────────
const FAULT_COLORS = ['#22c55e', '#ef4444', '#f97316', '#3b82f6'];
const FAULT_COLOR_CLASSES = ['normal', 'red', 'orange', 'blue'];
const FAULT_BADGE_LABELS = ['NORMAL', 'BEARING FAIL', 'OVERHEATING', 'DOOR FAULT'];

let lastFaultClass = 0;
let faultShown = false;

function updatePrediction(pred) {
  const cls = pred.class_id;
  const color = FAULT_COLORS[cls];

  // Card glow class
  const card = document.getElementById('predCard');
  card.className = `card prediction-card ${cls !== 0 ? 'fault-' + cls : ''}`;

  // Status text and icon
  document.getElementById('predIcon').textContent = pred.icon;
  const statusEl = document.getElementById('predStatus');
  statusEl.textContent = pred.class_name;
  statusEl.style.color = color;
  document.getElementById('predConfidence').textContent = `${pred.confidence}% confidence`;

  // Badge
  const badge = document.getElementById('predBadge');
  badge.textContent = FAULT_BADGE_LABELS[cls];
  badge.className = `badge ${FAULT_COLOR_CLASSES[cls] !== 'normal' ? FAULT_COLOR_CLASSES[cls] : ''}`;

  // Probability bars
  const classes = ['Normal', 'Bearing Failure', 'Motor Overheating', 'Door Malfunction'];
  for (let i = 0; i < 4; i++) {
    const pct = pred.probabilities[classes[i]] || 0;
    document.getElementById(`pb${i}`).style.width = pct + '%';
    document.getElementById(`pp${i}`).textContent = pct + '%';
  }

  // Show fault overlay on new fault
  if (cls !== 0 && cls !== lastFaultClass && !faultShown) {
    showFaultOverlay(pred);
    faultShown = true;
    setTimeout(() => { faultShown = false; }, 10000);
  }
  if (cls === 0) faultShown = false;
  lastFaultClass = cls;
}

function showFaultOverlay(pred) {
  document.getElementById('faultModalIcon').textContent = pred.icon;
  document.getElementById('faultModalTitle').textContent = 'FAULT DETECTED';
  document.getElementById('faultModalSub').textContent = pred.class_name;
  document.getElementById('faultOverlay').classList.add('show');
  // Auto-dismiss after 5 seconds
  setTimeout(dismissFault, 5000);
}

function dismissFault() {
  document.getElementById('faultOverlay').classList.remove('show');
}

// ─────────────────────────────────────────
// Health Ring
// ─────────────────────────────────────────
function updateHealthRing(health) {
  const circumference = 2 * Math.PI * 50; // r=50
  const offset = circumference * (1 - health / 100);
  const ring = document.getElementById('healthRing');
  ring.style.strokeDashoffset = offset;

  const color = health >= 80 ? '#22c55e' : health >= 50 ? '#f97316' : '#ef4444';
  ring.style.stroke = color;

  document.getElementById('healthPct').textContent = Math.round(health) + '%';
  document.getElementById('healthPct').style.color = color;
}

// ─────────────────────────────────────────
// Alert Feed
// ─────────────────────────────────────────
let alertCount = 0;

function addAlertItem(alert) {
  alertCount++;
  document.getElementById('alertCount').textContent = alertCount;

  const feed = document.getElementById('alertFeed');
  const empty = feed.querySelector('.alert-empty');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'alert-item';
  item.style.borderColor = alert.color;
  item.innerHTML = `
    <div class="alert-item-icon">${alert.icon}</div>
    <div class="alert-item-body">
      <div class="alert-item-fault" style="color:${alert.color}">${alert.fault}</div>
      <div class="alert-item-time">${alert.timestamp}</div>
    </div>
    <div class="alert-item-conf" style="color:${alert.color}">${alert.confidence}%</div>
  `;
  feed.insertBefore(item, feed.firstChild);

  // Keep only last 20 items
  while (feed.children.length > 20) feed.removeChild(feed.lastChild);
}

// ─────────────────────────────────────────
// Maintenance Log Table
// ─────────────────────────────────────────
const logRows = [];

function addLogRow(data) {
  logRows.unshift(data);
  if (logRows.length > 100) logRows.pop();

  // Show last 20 in table
  const tbody = document.getElementById('logTableBody');
  const visible = logRows.slice(0, 20);

  tbody.innerHTML = visible.map(r => {
    const isFault = r.prediction.class_id !== 0;
    const action = isFault ? '🔧 Inspect Required' : '✅ No Action';
    const statusEl = isFault
      ? `<span class="td-status fault">${r.prediction.icon} ${r.prediction.class_name}</span>`
      : `<span class="td-status normal">✅ Normal</span>`;

    return `<tr>
      <td style="font-family:'JetBrains Mono',monospace;font-size:0.72rem">${r.timestamp}</td>
      <td>${isFault ? '<span style="color:#ef4444">⚠️ FAULT</span>' : '<span style="color:#22c55e">● OK</span>'}</td>
      <td>${statusEl}</td>
      <td style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:${r.prediction.color}">${r.prediction.confidence}%</td>
      <td style="font-size:0.72rem">${action}</td>
    </tr>`;
  }).join('');
}

// ─────────────────────────────────────────
// Stats Update
// ─────────────────────────────────────────
function updateStats(stats) {
  document.getElementById('st-readings').textContent  = stats.total_readings;
  document.getElementById('st-faults').textContent    = stats.total_faults;
  document.getElementById('st-faultrate').textContent = stats.fault_rate + '%';
  document.getElementById('st-alerts').textContent    = stats.alerts_count;
  document.getElementById('headerUptime').textContent = stats.uptime;
  updateHealthRing(stats.health_score);
}

// ─────────────────────────────────────────
// Main Polling Loop
// ─────────────────────────────────────────
let pollCount = 0;
let lastAlertTimestamps = new Set();

async function poll() {
  try {
    const [liveResp, statsResp] = await Promise.all([
      fetch('/live-data'),
      fetch('/stats'),
    ]);

    if (!liveResp.ok || !statsResp.ok) return;

    const live  = await liveResp.json();
    const stats = await statsResp.json();
    const s     = live.sensors;
    const pred  = live.prediction;
    const ts    = live.timestamp;

    pollCount++;

    // Update gauges
    updateGauge('vibration', s.vibration_rms, 3);
    updateGauge('temp',      s.temperature_mean, 1);
    updateGauge('current',   s.motor_current_mean, 2);
    updateGauge('door',      s.door_signal_variance, 3);
    updateGauge('accel',     s.acceleration_peak, 3);

    // Update charts
    pushChartPoint(charts.vibration, ts, s.vibration_rms);
    pushChartPoint(charts.temp,      ts, s.temperature_mean);
    pushChartPoint(charts.current,   ts, s.motor_current_mean);
    pushChartPoint(charts.door,      ts, s.door_signal_variance);

    // Update prediction
    updatePrediction(pred);

    // Add to log
    addLogRow(live);

    // Update stats & health
    updateStats(stats);

    // Update last update time
    document.getElementById('lastUpdate').textContent = `Last update: ${ts}`;

    // New fault alert
    if (live.is_fault) {
      const alertKey = `${ts}-${pred.class_name}`;
      if (!lastAlertTimestamps.has(alertKey)) {
        lastAlertTimestamps.add(alertKey);
        addAlertItem({
          timestamp: ts,
          fault: pred.class_name,
          confidence: pred.confidence,
          color: pred.color,
          icon: pred.icon,
        });
        // Keep set from growing
        if (lastAlertTimestamps.size > 100) {
          lastAlertTimestamps = new Set([...lastAlertTimestamps].slice(-50));
        }
      }
    }

  } catch (err) {
    console.warn('Poll error:', err);
  }
}

// ─────────────────────────────────────────
// Initialize
// ─────────────────────────────────────────
// Start polling immediately, then every 2 seconds
poll();
setInterval(poll, 2000);

console.log('🛗 ElevatorAI Dashboard loaded — polling every 2 seconds');
