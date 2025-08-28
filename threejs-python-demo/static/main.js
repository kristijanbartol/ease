// static/main.js
const fields = [
  "mid","neck","shoulder","upper_side","sleeve","upper_bottom",
  "lower_side","between","lower_bottom","upper_scale","lower_scale"
];

const statusEl = document.getElementById('paramsStatus');
const runBtn   = document.getElementById('runParamsBtn');

function linkPair(name) {
  const r = document.getElementById(`${name}_range`);
  const n = document.getElementById(`${name}_num`);
  if (!r || !n) return;
  r.addEventListener('input', () => { n.value = r.value; });
  n.addEventListener('input', () => {
    const min = parseFloat(n.min), max = parseFloat(n.max);
    let v = parseFloat(n.value);
    if (Number.isFinite(min)) v = Math.max(min, v);
    if (Number.isFinite(max)) v = Math.min(max, v);
    n.value = v; r.value = v;
  });
}

function setValue(name, val) {
  const r = document.getElementById(`${name}_range`);
  const n = document.getElementById(`${name}_num`);
  if (r) r.value = val;
  if (n) n.value = val;
}

function readConfigFromUI() {
  const cfg = {};
  for (const k of fields) {
    const n = document.getElementById(`${k}_num`);
    if (!n) continue;
    const v = parseFloat(n.value);
    cfg[k] = Number.isFinite(v) ? v : 0;
  }
  return cfg;
}

async function loadConfig() {
  statusEl.textContent = 'Loading config…';
  try {
    const res = await fetch('/config');
    if (!res.ok) {
      statusEl.textContent = 'Failed to load config';
      console.error(await res.text());
      return;
    }
    const cfg = await res.json();
    for (const k of fields) if (cfg[k] != null) setValue(k, cfg[k]);
    statusEl.textContent = 'Ready';
  } catch (e) {
    statusEl.textContent = 'Network error';
    console.error(e);
  }
}

async function runWithParams() {
  const cfg = readConfigFromUI();
  runBtn.disabled = true;
  statusEl.textContent = 'Saving…';
  try {
    // 1) Save config to disk (script will read it)
    let res = await fetch('/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg)
    });
    if (!res.ok) {
      const txt = await res.text();
      statusEl.textContent = 'Save failed';
      console.error(txt); alert(txt); return;
    }

    // 2) Run pipeline
    statusEl.textContent = 'Running…';
    res = await fetch('/run', { method: 'POST' });
    const txt = await res.text();
    if (!res.ok) {
      statusEl.textContent = 'Error';
      console.error(txt); alert(txt); return;
    }
    statusEl.textContent = 'Done';

    // 3) Optional: refresh garments
    if (window.refreshGarments) {
      try { await window.refreshGarments(); } catch {}
    }
  } catch (e) {
    statusEl.textContent = 'Network error';
    console.error(e); alert(e);
  } finally {
    runBtn.disabled = false;
  }
}

// init
for (const k of fields) linkPair(k);
loadConfig();
runBtn.addEventListener('click', runWithParams);
