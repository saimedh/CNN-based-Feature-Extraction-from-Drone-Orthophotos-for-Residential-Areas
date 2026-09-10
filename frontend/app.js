/**
 * OrthoSeg — app.js
 * Frontend logic: file upload, API call, Leaflet map, stats display
 */

'use strict';

// ═══════════════════════════════════════════════════════════ State ═══
const state = {
  mode: 'single',         // 'single' | 'folder'
  folderSource: null,     // 'client' | 'server'
  folderFiles: [],        // [{ id, name, size, fileObj, path, checked, status, results }]
  activeBatchIndex: -1,
  batchResults: {},       // { [fileName]: geojsonData }
  file: null,
  geojsonData: {},        // { building: FeatureCollection, road: ..., vegetation: ... }
  layers: {},             // { building: L.GeoJSON, road: ..., vegetation: ... }
  baseTile: null,
  featureOpacity: 0.6,
};

// Class config
const CLASS_CONFIG = {
  building: {
    color: '#f59e0b',
    fillColor: '#f59e0b',
    fillOpacity: 0.32,
    weight: 2,
    layerCheckbox: 'layer-buildings',
    count: 'count-buildings',
  },
  road: {
    color: '#38bdf8',
    fillColor: '#38bdf8',
    fillOpacity: 0.25,
    weight: 2.5,
    layerCheckbox: 'layer-roads',
    count: 'count-roads',
  },
  vegetation: {
    color: '#10b981',
    fillColor: '#10b981',
    fillOpacity: 0.28,
    weight: 1.5,
    layerCheckbox: 'layer-vegetation',
    count: 'count-vegetation',
  },
};

// ═══════════════════════════════════════════════════════════ MAP ════
let map;

function initMap() {
  map = L.map('map', {
    center: [20, 78],
    zoom: 4,
    zoomControl: true,
    attributionControl: true,
  });

  state.baseTile = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'Esri World Imagery', maxZoom: 20 }
  ).addTo(map);

  // Coords display
  map.on('mousemove', (e) => {
    document.getElementById('map-coords').textContent =
      `${e.latlng.lat.toFixed(6)}°N, ${e.latlng.lng.toFixed(6)}°E`;
  });
  map.on('zoomend', () => {
    document.getElementById('map-zoom').textContent = `Zoom: ${map.getZoom()}`;
  });
  document.getElementById('map-zoom').textContent = `Zoom: ${map.getZoom()}`;
}

// ═══════════════════════════════════════════ Basemap switcher ════
const BASEMAPS = {
  satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  osm: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
};

function changeBasemap() {
  const val = document.getElementById('s-basemap').value;
  if (state.baseTile) { state.baseTile.remove(); }
  state.baseTile = L.tileLayer(BASEMAPS[val], { maxZoom: 20 }).addTo(map);
}

// ═══════════════════════════════════════════ API status ════
function getApiEndpoint() {
  const el = document.getElementById('api-endpoint');
  if (el && el.value && el.value.trim()) {
    return el.value.trim().replace(/\/+$/, '');
  }
  if (window.location && window.location.protocol && window.location.protocol.startsWith('http')) {
    return window.location.origin;
  }
  return 'http://127.0.0.1:8000';
}

async function checkApiStatus() {
  const endpoint = getApiEndpoint();
  const dot = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  try {
    const res = await fetch(`${endpoint}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      if (dot) dot.className = 'status-dot online';
      if (label) label.textContent = 'API Online';
    } else {
      throw new Error('not ok');
    }
  } catch {
    if (dot) dot.className = 'status-dot offline';
    if (label) label.textContent = 'API Offline';
  }
}

// ═══════════════════════════════════════════ File handling ════
function onFileSelect(event) {
  const file = event.target.files[0];
  if (file) setFile(file);
}

function onDragOver(event) {
  event.preventDefault();
  document.getElementById('drop-zone').classList.add('dragover');
}
function onDragLeave(event) {
  document.getElementById('drop-zone').classList.remove('dragover');
}
function onDrop(event) {
  event.preventDefault();
  document.getElementById('drop-zone').classList.remove('dragover');
  const file = event.dataTransfer.files[0];
  if (file) {
    setFile(file);
  } else {
    showToast('Please drop a valid file.', 'error');
  }
}

function setFile(file) {
  state.file = file;
  document.getElementById('drop-zone').style.display = 'none';
  document.getElementById('file-info').style.display = 'block';
  document.getElementById('file-name').textContent = file.name;
  document.getElementById('file-size').textContent = formatBytes(file.size);
  updateProcessButton();
}

function clearFile() {
  state.file = null;
  document.getElementById('drop-zone').style.display = 'block';
  document.getElementById('file-info').style.display = 'none';
  document.getElementById('file-input').value = '';
  updateProcessButton();
  document.getElementById('progress-wrap').style.display = 'none';
}

// ═══════════════════════════════════════════ Folder Mode & Batch ════
function setUploadMode(mode) {
  state.mode = mode;
  document.getElementById('mode-single-btn').classList.toggle('active', mode === 'single');
  document.getElementById('mode-folder-btn').classList.toggle('active', mode === 'folder');
  document.getElementById('mode-single-section').style.display = mode === 'single' ? 'block' : 'none';
  document.getElementById('mode-folder-section').style.display = mode === 'folder' ? 'block' : 'none';
  updateProcessButton();
}

function updateProcessButton() {
  const btn = document.getElementById('btn-process');
  const label = document.getElementById('btn-label');
  if (state.mode === 'single') {
    if (state.file) {
      btn.disabled = false;
      label.textContent = 'Extract Features';
    } else {
      btn.disabled = true;
      label.textContent = 'Select a file to start';
    }
  } else {
    const checkedCount = state.folderFiles.filter(f => f.checked).length;
    if (checkedCount > 0) {
      btn.disabled = false;
      label.textContent = `Process Batch (${checkedCount} file${checkedCount > 1 ? 's' : ''})`;
    } else {
      btn.disabled = true;
      label.textContent = state.folderFiles.length === 0 ? 'Select or scan a folder' : 'Select files to process';
    }
  }
}

function onFolderSelect(event) {
  const rawFiles = Array.from(event.target.files);
  const SUPPORTED_EXTS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.img', '.vrt', '.ecw', '.jp2'];
  const validFiles = rawFiles.filter(f => {
    const lower = f.name.toLowerCase();
    if (lower.includes('_mask')) return false;
    return SUPPORTED_EXTS.some(ext => lower.endsWith(ext));
  });

  if (validFiles.length === 0) {
    showToast('No valid files found in folder.', 'error');
    return;
  }

  state.folderSource = 'client';
  state.folderFiles = validFiles.map((f, idx) => ({
    id: idx,
    name: f.name,
    size: f.size,
    fileObj: f,
    checked: true,
    status: 'pending',
    results: null,
  }));

  renderFolderFileList();
  updateProcessButton();
  showToast(`Found ${validFiles.length} file(s)`, 'info');
}

async function scanServerFolder() {
  const endpoint = getApiEndpoint();
  const folderPath = document.getElementById('server-folder-path').value.trim();
  const scanBtn = document.getElementById('btn-scan');
  scanBtn.disabled = true;
  scanBtn.textContent = 'Scanning...';

  try {
    const res = await fetch(`${endpoint}/folders/scan?folder_path=${encodeURIComponent(folderPath)}`);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Folder scan failed');
    }
    const data = await res.json();
    const files = data.primary_files || [];
    if (files.length === 0) {
      showToast(`No primary GeoTIFF files found in "${folderPath}".`, 'info');
    } else {
      showToast(`Found ${files.length} orthophoto(s) in "${folderPath}"`, 'success');
    }

    state.folderSource = 'server';
    state.folderFiles = files.map((f, idx) => ({
      id: idx,
      name: f.name,
      size: f.size_bytes,
      path: f.path,
      checked: true,
      status: 'pending',
      has_paired_mask: f.has_paired_mask,
      results: null,
    }));

    renderFolderFileList();
    updateProcessButton();
  } catch (err) {
    showToast(`Scan error: ${err.message}`, 'error');
  } finally {
    scanBtn.disabled = false;
    scanBtn.textContent = 'Scan';
  }
}

function renderFolderFileList() {
  const wrap = document.getElementById('folder-files-wrap');
  const listEl = document.getElementById('folder-file-list');
  const countEl = document.getElementById('folder-files-count');
  if (state.folderFiles.length === 0) {
    wrap.style.display = 'none';
    return;
  }
  wrap.style.display = 'block';
  countEl.textContent = state.folderFiles.length;

  listEl.innerHTML = state.folderFiles.map((item, idx) => `
    <div class="folder-file-item ${state.activeBatchIndex === idx ? 'active' : ''}" onclick="selectBatchItem(${idx})">
      <input type="checkbox" class="folder-file-checkbox" ${item.checked ? 'checked' : ''} onclick="event.stopPropagation(); toggleFolderFile(${idx})" />
      <span class="folder-file-name" title="${item.name}">${item.name}</span>
      <span class="file-size">${formatBytes(item.size)}</span>
      <span class="file-status-chip ${item.status}">${item.status}</span>
    </div>
  `).join('');
}

function toggleFolderFile(idx) {
  if (state.folderFiles[idx]) {
    state.folderFiles[idx].checked = !state.folderFiles[idx].checked;
    renderFolderFileList();
    updateProcessButton();
  }
}

function toggleAllFolderFiles(selectAll) {
  state.folderFiles.forEach(f => { f.checked = selectAll; });
  renderFolderFileList();
  updateProcessButton();
}

function selectBatchItem(idx) {
  const item = state.folderFiles[idx];
  if (!item || !item.results) return;
  state.activeBatchIndex = idx;
  renderFolderFileList();

  const select = document.getElementById('batch-active-file-select');
  if (select) select.value = item.name;

  renderLayers(item.results);
  updateStats(item.results);
  document.getElementById('map-empty').style.display = 'none';
  document.getElementById('download-row').style.display = 'flex';
}

function onBatchActiveFileChange() {
  const select = document.getElementById('batch-active-file-select');
  const chosenName = select.value;
  const idx = state.folderFiles.findIndex(f => f.name === chosenName);
  if (idx >= 0) {
    selectBatchItem(idx);
  }
}

function updateBatchSwitcher() {
  const card = document.getElementById('batch-switcher-card');
  const select = document.getElementById('batch-active-file-select');
  const finished = state.folderFiles.filter(f => f.results !== null);
  if (finished.length === 0) {
    card.style.display = 'none';
    return;
  }
  card.style.display = 'block';
  select.innerHTML = finished.map(f => `
    <option value="${f.name}">${f.name}</option>
  `).join('');
  if (state.activeBatchIndex >= 0 && state.folderFiles[state.activeBatchIndex]) {
    select.value = state.folderFiles[state.activeBatchIndex].name;
  }
}

function handleProcessClick() {
  if (state.mode === 'single') {
    processFile();
  } else {
    processFolderBatch();
  }
}

async function parseZipFeatures(zipBlob, prefixFilter = '') {
  const zip = await JSZip.loadAsync(zipBlob);
  const geojsonData = {};
  for (const [filename, zipEntry] of Object.entries(zip.files)) {
    if (zipEntry.dir) continue;
    if (prefixFilter && !filename.startsWith(prefixFilter)) continue;
    if (filename.endsWith('.geojson')) {
      const text = await zipEntry.async('text');
      const geojson = JSON.parse(text);
      const base = filename.split('/').pop().replace('.geojson', '').toLowerCase();
      if (['building', 'road', 'vegetation'].includes(base)) {
        geojsonData[base] = geojson;
      }
    }
  }
  return geojsonData;
}

// ═══════════════════════════════════════════ Process / API call ════
async function processFile() {
  if (!state.file) return;

  const endpoint = getApiEndpoint();
  const btn = document.getElementById('btn-process');
  const spinner = document.getElementById('btn-spinner');
  const label = document.getElementById('btn-label');
  const progressWrap = document.getElementById('progress-wrap');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');

  showOverlay(true);
  setStep('upload');

  btn.disabled = true;
  spinner.style.display = 'block';
  label.textContent = 'Processing...';
  progressWrap.style.display = 'block';

  try {
    progressFill.style.width = '15%';
    progressText.textContent = 'Uploading orthophoto...';
    setStep('upload');

    const formData = new FormData();
    formData.append('file', state.file);

    const regularize = document.getElementById('setting-regularize')?.checked ?? true;
    const zipBlob = await uploadWithProgress(
      `${endpoint}/predict?regularize=${regularize}`,
      formData,
      (pct) => {
        const capped = Math.min(40, pct * 0.4);
        progressFill.style.width = `${capped}%`;
      }
    );

    progressFill.style.width = '50%';
    progressText.textContent = 'Tiling & running inference...';
    setStep('tile');
    await delay(600);
    setStep('infer');
    progressFill.style.width = '70%';

    progressText.textContent = 'Vectorizing results...';
    setStep('vec');
    const geojsonData = await parseZipFeatures(zipBlob);

    progressFill.style.width = '100%';
    progressText.textContent = 'Done!';
    await delay(400);

    renderLayers(geojsonData);
    updateStats(geojsonData);

    showOverlay(false);
    document.getElementById('map-empty').style.display = 'none';
    document.getElementById('download-row').style.display = 'flex';

    showToast('Feature extraction complete!', 'success');
    addHistoryEntry(state.file.name, geojsonData);

  } catch (err) {
    console.error(err);
    showOverlay(false);
    showToast(`Error: ${err.message}`, 'error');
    progressText.textContent = 'Failed.';
  } finally {
    btn.disabled = false;
    spinner.style.display = 'none';
    label.textContent = 'Extract Features';
  }
}

async function processFolderBatch() {
  const checkedFiles = state.folderFiles.filter(f => f.checked);
  if (checkedFiles.length === 0) return;

  const endpoint = getApiEndpoint();
  const regularize = document.getElementById('setting-regularize')?.checked ?? true;
  const btn = document.getElementById('btn-process');
  const spinner = document.getElementById('btn-spinner');
  const label = document.getElementById('btn-label');
  const progressWrap = document.getElementById('progress-wrap');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');

  btn.disabled = true;
  spinner.style.display = 'block';
  label.textContent = 'Processing Batch...';
  progressWrap.style.display = 'block';

  let completedCount = 0;
  let firstResult = null;

  for (let i = 0; i < checkedFiles.length; i++) {
    const item = checkedFiles[i];
    item.status = 'processing';
    renderFolderFileList();

    const pct = Math.round((i / checkedFiles.length) * 100);
    progressFill.style.width = `${pct}%`;
    progressText.textContent = `(${i + 1}/${checkedFiles.length}) Processing ${item.name}...`;

    try {
      let geojsonData = {};

      if (state.folderSource === 'client' && item.fileObj) {
        const formData = new FormData();
        formData.append('file', item.fileObj);
        const res = await fetch(`${endpoint}/predict?regularize=${regularize}`, {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const zipBlob = await res.blob();
        geojsonData = await parseZipFeatures(zipBlob);
      } else if (state.folderSource === 'server' && item.path) {
        const res = await fetch(`${endpoint}/predict/folder?folder_path=${encodeURIComponent(document.getElementById('server-folder-path').value.trim())}&regularize=${regularize}`, {
          method: 'POST',
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const stem = item.name.replace(/\.[^/.]+$/, '');
        for (const cls of ['building', 'road', 'vegetation']) {
          try {
            const gRes = await fetch(`${endpoint}/outputs/${stem}/${cls}.geojson`);
            if (gRes.ok) {
              geojsonData[cls] = await gRes.json();
            }
          } catch (_) {}
        }
      }

      item.status = 'done';
      item.results = geojsonData;
      state.batchResults[item.name] = geojsonData;
      addHistoryEntry(item.name, geojsonData);
      completedCount++;

      if (!firstResult && Object.keys(geojsonData).length > 0) {
        firstResult = { idx: item.id, data: geojsonData };
      }
    } catch (err) {
      console.error(`Error processing ${item.name}:`, err);
      item.status = 'error';
    }

    renderFolderFileList();
  }

  progressFill.style.width = '100%';
  progressText.textContent = `Batch complete: ${completedCount}/${checkedFiles.length} succeeded`;
  btn.disabled = false;
  spinner.style.display = 'none';
  label.textContent = `Process Batch (${checkedFiles.length} files)`;

  updateBatchSwitcher();

  if (firstResult) {
    selectBatchItem(firstResult.idx);
    showToast(`Batch finished! Displaying ${state.folderFiles[firstResult.idx].name}`, 'success');
  } else {
    showToast(`Batch completed (${completedCount}/${checkedFiles.length})`, 'info');
  }
}

function uploadWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.responseType = 'blob';
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response);
      } else {
        reject(new Error(`Server returned ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error('Network error'));
    xhr.send(formData);
  });
}

// ═══════════════════════════════════════════ Layer rendering ════
function renderLayers(geojsonData) {
  // Clear existing
  Object.values(state.layers).forEach(l => map.removeLayer(l));
  state.layers = {};
  state.geojsonData = geojsonData;

  const bounds = [];

  for (const [cls, cfg] of Object.entries(CLASS_CONFIG)) {
    const data = geojsonData[cls];
    if (!data || !data.features || data.features.length === 0) continue;

    const layer = L.geoJSON(data, {
      style: {
        color: cfg.color,
        fillColor: cfg.fillColor,
        fillOpacity: cfg.fillOpacity * state.featureOpacity,
        weight: cfg.weight,
        opacity: 0.85,
      },
      onEachFeature: (feature, featureLayer) => {
        const p = feature.properties || {};
        let extraSolarHtml = '';
        if (cls === 'building') {
          const usable = p.solar_usable_m2 ?? Math.round((p.area_sq_m || 0) * 0.70);
          const kwp = p.solar_kwp ?? (usable * 0.15).toFixed(2);
          const annualKwh = p.annual_kwh ?? Math.round(kwp * 1450);
          const co2 = p.co2_offset_tons_yr ?? ((annualKwh * 0.82) / 1000).toFixed(2);
          extraSolarHtml = `
            <div class="popup-solar-section">
              <div class="popup-solar-title">
                <i data-lucide="sun" class="lucide-xs"></i> Rooftop Solar Potential
              </div>
              <div class="popup-row"><span class="popup-key">Capacity</span><span class="popup-val" style="color:var(--solar-amber);font-weight:700;">${kwp} kWp</span></div>
              <div class="popup-row"><span class="popup-key">Est. Generation</span><span class="popup-val">${formatNum(annualKwh)} kWh/yr</span></div>
              <div class="popup-row"><span class="popup-key">CO₂ Offset</span><span class="popup-val" style="color:var(--success);font-weight:600;">${co2} t/yr</span></div>
            </div>
          `;
        }
        featureLayer.bindPopup(`
          <div class="popup-title">${(p.class || cls).toUpperCase()}</div>
          <div class="popup-row"><span class="popup-key">Area</span><span class="popup-val">${formatNum(p.area_sq_m)} m²</span></div>
          <div class="popup-row"><span class="popup-key">Centroid Lat</span><span class="popup-val">${p.centroid_lat?.toFixed(6) ?? '-'}</span></div>
          <div class="popup-row"><span class="popup-key">Centroid Lon</span><span class="popup-val">${p.centroid_lon?.toFixed(6) ?? '-'}</span></div>
          ${extraSolarHtml}
          <div class="popup-row"><span class="popup-key">ID</span><span class="popup-val" style="font-size:0.65rem">${(p.feature_id || '').slice(0, 12)}...</span></div>
        `);
        featureLayer.on('popupopen', () => {
          if (window.lucide) lucide.createIcons();
        });
        featureLayer.on('mouseover', function () { this.setStyle({ weight: 3.5, fillOpacity: (cfg.fillOpacity * state.featureOpacity) + 0.2 }); });
        featureLayer.on('mouseout', function () { this.setStyle({ weight: cfg.weight, fillOpacity: cfg.fillOpacity * state.featureOpacity }); });
      },
    }).addTo(map);

    state.layers[cls] = layer;

    try {
      const b = layer.getBounds();
      if (b.isValid()) bounds.push(b);
    } catch(e) {}
  }

  // Fly to features
  if (bounds.length > 0) {
    let combined = bounds[0];
    bounds.slice(1).forEach(b => { combined = combined.extend(b); });
    map.flyToBounds(combined, { padding: [40, 40], duration: 1.2 });
  }
}

function toggleLayer(cls) {
  const layer = state.layers[cls];
  if (!layer) return;
  const checkbox = document.getElementById(CLASS_CONFIG[cls].layerCheckbox);
  if (checkbox.checked) {
    map.addLayer(layer);
  } else {
    map.removeLayer(layer);
  }
}

function updateOpacity(val) {
  state.featureOpacity = parseFloat(val);
  for (const [cls, layer] of Object.entries(state.layers)) {
    const cfg = CLASS_CONFIG[cls];
    layer.setStyle({ fillOpacity: cfg.fillOpacity * state.featureOpacity });
  }
}

// ═══════════════════════════════════════════ Stats ════
function updateStats(geojsonData) {
  let totalFeatures = 0;
  let totalArea = 0;
  const classStats = {};

  for (const [cls, data] of Object.entries(geojsonData)) {
    if (!data.features) continue;
    const count = data.features.length;
    const area = data.features.reduce((s, f) => s + (f.properties?.area_sq_m || 0), 0);
    classStats[cls] = { count, area };
    totalFeatures += count;
    totalArea += area;
  }

  // Update totals
  document.getElementById('total-features').textContent = formatNum(totalFeatures);
  document.getElementById('total-area').textContent = formatNum(Math.round(totalArea));

  // Solar Rooftop Potential calculations
  const buildings = geojsonData['building']?.features || [];
  let totalKwp = 0;
  let totalKwh = 0;
  let totalCo2 = 0;
  let totalUsableM2 = 0;

  buildings.forEach(f => {
    const p = f.properties || {};
    const area = p.area_sq_m || 0;
    const usable = p.solar_usable_m2 ?? (area * 0.70);
    const kwp = p.solar_kwp ?? (usable * 0.15);
    const kwh = p.annual_kwh ?? (kwp * 1450);
    const co2 = p.co2_offset_tons_yr ?? ((kwh * 0.82) / 1000);

    totalUsableM2 += usable;
    totalKwp += kwp;
    totalKwh += kwh;
    totalCo2 += co2;
  });

  const mwh = totalKwh / 1000;
  setEl('solar-kwp', totalKwp > 0 ? `${totalKwp.toFixed(1)} kWp` : '0 kWp');
  setEl('solar-mwh', mwh > 0 ? `${mwh.toFixed(1)} MWh` : '0 MWh');
  setEl('solar-co2', totalCo2 > 0 ? `${totalCo2.toFixed(1)} t` : '0 t');
  setEl('solar-usable', totalUsableM2 > 0 ? `${formatNum(Math.round(totalUsableM2))} m²` : '0 m²');

  // Update per-class
  for (const [cls, stats] of Object.entries(classStats)) {
    const pct = totalArea > 0 ? Math.round((stats.area / totalArea) * 100) : 0;
    const key = cls === 'building' ? 'building' : cls === 'road' ? 'road' : 'vegetation';

    setEl(`stat-${key}-count`, `${formatNum(stats.count)} features`);
    setEl(`stat-${key}-area`, formatNum(Math.round(stats.area)));
    setEl(`count-${key}s`, formatNum(stats.count));

    // Animate bar
    setTimeout(() => {
      const bar = document.getElementById(`bar-${key}`);
      const pctEl = document.getElementById(`pct-${key}`);
      if (bar) bar.style.width = `${pct}%`;
      if (pctEl) pctEl.textContent = `${pct}%`;
    }, 100);
  }
}

// ═══════════════════════════════════════════ Download ════
function downloadGeoJSON(cls) {
  const data = state.geojsonData[cls];
  if (!data) { showToast(`No ${cls} data loaded.`, 'error'); return; }
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${cls}.geojson`;
  a.click();
  URL.revokeObjectURL(url);
  showToast(`Downloaded ${cls}.geojson`, 'success');
}

// ═══════════════════════════════════════════ UI helpers ════
function showOverlay(visible) {
  document.getElementById('processing-overlay').style.display = visible ? 'flex' : 'none';
}

const STEPS = ['upload', 'tile', 'infer', 'vec'];
function setStep(activeStep) {
  const idx = STEPS.indexOf(activeStep);
  STEPS.forEach((s, i) => {
    const el = document.getElementById(`step-${s}`);
    if (!el) return;
    if (i < idx) el.className = 'step done';
    else if (i === idx) el.className = 'step active';
    else el.className = 'step';
  });
  const subtexts = {
    upload: 'Uploading orthophoto to API...',
    tile: 'Tiling into 256×256 patches...',
    infer: 'Running U-Net CNN inference...',
    vec: 'Vectorizing & building GeoJSON...',
  };
  setEl('processing-sub', subtexts[activeStep] || '');
}

function switchTab(tab) {
  ['dashboard', 'models', 'history', 'settings'].forEach(t => {
    const btn = document.getElementById(`tab-${t}`);
    const panel = document.getElementById(`panel-${t}`);
    if (btn) btn.className = 'nav-tab' + (t === tab ? ' active' : '');
    if (panel) panel.style.display = t === tab ? '' : 'none';
  });
  if (window.lucide) lucide.createIcons();
  // Invalidate map size when switching back to dashboard
  if (tab === 'dashboard' && map) {
    setTimeout(() => map.invalidateSize(), 50);
  }
  // Populate models page with live data
  if (tab === 'models') {
    populateModelsPage();
  }
}

async function populateModelsPage() {
  const endpoint = getApiEndpoint();
  try {
    const url = currentRunId
      ? `${endpoint}/model/accuracy?run_id=${encodeURIComponent(currentRunId)}`
      : `${endpoint}/model/accuracy`;
    const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return;
    const data = await res.json();

    // Hero stats
    setEl('mh-miou', `${(data.mean_iou * 100).toFixed(1)}%`);
    setEl('mh-dice', `${(data.mean_dice * 100).toFixed(1)}%`);
    setEl('mh-conf', `${(data.mean_confidence * 100).toFixed(1)}%`);

    // Per-class detailed metrics
    for (const cls of ['building', 'road', 'vegetation']) {
      const d = data.per_class?.[cls];
      if (!d) continue;

      // Feature counts
      setEl(`md-${cls}-count`, `${d.feature_count} features`);

      // Animate bars with delay for visual effect
      setTimeout(() => {
        const iouBar = document.getElementById(`md-bar-${cls}-iou`);
        const diceBar = document.getElementById(`md-bar-${cls}-dice`);
        const confBar = document.getElementById(`md-bar-${cls}-conf`);
        if (iouBar) iouBar.style.width = `${(d.iou * 100).toFixed(0)}%`;
        if (diceBar) diceBar.style.width = `${(d.dice * 100).toFixed(0)}%`;
        if (confBar) confBar.style.width = `${(d.confidence * 100).toFixed(0)}%`;
      }, 150);

      setEl(`md-val-${cls}-iou`, `${(d.iou * 100).toFixed(1)}%`);
      setEl(`md-val-${cls}-dice`, `${(d.dice * 100).toFixed(1)}%`);
      setEl(`md-val-${cls}-conf`, `${(d.confidence * 100).toFixed(1)}%`);
    }
  } catch (e) {
    console.warn('Could not populate models page:', e);
  }
}

function showToast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  const iconNames = { success: 'check-circle-2', error: 'alert-circle', info: 'info' };
  const iconName = iconNames[type] || 'info';
  toast.innerHTML = `<i data-lucide="${iconName}" class="lucide-sm"></i><span>${msg}</span>`;
  container.appendChild(toast);
  if (window.lucide) lucide.createIcons();
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(30px)';
    toast.style.transition = 'all 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ═══════════════════════════════════════════ History ════
const historyEntries = [];
function addHistoryEntry(filename, geojsonData) {
  const counts = Object.fromEntries(
    Object.entries(geojsonData).map(([cls, d]) => [cls, d.features?.length || 0])
  );
  historyEntries.unshift({ filename, timestamp: new Date().toLocaleString(), counts });
  renderHistory();
}
function renderHistory() {
  const panel = document.getElementById('panel-history');
  if (historyEntries.length === 0) return;
  panel.innerHTML = `
    <div style="padding:24px">
      <h2 style="font-size:1rem;font-weight:700;margin-bottom:16px;color:var(--text-secondary)">Processing History</h2>
      ${historyEntries.map(e => `
        <div class="card" style="margin-bottom:10px">
          <div style="display:flex;align-items:center;gap:12px">
            <span style="display:flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:8px;background:rgba(56,189,248,0.12);color:var(--accent-primary)">
              <i data-lucide="map-pin" class="lucide-md"></i>
            </span>
            <div style="flex:1">
              <div style="font-size:0.85rem;font-weight:600">${e.filename}</div>
              <div style="font-size:0.72rem;color:var(--text-muted)">${e.timestamp}</div>
            </div>
          </div>
          <div style="display:flex;gap:12px;margin-top:10px">
            ${Object.entries(e.counts).map(([cls,n]) => `
              <span style="font-size:0.72rem;padding:3px 10px;border-radius:99px;background:rgba(255,255,255,0.05);border:1px solid var(--border)">
                ${cls}: ${n}
              </span>`).join('')}
          </div>
        </div>
      `).join('')}
    </div>
  `;
  if (window.lucide) lucide.createIcons();
}

// ═══════════════════════════════════════════ Demo mode ════
/**
 * Loads synthetic demo data so the map is populated even without an API.
 * Generates random polygons near Bengaluru, India.
 */
function loadDemoData() {
  const center = [12.9716, 77.5946];

  function randomPoly(lat, lon, radiusDeg, nPoints = 6) {
    const coords = [];
    for (let i = 0; i < nPoints; i++) {
      const angle = (i / nPoints) * 2 * Math.PI;
      const r = radiusDeg * (0.6 + Math.random() * 0.4);
      coords.push([lon + r * Math.cos(angle), lat + r * Math.sin(angle)]);
    }
    coords.push(coords[0]);
    return { type: 'Polygon', coordinates: [coords] };
  }

  const geojsonData = {};

  const defs = [
    { cls: 'building', n: 120, r: 0.0008 },
    { cls: 'road',     n: 30,  r: 0.002  },
    { cls: 'vegetation', n: 50, r: 0.0015 },
  ];

  for (const { cls, n, r } of defs) {
    const features = Array.from({ length: n }, (_, i) => {
      const lat = center[0] + (Math.random() - 0.5) * 0.08;
      const lon = center[1] + (Math.random() - 0.5) * 0.08;
      const area = (r * 111000) ** 2 * Math.PI * (0.5 + Math.random());
      const props = {
        feature_id: `demo-${cls}-${i}`,
        class: cls,
        area_sq_m: Math.round(area),
        centroid_lat: parseFloat(lat.toFixed(6)),
        centroid_lon: parseFloat(lon.toFixed(6)),
      };
      if (cls === 'building') {
        const usable = Math.round(area * 0.70);
        const kwp = parseFloat((usable * 0.15).toFixed(2));
        const kwh = Math.round(kwp * 1450);
        const co2 = parseFloat(((kwh * 0.82) / 1000).toFixed(2));
        props.solar_usable_m2 = usable;
        props.solar_kwp = kwp;
        props.annual_kwh = kwh;
        props.co2_offset_tons_yr = co2;
      }
      return {
        type: 'Feature',
        geometry: randomPoly(lat, lon, r, cls === 'road' ? 4 : 6),
        properties: props,
      };
    });
    geojsonData[cls] = { type: 'FeatureCollection', features };
  }

  renderLayers(geojsonData);
  updateStats(geojsonData);
  document.getElementById('map-empty').style.display = 'none';
  document.getElementById('download-row').style.display = 'flex';
  showToast('Demo data loaded — connect the API to use real orthophotos', 'info');
}

// ═══════════════════════════════════════════ Utilities ════
function setEl(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}
function formatNum(n) {
  if (n === null || n === undefined || n === '-') return '-';
  return Number(n).toLocaleString();
}
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ═══════════════════════════════════════════ Geo Heatmap ════
let heatmapLayer = null;
let currentRunId = 'odm_aukerman';

function toggleHeatmap() {
  const cb = document.getElementById('layer-heatmap');
  const opts = document.getElementById('heatmap-options');
  const countEl = document.getElementById('count-heatmap');

  if (cb.checked) {
    opts.style.display = 'block';
    countEl.textContent = '...';
    loadHeatmapFromFeatures();
  } else {
    opts.style.display = 'none';
    countEl.textContent = 'off';
    if (heatmapLayer) {
      map.removeLayer(heatmapLayer);
      heatmapLayer = null;
    }
  }
}

function loadHeatmapFromFeatures() {
  const countEl = document.getElementById('count-heatmap');
  const intensityBy = document.getElementById('heatmap-intensity')?.value || 'area';
  const radius = parseInt(document.getElementById('heatmap-radius')?.value || '25');

  // Build heatmap points from current geojsonData (client-side, no API call needed)
  const points = [];

  for (const [cls, data] of Object.entries(state.geojsonData)) {
    if (!data || !data.features) continue;
    for (const feat of data.features) {
      const p = feat.properties || {};
      const lat = p.centroid_lat;
      const lon = p.centroid_lon;
      if (lat == null || lon == null) continue;

      let intensity = 1;
      if (intensityBy === 'area') {
        intensity = p.area_sq_m || 1;
      } else if (intensityBy === 'solar_kwp' && cls === 'building') {
        intensity = p.solar_kwp || (p.area_sq_m || 0) * 0.15;
      } else if (intensityBy === 'solar_kwp') {
        intensity = p.area_sq_m || 1;
      } else {
        intensity = 1;
      }
      points.push([lat, lon, intensity]);
    }
  }

  // Normalize intensities
  if (points.length > 0) {
    const maxI = Math.max(...points.map(p => p[2])) || 1;
    points.forEach(p => { p[2] = Math.max(0.1, p[2] / maxI); });
  }

  // Remove existing heatmap
  if (heatmapLayer) {
    map.removeLayer(heatmapLayer);
  }

  if (points.length > 0 && typeof L.heatLayer === 'function') {
    heatmapLayer = L.heatLayer(points, {
      radius: radius,
      blur: 20,
      maxZoom: 20,
      max: 1.0,
      gradient: {
        0.0: '#000080',
        0.25: '#0000ff',
        0.5: '#00ff00',
        0.75: '#ffff00',
        1.0: '#ff0000',
      },
    }).addTo(map);
    countEl.textContent = `${points.length} pts`;
  } else {
    countEl.textContent = '0 pts';
  }
}

function refreshHeatmap() {
  const cb = document.getElementById('layer-heatmap');
  if (cb && cb.checked) {
    loadHeatmapFromFeatures();
  }
}

function updateHeatmapRadius(val) {
  if (heatmapLayer) {
    heatmapLayer.setOptions({ radius: parseInt(val) });
    heatmapLayer.redraw();
  }
}

// ═══════════════════════════════════════════ Model Accuracy ════
async function fetchModelAccuracy(runId) {
  const endpoint = getApiEndpoint();
  const badge = document.getElementById('accuracy-badge');

  try {
    const url = runId
      ? `${endpoint}/model/accuracy?run_id=${encodeURIComponent(runId)}`
      : `${endpoint}/model/accuracy`;
    const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Update gauges
    animateGauge('gauge-iou-arc', 'gauge-iou-val', data.mean_iou);
    animateGauge('gauge-dice-arc', 'gauge-dice-val', data.mean_dice);
    animateGauge('gauge-conf-arc', 'gauge-conf-val', data.mean_confidence);

    // Update per-class bars
    for (const cls of ['building', 'road', 'vegetation']) {
      const clsData = data.per_class?.[cls];
      if (clsData) {
        const bar = document.getElementById(`acc-bar-${cls}`);
        const val = document.getElementById(`acc-val-${cls}`);
        if (bar) bar.style.width = `${Math.round(clsData.iou * 100)}%`;
        if (val) val.textContent = `${(clsData.iou * 100).toFixed(1)}%`;
      }
    }

    // Update model info
    setEl('acc-model-arch', data.model_architecture ? `Arch: ${data.model_architecture}` : '-');
    setEl('acc-model-encoder', data.encoder ? `Enc: ${data.encoder}` : '-');

    // Update badge
    const overallPct = Math.round(data.mean_confidence * 100);
    if (badge) {
      badge.textContent = `${overallPct}% Conf`;
      badge.style.background = overallPct > 75
        ? 'rgba(16,185,129,0.18)' : overallPct > 50
          ? 'rgba(245,158,11,0.18)' : 'rgba(239,68,68,0.18)';
      badge.style.color = overallPct > 75
        ? '#10b981' : overallPct > 50
          ? '#f59e0b' : '#ef4444';
    }

  } catch (err) {
    console.warn('Could not fetch accuracy:', err);
    if (badge) {
      badge.textContent = 'Unavailable';
      badge.style.background = 'rgba(100,116,139,0.18)';
      badge.style.color = '#64748b';
    }
  }
}

function animateGauge(arcId, valId, value) {
  const arc = document.getElementById(arcId);
  const valEl = document.getElementById(valId);
  if (!arc || !valEl) return;

  // Semi-circle arc length (~94.25 for r=30 half-circle)
  const totalArc = 94.25;
  const targetDash = value * totalArc;

  // Animate
  let current = 0;
  const step = targetDash / 30;
  const interval = setInterval(() => {
    current = Math.min(current + step, targetDash);
    arc.setAttribute('stroke-dasharray', `${current} ${totalArc}`);
    valEl.textContent = `${Math.round((current / totalArc) * 100)}%`;
    if (current >= targetDash) clearInterval(interval);
  }, 20);
}

// ═══════════════════════════════════════════ Live Dataset Loading ════
async function loadProcessedRun(runId) {
  const endpoint = getApiEndpoint();
  currentRunId = runId;
  showToast(`Loading dataset: ${runId}...`, 'info');
  
  const select = document.getElementById('dataset-select');
  if (select && select.value !== runId) {
    select.value = runId;
  }

  try {
    const geojsonData = {};
    const classes = ['building', 'road', 'vegetation'];
    let totalFeats = 0;

    for (const cls of classes) {
      try {
        const res = await fetch(`${endpoint}/outputs/${runId}/${cls}.geojson`);
        if (res.ok) {
          const data = await res.json();
          geojsonData[cls] = data;
          totalFeats += (data.features || []).length;
        }
      } catch (e) {
        console.warn(`Layer ${cls} could not be loaded:`, e);
      }
    }

    renderLayers(geojsonData);
    updateStats(geojsonData);

    const emptyOverlay = document.getElementById('map-empty');
    if (emptyOverlay) emptyOverlay.style.display = 'none';
    const downloadRow = document.getElementById('download-row');
    if (downloadRow) downloadRow.style.display = 'flex';

    showToast(`Loaded ${runId} (${totalFeats} features on map)!`, 'success');

    // Auto-fetch accuracy for this run
    fetchModelAccuracy(runId);

    // Refresh heatmap if active
    const heatCb = document.getElementById('layer-heatmap');
    if (heatCb && heatCb.checked) {
      loadHeatmapFromFeatures();
    }
  } catch (err) {
    console.error('Failed to load dataset:', err);
    showToast(`Error loading dataset: ${err.message}`, 'error');
  }
}

function onDatasetSelectChange() {
  const select = document.getElementById('dataset-select');
  if (select && select.value) {
    loadProcessedRun(select.value);
  }
}

async function fetchRunsAndPopulate() {
  const endpoint = getApiEndpoint();
  const select = document.getElementById('dataset-select');
  if (!select) return;

  try {
    const res = await fetch(`${endpoint}/runs`);
    if (res.ok) {
      const data = await res.json();
      const runs = data.runs || [];
      if (runs.length > 0) {
        select.innerHTML = runs.map(r => `
          <option value="${r.id}">${r.name || r.id}</option>
        `).join('');
      }
    }
  } catch (err) {
    console.warn('Could not fetch runs:', err);
  }
}

// ═══════════════════════════════════════════ Global Window Exports ════
Object.assign(window, {
  setUploadMode,
  updateProcessButton,
  onFolderSelect,
  scanServerFolder,
  renderFolderFileList,
  toggleFolderFile,
  toggleAllFolderFiles,
  selectBatchItem,
  onBatchActiveFileChange,
  handleProcessClick,
  processFile,
  processFolderBatch,
  setFile,
  clearFile,
  onFileSelect,
  onDragOver,
  onDragLeave,
  onDrop,
  switchTab,
  toggleLayer,
  updateOpacity,
  downloadGeoJSON,
  changeBasemap,
  loadDemoData,
  loadProcessedRun,
  onDatasetSelectChange,
  toggleHeatmap,
  refreshHeatmap,
  updateHeatmapRadius,
  fetchModelAccuracy,
});

// ═══════════════════════════════════════════ Init ════
document.addEventListener('DOMContentLoaded', async () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  initMap();
  checkApiStatus();

  // Direct event bindings for robustness
  document.getElementById('mode-single-btn')?.addEventListener('click', () => setUploadMode('single'));
  document.getElementById('mode-folder-btn')?.addEventListener('click', () => setUploadMode('folder'));
  document.getElementById('btn-scan')?.addEventListener('click', scanServerFolder);
  document.getElementById('btn-process')?.addEventListener('click', handleProcessClick);

  // Re-check API status when endpoint changes
  document.getElementById('api-endpoint')?.addEventListener('change', checkApiStatus);
  // Poll API status every 15s
  setInterval(checkApiStatus, 15000);

  // Fetch available server runs and auto-load the latest run
  try {
    await fetchRunsAndPopulate();
    // Automatically load Aukerman (or first available run) so the live view is instantly populated
    await loadProcessedRun('odm_aukerman');
  } catch (e) {
    console.warn('Could not auto-load run:', e);
  }
});

