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
  if (window.location && window.location.protocol && window.location.protocol.startsWith('http')) {
    const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    if (!isLocal) {
      if (el && (!el.value || el.value.includes('localhost') || el.value.includes('127.0.0.1'))) {
        el.value = window.location.origin;
      }
      return window.location.origin;
    }
  }
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
    const maskInput = document.getElementById('mask-input');
    if (maskInput?.files?.[0]) {
      formData.append('mask', maskInput.files[0]);
    }

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

    const runId = sanitizeRunId(state.file.name);
    currentRunId = runId;
    await fetchRunsAndPopulate();
    const select = document.getElementById('dataset-select');
    if (select) {
      const match = Array.from(select.options).find(o => o.value === runId);
      if (match) select.value = runId;
    }
    fetchModelAccuracy(runId);

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
        const blob = xhr.response;
        if (blob && typeof blob.text === 'function') {
          blob.text().then((t) => {
            try {
              const parsed = JSON.parse(t);
              reject(new Error(parsed.detail || `Server returned ${xhr.status}`));
            } catch {
              reject(new Error(t || `Server returned ${xhr.status}`));
            }
          }).catch(() => reject(new Error(`Server returned ${xhr.status}`)));
        } else {
          reject(new Error(`Server returned ${xhr.status}`));
        }
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
  ['dashboard', 'models', 'history', 'settings', 'analysis'].forEach(t => {
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
    populateLiveBenchmarkTable();
  }
  // Populate analysis run selector when switching to analysis tab
  if (tab === 'analysis') {
    populateAnalysisRunSelector();
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
    setEl('mh-miou', formatMetricPct(data.mean_iou));
    setEl('mh-dice', formatMetricPct(data.mean_dice));
    setEl('mh-conf', formatMetricPct(data.mean_confidence));

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
        if (iouBar) iouBar.style.width = `${metricBarWidth(d.iou)}%`;
        if (diceBar) diceBar.style.width = `${metricBarWidth(d.dice)}%`;
        if (confBar) confBar.style.width = `${metricBarWidth(d.confidence)}%`;
      }, 150);

      setEl(`md-val-${cls}-iou`, formatMetricPct(d.iou));
      setEl(`md-val-${cls}-dice`, formatMetricPct(d.dice));
      setEl(`md-val-${cls}-conf`, formatMetricPct(d.confidence));
    }
  } catch (e) {
    console.warn('Could not populate models page:', e);
  }
}

function formatMaybePct(value, alreadyPercent) {
  if (value == null || value === '' || Number.isNaN(Number(value))) return 'N/A';
  const n = Number(value);
  return `${(alreadyPercent ? n : n * 100).toFixed(1)}%`;
}

async function populateLiveBenchmarkTable() {
  const tbody = document.getElementById('live-benchmark-tbody');
  if (!tbody) return;
  const endpoint = getApiEndpoint();
  const rows = [];

  try {
    const url = currentRunId
      ? `${endpoint}/model/accuracy?run_id=${encodeURIComponent(currentRunId)}`
      : `${endpoint}/model/accuracy`;
    const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (res.ok) {
      const data = await res.json();
      if (data.mean_iou != null || data.metrics_source === 'test_split_evaluation' || data.metrics_source === 'ground_truth') {
        rows.push(`
          <tr class="bt-highlight" style="background: rgba(99,102,241,0.16); border-left: 3px solid #6366f1;">
            <td><strong>${data.model_architecture || 'CNN'} / ${data.encoder || ''}</strong></td>
            <td>${data.metrics_source || 'live'}</td>
            <td>${data.split || data.run_id || 'checkpoint'}</td>
            <td><strong>${formatMaybePct(data.mean_iou)}</strong></td>
            <td>${formatMaybePct(data.per_class?.building?.iou)}</td>
            <td>${formatMaybePct(data.per_class?.road?.iou)}</td>
            <td>${formatMaybePct(data.per_class?.vegetation?.iou)}</td>
            <td>${formatMaybePct(data.mean_f1)}</td>
            <td>—</td>
          </tr>`);
      }
    }
  } catch (e) {
    console.warn('Live checkpoint accuracy unavailable', e);
  }

  try {
    const res = await fetch(`${endpoint}/analysis/multi-model-accuracy`, { signal: AbortSignal.timeout(8000) });
    if (res.ok) {
      const report = await res.json();
      (report.models || []).forEach((m) => {
        rows.push(`
          <tr>
            <td><strong>${m.name}</strong></td>
            <td>${m.metrics_source || 'tiled_dataset_eval'}</td>
            <td>${report.epochs || ''} epoch train</td>
            <td><strong>${formatMaybePct(m.mean_iou_pct, true)}</strong></td>
            <td>${formatMaybePct(m.class_breakdown?.building?.iou, true)}</td>
            <td>${formatMaybePct(m.class_breakdown?.road?.iou, true)}</td>
            <td>${formatMaybePct(m.class_breakdown?.vegetation?.iou, true)}</td>
            <td>${formatMaybePct(m.mean_f1_pct, true)}</td>
            <td>${m.inference_fps != null ? `${m.inference_fps} fps` : 'N/A'}</td>
          </tr>`);
      });
    }
  } catch (e) {
    console.warn('Multi-model live accuracy unavailable', e);
  }

  tbody.innerHTML = rows.length
    ? rows.join('')
    : '<tr><td colspan="9" style="text-align:center;color:var(--text-muted);padding:16px;">No measured accuracy yet. Process an orthophoto with a mask, run evaluation/evaluate.py, or Train &amp; Benchmark on tiled labels.</td></tr>';
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

function sanitizeRunId(filename) {
  const stem = String(filename || 'run').replace(/\.[^/.]+$/, '');
  return stem.replace(/[^A-Za-z0-9_-]/g, '_') || `run_${Date.now()}`;
}

function formatMetricPct(value) {
  if (value == null || value === '' || Number.isNaN(Number(value))) return 'N/A';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function metricBarWidth(value) {
  if (value == null || Number.isNaN(Number(value))) return 0;
  return Math.max(0, Math.min(100, Number(value) * 100));
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
let currentRunId = '';

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

    // Update gauges (IoU/Dice only when ground truth exists)
    animateGauge('gauge-iou-arc', 'gauge-iou-val', data.mean_iou);
    animateGauge('gauge-dice-arc', 'gauge-dice-val', data.mean_dice);
    animateGauge('gauge-conf-arc', 'gauge-conf-val', data.mean_confidence);

    // Update per-class bars — IoU if available, else softmax confidence
    for (const cls of ['building', 'road', 'vegetation']) {
      const clsData = data.per_class?.[cls];
      if (clsData) {
        const metric = clsData.iou != null ? clsData.iou : clsData.confidence;
        const bar = document.getElementById(`acc-bar-${cls}`);
        const val = document.getElementById(`acc-val-${cls}`);
        if (bar) bar.style.width = `${metricBarWidth(metric).toFixed(0)}%`;
        if (val) val.textContent = formatMetricPct(metric);
      }
    }

    // Update model info
    const source = data.metrics_source === 'ground_truth' || data.has_ground_truth
      ? 'IoU vs mask'
      : data.metrics_source === 'test_split_evaluation'
        ? 'Test-split eval'
        : data.metrics_source === 'softmax_confidence'
          ? 'Softmax confidence'
          : 'No GT yet';
    setEl('acc-model-arch', data.model_architecture ? `Arch: ${data.model_architecture}` : '-');
    setEl('acc-model-encoder', source);

    const headline = data.mean_iou != null ? data.mean_iou : data.mean_confidence;
    if (badge) {
      if (headline == null) {
        badge.textContent = 'N/A';
        badge.style.background = 'rgba(100,116,139,0.18)';
        badge.style.color = '#64748b';
      } else {
        const overallPct = Math.round(Number(headline) * 100);
        badge.textContent = data.mean_iou != null ? `${overallPct}% mIoU` : `${overallPct}% Conf`;
        badge.style.background = overallPct > 75
          ? 'rgba(16,185,129,0.18)' : overallPct > 50
            ? 'rgba(245,158,11,0.18)' : 'rgba(239,68,68,0.18)';
        badge.style.color = overallPct > 75
          ? '#10b981' : overallPct > 50
            ? '#f59e0b' : '#ef4444';
      }
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

  if (value == null || Number.isNaN(Number(value))) {
    arc.setAttribute('stroke-dasharray', `0 94.25`);
    valEl.textContent = 'N/A';
    return;
  }

  // Semi-circle arc length (~94.25 for r=30 half-circle)
  const totalArc = 94.25;
  const targetDash = Number(value) * totalArc;

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
        select.innerHTML = '<option value="">— Select a live run —</option>' + runs.map(r => `
          <option value="${r.id}">${r.name || r.id}</option>
        `).join('');
        const preferred = currentRunId && runs.some(r => r.id === currentRunId)
          ? currentRunId
          : runs[0].id;
        select.value = preferred;
      } else {
        select.innerHTML = '<option value="">— Upload or process an orthophoto —</option>';
      }
    }
  } catch (err) {
    console.warn('Could not fetch runs:', err);
  }
}

// ═══════════════════════════════════════════ Analysis Module ════

function switchAnalysisSubtab(sub) {
  const dsBtn = document.getElementById('subtab-dataset');
  const prBtn = document.getElementById('subtab-prediction');
  const mmBtn = document.getElementById('subtab-multimodel');
  const resBtn = document.getElementById('subtab-resources');
  const bmBtn = document.getElementById('subtab-benchmarks');
  const dsPanel = document.getElementById('analysis-dataset-panel');
  const prPanel = document.getElementById('analysis-prediction-panel');
  const mmPanel = document.getElementById('analysis-multimodel-panel');
  const resPanel = document.getElementById('analysis-resources-panel');
  const bmPanel = document.getElementById('analysis-benchmarks-panel');

  // Reset active classes
  dsBtn?.classList.remove('active');
  prBtn?.classList.remove('active');
  mmBtn?.classList.remove('active');
  resBtn?.classList.remove('active');
  bmBtn?.classList.remove('active');
  if (dsPanel) dsPanel.style.display = 'none';
  if (prPanel) prPanel.style.display = 'none';
  if (mmPanel) mmPanel.style.display = 'none';
  if (resPanel) resPanel.style.display = 'none';
  if (bmPanel) bmPanel.style.display = 'none';

  if (sub === 'dataset') {
    dsBtn?.classList.add('active');
    if (dsPanel) dsPanel.style.display = '';
  } else if (sub === 'prediction') {
    prBtn?.classList.add('active');
    if (prPanel) prPanel.style.display = '';
    populateAnalysisRunSelector();
  } else if (sub === 'multimodel') {
    mmBtn?.classList.add('active');
    if (mmPanel) mmPanel.style.display = '';
    loadMultiModelAccuracy();
  } else if (sub === 'resources') {
    resBtn?.classList.add('active');
    if (resPanel) resPanel.style.display = '';
    loadDataResources();
  } else if (sub === 'benchmarks') {
    bmBtn?.classList.add('active');
    if (bmPanel) bmPanel.style.display = '';
    loadBenchmarkDatasets();
  }
  if (window.lucide) lucide.createIcons();
}

// ═══════════════════════════════════════════ Benchmark Datasets & Outputs Catalog ════

async function loadBenchmarkDatasets() {
  const statusEl = document.getElementById('benchmarks-status');
  const tableBody = document.getElementById('benchmarks-table-body');
  const outputsBody = document.getElementById('outputs-table-body');
  const cardsContainer = document.getElementById('benchmarks-detail-cards');
  const endpoint = getApiEndpoint();

  if (statusEl) {
    statusEl.textContent = 'Loading benchmark datasets & outputs...';
    statusEl.style.color = '';
  }

  try {
    const [resDatasets, resRuns] = await Promise.all([
      fetch(`${endpoint}/datasets`, { signal: AbortSignal.timeout(10000) }),
      fetch(`${endpoint}/runs`, { signal: AbortSignal.timeout(10000) })
    ]);

    if (!resDatasets.ok) throw new Error(`Datasets API error: ${resDatasets.status}`);
    const data = await resDatasets.json();
    const runsData = resRuns.ok ? await resRuns.json() : { runs: [] };
    const runs = runsData.runs || [];

    // KPI cards
    setEl('bm-total-datasets', data.total_datasets ?? 0);
    setEl('bm-available', data.available_locally ?? 0);
    setEl('bm-total-tiles', data.total_tiles ?? 0);
    setEl('bm-outputs-count', runs.length);

    // Benchmarks table
    if (tableBody) {
      if (!data.catalog || data.catalog.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:16px;">No datasets found in catalog.</td></tr>';
      } else {
        tableBody.innerHTML = data.catalog.map(d => {
          const isAvail = d.local_available;
          const statusBadge = isAvail
            ? `<span style="font-size:0.72rem;padding:3px 8px;border-radius:99px;background:rgba(34,197,94,0.15);color:#22c55e;font-weight:700;">✓ Available</span>`
            : `<span style="font-size:0.72rem;padding:3px 8px;border-radius:99px;background:rgba(148,163,184,0.15);color:#94a3b8;font-weight:600;">Not Downloaded</span>`;

          const tileCount = (d.tile_count_images > 0)
            ? `<span style="color:#38bdf8;font-weight:700;">${d.tile_count_images}</span> tiles <span style="font-size:0.7rem;color:var(--text-muted);">(${d.tile_count_masks} masks)</span>`
            : `<span style="color:var(--text-muted);">0 tiles</span>`;

          const classPills = (d.classes || []).slice(0, 4).map(c =>
            `<span style="font-size:0.68rem;padding:1px 5px;background:rgba(255,255,255,0.06);border-radius:4px;color:var(--text);">${c}</span>`
          ).join(' ') + (d.classes && d.classes.length > 4 ? ` <span style="font-size:0.68rem;color:var(--text-muted);">+${d.classes.length - 4}</span>` : '');

          let actionBtn = '';
          if (isAvail) {
            actionBtn = `
              <button class="btn-ghost" style="padding:3px 8px;font-size:0.72rem;color:#38bdf8;" onclick="toggleDatasetCard('${d.id}')">
                <i data-lucide="info" style="width:12px;height:12px;"></i> Inspect
              </button>
            `;
          } else {
            actionBtn = `
              <button class="btn-ghost" style="padding:3px 8px;font-size:0.72rem;color:#f59e0b;" onclick="toggleDatasetCard('${d.id}')">
                <i data-lucide="download" style="width:12px;height:12px;"></i> Setup Info
              </button>
            `;
          }

          if (d.access_url) {
            actionBtn += `
              <a href="${d.access_url}" target="_blank" rel="noopener noreferrer" class="btn-ghost" style="padding:3px 8px;font-size:0.72rem;text-decoration:none;margin-left:4px;">
                <i data-lucide="external-link" style="width:12px;height:12px;"></i> Link
              </a>
            `;
          }

          return `
            <tr>
              <td>
                <div style="font-weight:700;color:var(--text);">${d.name}</div>
                <div style="font-size:0.72rem;color:var(--text-muted);max-width:280px;line-height:1.3;margin-top:2px;">${d.description}</div>
              </td>
              <td><span style="font-size:0.78rem;font-weight:600;">${d.platform}</span><br><span style="font-size:0.7rem;color:var(--text-muted);">${d.sensor}</span></td>
              <td><code style="font-size:0.75rem;">${d.resolution_gsd}</code></td>
              <td><div style="display:flex;flex-wrap:wrap;gap:3px;max-width:200px;">${classPills}</div></td>
              <td>${statusBadge}</td>
              <td>${tileCount}</td>
              <td style="white-space:nowrap;">${actionBtn}</td>
            </tr>
          `;
        }).join('');
      }
    }

    // Outputs table (all output displayed on website)
    if (outputsBody) {
      if (runs.length === 0) {
        outputsBody.innerHTML = '<tr><td colspan="5" style="text-align:center;padding:16px;">No output runs found yet. Run an inference job or select a demo orthophoto.</td></tr>';
      } else {
        outputsBody.innerHTML = runs.map(r => {
          const runId = r.id;
          const runName = r.name || runId;
          const features = (r.classes || ['building', 'road', 'vegetation']).map(c => {
            const color = c === 'building' ? '#ef4444' : c === 'road' ? '#a855f7' : '#22c55e';
            return `<span style="font-size:0.7rem;padding:2px 6px;border-radius:4px;background:${color}22;color:${color};font-weight:600;">${c}</span>`;
          }).join(' ');

          const metrics = r.metrics;
          let metricStr = '—';
          if (metrics) {
            const miou = metrics.mean_iou ?? metrics.mIoU;
            const f1 = metrics.mean_dice ?? metrics.f1_score;
            if (miou !== undefined) {
              metricStr = `<span style="color:#22c55e;font-weight:700;">${(miou * 100).toFixed(1)}% mIoU</span>`;
              if (f1 !== undefined) {
                metricStr += ` <span style="color:var(--text-muted);font-size:0.7rem;">(F1: ${(f1 * 100).toFixed(1)}%)</span>`;
              }
            }
          }

          return `
            <tr>
              <td>
                <div style="font-weight:700;color:#38bdf8;">${runName}</div>
                <div style="font-size:0.7rem;color:var(--text-muted);"><code style="font-size:0.68rem;">outputs/${runId}</code></div>
              </td>
              <td><div style="display:flex;flex-wrap:wrap;gap:4px;">${features}</div></td>
              <td>${metricStr}</td>
              <td><span style="font-size:0.75rem;color:var(--text-muted);">${r.created_at ? new Date(r.created_at * 1000).toLocaleString() : 'Ready'}</span></td>
              <td style="white-space:nowrap;">
                <button class="btn-primary" style="padding:4px 10px;font-size:0.72rem;display:inline-flex;align-items:center;gap:4px;" onclick="viewRunOnMap('${runId}')">
                  <i data-lucide="eye" style="width:12px;height:12px;"></i> View on Map
                </button>
                <button class="btn-ghost" style="padding:4px 8px;font-size:0.72rem;margin-left:4px;" onclick="viewRunInAnalysis('${runId}')">
                  <i data-lucide="bar-chart-2" style="width:12px;height:12px;"></i> Analytics
                </button>
              </td>
            </tr>
          `;
        }).join('');
      }
    }

    // Detail & setup cards
    if (cardsContainer && data.catalog) {
      cardsContainer.innerHTML = data.catalog.map(d => `
        <div class="analysis-split-card" id="dataset-card-${d.id}" style="margin-bottom:16px;display:none;border-left:4px solid ${d.local_available ? '#22c55e' : '#6366f1'};">
          <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:12px;">
            <div>
              <h4 style="margin:0 0 4px 0;font-size:1rem;color:var(--text);font-weight:700;">
                ${d.name} <span style="font-size:0.72rem;font-weight:normal;color:var(--text-muted);">(${d.spatial_coverage})</span>
              </h4>
              <p style="margin:0;font-size:0.78rem;color:var(--text-muted);line-height:1.4;">${d.description}</p>
            </div>
            <button class="btn-ghost" onclick="toggleDatasetCard('${d.id}')" style="padding:3px 8px;font-size:0.72rem;">Close</button>
          </div>

          <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:12px;margin-bottom:14px;background:rgba(255,255,255,0.02);padding:10px;border-radius:6px;">
            <div><span style="font-size:0.7rem;color:var(--text-muted);">Sensor & Platform:</span><br><strong style="font-size:0.8rem;">${d.sensor} (${d.platform})</strong></div>
            <div><span style="font-size:0.7rem;color:var(--text-muted);">GSD Resolution:</span><br><strong style="font-size:0.8rem;">${d.resolution_gsd}</strong></div>
            <div><span style="font-size:0.7rem;color:var(--text-muted);">Local Directory:</span><br><code style="font-size:0.75rem;">${d.local_dir}</code></div>
            <div><span style="font-size:0.7rem;color:var(--text-muted);">Adapter Script:</span><br><code style="font-size:0.75rem;">${d.adapter_script || 'Standard pipeline'}</code></div>
          </div>

          ${d.download_instructions ? `
            <div style="background:rgba(0,0,0,0.3);padding:10px 12px;border-radius:6px;margin-bottom:10px;border:1px solid var(--border);">
              <div style="font-size:0.7rem;color:#f59e0b;font-weight:700;margin-bottom:4px;display:flex;align-items:center;gap:4px;">
                <i data-lucide="terminal" style="width:12px;height:12px;"></i> Ingestion & Download Instructions:
              </div>
              <pre style="margin:0;font-family:monospace;font-size:0.75rem;color:#e2e8f0;white-space:pre-wrap;word-break:break-all;">${d.download_instructions}</pre>
            </div>
          ` : ''}

          <div style="display:flex;align-items:center;gap:8px;margin-top:10px;">
            ${d.adapter_script ? `
              <span style="font-size:0.72rem;color:var(--text-muted);">Run adapter: <code>python ${d.adapter_script}</code></span>
            ` : ''}
            ${d.access_url ? `
              <a href="${d.access_url}" target="_blank" rel="noopener noreferrer" class="btn-ghost" style="padding:3px 8px;font-size:0.72rem;text-decoration:none;margin-left:auto;">
                Visit Official Portal <i data-lucide="external-link" style="width:11px;height:11px;"></i>
              </a>
            ` : ''}
          </div>
        </div>
      `).join('');
    }

    if (statusEl) {
      statusEl.textContent = `✓ ${data.available_locally}/${data.total_datasets} datasets available (${runs.length} output runs loaded)`;
      statusEl.style.color = '#22c55e';
    }
  } catch (err) {
    console.error('Failed to load benchmark datasets:', err);
    if (statusEl) {
      statusEl.textContent = `✗ Failed to load: ${err.message}`;
      statusEl.style.color = '#ef4444';
    }
  }
  if (window.lucide) lucide.createIcons();
}

function toggleDatasetCard(id) {
  const card = document.getElementById(`dataset-card-${id}`);
  if (card) {
    const isHidden = card.style.display === 'none';
    card.style.display = isHidden ? '' : 'none';
    if (isHidden) {
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }
}

async function viewRunOnMap(runId) {
  switchTab('dashboard');
  const sel = document.getElementById('dataset-select');
  if (sel) {
    sel.value = runId;
  }
  await loadProcessedRun(runId);
  showToast(`Loaded output run '${runId}' onto dashboard map!`, 'success');
}

function viewRunInAnalysis(runId) {
  switchAnalysisSubtab('prediction');
  const sel = document.getElementById('analysis-run-select');
  if (sel) {
    sel.value = runId;
  }
  loadPredictionAnalysis(runId);
}

// ═══════════════════════════════════════════ Multi-Model Training & Benchmark ════

async function loadMultiModelAccuracy() {
  const statusEl = document.getElementById('multimodel-status');
  const tbody = document.getElementById('mm-leaderboard-body');
  const statsGrid = document.getElementById('mm-stats-grid');
  const chartCard = document.getElementById('mm-chart-card');
  const chartImg = document.getElementById('mm-chart-img');

  if (statusEl) { statusEl.textContent = 'Loading accuracy report...'; statusEl.style.color = ''; }

  const endpoint = getApiEndpoint();
  try {
    const res = await fetch(`${endpoint}/analysis/multi-model-accuracy`, { signal: AbortSignal.timeout(30000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const report = await res.json();
    const models = report.models || [];

    if (models.length > 0) {
      // Top stats
      const best = models[0];
      setEl('mm-top-model', best.name);
      setEl('mm-top-miou', `${best.mean_iou_pct}%`);
      
      const bestBldg = [...models].sort((a,b) => (b.class_breakdown?.building?.iou || 0) - (a.class_breakdown?.building?.iou || 0))[0];
      setEl('mm-top-bldg', `${bestBldg.class_breakdown?.building?.iou || 0}% (${bestBldg.name.split('(')[0].trim()})`);

      const bestRoad = [...models].sort((a,b) => (b.class_breakdown?.road?.iou || 0) - (a.class_breakdown?.road?.iou || 0))[0];
      setEl('mm-top-road', `${bestRoad.class_breakdown?.road?.iou || 0}% (${bestRoad.name.split('(')[0].trim()})`);

      const fastest = [...models].sort((a,b) => b.inference_fps - a.inference_fps)[0];
      setEl('mm-top-fps', `${fastest.inference_fps} fps (${fastest.name.split('(')[0].trim()})`);

      if (statsGrid) statsGrid.style.display = '';

      // Leaderboard table
      if (tbody) {
        tbody.innerHTML = models.map((m, idx) => {
          const rank = m.rank || (idx + 1);
          let rankBadge = `<span style="font-weight:700;color:var(--text-muted);">${rank}</span>`;
          if (rank === 1) rankBadge = `<span style="background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#000;font-size:0.7rem;font-weight:800;padding:2px 8px;border-radius:99px;">★ 1st</span>`;
          else if (rank === 2) rankBadge = `<span style="background:rgba(56,189,248,0.2);color:#38bdf8;font-size:0.7rem;font-weight:800;padding:2px 8px;border-radius:99px;">2nd</span>`;
          else if (rank === 3) rankBadge = `<span style="background:rgba(99,102,241,0.2);color:#6366f1;font-size:0.7rem;font-weight:800;padding:2px 8px;border-radius:99px;">3rd</span>`;

          const bldg = m.class_breakdown?.building?.iou || 0;
          const road = m.class_breakdown?.road?.iou || 0;
          const veg = m.class_breakdown?.vegetation?.iou || 0;

          return `
            <tr style="${rank === 1 ? 'background:rgba(99,102,241,0.12); font-weight:600;' : ''}">
              <td style="text-align:center;">${rankBadge}</td>
              <td>
                <strong>${m.name}</strong>
                <div style="font-size:0.7rem;color:var(--text-muted);">${m.description}</div>
              </td>
              <td><code>${m.parameters_m}M</code></td>
              <td><strong style="color:#10b981;font-size:0.95rem;">${m.mean_iou_pct}%</strong></td>
              <td><span style="color:#f59e0b;font-weight:600;">${bldg}%</span></td>
              <td><span style="color:#38bdf8;font-weight:600;">${road}%</span></td>
              <td><span style="color:#22c55e;font-weight:600;">${veg}%</span></td>
              <td>${m.mean_f1_pct}%</td>
              <td>${m.pixel_accuracy_pct}%</td>
              <td><span style="color:#ec4899;font-weight:600;">${m.inference_fps} fps</span></td>
            </tr>
          `;
        }).join('');
      }

      // Chart
      if (chartCard && chartImg) {
        chartImg.src = `${endpoint}/analysis/multi-model-chart?t=${Date.now()}`;
        chartCard.style.display = '';
      }

      if (statusEl) { statusEl.textContent = `✓ Evaluated ${models.length} architectures`; statusEl.style.color = '#22c55e'; }
    } else {
      if (tbody) tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;color:var(--text-muted);padding:16px;">No benchmark data available. Click "Train & Benchmark Models" to run.</td></tr>';
      if (statusEl) statusEl.textContent = '';
    }
  } catch (e) {
    console.error('Failed to load multi-model accuracy:', e);
    if (tbody) tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;color:var(--text-muted);padding:16px;">No live multi-model report. Train on tiled labelled data (not demo scores).</td></tr>';
    if (statusEl) {
      statusEl.textContent = 'No live report yet — train models on real tiles.';
      statusEl.style.color = '#94a3b8';
    }
  }
  if (window.lucide) lucide.createIcons();
}

async function executeMultiModelTraining() {
  const checkboxes = document.querySelectorAll('input[name="mm-arch"]:checked');
  const selectedModels = Array.from(checkboxes).map(cb => cb.value);
  const epochs = parseInt(document.getElementById('mm-epochs')?.value || '3', 10);
  const lr = parseFloat(document.getElementById('mm-lr')?.value || '0.0001');
  const batchSize = parseInt(document.getElementById('mm-batch-size')?.value || '4', 10);
  const statusEl = document.getElementById('multimodel-status');
  const btn = document.getElementById('btn-run-multimodel');

  if (selectedModels.length === 0) {
    showToast('Please select at least one architecture to train/benchmark.', 'warning');
    return;
  }

  if (btn) btn.disabled = true;
  if (statusEl) { statusEl.textContent = `Training ${selectedModels.length} models (${epochs} epochs)...`; statusEl.style.color = ''; }

  const endpoint = getApiEndpoint();
  const url = `${endpoint}/analysis/train-multi-model?models=${encodeURIComponent(selectedModels.join(','))}&epochs=${epochs}&lr=${lr}&batch_size=${batchSize}`;

  try {
    const res = await fetch(url, { method: 'POST', signal: AbortSignal.timeout(120000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const report = await res.json();
    showToast(`Multi-model training complete! Best: ${report.best_model} (${report.best_miou}% mIoU)`, 'success');
    await loadMultiModelAccuracy();
  } catch (e) {
    showToast(`Multi-model training error: ${e.message}`, 'error');
    if (statusEl) { statusEl.textContent = `✗ ${e.message}`; statusEl.style.color = '#ef4444'; }
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function populateAnalysisRunSelector() {
  const sel = document.getElementById('analysis-run-select');
  if (!sel) return;
  const endpoint = getApiEndpoint();
  try {
    const res = await fetch(`${endpoint}/runs`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return;
    const data = await res.json();
    const runs = data.runs || [];
    sel.innerHTML = '<option value="">— Select a run —</option>';
    runs.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.id;
      opt.textContent = r.name || r.id;
      sel.appendChild(opt);
    });

    if (runs.length > 0) {
      const preferred = currentRunId && runs.some(r => r.id === currentRunId)
        ? currentRunId
        : runs[0].id;
      sel.value = preferred;
      loadPredictionAnalysis(preferred);
    }
  } catch (e) {
    console.warn('Could not fetch runs for analysis:', e);
  }
}

// ═══════════════════════════════════════════ Data Resources Management ════

async function loadDataResources() {
  const statusEl = document.getElementById('resources-status');
  const tbody = document.getElementById('resources-table-body');
  const orthoSelect = document.getElementById('tile-ortho-select');
  const maskSelect = document.getElementById('tile-mask-select');

  if (statusEl) { statusEl.textContent = 'Refreshing...'; statusEl.style.color = ''; }

  const endpoint = getApiEndpoint();
  try {
    const res = await fetch(`${endpoint}/analysis/resources`, { signal: AbortSignal.timeout(10000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const items = data.resources || [];

    if (tbody) {
      if (items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:16px;">No resources found in workspace.</td></tr>';
      } else {
        tbody.innerHTML = items.map(item => {
          let badgeColor = '#38bdf8';
          let badgeBg = 'rgba(56,189,248,0.12)';
          if (item.category.includes('Mask')) { badgeColor = '#f59e0b'; badgeBg = 'rgba(245,158,11,0.12)'; }
          else if (item.category.includes('Tiled') || item.category.includes('Benchmark')) { badgeColor = '#22c55e'; badgeBg = 'rgba(34,197,94,0.12)'; }
          else if (item.category.includes('Model')) { badgeColor = '#a855f7'; badgeBg = 'rgba(168,85,247,0.12)'; }

          let sizeDetail = item.tile_count ? `${item.tile_count} tiles (${item.size_mb} MB)` : `${item.size_mb} MB`;

          let actionsHtml = '';
          if (item.category === 'Raw Orthophoto') {
            actionsHtml = `
              <button class="btn-ghost" style="padding:3px 8px;font-size:0.72rem;" onclick="selectForTiling('${item.name}')">
                <i data-lucide="scissors" style="width:12px;height:12px;"></i> Tile
              </button>
              <button class="btn-ghost" style="padding:3px 8px;font-size:0.72rem;color:#ef4444;" onclick="deleteDataResource('${item.name}')">
                <i data-lucide="trash-2" style="width:12px;height:12px;"></i>
              </button>
            `;
          } else if (item.category.includes('Tiled')) {
            actionsHtml = `
              <button class="btn-ghost" style="padding:3px 8px;font-size:0.72rem;color:#22c55e;" onclick="switchAnalysisSubtab('dataset'); loadDatasetAnalysis();">
                <i data-lucide="bar-chart-2" style="width:12px;height:12px;"></i> Analyze
              </button>
            `;
          } else {
            actionsHtml = `
              <button class="btn-ghost" style="padding:3px 8px;font-size:0.72rem;color:#ef4444;" onclick="deleteDataResource('${item.name}')">
                <i data-lucide="trash-2" style="width:12px;height:12px;"></i>
              </button>
            `;
          }

          return `
            <tr>
              <td><strong>${item.name}</strong></td>
              <td><span style="font-size:0.72rem;padding:2px 8px;border-radius:99px;background:${badgeBg};color:${badgeColor};font-weight:600;">${item.category}</span></td>
              <td><code style="font-size:0.75rem;color:var(--text-muted);">${item.path}</code></td>
              <td>${sizeDetail}</td>
              <td style="white-space:nowrap;">${actionsHtml}</td>
            </tr>
          `;
        }).join('');
      }
    }

    // Populate tiling dropdowns
    if (orthoSelect) {
      const rawOrthos = items.filter(i => i.category === 'Raw Orthophoto');
      orthoSelect.innerHTML = '<option value="">— Select from data/raw/ —</option>' + 
        rawOrthos.map(o => `<option value="${o.name}">${o.name} (${o.size_mb} MB)</option>`).join('');
    }

    if (maskSelect) {
      const rawMasks = items.filter(i => i.category === 'Ground Truth Mask');
      maskSelect.innerHTML = '<option value="">(Auto-detect matching mask)</option>' + 
        rawMasks.map(m => `<option value="${m.name}">${m.name} (${m.size_mb} MB)</option>`).join('');
    }

    if (statusEl) { statusEl.textContent = `✓ ${items.length} resource(s) active`; statusEl.style.color = '#22c55e'; }
  } catch (e) {
    console.error('Failed to load resources:', e);
    if (statusEl) { statusEl.textContent = `✗ Failed: ${e.message}`; statusEl.style.color = '#ef4444'; }
  }
  if (window.lucide) lucide.createIcons();
}

function selectForTiling(orthoName) {
  const sel = document.getElementById('tile-ortho-select');
  if (sel) {
    sel.value = orthoName;
    sel.scrollIntoView({ behavior: 'smooth', block: 'center' });
    showToast(`Selected '${orthoName}' for tiling.`);
  }
}

async function handleResourceUpload(event, type) {
  event.preventDefault();
  const endpoint = getApiEndpoint();
  const statusEl = document.getElementById('resources-status');

  const formData = new FormData();

  if (type === 'raw_ortho') {
    const orthoInput = document.getElementById('res-ortho-file');
    const maskInput = document.getElementById('res-mask-file');
    const btn = document.getElementById('btn-upload-raw');

    if (!orthoInput || !orthoInput.files || orthoInput.files.length === 0) {
      showToast('Please select a raw orthophoto file.', 'warning');
      return;
    }

    formData.append('file', orthoInput.files[0]);
    if (maskInput && maskInput.files && maskInput.files.length > 0) {
      formData.append('mask_file', maskInput.files[0]);
    }

    if (btn) btn.disabled = true;
    if (statusEl) { statusEl.textContent = 'Uploading orthophoto to data/raw/...'; statusEl.style.color = ''; }

    try {
      const res = await fetch(`${endpoint}/analysis/upload-resource?resource_type=raw_ortho`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      showToast(data.message || 'Orthophoto uploaded successfully!', 'success');
      document.getElementById('form-upload-raw')?.reset();
      await loadDataResources();
    } catch (e) {
      showToast(`Upload failed: ${e.message}`, 'error');
      if (statusEl) { statusEl.textContent = `✗ ${e.message}`; statusEl.style.color = '#ef4444'; }
    } finally {
      if (btn) btn.disabled = false;
    }

  } else if (type === 'dataset_zip') {
    const zipInput = document.getElementById('res-zip-file');
    const btn = document.getElementById('btn-upload-zip');

    if (!zipInput || !zipInput.files || zipInput.files.length === 0) {
      showToast('Please select a ZIP dataset archive.', 'warning');
      return;
    }

    formData.append('file', zipInput.files[0]);

    if (btn) btn.disabled = true;
    if (statusEl) { statusEl.textContent = 'Extracting ZIP archive into data/tiles/...'; statusEl.style.color = ''; }

    try {
      const res = await fetch(`${endpoint}/analysis/upload-resource?resource_type=dataset_zip`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      showToast(data.message || 'Dataset ZIP extracted successfully!', 'success');
      document.getElementById('form-upload-zip')?.reset();
      await loadDataResources();
    } catch (e) {
      showToast(`ZIP ingest failed: ${e.message}`, 'error');
      if (statusEl) { statusEl.textContent = `✗ ${e.message}`; statusEl.style.color = '#ef4444'; }
    } finally {
      if (btn) btn.disabled = false;
    }
  }
}

async function executeTilingProcess() {
  const orthoSelect = document.getElementById('tile-ortho-select');
  const maskSelect = document.getElementById('tile-mask-select');
  const sizeSelect = document.getElementById('tile-size-select');
  const overlapSelect = document.getElementById('tile-overlap-select');
  const statusEl = document.getElementById('tiling-status');
  const btn = document.getElementById('btn-run-tiling');

  const orthoName = orthoSelect?.value;
  if (!orthoName) {
    showToast('Please select an orthophoto to tile.', 'warning');
    return;
  }

  const maskName = maskSelect?.value || '';
  const tileSize = parseInt(sizeSelect?.value || '256', 10);
  const overlap = parseFloat(overlapSelect?.value || '0.15');

  if (btn) btn.disabled = true;
  if (statusEl) { statusEl.textContent = `Tiling '${orthoName}'...`; statusEl.style.color = ''; }

  const endpoint = getApiEndpoint();
  const url = `${endpoint}/analysis/tile-resource?ortho_name=${encodeURIComponent(orthoName)}&mask_name=${encodeURIComponent(maskName)}&tile_size=${tileSize}&overlap=${overlap}`;

  try {
    const res = await fetch(url, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    showToast(data.message || 'Tiling completed successfully!', 'success');
    if (statusEl) { statusEl.textContent = `✓ Generated ${data.total_tiles} tiles (${tileSize}x${tileSize})`; statusEl.style.color = '#22c55e'; }
    await loadDataResources();
  } catch (e) {
    showToast(`Tiling failed: ${e.message}`, 'error');
    if (statusEl) { statusEl.textContent = `✗ Failed: ${e.message}`; statusEl.style.color = '#ef4444'; }
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function deleteDataResource(filename) {
  if (!confirm(`Are you sure you want to delete '${filename}'?`)) return;
  const endpoint = getApiEndpoint();
  try {
    const res = await fetch(`${endpoint}/analysis/resources/${encodeURIComponent(filename)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    showToast(`Deleted '${filename}'`, 'info');
    await loadDataResources();
  } catch (e) {
    showToast(`Delete failed: ${e.message}`, 'error');
  }
}

async function loadDatasetAnalysis() {
  const statusEl = document.getElementById('dataset-analysis-status');
  const btn = document.getElementById('btn-run-dataset-analysis');
  if (btn) btn.disabled = true;
  if (statusEl) statusEl.textContent = 'Analyzing...';

  const endpoint = getApiEndpoint();
  try {
    const res = await fetch(`${endpoint}/analysis/dataset`, { signal: AbortSignal.timeout(30000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Populate stat cards
    const ts = data.tile_stats || {};
    setEl('ds-tile-count', ts.count || 0);
    setEl('ds-masks-found', ts.masks_found || 0);
    setEl('ds-total-size', `${ts.total_size_mb || 0} MB`);
    setEl('ds-imbalance', data.imbalance_ratio || '—');
    document.getElementById('dataset-stats-grid').style.display = '';

    // Draw class distribution chart
    drawClassDistChart(data);
    drawClassWeightsChart(data);
    document.getElementById('dataset-charts-row').style.display = '';

    // Split summary
    const ss = data.split_summary || {};
    drawSplitBars(ss);
    document.getElementById('dataset-split-card').style.display = '';

    // Load sample grid image
    const gridImg = document.getElementById('ds-sample-grid-img');
    if (gridImg) {
      gridImg.src = `${endpoint}/analysis/dataset/chart?t=${Date.now()}`;
      gridImg.onerror = () => { document.getElementById('dataset-sample-grid').style.display = 'none'; };
      gridImg.onload = () => { document.getElementById('dataset-sample-grid').style.display = ''; };
    }

    if (statusEl) statusEl.textContent = '✓ Analysis complete';
    if (statusEl) statusEl.style.color = '#22c55e';
  } catch (e) {
    console.error('Dataset analysis error:', e);
    if (statusEl) { statusEl.textContent = `✗ Failed: ${e.message}`; statusEl.style.color = '#ef4444'; }
  } finally {
    if (btn) btn.disabled = false;
  }
  if (window.lucide) lucide.createIcons();
}

async function loadPredictionAnalysis(runId) {
  if (!runId) return;
  const statusEl = document.getElementById('prediction-analysis-status');
  if (statusEl) { statusEl.textContent = 'Analyzing...'; statusEl.style.color = ''; }

  const endpoint = getApiEndpoint();
  try {
    const res = await fetch(`${endpoint}/analysis/prediction/${encodeURIComponent(runId)}`, {
      signal: AbortSignal.timeout(30000)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Stats cards
    setEl('pred-total-features', data.total_features || 0);
    setEl('pred-total-area', (data.total_mapped_area_m2 || 0).toLocaleString(undefined, { maximumFractionDigits: 1 }));
    setEl('pred-density', data.features_per_hectare || 0);
    setEl('pred-hectares', (data.total_mapped_area_hectares || 0).toFixed(4));
    document.getElementById('pred-stats-grid').style.display = '';

    // Charts
    drawLandUsePie(data);
    drawFeatureCountBars(data);
    document.getElementById('pred-charts-row').style.display = '';

    // Per-class insights
    const pc = data.per_class || {};
    const b = pc.building || {};
    setEl('pi-bldg-count', b.count || 0);
    setEl('pi-bldg-area', `${(b.total_area_m2 || 0).toLocaleString()} m²`);
    setEl('pi-bldg-mean', `${(b.mean_area_m2 || 0).toFixed(1)} m²`);
    setEl('pi-bldg-regularity', b.regularity || 'N/A');
    setEl('pi-bldg-compact', (b.mean_compactness || 0).toFixed(4));
    setEl('pi-bldg-solar', `${(b.total_solar_kwp || 0).toFixed(1)} kWp`);

    const r = pc.road || {};
    setEl('pi-road-count', r.count || 0);
    setEl('pi-road-area', `${(r.total_area_m2 || 0).toLocaleString()} m²`);
    setEl('pi-road-length', `${(r.total_road_length_m || 0).toFixed(1)} m`);
    setEl('pi-road-width', `${(r.avg_width_m || 0).toFixed(1)} m`);
    setEl('pi-road-segment', `${(r.mean_segment_length_m || 0).toFixed(1)} m`);

    const v = pc.vegetation || {};
    setEl('pi-veg-count', v.count || 0);
    setEl('pi-veg-canopy', `${(v.canopy_area_m2 || 0).toLocaleString()} m²`);
    setEl('pi-veg-frag', (v.fragmentation_index || 0).toFixed(4));
    setEl('pi-veg-density', `${(v.patch_density || 0).toFixed(1)} /ha`);
    setEl('pi-veg-mean', `${(v.mean_area_m2 || 0).toFixed(1)} m²`);

    document.getElementById('pred-insights-grid').style.display = '';
    if (statusEl) { statusEl.textContent = '✓ Analysis complete'; statusEl.style.color = '#22c55e'; }
  } catch (e) {
    console.error('Prediction analysis error:', e);
    if (statusEl) { statusEl.textContent = `✗ Failed: ${e.message}`; statusEl.style.color = '#ef4444'; }
  }
  if (window.lucide) lucide.createIcons();
}


// ─── Canvas chart helpers ───────────────────────────────────

function drawClassDistChart(data) {
  const canvas = document.getElementById('chart-class-dist');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const dist = data.class_distribution || {};
  const names = Object.keys(dist);
  const pcts = names.map(n => dist[n].percentage || 0);
  const colors = ['#64748b', '#ef4444', '#a855f7', '#22c55e'];

  const barW = Math.min(60, (W - 80) / names.length - 10);
  const maxVal = Math.max(...pcts, 1);
  const chartH = H - 60;
  const startX = 50;

  // Y axis
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(startX, 10); ctx.lineTo(startX, chartH + 10); ctx.stroke();

  names.forEach((name, i) => {
    const x = startX + 20 + i * (barW + 20);
    const barH = (pcts[i] / maxVal) * (chartH - 20);
    const y = chartH + 10 - barH;

    // Bar
    const grad = ctx.createLinearGradient(x, y, x, chartH + 10);
    grad.addColorStop(0, colors[i] || '#38bdf8');
    grad.addColorStop(1, (colors[i] || '#38bdf8') + '44');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, [4, 4, 0, 0]);
    ctx.fill();

    // Value label
    ctx.fillStyle = '#e2e8f0'; ctx.font = 'bold 11px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(`${pcts[i].toFixed(1)}%`, x + barW / 2, y - 6);

    // X label
    ctx.fillStyle = '#94a3b8'; ctx.font = '10px Inter, sans-serif';
    ctx.fillText(name, x + barW / 2, chartH + 28);
  });
}

function drawClassWeightsChart(data) {
  const canvas = document.getElementById('chart-class-weights');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const names = data.class_names || [];
  const weights = data.class_weights || [];
  const colors = ['#64748b', '#ef4444', '#a855f7', '#22c55e'];

  const barW = Math.min(60, (W - 80) / names.length - 10);
  const maxVal = Math.max(...weights, 1);
  const chartH = H - 60;
  const startX = 50;

  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(startX, 10); ctx.lineTo(startX, chartH + 10); ctx.stroke();

  // Reference line at 1.0
  const refY = chartH + 10 - (1.0 / maxVal) * (chartH - 20);
  ctx.strokeStyle = '#f59e0b44'; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(startX, refY); ctx.lineTo(W - 10, refY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#f59e0b'; ctx.font = '9px Inter, sans-serif'; ctx.textAlign = 'left';
  ctx.fillText('1.0 (balanced)', startX + 4, refY - 4);

  names.forEach((name, i) => {
    const x = startX + 20 + i * (barW + 20);
    const barH = (weights[i] / maxVal) * (chartH - 20);
    const y = chartH + 10 - barH;

    const grad = ctx.createLinearGradient(x, y, x, chartH + 10);
    grad.addColorStop(0, colors[i] || '#38bdf8');
    grad.addColorStop(1, (colors[i] || '#38bdf8') + '44');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, [4, 4, 0, 0]);
    ctx.fill();

    ctx.fillStyle = '#e2e8f0'; ctx.font = 'bold 11px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(weights[i].toFixed(2), x + barW / 2, y - 6);

    ctx.fillStyle = '#94a3b8'; ctx.font = '10px Inter, sans-serif';
    ctx.fillText(name, x + barW / 2, chartH + 28);
  });
}

function drawSplitBars(ss) {
  const container = document.getElementById('ds-split-bars');
  if (!container) return;
  const total = (ss.train || 0) + (ss.val || 0) + (ss.test || 0);
  if (total === 0) { container.innerHTML = '<span style="color:#94a3b8;">No data</span>'; return; }

  const pTrain = ((ss.train / total) * 100).toFixed(0);
  const pVal = ((ss.val / total) * 100).toFixed(0);
  const pTest = ((ss.test / total) * 100).toFixed(0);

  container.innerHTML = `
    <div class="split-bar-row">
      <div class="split-bar" style="flex:${ss.train};background:linear-gradient(135deg,#38bdf8,#0ea5e9);border-radius:6px 0 0 6px;">
        <span>Train ${pTrain}% (${ss.train})</span>
      </div>
      <div class="split-bar" style="flex:${ss.val};background:linear-gradient(135deg,#a855f7,#7c3aed);">
        <span>Val ${pVal}% (${ss.val})</span>
      </div>
      <div class="split-bar" style="flex:${ss.test};background:linear-gradient(135deg,#f59e0b,#d97706);border-radius:0 6px 6px 0;">
        <span>Test ${pTest}% (${ss.test})</span>
      </div>
    </div>
  `;
}

function drawLandUsePie(data) {
  const canvas = document.getElementById('chart-land-use');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const lu = data.land_use || {};
  const classes = ['building', 'road', 'vegetation'];
  const colors = ['#ef4444', '#a855f7', '#22c55e'];
  const labels = classes.map(c => `${c} ${(lu[c]?.percentage || 0).toFixed(1)}%`);
  const values = classes.map(c => lu[c]?.percentage || 0);
  const total = values.reduce((a, b) => a + b, 0);

  if (total === 0) {
    ctx.fillStyle = '#94a3b8'; ctx.font = '14px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('No features detected', W / 2, H / 2);
    return;
  }

  const cx = W / 2 - 40, cy = H / 2, radius = Math.min(W, H) / 2 - 40;
  let startAngle = -Math.PI / 2;

  values.forEach((val, i) => {
    const sliceAngle = (val / total) * 2 * Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, startAngle, startAngle + sliceAngle);
    ctx.closePath();
    ctx.fillStyle = colors[i];
    ctx.fill();
    ctx.strokeStyle = '#0f1117'; ctx.lineWidth = 3; ctx.stroke();
    startAngle += sliceAngle;
  });

  // Inner circle for donut effect
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 0.5, 0, 2 * Math.PI);
  ctx.fillStyle = '#14141f'; ctx.fill();

  // Total in center
  ctx.fillStyle = '#e2e8f0'; ctx.font = 'bold 16px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText(`${total.toFixed(0)}%`, cx, cy + 5);

  // Legend
  const legendX = W - 110;
  classes.forEach((cls, i) => {
    const ly = 40 + i * 30;
    ctx.fillStyle = colors[i];
    ctx.fillRect(legendX, ly, 12, 12);
    ctx.fillStyle = '#e2e8f0'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(labels[i], legendX + 18, ly + 10);
  });
}

function drawFeatureCountBars(data) {
  const canvas = document.getElementById('chart-feature-count');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const pc = data.per_class || {};
  const classes = ['building', 'road', 'vegetation'];
  const colors = ['#ef4444', '#a855f7', '#22c55e'];
  const counts = classes.map(c => pc[c]?.count || 0);

  const barW = Math.min(70, (W - 80) / classes.length - 20);
  const maxVal = Math.max(...counts, 1);
  const chartH = H - 60;
  const startX = 50;

  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(startX, 10); ctx.lineTo(startX, chartH + 10); ctx.stroke();

  classes.forEach((cls, i) => {
    const x = startX + 25 + i * (barW + 30);
    const barH = (counts[i] / maxVal) * (chartH - 20);
    const y = chartH + 10 - barH;

    const grad = ctx.createLinearGradient(x, y, x, chartH + 10);
    grad.addColorStop(0, colors[i]);
    grad.addColorStop(1, colors[i] + '44');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, [4, 4, 0, 0]);
    ctx.fill();

    ctx.fillStyle = '#e2e8f0'; ctx.font = 'bold 12px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(counts[i], x + barW / 2, y - 8);

    ctx.fillStyle = '#94a3b8'; ctx.font = '11px Inter, sans-serif';
    ctx.fillText(cls, x + barW / 2, chartH + 28);
  });
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
  loadProcessedRun,
  onDatasetSelectChange,
  toggleHeatmap,
  refreshHeatmap,
  updateHeatmapRadius,
  fetchModelAccuracy,
  switchAnalysisSubtab,
  loadDatasetAnalysis,
  loadPredictionAnalysis,
  loadMultiModelAccuracy,
  executeMultiModelTraining,
  loadDataResources,
  handleResourceUpload,
  executeTilingProcess,
  deleteDataResource,
  selectForTiling,
  loadBenchmarkDatasets,
  toggleDatasetCard,
  viewRunOnMap,
  viewRunInAnalysis,
});

// ═══════════════════════════════════════════ Init ════
async function initApp() {
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

  // Load the newest live inference run (no synthetic demo data)
  try {
    await fetchRunsAndPopulate();
    const select = document.getElementById('dataset-select');
    if (select && select.value) {
      await loadProcessedRun(select.value);
    } else {
      fetchModelAccuracy('');
    }
  } catch (e) {
    console.warn('Could not auto-load run:', e);
    fetchModelAccuracy('');
  }

  setInterval(async () => {
    const previous = currentRunId;
    await fetchRunsAndPopulate();
    const select = document.getElementById('dataset-select');
    const nextId = select && select.value;
    if (nextId && nextId !== previous) {
      await loadProcessedRun(nextId);
    } else if (currentRunId) {
      fetchModelAccuracy(currentRunId);
    } else {
      fetchModelAccuracy('');
    }
  }, 20000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

