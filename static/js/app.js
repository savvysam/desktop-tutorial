let selectedFile = null;

const fileInput     = document.getElementById('fileInput');
const dropZone      = document.getElementById('dropZone');
const previewImg    = document.getElementById('previewImg');
const uploadIcon    = document.getElementById('uploadIcon');
const analyzeSection = document.getElementById('analyzeSection');
const loading       = document.getElementById('loading');
const errorMsg      = document.getElementById('errorMsg');
const resultsSection = document.getElementById('resultsSection');
const dropText      = document.querySelector('.drop-text');
const dropSubtext   = document.getElementById('dropSubtext');

fileInput.addEventListener('change', e => { if (e.target.files[0]) loadPreview(e.target.files[0]); });
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) loadPreview(e.dataTransfer.files[0]);
});
dropZone.addEventListener('click', e => { if (!e.target.closest('.btn-secondary')) fileInput.click(); });

function loadPreview(file) {
    if (!file.type.startsWith('image/')) { showError('Please upload an image file.'); return; }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        previewImg.style.display = 'block';
        uploadIcon.style.display = 'none';
        dropText.textContent = file.name;
        dropSubtext.style.display = 'none';
        analyzeSection.style.display = 'flex';
        hideError();
        resultsSection.style.display = 'none';
        resultsSection.innerHTML = '';
    };
    reader.readAsDataURL(file);
}

function resetPhoto() {
    selectedFile = null;
    previewImg.style.display = 'none'; previewImg.src = '';
    uploadIcon.style.display = 'block';
    dropText.textContent = 'Drag & drop a clear face photo here';
    dropSubtext.style.display = 'block';
    analyzeSection.style.display = 'none';
    resultsSection.style.display = 'none'; resultsSection.innerHTML = '';
    fileInput.value = ''; hideError();
}

function showError(msg) { errorMsg.textContent = msg; errorMsg.style.display = 'block'; }
function hideError() { errorMsg.style.display = 'none'; }

async function analyze() {
    if (!selectedFile) return;
    hideError();
    analyzeSection.style.display = 'none';
    loading.style.display = 'block';
    const formData = new FormData();
    formData.append('photo', selectedFile);
    try {
        const res  = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) { showError(data.error); analyzeSection.style.display = 'flex'; return; }
        renderResults(data);
    } catch {
        showError('An error occurred. Please try again.');
        analyzeSection.style.display = 'flex';
    } finally {
        loading.style.display = 'none';
    }
}

function modeBadge(mode) {
    if (mode === 'ml_model_finetune') return `<span class="mode-note mode-note--finetune">ML Model (Fine-tuned)</span>`;
    if (mode === 'ml_model')          return `<span class="mode-note mode-note--ml">ML Model (SVM Head)</span>`;
    if (mode === 'color_theory')      return `<span class="mode-note mode-note--theory">SIVC Colour Theory</span>`;
    return `<span class="mode-note mode-note--mock">Demo Mode</span>`;
}

function metricsStrip(metrics) {
    if (!metrics) return '';
    const subtoneClass = metrics.subtone === 'warm' ? 'dot-warm' : 'dot-cold';

    // Lip-season vote chip (from Colorinsight L2 distance voting)
    let lipVoteChip = '';
    if (metrics.lip_vote) {
        const lv = metrics.lip_vote;
        const topSeason = Object.entries(lv).sort((a, b) => b[1] - a[1])[0];
        const seasonEmoji = { spring: '🌸', summer: '☀️', autumn: '🍂', winter: '❄️' };
        lipVoteChip = `
        <div class="metric-chip">
            <span class="label">Lip signal</span>
            <span class="value">${seasonEmoji[topSeason[0]] || ''} ${topSeason[0]} ${Math.round(topSeason[1] * 100)}%</span>
        </div>`;
    }

    return `
    <div class="metrics-strip">
        <div class="metric-chip">
            <div class="dot ${subtoneClass}"></div>
            <span class="label">Subtone</span>
            <span class="value">${metrics.subtone}</span>
        </div>
        <div class="metric-chip">
            <span class="label">Intensity</span>
            <span class="value">${Math.round(metrics.intensity * 100)}%</span>
        </div>
        <div class="metric-chip">
            <span class="label">Value</span>
            <span class="value">${Math.round(metrics.value * 100)}%</span>
        </div>
        <div class="metric-chip">
            <span class="label">Contrast</span>
            <span class="value">${Math.round(metrics.contrast * 100)}%</span>
        </div>
        ${lipVoteChip}
    </div>`;
}

function dominantStrip(dc) {
    if (!dc) return '';
    const parts = [];
    const labels = { skin: 'Skin', hair: 'Hair', eye: 'Eye', lips: 'Lips' };
    for (const [key, hex] of Object.entries(dc)) {
        if (!hex) continue;
        parts.push(`
            <div class="dominant-chip">
                <div class="swatch-lg" style="background:${hex};"></div>
                <div class="chip-label">${labels[key] || key}</div>
            </div>`);
    }
    return parts.length ? `<div class="dominant-strip">${parts.join('')}</div>` : '';
}

function renderResults(data) {
    const { primary, secondary, predictions, image_data_url: imgUrl,
            analysis_mode: mode, metrics, dominant_colors: dc, explanation } = data;

    const bestName = secondary
        ? `${primary.palette.name} / ${secondary.palette.name}`
        : primary.palette.name;

    const secondaryBlock = secondary ? `
        <div class="primary-palette">
            <div class="face-img-wrapper">
                <img src="${imgUrl}" alt="face">
                <div class="overlay-layer" style="background:${secondary.palette.overlay};"></div>
            </div>
            <div class="primary-info">
                <span class="badge-secondary">Secondary</span>
                <h3>${secondary.palette.name}</h3>
                <p>${secondary.palette.description}</p>
                <div class="swatches">
                    ${secondary.palette.swatches.map(c=>`<div class="swatch" style="background:${c};"></div>`).join('')}
                </div>
            </div>
        </div>` : '';

    const gridItems = predictions.map(p => {
        const isH = p.palette.name === primary.palette.name ||
                    (secondary && p.palette.name === secondary.palette.name);
        const pct = Math.round(p.confidence * 100);
        return `
            <div class="palette-item ${isH ? 'highlighted' : ''}">
                <div class="palette-thumb">
                    <img src="${imgUrl}" alt="${p.palette.name}">
                    <div class="overlay-layer" style="background:${p.palette.overlay};"></div>
                </div>
                <div class="palette-info">
                    <div class="palette-name">${p.palette.name}</div>
                    <div class="palette-confidence">${pct}% match</div>
                    <div class="mini-swatches">
                        ${p.palette.swatches.map(c=>`<div class="mini-swatch" style="background:${c};"></div>`).join('')}
                    </div>
                </div>
            </div>`;
    }).join('');

    resultsSection.style.display = 'block';
    resultsSection.innerHTML = `
        <div class="results-card">
            <div class="best-match">
                <div class="best-match-label">Your Colour Season</div>
                <div class="best-match-name">${bestName}</div>
                <div>${modeBadge(mode)}</div>
            </div>

            ${metricsStrip(metrics)}
            ${dominantStrip(dc)}

            <p class="explanation">${explanation}</p>

            <div class="primary-palette">
                <div class="face-img-wrapper">
                    <img src="${imgUrl}" alt="face">
                    <div class="overlay-layer" style="background:${primary.palette.overlay};"></div>
                </div>
                <div class="primary-info">
                    ${secondary ? '<span class="badge-primary">Primary</span>' : ''}
                    <h3>${primary.palette.name}</h3>
                    <p>${primary.palette.description}</p>
                    <div class="swatches">
                        ${primary.palette.swatches.map(c=>`<div class="swatch" style="background:${c};"></div>`).join('')}
                    </div>
                </div>
            </div>
            ${secondaryBlock}

            <div class="section-title">All 12 Seasons</div>
            <div class="palette-grid">${gridItems}</div>

            <div class="result-actions">
                <button class="btn btn-primary" onclick="downloadResult('${bestName}')">Save Results</button>
                <button class="btn btn-outline" onclick="resetPhoto()">Try Another Photo</button>
            </div>
        </div>`;

    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function downloadResult(bestMatch) {
    const text = `My Colour Season: ${bestMatch}\n\nAnalyzed by Seasonal Colour Analysis App (armocromia 12-season SIVC algorithm)`;
    const blob = new Blob([text], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'my-colour-season.txt';
    a.click();
}
