/* frontend/app.js */

let allCalls = [];
let selectedCall = null;
let currentSort = 'date-desc';

// API Base URL helper
const API_URL = '';

// Helper to format timestamps
function formatDateTime(isoString) {
    if (!isoString) return '--';
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

// Helper to format duration
function formatDuration(seconds) {
    if (seconds === undefined || seconds === null) return '0.0s';
    return `${parseFloat(seconds).toFixed(1)}s`;
}

// Helper to format WER
function formatWER(wer) {
    if (wer === undefined || wer === null) return 'N/A';
    return `${(wer * 100).toFixed(1)}%`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Home Page Logic
// ─────────────────────────────────────────────────────────────────────────────
async function loadHomeStats() {
    try {
        const statsRes = await fetch(`${API_URL}/stats`);
        const stats = await statsRes.json();
        
        // Update stats row
        const avgWer = stats.avg_wer !== null ? `${(stats.avg_wer * 100).toFixed(1)}%` : 'N/A';
        document.getElementById('avg-wer-val').innerText = avgWer;
        document.getElementById('today-calls-count').innerText = `Processed Today: ${stats.total_calls} calls`;
        
        // Load recent calls
        const callsRes = await fetch(`${API_URL}/calls`);
        allCalls = await callsRes.json();
        renderRecentCalls();
    } catch (err) {
        console.error("Failed to load home page stats:", err);
    }
}

function renderRecentCalls() {
    const tbody = document.getElementById('recent-calls-body');
    if (!tbody) return;

    if (allCalls.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-secondary);">No calls processed yet. Click "Open App" to upload!</td></tr>`;
        return;
    }

    // Limit to top 5 recent calls
    const recent = allCalls.slice(0, 5);
    tbody.innerHTML = recent.map(call => {
        const dateStr = formatDateTime(call.processed_at);
        const durationStr = formatDuration(call.duration_seconds);
        const werStr = formatWER(call.wer_score);
        
        const sentimentClass = call.sentiment === 'Positive' ? 'badge-success' : (call.sentiment === 'Negative' ? 'badge-danger' : 'badge-warning');
        const alertHtml = call.flagged ? `<span class="badge badge-danger">FLAGGED</span>` : `<span class="badge badge-success">OK</span>`;
        
        return `
            <tr>
                <td>${dateStr}</td>
                <td style="font-weight: 500;">${call.audio_file}</td>
                <td>${durationStr}</td>
                <td><span class="badge badge-info">${call.asr_backend || 'whisper'}</span></td>
                <td style="font-weight: 600;">${werStr}</td>
                <td><span class="badge ${sentimentClass}">${call.sentiment || 'Neutral'}</span></td>
                <td>${alertHtml}</td>
            </tr>
        `;
    }).join('');
}

function filterRecentCalls() {
    const query = document.getElementById('recent-search').value.toLowerCase();
    const tbody = document.getElementById('recent-calls-body');
    if (!tbody) return;

    const filtered = allCalls.filter(call => 
        call.audio_file.toLowerCase().includes(query) ||
        (call.sentiment || '').toLowerCase().includes(query)
    );

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-secondary);">No matching calls found.</td></tr>`;
        return;
    }

    tbody.innerHTML = filtered.slice(0, 5).map(call => {
        const dateStr = formatDateTime(call.processed_at);
        const durationStr = formatDuration(call.duration_seconds);
        const werStr = formatWER(call.wer_score);
        
        const sentimentClass = call.sentiment === 'Positive' ? 'badge-success' : (call.sentiment === 'Negative' ? 'badge-danger' : 'badge-warning');
        const alertHtml = call.flagged ? `<span class="badge badge-danger">FLAGGED</span>` : `<span class="badge badge-success">OK</span>`;
        
        return `
            <tr>
                <td>${dateStr}</td>
                <td style="font-weight: 500;">${call.audio_file}</td>
                <td>${durationStr}</td>
                <td><span class="badge badge-info">${call.asr_backend || 'whisper'}</span></td>
                <td style="font-weight: 600;">${werStr}</td>
                <td><span class="badge ${sentimentClass}">${call.sentiment || 'Neutral'}</span></td>
                <td>${alertHtml}</td>
            </tr>
        `;
    }).join('');
}


// ─────────────────────────────────────────────────────────────────────────────
// Dashboard Operations
// ─────────────────────────────────────────────────────────────────────────────
async function initDashboard() {
    await reloadDashboardData();
}

async function reloadDashboardData() {
    try {
        // Fetch stats
        const statsRes = await fetch(`${API_URL}/stats`);
        const stats = await statsRes.json();
        
        // Update cards
        document.getElementById('card-total-calls').innerText = stats.total_calls;
        document.getElementById('card-avg-wer').innerText = stats.avg_wer !== null ? `${(stats.avg_wer * 100).toFixed(1)}%` : 'N/A';
        document.getElementById('card-flagged-calls').innerText = stats.flagged_calls;
        document.getElementById('card-sentiment-ratio').innerText = `${stats.positive_calls} / ${stats.negative_calls} / ${stats.neutral_calls}`;
        
        // Fetch call logs
        const callsRes = await fetch(`${API_URL}/calls`);
        allCalls = await callsRes.json();
        
        // Sort & Render
        sortCallHistory();
        
        // Draw Canvas Charts
        drawWERChart(allCalls);
        drawSentimentChart(stats);
        
        // Update banner
        const banner = document.getElementById('manager-alert-banner');
        if (stats.flagged_calls > 0) {
            banner.style.display = 'flex';
        } else {
            banner.style.display = 'none';
        }
    } catch (err) {
        console.error("Failed to reload dashboard data:", err);
    }
}

// Track which job_id is pending delete confirmation
let _pendingDeleteId = null;

function renderCallHistoryTable(callsToRender) {
    const tbody = document.getElementById('call-history-body');
    if (!tbody) return;

    if (callsToRender.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">No calls matches search query.</td></tr>`;
        return;
    }

    tbody.innerHTML = callsToRender.map(call => {
        const werStr = formatWER(call.wer_score);
        const sentimentClass = call.sentiment === 'Positive' ? 'badge-success' : (call.sentiment === 'Negative' ? 'badge-danger' : 'badge-warning');
        const alertHtml = call.flagged ? `<span class="badge badge-danger">FLAGGED</span>` : `<span class="badge badge-success">OK</span>`;

        const isSelected = selectedCall && selectedCall.job_id === call.job_id;
        const isPending = _pendingDeleteId === call.job_id;
        const rowStyle = isSelected
            ? 'style="background-color: rgba(59, 130, 246, 0.15); border-left: 3px solid var(--accent-blue);"'
            : '';

        // Inline confirm row shown instead of browser confirm dialog
        const confirmRow = isPending ? `
            <tr id="confirm-row-${call.job_id}" style="background:rgba(239,68,68,0.12); border-left:3px solid #ef4444;">
                <td colspan="5" style="padding:0.5rem 0.75rem; font-size:0.8125rem;">
                    <span style="color:#fca5a5; margin-right:0.75rem;">🗑 Delete <strong>${call.audio_file}</strong>? This cannot be undone.</span>
                    <button onclick="event.stopPropagation(); confirmDelete('${call.job_id}')"
                        style="background:#ef4444;color:#fff;border:none;padding:0.25rem 0.75rem;border-radius:0.375rem;cursor:pointer;font-size:0.8rem;margin-right:0.5rem;">Yes, Delete</button>
                    <button onclick="event.stopPropagation(); cancelDelete()"
                        style="background:transparent;color:#94a3b8;border:1px solid #475569;padding:0.25rem 0.75rem;border-radius:0.375rem;cursor:pointer;font-size:0.8rem;">Cancel</button>
                </td>
            </tr>` : '';

        return `
            <tr ${rowStyle} onclick="selectCall('${call.job_id}')" class="clickable-row">
                <td style="font-weight: 500; cursor: pointer;">${call.audio_file}</td>
                <td>${werStr}</td>
                <td><span class="badge ${sentimentClass}">${call.sentiment || 'Neutral'}</span></td>
                <td>${alertHtml}</td>
                <td>
                    <button class="action-btn" onclick="event.stopPropagation(); deleteCall('${call.job_id}')" title="Delete record">🗑</button>
                </td>
            </tr>
            ${confirmRow}
        `;
    }).join('');
}

function getProcessedCalls() {
    let processed = [...allCalls];
    const selector = document.getElementById('sort-selector');
    const currentSort = selector ? selector.value : 'date-desc';
    
    if (currentSort === 'date-desc') {
        processed.sort((a, b) => new Date(b.processed_at) - new Date(a.processed_at));
    } else if (currentSort === 'date-asc') {
        processed.sort((a, b) => new Date(a.processed_at) - new Date(b.processed_at));
    } else if (currentSort === 'wer-desc') {
        processed.sort((a, b) => (b.wer_score || 0) - (a.wer_score || 0));
    } else if (currentSort === 'wer-asc') {
        processed.sort((a, b) => (a.wer_score || 999) - (a.wer_score || 999));
    }
    
    return processed;
}

function filterCallHistory() {
    triggerSearch();
}

function sortCallHistory() {
    renderCallHistoryTable(getProcessedCalls());
}

async function triggerSearch() {
    const q = document.getElementById('db-search')?.value.trim() || '';
    const sentiment = document.getElementById('filter-sentiment')?.value || '';
    const flagged = document.getElementById('filter-flagged')?.value || '';
    const tag = document.getElementById('filter-tag')?.value.trim() || '';
    const werMinVal = document.getElementById('filter-wer-min')?.value || '';
    const werMaxVal = document.getElementById('filter-wer-max')?.value || '';
    const durMinVal = document.getElementById('filter-dur-min')?.value || '';
    const durMaxVal = document.getElementById('filter-dur-max')?.value || '';

    // Convert WER % from input (e.g. 15%) to decimal (0.15) for database search
    const wer_min = werMinVal ? parseFloat(werMinVal) / 100 : '';
    const wer_max = werMaxVal ? parseFloat(werMaxVal) / 100 : '';

    const params = new URLSearchParams();
    if (q) params.append('q', q);
    if (sentiment) params.append('sentiment', sentiment);
    if (flagged !== '') params.append('flagged', flagged === '1' ? 'true' : 'false');
    if (tag) params.append('tag', tag);
    if (wer_min !== '') params.append('wer_min', wer_min);
    if (wer_max !== '') params.append('wer_max', wer_max);
    if (durMinVal !== '') params.append('duration_min', durMinVal);
    if (durMaxVal !== '') params.append('duration_max', durMaxVal);

    try {
        const res = await fetch(`${API_URL}/calls?${params.toString()}`);
        if (res.ok) {
            allCalls = await res.json();
            sortCallHistory();
        }
    } catch (err) {
        console.error("Failed to query calls:", err);
    }
}

function toggleAdvancedSearch() {
    const panel = document.getElementById('advanced-search-panel');
    if (!panel) return;
    if (panel.style.display === 'none') {
        panel.style.display = 'flex';
    } else {
        panel.style.display = 'none';
    }
}

function resetAdvancedSearch() {
    const elements = [
        'filter-sentiment', 'filter-flagged', 'filter-tag',
        'filter-wer-min', 'filter-wer-max', 'filter-dur-min', 'filter-dur-max'
    ];
    elements.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = '';
    });
    triggerSearch();
}


// ─────────────────────────────────────────────────────────────────────────────
// Call details pane
// ─────────────────────────────────────────────────────────────────────────────
async function selectCall(jobId) {
    try {
        const res = await fetch(`${API_URL}/calls/${jobId}`);
        if (!res.ok) throw new Error("Call not found");
        selectedCall = await res.json();
        
        // Refresh table highlights
        sortCallHistory();
        
        // Render details pane
        renderDetailsPane();
    } catch (err) {
        console.error("Failed to select call:", err);
    }
}

function renderDetailsPane() {
    const pane = document.getElementById('details-pane');
    if (!pane || !selectedCall) return;

    // Reset player state variables
    useTTS = false;
    currentHighlightedIndex = -1;

    // Parse segments out of speakers or JSON representation
    let segments = selectedCall.segments || [];
    if (typeof segments === 'string') {
        try { segments = JSON.parse(segments); } catch (e) {}
    }

    let actionItems = selectedCall.action_items || [];
    if (typeof actionItems === 'string') {
        try { actionItems = JSON.parse(actionItems); } catch (e) {}
    }

    const flaggedHtml = selectedCall.flagged 
        ? `<div class="alert-banner" style="margin-bottom: 1rem;">
             <span>🚨</span>
             <div>
                <strong>Flagged Call Alert:</strong> Customer support issues detected.
             </div>
           </div>`
        : '';

    const actionListHtml = actionItems.length > 0
        ? actionItems.map(item => `<li>${item}</li>`).join('')
        : `<li style="list-style:none; color:var(--text-secondary);">No action items extracted.</li>`;

    // Map segments with data-start and data-end attributes
    const transcriptHtml = segments.map((seg, idx) => {
        const speakerClass = (seg.speaker || '').toLowerCase() === 'agent' ? 'agent' : 'customer';
        const formattedText = highlightPII(seg.text);
        return `
            <div class="segment-item" id="seg-${idx}" data-start="${seg.start}" data-end="${seg.end}" onclick="playAudioSegment(${seg.start}, ${seg.end}, ${idx})">
                <div class="segment-header">
                    <span class="segment-speaker ${speakerClass}">${seg.speaker || 'Speaker A'}</span>
                    <span>${formatDuration(seg.start)} - ${formatDuration(seg.end)}</span>
                </div>
                <div class="segment-text">${formattedText}</div>
            </div>
        `;
    }).join('');

    pane.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-color); padding-bottom:1rem;">
            <div>
                <h2 style="font-size: 1.25rem; font-weight: 700; word-break: break-all;">${selectedCall.audio_file}</h2>
                <p style="color:var(--text-secondary); font-size:0.8125rem;">Processed: ${formatDateTime(selectedCall.processed_at)}</p>
            </div>
            <div style="display: flex; gap: 0.5rem;">
                <button class="btn-secondary" onclick="exportReport('txt')" style="padding: 0.25rem 0.5rem; font-size:0.8125rem;">TXT</button>
                <button class="btn-secondary" onclick="exportReport('json')" style="padding: 0.25rem 0.5rem; font-size:0.8125rem;">JSON</button>
                <button class="nav-btn" onclick="printQualityReport()" style="padding: 0.25rem 0.75rem; font-size:0.8125rem; background: var(--accent-purple); box-shadow: none;">Print PDF</button>
            </div>
        </div>

        ${flaggedHtml}

        <!-- Key Metrics Panel -->
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:1rem; background:rgba(0,0,0,0.2); padding:1rem; border-radius:0.5rem; border:1px solid var(--border-color);">
            <div>
                <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Word Error Rate</div>
                <div style="font-size:1.5rem; font-weight:700; color:var(--accent-blue);">${formatWER(selectedCall.wer_score)}</div>
            </div>
            <div>
                <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Sentiment</div>
                <div style="font-size:1.5rem; font-weight:700; color:var(--accent-green);">${selectedCall.sentiment || 'Neutral'} (${(selectedCall.sentiment_score || 0).toFixed(0)}%)</div>
            </div>
            <div>
                <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Duration</div>
                <div style="font-size:1rem; font-weight:600;">${formatDuration(selectedCall.duration_seconds)}</div>
            </div>
            <div>
                <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Backend ASR</div>
                <div style="font-size:1rem; font-weight:600;"><span class="badge badge-info">${selectedCall.asr_backend || 'whisper'}</span></div>
            </div>
        </div>

        <!-- Tags Manager -->
        <div style="margin-top: 0.75rem; background:rgba(0,0,0,0.1); padding:0.75rem; border-radius:0.5rem; border:1px solid var(--border-color);">
            <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; margin-bottom: 0.5rem; font-weight: 600;">Tags</div>
            <div id="call-tags-list" style="display: flex; flex-wrap: wrap; align-items: center; gap: 0.25rem;">
                ${(selectedCall.tags || []).map(tag => `
                    <span class="tag-badge">
                        ${tag}
                        <button class="tag-delete-btn" onclick="event.stopPropagation(); removeCallTag('${selectedCall.job_id}', '${tag}')">×</button>
                    </span>
                `).join('')}
                <div id="tag-add-container" style="display: inline-flex; align-items: center;">
                    <button class="tag-add-btn" onclick="showAddTagInput()">+ Add Tag</button>
                </div>
            </div>
        </div>

        <!-- Week 2: Intelligence Layer — Call Summary Panel -->
        ${renderIntelligencePanel(selectedCall)}

        <!-- Real Audio Streaming & Sync Player -->
        <div class="audio-player-container">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.8125rem; font-weight:600; display:flex; align-items:center; gap:0.5rem;">
                    <span id="player-mode-badge" class="badge badge-success">📻 Live Stream</span>
                    <span id="player-title">Call Recording Playback</span>
                </span>
                <span style="font-size:0.75rem; color:var(--text-secondary); font-family:monospace;" id="audio-time-label">0.0s / ${formatDuration(selectedCall.duration_seconds)}</span>
            </div>
            <div style="display:flex; gap:0.75rem; align-items:center; margin-top:0.75rem;">
                <button class="nav-btn" id="audio-play-btn" onclick="toggleAudioPlayback()" style="padding: 0.35rem 0.85rem; font-size:0.75rem; background: var(--accent-blue); box-shadow:none; display:flex; align-items:center; gap:0.25rem;">
                    <span id="play-btn-icon">▶</span> <span id="play-btn-text">Play Call</span>
                </button>
                <button class="btn-secondary" id="audio-stop-btn" onclick="stopAudioPlayback()" style="padding: 0.35rem 0.85rem; font-size:0.75rem;">⏹ Stop</button>
                
                <select id="playback-speed-selector" class="search-input" onchange="changePlaybackSpeed()" style="width: auto; max-width: 75px; padding: 0.25rem; font-size: 0.75rem; height: 30px; margin: 0; line-height: 1;">
                    <option value="0.5">0.5x</option>
                    <option value="1.0" selected>1.0x</option>
                    <option value="1.25">1.25x</option>
                    <option value="1.5">1.5x</option>
                    <option value="2.0">2.0x</option>
                </select>

                <!-- Clickable Progress Track -->
                <div id="audio-progress-track" onclick="onProgressTrackClick(event)" style="flex:1; height:6px; background:var(--border-color); border-radius:3px; position:relative; cursor:pointer; overflow:hidden;">
                    <div id="audio-progress-bar" style="position:absolute; top:0; left:0; width:0%; height:100%; background:var(--accent-blue); transition: width 0.05s linear;"></div>
                </div>
            </div>
            
            <!-- Hidden native audio element -->
            <audio id="active-call-audio" src="${API_URL}/calls/${selectedCall.job_id}/audio" style="display:none;" ontimeupdate="onAudioTimeUpdate()" onended="onAudioEnded()" onerror="onAudioError()"></audio>
            
            <p style="font-size:0.6875rem; color:var(--text-secondary); margin-top:0.35rem;">💡 Click any segment below to seek &amp; play from that moment. Tracks active segment in real-time.</p>
        </div>

        <!-- Action Items List -->
        <div>
            <h3 style="font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem;">Extracted Action Items</h3>
            <ul class="action-list">
                ${actionListHtml}
            </ul>
        </div>

        <!-- Dialog Script / Transcript -->
        <div style="display:flex; flex-direction:column; gap:0.5rem;">
            <h3 style="font-size: 1rem; font-weight: 600;">Transcript Conversation</h3>
            <div class="transcript-box">
                ${transcriptHtml}
            </div>
        </div>
    `;
}


// ─────────────────────────────────────────────────────────────────────────────
// Week 2: Intelligence Panel Renderer
// ─────────────────────────────────────────────────────────────────────────────
function renderIntelligencePanel(call) {
    const summary = call.summary || '';
    const issue = call.summary_issue || '';
    const resolution = call.summary_resolution || '';
    const followUp = call.summary_follow_up || 'None';
    const engine = call.summary_engine || 'extractive';
    const latency = call.latency_report || null;

    // If no summary data at all (old records), skip panel
    if (!summary && !issue) return '';

    const engineBadge = engine === 'gpt'
        ? `<span class="badge badge-success" title="Powered by GPT">🤖 GPT</span>`
        : `<span class="badge badge-warning" title="Extractive rule-based">⚙️ Extractive</span>`;

    let latencyHtml = '';
    if (latency) {
        const intelOk = latency.intelligence_within_target;
        const intelColor = intelOk ? 'var(--accent-green)' : 'var(--accent-red)';
        const intelIcon = intelOk ? '✅' : '⚠️';
        latencyHtml = `
            <div style="margin-top:0.75rem; padding:0.625rem 0.75rem; background:rgba(0,0,0,0.25); border-radius:0.375rem; border:1px solid var(--border-color);">
                <div style="font-size:0.7rem; text-transform:uppercase; color:var(--text-secondary); margin-bottom:0.4rem; letter-spacing:0.05em;">⏱ Pipeline Latency</div>
                <div style="display:flex; flex-wrap:wrap; gap:0.75rem; font-size:0.8rem;">
                    <span>🔧 Preprocess: <strong>${latency.preprocess_ms}ms</strong></span>
                    <span>🎙 Transcribe: <strong>${latency.transcribe_ms}ms</strong></span>
                    <span style="color:${intelColor}">${intelIcon} Intelligence: <strong>${latency.intelligence_ms}ms</strong> (target ≤${latency.target_s * 1000}ms)</span>
                    <span>📊 Total: <strong>${latency.total_ms}ms</strong></span>
                </div>
            </div>
        `;
    }

    return `
        <div style="background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(59,130,246,0.08) 100%);
                    border: 1px solid rgba(99,102,241,0.25); border-radius: 0.75rem; padding: 1rem; margin: 0;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <h3 style="font-size:1rem; font-weight:700; margin:0; display:flex; align-items:center; gap:0.5rem;">
                    🧠 Intelligence Layer Summary
                </h3>
                ${engineBadge}
            </div>

            <!-- Summary Text -->
            ${ summary ? `<p style="font-size:0.875rem; color:var(--text-primary); line-height:1.6; margin-bottom:0.75rem; font-style:italic;">&ldquo;${summary}&rdquo;</p>` : '' }

            <!-- Structured Cards -->
            <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:0.5rem;">
                <div style="background:rgba(0,0,0,0.2); border-radius:0.5rem; padding:0.625rem; border-left: 3px solid #f59e0b;">
                    <div style="font-size:0.65rem; text-transform:uppercase; color:#f59e0b; letter-spacing:0.08em; margin-bottom:0.25rem;">🔎 Issue</div>
                    <div style="font-size:0.8125rem; font-weight:500;">${issue || 'N/A'}</div>
                </div>
                <div style="background:rgba(0,0,0,0.2); border-radius:0.5rem; padding:0.625rem; border-left: 3px solid #10b981;">
                    <div style="font-size:0.65rem; text-transform:uppercase; color:#10b981; letter-spacing:0.08em; margin-bottom:0.25rem;">✅ Resolution</div>
                    <div style="font-size:0.8125rem; font-weight:500;">${resolution || 'N/A'}</div>
                </div>
                <div style="background:rgba(0,0,0,0.2); border-radius:0.5rem; padding:0.625rem; border-left: 3px solid #3b82f6;">
                    <div style="font-size:0.65rem; text-transform:uppercase; color:#3b82f6; letter-spacing:0.08em; margin-bottom:0.25rem;">📌 Follow-up</div>
                    <div style="font-size:0.8125rem; font-weight:500;">${followUp}</div>
                </div>
            </div>

            ${latencyHtml}
        </div>
    `;
}

// Regex PII Highlight wrapper
function highlightPII(text) {
    if (!text) return '';
    // Format [REDACTED PHONE] -> styled span
    return text.replace(/\[REDACTED ([A-Z]+)\]/g, (match, p1) => {
        return `<span class="pii-highlight" title="Sensitive PII Redacted">[REDACTED ${p1}]</span>`;
    });
}

// Real Audio Player & Sync / Speech Synthesis Fallback
let useTTS = false;
let currentHighlightedIndex = -1;
let speechSynthUtterance = null;


function onAudioError() {
    console.warn("Audio file not found or failed to load. Falling back to SpeechSynthesis TTS.");
    useTTS = true;
    const badge = document.getElementById('player-mode-badge');
    if (badge) {
        badge.innerText = "🤖 TTS Simulation";
        badge.className = "badge badge-warning";
    }
    const title = document.getElementById('player-title');
    if (title) title.innerText = "TTS Voice Reader (Audio Missing)";
}

function toggleAudioPlayback() {
    if (useTTS) {
        speakFullTranscript();
        return;
    }

    const audio = document.getElementById('active-call-audio');
    const playBtnIcon = document.getElementById('play-btn-icon');
    const playBtnText = document.getElementById('play-btn-text');
    if (!audio) return;

    if (audio.paused) {
        audio.play().catch(err => {
            console.error("Audio play failed, falling back to TTS:", err);
            onAudioError();
            speakFullTranscript();
        });
        if (playBtnIcon) playBtnIcon.innerText = "⏸";
        if (playBtnText) playBtnText.innerText = "Pause";
    } else {
        audio.pause();
        if (playBtnIcon) playBtnIcon.innerText = "▶";
        if (playBtnText) playBtnText.innerText = "Play Call";
    }
}

function stopAudioPlayback() {
    if (useTTS) {
        stopSpeech();
        return;
    }

    const audio = document.getElementById('active-call-audio');
    if (audio) {
        audio.pause();
        audio.currentTime = 0;
    }
    const playBtnIcon = document.getElementById('play-btn-icon');
    const playBtnText = document.getElementById('play-btn-text');
    if (playBtnIcon) playBtnIcon.innerText = "▶";
    if (playBtnText) playBtnText.innerText = "Play Call";
    
    const progressBar = document.getElementById('audio-progress-bar');
    if (progressBar) progressBar.style.width = '0%';
    
    // Clear highlights
    clearSegmentHighlights();
}

function onAudioTimeUpdate() {
    const audio = document.getElementById('active-call-audio');
    if (!audio || !selectedCall) return;

    const currentTime = audio.currentTime;
    const duration = audio.duration || selectedCall.duration_seconds || 1;
    const percent = (currentTime / duration) * 100;

    const progressBar = document.getElementById('audio-progress-bar');
    const timeLabel = document.getElementById('audio-time-label');

    if (progressBar) progressBar.style.width = `${percent}%`;
    if (timeLabel) {
        timeLabel.innerText = `${currentTime.toFixed(1)}s / ${formatDuration(duration)}`;
    }

    // Highlight active transcript segment
    highlightActiveSegment(currentTime);
}

function highlightActiveSegment(currentTime) {
    const segments = selectedCall.segments || [];
    let activeIdx = -1;

    for (let i = 0; i < segments.length; i++) {
        if (currentTime >= segments[i].start && currentTime <= segments[i].end) {
            activeIdx = i;
            break;
        }
    }

    if (activeIdx !== currentHighlightedIndex) {
        // Clear previous highlight
        clearSegmentHighlights();
        
        if (activeIdx !== -1) {
            const activeElem = document.getElementById(`seg-${activeIdx}`);
            if (activeElem) {
                activeElem.classList.add('active-segment');
                // Smooth scroll segment into view if needed
                activeElem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
        currentHighlightedIndex = activeIdx;
    }
}

function clearSegmentHighlights() {
    const items = document.querySelectorAll('.segment-item');
    items.forEach(item => item.classList.remove('active-segment'));
    currentHighlightedIndex = -1;
}

function onAudioEnded() {
    stopAudioPlayback();
}

function onProgressTrackClick(event) {
    if (useTTS) return;
    const track = document.getElementById('audio-progress-track');
    const audio = document.getElementById('active-call-audio');
    if (!track || !audio || !selectedCall) return;

    const rect = track.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const width = rect.width;
    const duration = audio.duration || selectedCall.duration_seconds || 1;

    const clickPercent = clickX / width;
    audio.currentTime = clickPercent * duration;
}

function playAudioSegment(start, end, idx) {
    if (useTTS) {
        // Fallback: Read segment text using SpeechSynthesis
        const elem = document.getElementById(`seg-${idx}`);
        if (elem) {
            const text = elem.querySelector('.segment-text').innerText;
            playSegmentText(text, start);
            clearSegmentHighlights();
            elem.classList.add('active-segment');
        }
        return;
    }

    const audio = document.getElementById('active-call-audio');
    if (!audio) return;

    audio.currentTime = start;
    if (audio.paused) {
        audio.play().catch(() => onAudioError());
        const playBtnIcon = document.getElementById('play-btn-icon');
        const playBtnText = document.getElementById('play-btn-text');
        if (playBtnIcon) playBtnIcon.innerText = "⏸";
        if (playBtnText) playBtnText.innerText = "Pause";
    }
    
    highlightActiveSegment(start);
}

function playSegmentText(text, startTime) {
    stopSpeech();

    // Clean text of redacted tokens for speech synthesis
    const spokenText = text.replace(/\[REDACTED [A-Z]+\]/g, "redacted information");

    speechSynthUtterance = new SpeechSynthesisUtterance(spokenText);
    
    speechSynthUtterance.rate = 1.0;
    speechSynthUtterance.pitch = 1.0;

    // Simulate progress bar movement
    const duration = Math.max(2, spokenText.split(' ').length * 0.4);
    const start = Date.now();
    const progressInterval = setInterval(() => {
        const elapsed = (Date.now() - start) / 1000;
        const percent = Math.min(100, (elapsed / duration) * 100);
        const progressBar = document.getElementById('audio-progress-bar');
        const timeLabel = document.getElementById('audio-time-label');
        
        if (progressBar) progressBar.style.width = `${percent}%`;
        if (timeLabel && selectedCall) {
            const currentSimTime = Math.min(selectedCall.duration_seconds, startTime + elapsed);
            timeLabel.innerText = `${currentSimTime.toFixed(1)}s / ${formatDuration(selectedCall.duration_seconds)}`;
        }

        if (elapsed >= duration) {
            clearInterval(progressInterval);
        }
    }, 100);

    speechSynthUtterance.onend = () => {
        clearInterval(progressInterval);
        const progressBar = document.getElementById('audio-progress-bar');
        if (progressBar) progressBar.style.width = '100%';
    };

    window.speechSynthesis.speak(speechSynthUtterance);
}

function speakFullTranscript() {
    if (!selectedCall) return;
    const fullText = selectedCall.redacted_transcript || selectedCall.full_transcript;
    playSegmentText(fullText, 0);
}

function stopSpeech() {
    window.speechSynthesis.cancel();
    const progressBar = document.getElementById('audio-progress-bar');
    if (progressBar) progressBar.style.width = '0%';
}


// ─────────────────────────────────────────────────────────────────────────────
// Upload File Logic
// ─────────────────────────────────────────────────────────────────────────────
function triggerFileInput() {
    document.getElementById('audio-file-input').click();
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        uploadFile(file);
    }
}

async function uploadFile(file) {
    const statusDiv = document.getElementById('upload-status');
    statusDiv.style.display = 'block';
    statusDiv.style.color = 'var(--text-primary)';
    
    // Setup FormData
    const formData = new FormData();
    formData.append('file', file);
    
    const lang = document.getElementById('hint-language').value;
    if (lang) {
        formData.append('language', lang);
    }
    
    const refText = document.getElementById('reference-text-input').value;
    if (refText) {
        formData.append('reference_text', refText);
    }

    try {
        // Step-by-step progress simulation
        const steps = [
            "1. Uploading audio file...",
            "2. Running audio preprocessor (16kHz mono conversion)...",
            "3. Calling Whisper ASR engine...",
            "4. Computing Word Error Rate (WER)...",
            "5. Running Intelligence Layer: LLM Summary + Sentiment..."
        ];

        let stepIndex = 0;
        statusDiv.innerText = steps[0];
        const stepInterval = setInterval(() => {
            if (stepIndex < steps.length - 1) {
                stepIndex++;
                statusDiv.innerText = steps[stepIndex];
            }
        }, 1500);

        const res = await fetch(`${API_URL}/transcribe`, {
            method: 'POST',
            body: formData
        });
        
        clearInterval(stepInterval);

        if (!res.ok) {
            const errBody = await res.json();
            throw new Error(errBody.detail || "Transcription failed");
        }

        const data = await res.json();
        statusDiv.innerText = "✓ Processing completed!";
        statusDiv.style.color = 'var(--accent-green)';
        
        // Reload statistics & history table
        await reloadDashboardData();
        
        // Automatically select the new call
        await selectCall(data.job_id);

        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 3000);

    } catch (err) {
        console.error("Upload error:", err);
        statusDiv.innerText = `Error: ${err.message}`;
        statusDiv.style.color = 'var(--accent-red)';
    }
}

// Drag & Drop event bindings
const uploadZone = document.getElementById('upload-zone');
if (uploadZone) {
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = 'var(--accent-blue)';
            uploadZone.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = 'var(--border-color)';
            uploadZone.style.backgroundColor = 'transparent';
        }, false);
    });

    uploadZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        if (file) {
            uploadFile(file);
        }
    }, false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Deletion Logic
// ─────────────────────────────────────────────────────────────────────────────
// Step 1 — show inline confirm row (no browser dialog)
function deleteCall(jobId) {
    _pendingDeleteId = jobId;
    sortCallHistory(); // re-render table with confirm row visible
}

// Step 2 — user clicked "Yes, Delete"
async function confirmDelete(jobId) {
    _pendingDeleteId = null;
    try {
        const res = await fetch(`${API_URL}/calls/${jobId}`, {
            method: 'DELETE'
        });
        if (res.ok) {
            // Clear details pane if the deleted call was selected
            if (selectedCall && selectedCall.job_id === jobId) {
                selectedCall = null;
                document.getElementById('details-pane').innerHTML = `
                    <div style="text-align: center; padding: 4rem 2rem; color: var(--text-secondary);">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                        <h3>No Call Selected</h3>
                        <p>Click on any call record from the history table or upload a new file to display the real-time transcription and intelligence report.</p>
                    </div>
                `;
            }
            await reloadDashboardData();
        } else {
            const body = await res.json().catch(() => ({}));
            const msg = body.detail || `Server error ${res.status}`;
            // Show error inline
            const confirmRow = document.getElementById(`confirm-row-${jobId}`);
            if (confirmRow) {
                confirmRow.innerHTML = `<td colspan="5" style="color:#fca5a5; padding:0.5rem 0.75rem; font-size:0.8125rem;">⚠️ Delete failed: ${msg}</td>`;
                setTimeout(() => { _pendingDeleteId = null; sortCallHistory(); }, 2500);
            }
        }
    } catch (err) {
        console.error('Delete call error:', err);
        _pendingDeleteId = null;
        sortCallHistory();
    }
}

// Step 2 (alt) — user clicked "Cancel"
function cancelDelete() {
    _pendingDeleteId = null;
    sortCallHistory(); // re-render table without confirm row
}

// ─────────────────────────────────────────────────────────────────────────────
// Export Call Reports
// ─────────────────────────────────────────────────────────────────────────────
function exportReport(format) {
    if (!selectedCall) return;

    let content = "";
    let filename = `${selectedCall.audio_file.replace(/\.[^/.]+$/, "")}_report.${format}`;

    if (format === 'json') {
        content = JSON.stringify(selectedCall, null, 2);
    } else if (format === 'txt') {
        let segments = selectedCall.segments || [];
        if (typeof segments === 'string') {
            try { segments = JSON.parse(segments); } catch (e) {}
        }
        
        let actionItems = selectedCall.action_items || [];
        if (typeof actionItems === 'string') {
            try { actionItems = JSON.parse(actionItems); } catch (e) {}
        }

        content = `VOICEOPS SENTINEL CALL TRANSCRIPT REPORT
==========================================
Filename: ${selectedCall.audio_file}
Processed At: ${formatDateTime(selectedCall.processed_at)}
Duration: ${formatDuration(selectedCall.duration_seconds)}
Word Error Rate (WER): ${formatWER(selectedCall.wer_score)}
Sentiment: ${selectedCall.sentiment} (${(selectedCall.sentiment_score || 0).toFixed(0)}%)
ASR Backend Engine: ${selectedCall.asr_backend}
==========================================

ACTION ITEMS:
${actionItems.map(item => `- ${item}`).join('\n')}

TRANSCRIPT DIALOGUE:
${segments.map(seg => `[${formatDuration(seg.start)} - ${formatDuration(seg.end)}] ${seg.speaker}: ${seg.text}`).join('\n')}
`;
    }

    // Trigger local download
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
}

function printQualityReport() {
    if (!selectedCall) return;
    
    let segments = selectedCall.segments || [];
    if (typeof segments === 'string') {
        try { segments = JSON.parse(segments); } catch (e) {}
    }
    
    let actionItems = selectedCall.action_items || [];
    if (typeof actionItems === 'string') {
        try { actionItems = JSON.parse(actionItems); } catch (e) {}
    }

    const printWindow = window.open('', '_blank');
    printWindow.document.write(`
        <html>
        <head>
            <title>VoiceOps Sentinel Quality Report - ${selectedCall.audio_file}</title>
            <style>
                body { font-family: 'Segoe UI', system-ui, sans-serif; padding: 2rem; color: #1e293b; line-height: 1.6; }
                h1 { font-size: 24px; color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: 0.5rem; margin-bottom: 1rem; }
                .meta-table { width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }
                .meta-table td { padding: 0.5rem; border: 1px solid #e2e8f0; }
                .meta-table td.label { font-weight: bold; background: #f8fafc; width: 25%; }
                .section-title { font-size: 18px; font-weight: bold; color: #1e293b; margin-top: 1.5rem; border-bottom: 1px solid #cbd5e1; padding-bottom: 0.25rem; }
                .action-item { margin: 0.25rem 0; font-weight: 500; color: #b45309; }
                .segment { margin-bottom: 0.75rem; padding-bottom: 0.5rem; border-bottom: 1px dashed #f1f5f9; }
                .speaker { font-weight: bold; }
                .speaker.agent { color: #2563eb; }
                .speaker.customer { color: #7c3aed; }
                .timestamp { font-size: 12px; color: #64748b; margin-left: 0.5rem; }
            </style>
        </head>
        <body>
            <h1>VoiceOps Call Quality Intelligence Report</h1>
            <table class="meta-table">
                <tr>
                    <td class="label">Filename</td>
                    <td>${selectedCall.audio_file}</td>
                    <td class="label">WER Score</td>
                    <td><strong>${formatWER(selectedCall.wer_score)}</strong></td>
                </tr>
                <tr>
                    <td class="label">Processed At</td>
                    <td>${formatDateTime(selectedCall.processed_at)}</td>
                    <td class="label">Sentiment Score</td>
                    <td>${selectedCall.sentiment} (${(selectedCall.sentiment_score || 0).toFixed(0)}%)</td>
                </tr>
                <tr>
                    <td class="label">Duration</td>
                    <td>${formatDuration(selectedCall.duration_seconds)}</td>
                    <td class="label">ASR engine</td>
                    <td>${selectedCall.asr_backend}</td>
                </tr>
            </table>

            <div class="section-title">Extracted Call Action Items</div>
            <ul>
                ${actionItems.map(item => `<li class="action-item">${item}</li>`).join('')}
            </ul>

            <div class="section-title">Redacted Call Transcript</div>
            <div style="margin-top:1rem;">
                ${segments.map(seg => `
                    <div class="segment">
                        <span class="speaker ${seg.speaker.toLowerCase() === 'agent' ? 'agent' : 'customer'}">${seg.speaker}</span>
                        <span class="timestamp">[${formatDuration(seg.start)} - ${formatDuration(seg.end)}]</span>
                        <p style="margin: 0.25rem 0 0;">${seg.text}</p>
                    </div>
                `).join('')}
            </div>
            
            <script>
                window.onload = function() { window.print(); }
            </script>
        </body>
        </html>
    `);
    printWindow.document.close();
}

// Modal management
function openModal(title, content) {
    document.getElementById('modal-title').innerText = title;
    document.getElementById('modal-body').innerText = content;
    document.getElementById('report-modal').classList.add('active');
    
    document.getElementById('modal-copy-btn').onclick = () => {
        navigator.clipboard.writeText(content);
        alert("Copied to clipboard!");
    };
}

function closeModal() {
    document.getElementById('report-modal').classList.remove('active');
}


// ─────────────────────────────────────────────────────────────────────────────
// Custom HTML5 Canvas Chart Renderers (No Library Needed!)
// ─────────────────────────────────────────────────────────────────────────────
function drawWERChart(calls) {
    const canvas = document.getElementById('werChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    // Clear & Resize
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Filter calls with WER scores
    const werCalls = calls.filter(c => c.wer_score !== null).slice(-10); // last 10
    if (werCalls.length === 0) {
        ctx.fillStyle = '#94a3b8';
        ctx.font = '14px Outfit';
        ctx.textAlign = 'center';
        ctx.fillText('No WER data available', width / 2, height / 2);
        return;
    }

    const padding = { top: 20, right: 20, bottom: 40, left: 40 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Draw Axes & Grid
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    // Max WER score
    const maxWER = Math.max(...werCalls.map(c => c.wer_score), 0.1);
    const stepX = chartWidth / Math.max(1, werCalls.length - 1);

    // Draw horizontal grid lines
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Outfit';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
        const val = (maxWER * (i / 4));
        const y = height - padding.bottom - (chartHeight * (i / 4));
        
        ctx.strokeStyle = '#1e293b';
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        
        ctx.fillText(`${(val * 100).toFixed(0)}%`, padding.left - 8, y);
    }

    // Plot Line and Glowing Gradient Fill
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    werCalls.forEach((call, index) => {
        const x = padding.left + index * stepX;
        const y = height - padding.bottom - (call.wer_score / maxWER) * chartHeight;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw Dots & labels
    ctx.fillStyle = '#3b82f6';
    ctx.textAlign = 'center';
    werCalls.forEach((call, index) => {
        const x = padding.left + index * stepX;
        const y = height - padding.bottom - (call.wer_score / maxWER) * chartHeight;
        
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();

        // label under dot
        ctx.fillStyle = '#94a3b8';
        ctx.font = '8px Outfit';
        const filenameLabel = call.audio_file.substring(0, 8) + '...';
        ctx.fillText(filenameLabel, x, height - padding.bottom + 16);
        ctx.fillStyle = '#3b82f6';
    });
}

function drawSentimentChart(stats) {
    const canvas = document.getElementById('sentimentChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    // Clear & Resize
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    const total = stats.positive_calls + stats.negative_calls + stats.neutral_calls;
    if (total === 0) {
        ctx.fillStyle = '#94a3b8';
        ctx.font = '14px Outfit';
        ctx.textAlign = 'center';
        ctx.fillText('No Sentiment data available', width / 2, height / 2);
        return;
    }

    const data = [
        { label: 'Positive', count: stats.positive_calls, color: '#10b981' },
        { label: 'Neutral', count: stats.neutral_calls, color: '#f59e0b' },
        { label: 'Negative', count: stats.negative_calls, color: '#ef4444' }
    ].filter(d => d.count > 0);

    const centerX = width / 2.5;
    const centerY = height / 2;
    const radius = Math.min(centerX, centerY) * 0.75;
    
    let currentAngle = -Math.PI / 2;

    data.forEach(slice => {
        const sliceAngle = (slice.count / total) * Math.PI * 2;
        
        ctx.fillStyle = slice.color;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
        ctx.closePath();
        ctx.fill();
        
        currentAngle += sliceAngle;
    });

    // Draw Donut Cutout dynamically using CSS variables
    const computedStyle = getComputedStyle(document.documentElement);
    ctx.fillStyle = computedStyle.getPropertyValue('--bg-card').trim() || '#ffffff';
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius * 0.5, 0, Math.PI * 2);
    ctx.fill();

    // Draw Legend
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.font = '11px Outfit';
    
    data.forEach((slice, idx) => {
        const lx = centerX + radius + 20;
        const ly = centerY - (data.length * 12) + (idx * 24);
        
        ctx.fillStyle = slice.color;
        ctx.fillRect(lx, ly - 5, 10, 10);
        
        ctx.fillStyle = computedStyle.getPropertyValue('--text-primary').trim() || '#0f172a';
        ctx.fillText(`${slice.label}: ${slice.count} (${((slice.count / total) * 100).toFixed(0)}%)`, lx + 16, ly);
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Premium Theme Toggle (Light / Dark Mode)
// ─────────────────────────────────────────────────────────────────────────────
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeToggleUI(newTheme);
    
    // Redraw charts to update colors for the new theme
    if (allCalls && allCalls.length > 0) {
        drawWERChart(allCalls);
        const stats = computeStatsFromCalls(allCalls);
        drawSentimentChart(stats);
    }
}

function updateThemeToggleUI(theme) {
    const btns = document.querySelectorAll('#theme-toggle-btn');
    btns.forEach(btn => {
        btn.innerText = theme === 'dark' ? '☀️' : '🌙';
    });
}

function computeStatsFromCalls(calls) {
    const total_calls = calls.length;
    let positive_calls = 0, negative_calls = 0, neutral_calls = 0;
    calls.forEach(c => {
        if (c.sentiment === 'Positive') positive_calls++;
        else if (c.sentiment === 'Negative') negative_calls++;
        else neutral_calls++;
    });
    return { total_calls, positive_calls, negative_calls, neutral_calls };
}

// Run immediately to load theme state
(function() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    document.addEventListener('DOMContentLoaded', () => {
        updateThemeToggleUI(savedTheme);
    });
})();

// ─────────────────────────────────────────────────────────────────────────────
// Playback Speed Control
// ─────────────────────────────────────────────────────────────────────────────
function changePlaybackSpeed() {
    const selector = document.getElementById('playback-speed-selector');
    const audio = document.getElementById('active-call-audio');
    if (selector && audio) {
        audio.playbackRate = parseFloat(selector.value);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tag Management
// ─────────────────────────────────────────────────────────────────────────────
function showAddTagInput() {
    const container = document.getElementById('tag-add-container');
    if (!container) return;
    container.innerHTML = `
        <div class="tag-add-container">
            <input type="text" id="new-tag-input" class="tag-input-field" placeholder="Tag..." onkeydown="handleTagInputKey(event)">
            <button class="tag-save-btn" onclick="saveCallTag()">Save</button>
            <button class="tag-cancel-btn" onclick="cancelAddTag()">Cancel</button>
        </div>
    `;
    const input = document.getElementById('new-tag-input');
    if (input) input.focus();
}

function cancelAddTag() {
    const container = document.getElementById('tag-add-container');
    if (!container) return;
    container.innerHTML = `<button class="tag-add-btn" onclick="showAddTagInput()">+ Add Tag</button>`;
}

function handleTagInputKey(event) {
    if (event.key === 'Enter') {
        saveCallTag();
    } else if (event.key === 'Escape') {
        cancelAddTag();
    }
}

async function saveCallTag() {
    const input = document.getElementById('new-tag-input');
    if (!input || !selectedCall) return;
    const tagVal = input.value.trim().toLowerCase();
    if (!tagVal) return;

    try {
        const formData = new FormData();
        formData.append('tag', tagVal);
        const res = await fetch(`${API_URL}/calls/${selectedCall.job_id}/tags`, {
            method: 'POST',
            body: formData
        });
        if (res.ok) {
            const updatedTags = await res.json();
            selectedCall.tags = updatedTags;
            const localCall = allCalls.find(c => c.job_id === selectedCall.job_id);
            if (localCall) localCall.tags = updatedTags;
            
            renderDetailsPane();
        } else {
            alert('Failed to save tag');
        }
    } catch (err) {
        console.error('Error saving tag:', err);
    }
}

async function removeCallTag(jobId, tagVal) {
    try {
        const res = await fetch(`${API_URL}/calls/${jobId}/tags/${encodeURIComponent(tagVal)}`, {
            method: 'DELETE'
        });
        if (res.ok) {
            const updatedTags = await res.json();
            if (selectedCall && selectedCall.job_id === jobId) {
                selectedCall.tags = updatedTags;
            }
            const localCall = allCalls.find(c => c.job_id === jobId);
            if (localCall) localCall.tags = updatedTags;
            
            renderDetailsPane();
        } else {
            alert('Failed to remove tag');
        }
    } catch (err) {
        console.error('Error removing tag:', err);
    }
}
