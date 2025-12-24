/**
 * Nepali + English ASR System - Main JavaScript
 */

// API Helper
const api = {
  async get(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  },
  
  async post(url, data, isFormData = false) {
    const options = {
      method: 'POST',
      headers: {},
      body: isFormData ? data : JSON.stringify(data)
    };
    
    if (!isFormData) {
      options.headers['Content-Type'] = 'application/json';
    }
    
    // Add CSRF token
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
    if (csrfToken) {
      options.headers['X-CSRFToken'] = csrfToken;
    }
    
    const response = await fetch(url, options);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  },
  
  async delete(url) {
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
    const response = await fetch(url, {
      method: 'DELETE',
      headers: {
        'X-CSRFToken': csrfToken || ''
      }
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response;
  }
};

// Format time helper
function formatTime(seconds) {
  if (!seconds && seconds !== 0) return '--:--';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatTimeMs(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

// Icons (SVG strings)
const icons = {
  mic: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>`,
  micOff: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="2" x2="22" y1="2" y2="22"/><path d="M18.89 13.23A7.12 7.12 0 0 0 19 12v-2"/><path d="M5 10v2a7 7 0 0 0 12 5"/><path d="M15 9.34V5a3 3 0 0 0-5.68-1.33"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12"/><line x1="12" x2="12" y1="19" y2="22"/></svg>`,
  play: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 3 20 12 6 21 6 3"/></svg>`,
  pause: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/></svg>`,
  upload: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>`,
  file: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 22h.5a2 2 0 0 0 2-2V7l-5-5H6a2 2 0 0 0-2 2v3"/><polyline points="14 2 14 7 19 7"/><path d="M10 20v-1a2 2 0 1 1 4 0v1a2 2 0 1 1-4 0Z"/><path d="M6 20v-1a2 2 0 1 0-4 0v1a2 2 0 1 0 4 0Z"/><path d="M2 19v-3a6 6 0 0 1 12 0v3"/></svg>`,
  x: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>`,
  check: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>`,
  edit: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>`,
  clock: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`,
  checkCircle: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>`,
  loader: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="spinner"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>`,
  trash: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>`,
  refresh: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>`,
  skipBack: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="19 20 9 12 19 4 19 20"/><line x1="5" x2="5" y1="19" y2="5"/></svg>`,
  skipForward: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 4 15 12 5 20 5 4"/><line x1="19" x2="19" y1="5" y2="19"/></svg>`,
  arrowLeft: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 19-7-7 7-7"/><path d="M19 12H5"/></svg>`,
};


/**
 * Audio Recorder Class
 */
class AudioRecorder {
  constructor(options = {}) {
    this.onStart = options.onStart || (() => {});
    this.onStop = options.onStop || (() => {});
    this.onDuration = options.onDuration || (() => {});
    
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.duration = 0;
    this.timer = null;
  }
  
  async start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const mimeType = MediaRecorder.isTypeSupported('audio/webm') 
        ? 'audio/webm' 
        : 'audio/mp4';
      
      this.mediaRecorder = new MediaRecorder(stream, { mimeType });
      this.audioChunks = [];
      
      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          this.audioChunks.push(e.data);
        }
      };
      
      this.mediaRecorder.onstop = () => {
        const blob = new Blob(this.audioChunks, { type: mimeType });
        stream.getTracks().forEach(track => track.stop());
        this.onStop(blob);
      };
      
      this.mediaRecorder.start(1000);
      this.isRecording = true;
      this.duration = 0;
      
      this.timer = setInterval(() => {
        this.duration++;
        this.onDuration(this.duration);
      }, 1000);
      
      this.onStart();
      
    } catch (err) {
      console.error('Recording error:', err);
      throw new Error('Could not access microphone. Please check permissions.');
    }
  }
  
  stop() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.isRecording = false;
      
      if (this.timer) {
        clearInterval(this.timer);
        this.timer = null;
      }
    }
  }
}


/**
 * Upload Page Controller
 */
class UploadController {
  constructor() {
    this.recorder = null;
    this.selectedFile = null;
    this.recordedBlob = null;
    
    this.init();
  }
  
  init() {
    // Elements
    this.recordBtn = document.getElementById('recordBtn');
    this.recordDuration = document.getElementById('recordDuration');
    this.waveform = document.getElementById('waveform');
    this.dropzone = document.getElementById('dropzone');
    this.fileInput = document.getElementById('fileInput');
    this.previewSection = document.getElementById('previewSection');
    this.audioPreview = document.getElementById('audioPreview');
    this.fileName = document.getElementById('fileName');
    this.clearBtn = document.getElementById('clearBtn');
    this.uploadBtn = document.getElementById('uploadBtn');
    this.statusSection = document.getElementById('statusSection');
    this.languageSelect = document.getElementById('language');
    this.titleInput = document.getElementById('title');
    
    if (!this.recordBtn) return; // Not on upload page
    
    this.setupRecorder();
    this.setupDropzone();
    this.setupEventListeners();
  }
  
  setupRecorder() {
    this.recorder = new AudioRecorder({
      onStart: () => {
        this.recordBtn.classList.remove('idle');
        this.recordBtn.classList.add('recording');
        this.recordBtn.innerHTML = icons.micOff;
        this.waveform.classList.remove('hidden');
        this.recordDuration.classList.remove('hidden');
      },
      onStop: (blob) => {
        this.recordBtn.classList.remove('recording');
        this.recordBtn.classList.add('idle');
        this.recordBtn.innerHTML = icons.mic;
        this.waveform.classList.add('hidden');
        
        this.recordedBlob = blob;
        this.selectedFile = null;
        this.showPreview(blob, `Recording (${formatTime(this.recorder.duration)})`);
      },
      onDuration: (duration) => {
        this.recordDuration.textContent = formatTime(duration);
      }
    });
  }
  
  setupDropzone() {
    this.dropzone.addEventListener('click', () => this.fileInput.click());
    
    this.dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.dropzone.classList.add('dragover');
    });
    
    this.dropzone.addEventListener('dragleave', () => {
      this.dropzone.classList.remove('dragover');
    });
    
    this.dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.dropzone.classList.remove('dragover');
      
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        this.handleFile(files[0]);
      }
    });
    
    this.fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        this.handleFile(e.target.files[0]);
      }
    });
  }
  
  setupEventListeners() {
    this.recordBtn.addEventListener('click', () => {
      if (this.recorder.isRecording) {
        this.recorder.stop();
      } else {
        this.recorder.start().catch(err => {
          this.showStatus('error', err.message);
        });
      }
    });
    
    this.clearBtn.addEventListener('click', () => this.clearSelection());
    this.uploadBtn.addEventListener('click', () => this.upload());
  }
  
  handleFile(file) {
    // Validate
    const allowedTypes = ['audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/wave', 
                         'audio/ogg', 'audio/flac', 'audio/m4a', 'audio/webm'];
    
    if (!allowedTypes.some(t => file.type.includes(t.split('/')[1]))) {
      this.showStatus('error', 'Unsupported audio format');
      return;
    }
    
    if (file.size > 100 * 1024 * 1024) {
      this.showStatus('error', 'File too large (max 100MB)');
      return;
    }
    
    this.selectedFile = file;
    this.recordedBlob = null;
    this.showPreview(file, file.name);
  }
  
  showPreview(fileOrBlob, name) {
    this.previewSection.classList.remove('hidden');
    this.fileName.textContent = name;
    this.audioPreview.src = URL.createObjectURL(fileOrBlob);
    this.clearStatus();
  }
  
  clearSelection() {
    this.selectedFile = null;
    this.recordedBlob = null;
    this.previewSection.classList.add('hidden');
    this.audioPreview.src = '';
    this.recordDuration.classList.add('hidden');
    this.recordDuration.textContent = '0:00';
    this.fileInput.value = '';
    this.clearStatus();
  }
  
  showStatus(type, message, showProgress = false) {
    this.statusSection.className = `alert alert-${type === 'error' ? 'error' : type === 'success' ? 'success' : 'info'}`;
    this.statusSection.innerHTML = `
      ${type === 'processing' ? icons.loader : type === 'success' ? icons.checkCircle : ''}
      <div>
        <p>${message}</p>
        ${showProgress ? '<div class="progress mt-2"><div class="progress-bar" id="uploadProgress" style="width: 0%"></div></div>' : ''}
      </div>
    `;
    this.statusSection.classList.remove('hidden');
  }
  
  clearStatus() {
    this.statusSection.classList.add('hidden');
  }
  
  async upload() {
    const audioFile = this.selectedFile || this.recordedBlob;
    if (!audioFile) return;
    
    this.showStatus('info', 'Uploading...', true);
    this.uploadBtn.disabled = true;
    
    try {
      const formData = new FormData();
      
      if (this.recordedBlob) {
        const ext = this.recordedBlob.type.includes('webm') ? 'webm' : 'm4a';
        formData.append('file', this.recordedBlob, `recording_${Date.now()}.${ext}`);
      } else {
        formData.append('file', this.selectedFile);
      }
      
      formData.append('language', this.languageSelect.value);
      if (this.titleInput.value) {
        formData.append('title', this.titleInput.value);
      }
      
      // Get CSRF token
      const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
      
      const response = await fetch('/api/audio/', {
        method: 'POST',
        headers: {
          'X-CSRFToken': csrfToken
        },
        body: formData
      });
      
      if (!response.ok) throw new Error('Upload failed');
      
      const data = await response.json();
      
      this.showStatus('processing', 'Transcribing with Whisper...');
      
      // Poll for status
      this.pollStatus(data.audio_file.id);
      
    } catch (err) {
      this.showStatus('error', err.message);
      this.uploadBtn.disabled = false;
    }
  }
  
  async pollStatus(audioId) {
    const check = async () => {
      try {
        const data = await api.get(`/api/audio/${audioId}/status/`);
        
        if (data.audio_status === 'transcribed' || data.audio_status === 'corrected') {
          this.showStatus('success', 'Transcription complete!');
          setTimeout(() => {
            window.location.href = `/editor/${audioId}/`;
          }, 1000);
          
        } else if (data.audio_status === 'failed') {
          this.showStatus('error', 'Transcription failed. Please try again.');
          this.uploadBtn.disabled = false;
          
        } else {
          setTimeout(check, 2000);
        }
      } catch (err) {
        this.showStatus('error', 'Could not check status');
        this.uploadBtn.disabled = false;
      }
    };
    
    check();
  }
}


/**
 * Audio List Controller
 */
class AudioListController {
  constructor() {
    this.audioFiles = [];
    this.filter = 'all';
    this.searchQuery = '';
    
    this.init();
  }
  
  init() {
    this.listContainer = document.getElementById('audioList');
    this.searchInput = document.getElementById('searchInput');
    this.filterSelect = document.getElementById('filterSelect');
    this.refreshBtn = document.getElementById('refreshBtn');
    
    if (!this.listContainer) return;
    
    this.setupEventListeners();
    this.loadAudioFiles();
  }
  
  setupEventListeners() {
    this.searchInput?.addEventListener('input', (e) => {
      this.searchQuery = e.target.value.toLowerCase();
      this.render();
    });
    
    this.filterSelect?.addEventListener('change', (e) => {
      this.filter = e.target.value;
      this.render();
    });
    
    this.refreshBtn?.addEventListener('click', () => this.loadAudioFiles());
  }
  
  async loadAudioFiles() {
    this.listContainer.innerHTML = `
      <div class="text-center" style="padding: 3rem;">
        <div class="spinner spinner-lg" style="margin: 0 auto;"></div>
      </div>
    `;
    
    try {
      const data = await api.get('/api/audio/');
      this.audioFiles = data.results || data;
      this.render();
    } catch (err) {
      this.listContainer.innerHTML = `
        <div class="alert alert-error">
          Failed to load audio files. <button class="btn btn-sm btn-secondary" onclick="audioListController.loadAudioFiles()">Try Again</button>
        </div>
      `;
    }
  }
  
  getFilteredFiles() {
    return this.audioFiles.filter(file => {
      if (this.filter !== 'all' && file.status !== this.filter) return false;
      
      if (this.searchQuery) {
        const query = this.searchQuery;
        return (
          file.original_filename?.toLowerCase().includes(query) ||
          file.title?.toLowerCase().includes(query)
        );
      }
      
      return true;
    });
  }
  
  render() {
    const files = this.getFilteredFiles();
    
    if (files.length === 0) {
      this.listContainer.innerHTML = `
        <div class="empty-state card">
          ${icons.file}
          <p>No audio files found</p>
          <small>${this.searchQuery || this.filter !== 'all' ? 'Try adjusting your filters' : 'Upload some audio to get started'}</small>
        </div>
      `;
      return;
    }
    
    this.listContainer.innerHTML = files.map(file => `
      <div class="audio-item" onclick="window.location.href='/editor/${file.id}/'">
        <div class="audio-icon">${icons.file}</div>
        <div class="audio-info">
          <div class="audio-title">
            ${file.title || file.original_filename}
            ${this.getStatusBadge(file.status)}
          </div>
          <div class="audio-meta">
            <span>${icons.clock} ${formatTime(file.duration_seconds)}</span>
            <span>${new Date(file.created_at).toLocaleDateString()}</span>
            ${file.segment_count !== undefined ? `<span>${file.segment_count} segments</span>` : ''}
            ${file.corrected_count > 0 ? `<span style="color: var(--accent-green);">${file.corrected_count} corrected</span>` : ''}
          </div>
        </div>
        <div class="audio-actions">
          <button class="btn-icon" onclick="event.stopPropagation(); audioListController.deleteAudio('${file.id}')" title="Delete">
            ${icons.trash}
          </button>
        </div>
      </div>
    `).join('');
  }
  
  getStatusBadge(status) {
    const badges = {
      pending: '<span class="badge badge-pending">Pending</span>',
      processing: '<span class="badge badge-processing">' + icons.loader + ' Processing</span>',
      transcribed: '<span class="badge badge-transcribed">Transcribed</span>',
      corrected: '<span class="badge badge-corrected">Corrected</span>',
      approved: '<span class="badge badge-approved">Approved</span>',
      failed: '<span class="badge badge-failed">Failed</span>',
    };
    return badges[status] || badges.pending;
  }
  
  async deleteAudio(id) {
    if (!confirm('Are you sure you want to delete this audio file?')) return;
    
    try {
      await api.delete(`/api/audio/${id}/`);
      this.audioFiles = this.audioFiles.filter(f => f.id !== id);
      this.render();
    } catch (err) {
      alert('Failed to delete audio file');
    }
  }
}


/**
 * Editor Controller
 */
class EditorController {
  constructor(audioId) {
    this.audioId = audioId;
    this.audioFile = null;
    this.segments = [];
    this.editingId = null;
    this.activeSegmentId = null;
    
    this.init();
  }
  
  async init() {
    this.audioPlayer = document.getElementById('audioPlayer');
    this.playBtn = document.getElementById('playBtn');
    this.timeDisplay = document.getElementById('timeDisplay');
    this.progressTrack = document.getElementById('progressTrack');
    this.progressFill = document.getElementById('progressFill');
    this.segmentList = document.getElementById('segmentList');
    this.correctedCount = document.getElementById('correctedCount');
    this.totalCount = document.getElementById('totalCount');
    this.progressRing = document.getElementById('progressRing');
    
    if (!this.audioPlayer) return;
    
    this.setupAudioPlayer();
    await this.loadAudioFile();
  }
  
  setupAudioPlayer() {
    this.audioPlayer.addEventListener('timeupdate', () => {
      const current = this.audioPlayer.currentTime;
      const duration = this.audioPlayer.duration || 0;
      
      this.timeDisplay.textContent = `${formatTimeMs(current)} / ${formatTimeMs(duration)}`;
      this.progressFill.style.width = `${(current / duration) * 100}%`;
      
      // Highlight active segment
      const active = this.segments.find(
        seg => current >= seg.start_time && current < seg.end_time
      );
      
      if (active && active.id !== this.activeSegmentId) {
        this.activeSegmentId = active.id;
        this.highlightSegment(active.id);
      }
    });
    
    this.audioPlayer.addEventListener('play', () => {
      this.playBtn.innerHTML = icons.pause;
    });
    
    this.audioPlayer.addEventListener('pause', () => {
      this.playBtn.innerHTML = icons.play;
    });
    
    this.playBtn.addEventListener('click', () => {
      if (this.audioPlayer.paused) {
        this.audioPlayer.play();
      } else {
        this.audioPlayer.pause();
      }
    });
    
    this.progressTrack.addEventListener('click', (e) => {
      const rect = this.progressTrack.getBoundingClientRect();
      const percent = (e.clientX - rect.left) / rect.width;
      this.audioPlayer.currentTime = percent * this.audioPlayer.duration;
    });
    
    // Skip buttons
    document.getElementById('skipBack')?.addEventListener('click', () => {
      this.audioPlayer.currentTime = Math.max(0, this.audioPlayer.currentTime - 5);
    });
    
    document.getElementById('skipForward')?.addEventListener('click', () => {
      this.audioPlayer.currentTime = Math.min(
        this.audioPlayer.duration,
        this.audioPlayer.currentTime + 5
      );
    });
  }
  
  async loadAudioFile() {
    try {
      this.audioFile = await api.get(`/api/audio/${this.audioId}/`);
      this.segments = this.audioFile.segments || [];
      
      this.audioPlayer.src = this.audioFile.file_url || this.audioFile.file;
      
      this.updateProgress();
      this.renderSegments();
      
    } catch (err) {
      console.error('Failed to load audio:', err);
      this.segmentList.innerHTML = `
        <div class="alert alert-error">Failed to load audio file</div>
      `;
    }
  }
  
  updateProgress() {
    const corrected = this.segments.filter(s => s.is_corrected).length;
    const total = this.segments.length;
    const percent = total > 0 ? Math.round((corrected / total) * 100) : 0;
    
    this.correctedCount.textContent = corrected;
    this.totalCount.textContent = total;
    this.progressRing.style.width = `${percent}%`;
  }
  
  highlightSegment(id) {
    document.querySelectorAll('.segment-item').forEach(el => {
      el.classList.remove('active');
    });
    
    const el = document.querySelector(`[data-segment-id="${id}"]`);
    if (el) {
      el.classList.add('active');
    }
  }
  
  renderSegments() {
    if (this.segments.length === 0) {
      this.segmentList.innerHTML = `
        <div class="empty-state">
          <p>No transcript segments found.</p>
        </div>
      `;
      return;
    }
    
    this.segmentList.innerHTML = this.segments.map((seg, index) => `
      <div class="segment-item ${seg.is_corrected ? 'corrected' : ''}" 
           data-segment-id="${seg.id}"
           onclick="editorController.jumpToSegment(${seg.start_time})">
        <div class="segment-index">
          <span>#${index + 1}</span>
          <span class="timestamp">${formatTimeMs(seg.start_time)}</span>
        </div>
        
        <div class="segment-content" id="segment-content-${seg.id}">
          ${this.editingId === seg.id ? this.renderEditMode(seg) : this.renderViewMode(seg)}
        </div>
        
        <div class="segment-duration">
          <span class="duration-badge">
            ${icons.clock}
            ${(seg.end_time - seg.start_time).toFixed(1)}s
          </span>
        </div>
        
        <div class="segment-edit-btn">
          <button class="btn-icon" onclick="event.stopPropagation(); editorController.startEditing('${seg.id}')" title="Edit">
            ${icons.edit}
          </button>
        </div>
      </div>
    `).join('');
  }
  
  renderViewMode(seg) {
    return `
      <p class="segment-text">${seg.corrected_text || seg.text}</p>
      ${seg.is_corrected ? `
        <div class="segment-status">
          ${icons.checkCircle}
          Corrected
        </div>
      ` : ''}
    `;
  }
  
  renderEditMode(seg) {
    return `
      <div class="segment-edit" onclick="event.stopPropagation()">
        <div class="original-text">
          <span>Original: </span>${seg.text}
        </div>
        <textarea class="form-textarea" id="editText-${seg.id}" rows="3">${seg.corrected_text || seg.text}</textarea>
        <div class="edit-actions">
          <button class="btn btn-primary btn-sm" onclick="editorController.saveCorrection('${seg.id}')">
            ${icons.check} Save
          </button>
          <button class="btn btn-secondary btn-sm" onclick="editorController.cancelEditing()">
            ${icons.x} Cancel
          </button>
        </div>
      </div>
    `;
  }
  
  jumpToSegment(startTime) {
    if (this.editingId) return; // Don't jump while editing
    this.audioPlayer.currentTime = startTime;
    this.audioPlayer.play();
  }
  
  startEditing(id) {
    this.editingId = id;
    this.renderSegments();
    
    // Focus the textarea
    setTimeout(() => {
      const textarea = document.getElementById(`editText-${id}`);
      if (textarea) textarea.focus();
    }, 50);
  }
  
  cancelEditing() {
    this.editingId = null;
    this.renderSegments();
  }
  
  async saveCorrection(id) {
    const textarea = document.getElementById(`editText-${id}`);
    const correctedText = textarea.value.trim();
    
    if (!correctedText) {
      alert('Please enter corrected text');
      return;
    }
    
    try {
      await api.post(`/api/segments/${id}/correct/`, {
        corrected_text: correctedText,
        correction_type: 'word'
      });
      
      // Update local state
      const segIndex = this.segments.findIndex(s => s.id === id);
      if (segIndex !== -1) {
        this.segments[segIndex].corrected_text = correctedText;
        this.segments[segIndex].is_corrected = true;
      }
      
      this.editingId = null;
      this.updateProgress();
      this.renderSegments();
      
    } catch (err) {
      alert('Failed to save correction');
    }
  }
}


// Initialize controllers based on page
document.addEventListener('DOMContentLoaded', () => {
  // Upload page
  if (document.getElementById('recordBtn')) {
    window.uploadController = new UploadController();
  }
  
  // Audio list page
  if (document.getElementById('audioList')) {
    window.audioListController = new AudioListController();
  }
  
  // Editor page
  const editorEl = document.getElementById('editorPage');
  if (editorEl) {
    const audioId = editorEl.dataset.audioId;
    window.editorController = new EditorController(audioId);
  }
});
