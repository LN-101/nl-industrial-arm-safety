"""Static HTML for the mobile web control surface."""

from __future__ import annotations

import html


def render_index_html(title: str, *, moss_pcm_buffer_seconds: float = 0.48) -> str:
    page_title = html.escape(title)
    return (
        INDEX_HTML.replace("__PAGE_TITLE__", page_title)
        .replace("__MOSS_PCM_BUFFER_SECONDS__", format(moss_pcm_buffer_seconds, ".3f"))
    )


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#111418">
  <title>__PAGE_TITLE__</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0e1114;
      --panel: #171c21;
      --panel-soft: #1d232a;
      --border: #2b3138;
      --text: #e7edf2;
      --muted: #9da7b1;
      --accent: #47c9a2;
      --accent-strong: #2ea07d;
      --danger: #d94b4b;
      --danger-strong: #b43a3a;
      --warning: #d6a44a;
      --input: #10151a;
      --shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
      --radius: 8px;
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #0e1114 0%, #0d1013 100%);
      color: var(--text);
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    button, input, textarea {
      font: inherit;
      color: inherit;
    }

    .app {
      min-height: 100%;
      display: flex;
      flex-direction: column;
      gap: 0;
    }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 14px 10px;
      background: rgba(14, 17, 20, 0.94);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(10px);
    }

    .brand {
      min-width: 0;
      display: grid;
      gap: 3px;
    }

    .brand h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .brand .status {
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      min-height: 1.1em;
    }

    .stop-btn {
      flex: 0 0 auto;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 44px;
      padding: 0 14px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: var(--radius);
      background: var(--danger);
      color: #fff;
      cursor: pointer;
      box-shadow: var(--shadow);
    }

    .stop-btn:active { background: var(--danger-strong); }
    .stop-btn[disabled] { opacity: 0.45; cursor: not-allowed; box-shadow: none; }

    .shell {
      width: min(100%, 960px);
      margin: 0 auto;
      padding: 14px;
      display: grid;
      gap: 14px;
      flex: 1 1 auto;
      min-height: 0;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }

    .login-panel {
      padding: 14px;
      display: grid;
      gap: 12px;
      max-width: 420px;
      width: 100%;
      align-self: center;
      margin: 12px auto 0;
    }

    .login-panel .title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .login-panel h2 {
      margin: 0;
      font-size: 16px;
      line-height: 1.2;
    }

    .fields {
      display: grid;
      gap: 10px;
    }

    .field {
      display: grid;
      gap: 6px;
    }

    .field label {
      color: var(--muted);
      font-size: 12px;
    }

    .field input, .composer textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--input);
      padding: 12px 12px;
      outline: none;
    }

    .field input:focus, .composer textarea:focus {
      border-color: rgba(71, 201, 162, 0.75);
      box-shadow: 0 0 0 2px rgba(71, 201, 162, 0.16);
    }

    .primary-row {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .primary-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 44px;
      padding: 0 14px;
      border: 1px solid transparent;
      border-radius: var(--radius);
      background: var(--accent);
      color: #07110d;
      font-weight: 650;
      cursor: pointer;
    }

    .primary-btn:active { background: var(--accent-strong); }

    .ghost-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 44px;
      padding: 0 14px;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--panel-soft);
      cursor: pointer;
    }

    .ghost-btn:active { background: #252c33; }

    .chat-panel {
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      min-height: calc(100dvh - 110px);
      overflow: hidden;
    }

    .messages {
      padding: 10px;
      overflow: auto;
      display: grid;
      gap: 10px;
      align-content: start;
      background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
    }

    .message {
      display: grid;
      gap: 4px;
      max-width: min(92%, 680px);
      width: fit-content;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--panel-soft);
      white-space: pre-wrap;
      word-break: break-word;
    }

    .message.user {
      justify-self: end;
      background: rgba(71, 201, 162, 0.12);
      border-color: rgba(71, 201, 162, 0.24);
    }

    .message.assistant {
      justify-self: start;
    }

    .message.system {
      justify-self: center;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.03);
      max-width: 100%;
    }

    .message .meta {
      color: var(--muted);
      font-size: 11px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .message-attachments {
      display: grid;
      gap: 6px;
      margin-top: 4px;
    }

    .message-image {
      display: block;
      width: min(100%, 420px);
      max-height: 320px;
      object-fit: contain;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #090c0f;
    }

    .image-caption {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.3;
    }

    .composer {
      position: sticky;
      bottom: 0;
      display: grid;
      gap: 10px;
      padding: 10px;
      background: rgba(14, 17, 20, 0.96);
      border-top: 1px solid var(--border);
      backdrop-filter: blur(10px);
    }

    .composer textarea {
      min-height: 56px;
      max-height: 160px;
      resize: vertical;
    }

    .composer-bar {
      display: grid;
      grid-template-columns: minmax(120px, 1fr) auto auto;
      gap: 10px;
      align-items: center;
    }

    .status-line {
      color: var(--muted);
      font-size: 12px;
      min-height: 1.2em;
    }

    .free-voice-panel {
      position: fixed;
      right: max(16px, env(safe-area-inset-right));
      bottom: calc(96px + env(safe-area-inset-bottom));
      z-index: 15;
      display: grid;
      place-items: center;
      gap: 8px;
      padding: 0;
      border: 0;
      background: transparent;
      pointer-events: none;
    }

    .voice-sphere {
      width: clamp(86px, 22vw, 128px);
      aspect-ratio: 1;
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 50%;
      background:
        radial-gradient(circle at 35% 30%, rgba(255,255,255,0.22), transparent 24%),
        radial-gradient(circle at center, rgba(71,201,162,0.72), rgba(36,111,91,0.92) 62%, rgba(16,44,38,1));
      box-shadow: 0 18px 50px rgba(0,0,0,0.32), inset 0 0 26px rgba(255,255,255,0.09);
      cursor: pointer;
      transform: translateZ(0);
      transition: transform 160ms ease, filter 160ms ease, background 160ms ease;
      pointer-events: auto;
    }

    .free-voice-panel .status-line {
      min-height: 0;
      padding: 3px 8px;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: rgba(14, 17, 20, 0.86);
      color: var(--text);
    }

    .voice-sphere.listening { animation: sphere-breathe 2.2s ease-in-out infinite; }
    .voice-sphere.speaking { animation: sphere-pulse 520ms ease-in-out infinite; filter: saturate(1.18); }
    .voice-sphere.submitting { animation: sphere-spin 1.1s linear infinite; }
    .voice-sphere.playback {
      animation: sphere-wave 820ms ease-in-out infinite;
      background:
        radial-gradient(circle at 40% 28%, rgba(255,255,255,0.22), transparent 24%),
        radial-gradient(circle at center, rgba(214,164,74,0.82), rgba(126,86,28,0.94) 64%, rgba(54,39,14,1));
    }
    .voice-sphere.interrupted { filter: grayscale(0.15) brightness(0.84); }
    .voice-sphere.error {
      background:
        radial-gradient(circle at 35% 30%, rgba(255,255,255,0.18), transparent 24%),
        radial-gradient(circle at center, rgba(217,75,75,0.8), rgba(104,34,34,0.94) 66%, rgba(45,17,17,1));
    }

    @keyframes sphere-breathe {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.035); }
    }

    @keyframes sphere-pulse {
      0%, 100% { transform: scale(1); box-shadow: 0 18px 50px rgba(0,0,0,0.32), 0 0 0 0 rgba(71,201,162,0.24); }
      50% { transform: scale(1.07); box-shadow: 0 18px 50px rgba(0,0,0,0.32), 0 0 0 16px rgba(71,201,162,0.02); }
    }

    @keyframes sphere-wave {
      0%, 100% { transform: scale(1.02); }
      33% { transform: scale(1.09) skewX(1.5deg); }
      66% { transform: scale(0.98) skewX(-1.5deg); }
    }

    @keyframes sphere-spin {
      from { transform: rotate(0deg) scale(1.02); }
      to { transform: rotate(360deg) scale(1.02); }
    }

    .confirmation-backdrop {
      position: fixed;
      inset: 0;
      z-index: 20;
      display: grid;
      place-items: center;
      padding: 16px;
      background: rgba(0,0,0,0.62);
    }

    .confirmation-dialog {
      width: min(100%, 460px);
      max-height: min(88dvh, 680px);
      overflow: auto;
      padding: 14px;
    }

    .confirmation-dialog h2 {
      margin: 0 0 8px;
      font-size: 18px;
      line-height: 1.2;
    }

    .confirmation-dialog p {
      margin: 8px 0;
      color: var(--muted);
    }

    .confirmation-list {
      margin: 10px 0;
      padding-left: 18px;
      color: var(--text);
    }

    .confirmation-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }

    .emergency-dialog {
      border: 1px solid var(--danger);
      box-shadow: 0 0 0 2px rgba(217, 75, 75, 0.28), var(--shadow);
    }

    .emergency-dialog h2 {
      color: var(--danger);
    }

    .emergency-details {
      white-space: pre-wrap;
      color: var(--text);
    }

    .emergency-actions {
      grid-template-columns: 1fr;
    }

    .hidden { display: none !important; }

    .icon {
      width: 18px;
      height: 18px;
      display: inline-block;
      vertical-align: middle;
      flex: 0 0 auto;
    }

    @media (max-width: 640px) {
      .shell { padding: 10px; }
      .chat-panel { min-height: calc(100dvh - 100px); }
      .composer-bar { grid-template-columns: 1fr 1fr; }
      .composer-bar .send-btn { grid-column: 1 / -1; }
      .free-voice-panel {
        right: 14px;
        bottom: calc(124px + env(safe-area-inset-bottom));
      }
      .message { max-width: 100%; }
      .topbar { align-items: stretch; }
      .stop-btn { min-width: 108px; justify-content: center; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <h1>__PAGE_TITLE__</h1>
        <p class="status" id="statusText">未登录</p>
      </div>
      <button class="stop-btn" id="stopBtn" type="button" disabled>
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true" fill="none">
          <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"></rect>
        </svg>
        <span>急停</span>
      </button>
    </header>

    <main class="shell">
      <section class="panel login-panel" id="loginPanel">
        <div class="title">
          <h2>登录</h2>
        </div>
        <div class="fields">
          <div class="field">
            <label for="username">用户名</label>
            <input id="username" name="username" autocomplete="username" value="admin">
          </div>
          <div class="field">
            <label for="password">密码</label>
            <input id="password" name="password" type="password" autocomplete="current-password">
          </div>
        </div>
        <div class="primary-row">
          <button class="primary-btn" id="loginBtn" type="button">
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true" fill="none">
              <path d="M10 17l5-5-5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path>
              <path d="M15 12H4" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path>
            </svg>
            <span>登录</span>
          </button>
        </div>
        <div class="status-line" id="loginStatus"></div>
      </section>

      <section class="panel chat-panel hidden" id="chatPanel">
        <div class="free-voice-panel hidden" id="freeVoicePanel">
          <button class="voice-sphere listening" id="voiceSphere" type="button" aria-label="自由语音"></button>
          <div class="status-line" id="freeVoiceStatus"></div>
        </div>
        <div class="messages" id="messages" aria-live="polite"></div>
        <form class="composer" id="chatForm">
          <textarea id="chatInput" placeholder=""></textarea>
          <div class="composer-bar">
            <button class="ghost-btn free-voice-toggle" id="freeVoiceBtn" type="button">
              <svg class="icon" viewBox="0 0 24 24" aria-hidden="true" fill="none">
                <rect x="9" y="4" width="6" height="12" rx="3" fill="currentColor"></rect>
                <path d="M5 12a7 7 0 0 0 14 0" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path>
                <path d="M12 19v3" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path>
              </svg>
              <span>自由语音</span>
            </button>
            <button class="ghost-btn" id="clearBtn" type="button">清空</button>
            <button class="primary-btn send-btn" id="sendBtn" type="submit">
              <svg class="icon" viewBox="0 0 24 24" aria-hidden="true" fill="none">
                <path d="M4 12h14" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path>
                <path d="M12 6l6 6-6 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path>
              </svg>
              <span>发送</span>
            </button>
          </div>
          <div class="status-line" id="chatStatus"></div>
        </form>
      </section>
    </main>
  </div>

  <div class="confirmation-backdrop hidden" id="confirmationModal" role="dialog" aria-modal="true" aria-labelledby="confirmationTitle">
    <section class="panel confirmation-dialog">
      <h2 id="confirmationTitle">确认操作</h2>
      <p id="confirmationOriginal"></p>
      <p id="confirmationPrompt"></p>
      <ul class="confirmation-list" id="confirmationChecklist"></ul>
      <div class="confirmation-actions">
        <button class="ghost-btn" id="confirmationCancelBtn" type="button">取消</button>
        <button class="primary-btn" id="confirmationConfirmBtn" type="button">确认</button>
      </div>
    </section>
  </div>

  <div class="confirmation-backdrop hidden" id="emergencyAlertModal" role="alertdialog" aria-modal="true" aria-labelledby="emergencyAlertTitle">
    <section class="panel confirmation-dialog emergency-dialog">
      <h2 id="emergencyAlertTitle">紧急停止告警</h2>
      <p class="emergency-details" id="emergencyAlertDetails"></p>
      <p class="status-line" id="emergencyAlertAudioStatus"></p>
      <p>关闭此提示不会解除急停；急停解除仍由 ROS2 停止源控制。</p>
      <div class="confirmation-actions emergency-actions">
        <button class="primary-btn" id="emergencyAlertDismissBtn" type="button">知道了</button>
      </div>
    </section>
  </div>

  <script>
    const FREE_VOICE_VAD_INTERVAL_MS = 80;
    const FREE_VOICE_SPEECH_THRESHOLD = 0.2;
    const FREE_VOICE_TRAILING_SILENCE_MS = 1300;
    const FREE_VOICE_MIN_SPEECH_MS = 260;
    const FREE_VOICE_PRE_ROLL_MS = 500;
    const FREE_VOICE_RECORDER_TIMESLICE_MS = 100;
    const PCM_TARGET_BUFFER_SECONDS = __MOSS_PCM_BUFFER_SECONDS__;
    const PCM_REBUFFER_SECONDS = 0.64;
    const PCM_MIN_SCHEDULE_LEAD_SECONDS = 0.10;
    const PCM_MIN_SCHEDULE_SECONDS = 0.16;
    const PCM_STREAM_FADE_SECONDS = 0.012;
    const PCM_FINAL_FADE_SECONDS = 0.018;
    const EMERGENCY_ALERT_POLL_MS = 2000;

    const state = {
      authenticated: false,
      playback: Promise.resolve(),
      activePlayback: null,
      activeRequestControllers: new Set(),
      requestEpoch: 0,
      cancelBarrier: Promise.resolve(),
      pendingConfirmation: null,
      emergency: {
        seenEventId: null,
        pollTimer: null,
        audio: null,
      },
      freeVoice: {
        active: false,
        stream: null,
        audioContext: null,
        analyser: null,
        vadTimer: null,
        recorder: null,
        chunks: [],
        preRollChunks: [],
        preRollHeaderChunk: null,
        speaking: false,
        finalizing: false,
        speechStartedAt: 0,
        silenceStartedAt: 0,
        submitting: false,
      },
    };

    const loginPanel = document.getElementById("loginPanel");
    const chatPanel = document.getElementById("chatPanel");
    const loginStatus = document.getElementById("loginStatus");
    const chatStatus = document.getElementById("chatStatus");
    const statusText = document.getElementById("statusText");
    const messages = document.getElementById("messages");
    const loginBtn = document.getElementById("loginBtn");
    const clearBtn = document.getElementById("clearBtn");
    const stopBtn = document.getElementById("stopBtn");
    const freeVoiceBtn = document.getElementById("freeVoiceBtn");
    const freeVoicePanel = document.getElementById("freeVoicePanel");
    const freeVoiceStatus = document.getElementById("freeVoiceStatus");
    const voiceSphere = document.getElementById("voiceSphere");
    const confirmationModal = document.getElementById("confirmationModal");
    const confirmationTitle = document.getElementById("confirmationTitle");
    const confirmationOriginal = document.getElementById("confirmationOriginal");
    const confirmationPrompt = document.getElementById("confirmationPrompt");
    const confirmationChecklist = document.getElementById("confirmationChecklist");
    const confirmationCancelBtn = document.getElementById("confirmationCancelBtn");
    const confirmationConfirmBtn = document.getElementById("confirmationConfirmBtn");
    const emergencyAlertModal = document.getElementById("emergencyAlertModal");
    const emergencyAlertDetails = document.getElementById("emergencyAlertDetails");
    const emergencyAlertAudioStatus = document.getElementById("emergencyAlertAudioStatus");
    const emergencyAlertDismissBtn = document.getElementById("emergencyAlertDismissBtn");
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const usernameInput = document.getElementById("username");
    const passwordInput = document.getElementById("password");

    function iconText(label) {
      return label;
    }

    function imageCaption(artifact) {
      const parts = [];
      if (artifact.caption) {
        parts.push(artifact.caption);
      }
      const metadata = artifact.metadata || {};
      if (metadata.stamp) {
        parts.push(String(metadata.stamp));
      }
      if (metadata.frame_id) {
        parts.push(String(metadata.frame_id));
      }
      return parts.join(" · ");
    }

    function appendMessage(role, text, meta, imageArtifacts = []) {
      const item = document.createElement("article");
      item.className = `message ${role}`;
      const body = document.createElement("div");
      body.textContent = text;
      item.appendChild(body);
      if (Array.isArray(imageArtifacts) && imageArtifacts.length > 0) {
        const attachments = document.createElement("div");
        attachments.className = "message-attachments";
        imageArtifacts.forEach((artifact) => {
          if (!artifact || !artifact.url) {
            return;
          }
          const image = document.createElement("img");
          image.className = "message-image";
          image.src = artifact.url;
          image.alt = artifact.caption || "视觉快照";
          image.loading = "lazy";
          attachments.appendChild(image);
          const caption = imageCaption(artifact);
          if (caption) {
            const captionLine = document.createElement("div");
            captionLine.className = "image-caption";
            captionLine.textContent = caption;
            attachments.appendChild(captionLine);
          }
        });
        item.appendChild(attachments);
      }
      if (meta) {
        const metaLine = document.createElement("div");
        metaLine.className = "meta";
        metaLine.textContent = meta;
        item.appendChild(metaLine);
      }
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
      return item;
    }

    function setStatus(text, target = statusText) {
      target.textContent = text || "";
    }

    function setFreeVoiceButtonLabel(label) {
      const labelNode = freeVoiceBtn.querySelector("span");
      if (labelNode) {
        labelNode.textContent = label;
      }
    }

    function setAuthenticated(flag) {
      state.authenticated = flag;
      loginPanel.classList.toggle("hidden", flag);
      chatPanel.classList.toggle("hidden", !flag);
      stopBtn.disabled = !flag;
      if (flag) {
        chatInput.focus();
      }
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, {
        credentials: "include",
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
        ...options,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const error = new Error(payload.error || response.statusText || "请求失败");
        error.payload = payload;
        error.status = response.status;
        throw error;
      }
      return payload;
    }

    function beginRequestSession() {
      const controller = new AbortController();
      const session = { epoch: state.requestEpoch, controller };
      state.activeRequestControllers.add(controller);
      return session;
    }

    function endRequestSession(session) {
      state.activeRequestControllers.delete(session.controller);
    }

    function isStaleRequest(session) {
      return session.epoch !== state.requestEpoch || session.controller.signal.aborted;
    }

    function abortActiveRequests() {
      state.requestEpoch += 1;
      state.activeRequestControllers.forEach((controller) => {
        try {
          controller.abort();
        } catch (error) {
          /* aborting an already-finished request is fine */
        }
      });
      state.activeRequestControllers.clear();
    }

    async function requestBackendCancel(reason) {
      try {
        await fetchJson("/api/turn/cancel", {
          method: "POST",
          body: JSON.stringify({ reason }),
        });
      } catch (error) {
        setStatus(error.message || "后端取消请求失败", chatStatus);
        throw error;
      }
    }

    async function cancelActiveWebWork(reason) {
      const cancellationEpoch = interruptActiveWebWorkLocally();
      showConfirmation(null);
      const backendCancel = state.cancelBarrier.then(() => requestBackendCancel(reason));
      state.cancelBarrier = backendCancel.catch(() => {});
      await backendCancel;
      return cancellationEpoch;
    }

    function interruptActiveWebWorkLocally() {
      stopActivePlayback();
      abortActiveRequests();
      return state.requestEpoch;
    }

    function showConfirmation(confirmation) {
      state.pendingConfirmation = confirmation || null;
      confirmationModal.classList.toggle("hidden", !confirmation);
      if (!confirmation) {
        return;
      }
      confirmationTitle.textContent = confirmation.summary || "确认操作";
      confirmationOriginal.textContent = confirmation.original_text ? `指令：${confirmation.original_text}` : "";
      confirmationPrompt.textContent = confirmation.prompt || "";
      confirmationChecklist.innerHTML = "";
      (confirmation.checklist || []).forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        confirmationChecklist.appendChild(li);
      });
      confirmationConfirmBtn.disabled = false;
      confirmationCancelBtn.disabled = false;
      confirmationConfirmBtn.focus();
    }

    async function confirmPending() {
      const confirmation = state.pendingConfirmation;
      if (!confirmation || !confirmation.id) {
        showConfirmation(null);
        return;
      }
      confirmationConfirmBtn.disabled = true;
      confirmationCancelBtn.disabled = true;
      interruptActiveWebWorkLocally();
      const session = beginRequestSession();
      setStatus("确认处理中...", chatStatus);
      try {
        const payload = await fetchJson("/api/confirmation/confirm", {
          method: "POST",
          signal: session.controller.signal,
          body: JSON.stringify({ id: confirmation.id }),
        });
        if (isStaleRequest(session)) {
          return;
        }
        showConfirmation(null);
        appendConfirmationAssistantMessage(payload, "已确认");
        setStatus(payload.status_line || "已确认", chatStatus);
        await followProgressiveTurn(payload, session);
      } catch (error) {
        if (error.name === "AbortError" || isStaleRequest(session)) {
          return;
        }
        showConfirmation(null);
        appendMessage("system", error.message, "错误");
        setStatus(error.message, chatStatus);
      } finally {
        endRequestSession(session);
      }
    }

    async function cancelPending() {
      const confirmation = state.pendingConfirmation;
      const id = confirmation && confirmation.id ? confirmation.id : null;
      confirmationConfirmBtn.disabled = true;
      confirmationCancelBtn.disabled = true;
      interruptActiveWebWorkLocally();
      const session = beginRequestSession();
      setStatus("取消确认中...", chatStatus);
      try {
        const payload = await fetchJson("/api/confirmation/cancel", {
          method: "POST",
          signal: session.controller.signal,
          body: JSON.stringify(id ? { id } : {}),
        });
        if (isStaleRequest(session)) {
          return;
        }
        showConfirmation(null);
        appendConfirmationAssistantMessage(payload, "已取消");
        setStatus(payload.status_line || "已取消", chatStatus);
        await followProgressiveTurn(payload, session);
      } catch (error) {
        if (error.name === "AbortError" || isStaleRequest(session)) {
          return;
        }
        showConfirmation(null);
        appendMessage("system", error.message, "错误");
        setStatus(error.message, chatStatus);
      } finally {
        endRequestSession(session);
      }
    }

    function appendConfirmationAssistantMessage(payload, fallbackText) {
      const text = payload.response_text || payload.message || fallbackText;
      if (!text) {
        return;
      }
      appendMessage("assistant", text, payload.ros2_error ? `ROS2: ${payload.ros2_error}` : "助手");
    }

    function handleConfirmationPayload(payload) {
      if (payload && payload.metadata && payload.metadata.confirmation_resolved) {
        showConfirmation(null);
        return;
      }
      if (payload && payload.confirmation) {
        showConfirmation(payload.confirmation);
        setStatus(payload.status_line || "等待确认", chatStatus);
      }
    }

    function formatEmergencyDistance(value) {
      return value.toFixed(2);
    }

    function formatEmergencyReason(reason) {
      return reason.replace(/(\\d+(?:\\.\\d+)?)m\\b/g, (match, value) => `${formatEmergencyDistance(Number(value))}m`);
    }

    function emergencyEventDetailLines(event) {
      const lines = [];
      lines.push(`来源：${event.source || "未知"}`);
      if (event.reason) {
        lines.push(`原因：${formatEmergencyReason(event.reason)}`);
      }
      if (typeof event.trigger_distance_m === "number") {
        lines.push(`触发距离：${formatEmergencyDistance(event.trigger_distance_m)} 米`);
      }
      if (typeof event.distance_m === "number") {
        lines.push(`当前人机距离：${formatEmergencyDistance(event.distance_m)} 米`);
      } else if (event.source === "min_distance_camera") {
        lines.push("当前人机距离：不可用");
      }
      if (typeof event.release_distance_m === "number") {
        lines.push(`解除门槛：${formatEmergencyDistance(event.release_distance_m)} 米`);
      }
      return lines;
    }

    function stopEmergencyAlertAudio() {
      if (state.emergency.audio) {
        try {
          state.emergency.audio.pause();
        } catch (error) {
          /* pausing an ended audio element is fine */
        }
        state.emergency.audio = null;
      }
    }

    function playEmergencyAlertAudio(audioUrl) {
      stopEmergencyAlertAudio();
      if (!audioUrl) {
        setStatus("未配置急停告警音频文件，无法播放语音提示。", emergencyAlertAudioStatus);
        return;
      }
      const audio = new Audio(audioUrl);
      state.emergency.audio = audio;
      setStatus("正在播放急停语音提示...", emergencyAlertAudioStatus);
      audio.addEventListener("ended", () => {
        setStatus("急停语音提示已播放。", emergencyAlertAudioStatus);
      });
      audio.play().catch((error) => {
        setStatus(`告警音频播放被浏览器拦截或失败：${error.message || error}`, emergencyAlertAudioStatus);
      });
    }

    function updateEmergencyAlertDetails(event) {
      emergencyAlertDetails.textContent = emergencyEventDetailLines(event).join("\\n");
    }

    function showEmergencyAlert(event) {
      updateEmergencyAlertDetails(event);
      emergencyAlertModal.classList.remove("hidden");
      emergencyAlertDismissBtn.focus();
    }

    function hideEmergencyAlert() {
      emergencyAlertModal.classList.add("hidden");
      stopEmergencyAlertAudio();
    }

    function handleEmergencyStatusPayload(payload) {
      const event = payload && payload.emergency_event;
      if (!event || !event.active) {
        return;
      }
      if (event.event_id === state.emergency.seenEventId) {
        if (!emergencyAlertModal.classList.contains("hidden")) {
          updateEmergencyAlertDetails(event);
        }
        return;
      }
      state.emergency.seenEventId = event.event_id;
      interruptActiveWebWorkLocally();
      showConfirmation(null);
      showEmergencyAlert(event);
      playEmergencyAlertAudio(payload.emergency_alert_audio_url);
      setStatus("外部急停已触发", chatStatus);
    }

    function startEmergencyStatusPolling() {
      if (state.emergency.pollTimer !== null) {
        return;
      }
      state.emergency.pollTimer = window.setInterval(async () => {
        try {
          const payload = await fetchJson("/api/status", { method: "GET", headers: undefined });
          handleEmergencyStatusPayload(payload);
        } catch (error) {
          /* transient status polling failures are ignored */
        }
      }, EMERGENCY_ALERT_POLL_MS);
    }

    async function refreshStatus() {
      const payload = await fetchJson("/api/status", { method: "GET", headers: undefined });
      setAuthenticated(Boolean(payload.authenticated));
      setStatus(payload.status_line || (payload.authenticated ? "已登录" : "未登录"));
      stopBtn.disabled = !payload.authenticated;
      if (payload.authenticated) {
        appendMessage("system", `已连接：${payload.resolved_tts_engine || "unknown"} / ${payload.openvino_devices?.join(", ") || "no device"}`, "状态");
        if (payload.pending_confirmation) {
          showConfirmation(payload.pending_confirmation);
        }
        startEmergencyStatusPolling();
      }
      handleEmergencyStatusPayload(payload);
    }

    async function login() {
      setStatus("登录中...", loginStatus);
      try {
        const payload = await fetchJson("/api/login", {
          method: "POST",
          body: JSON.stringify({
            username: usernameInput.value.trim(),
            password: passwordInput.value,
          }),
        });
        setAuthenticated(true);
        setStatus(payload.status_line || "已登录", statusText);
        setStatus("", loginStatus);
        appendMessage("system", "登录成功", "状态");
        startEmergencyStatusPolling();
        await refreshStatus();
      } catch (error) {
        setStatus(error.message, loginStatus);
      }
    }

    async function sendChat(text) {
      const content = text.trim();
      if (!content) {
        return;
      }
      let supersessionEpoch;
      try {
        supersessionEpoch = await cancelActiveWebWork("text_superseded");
      } catch (error) {
        appendMessage("system", error.message || "后端取消请求失败", "错误");
        return;
      }
      if (supersessionEpoch !== state.requestEpoch) {
        return;
      }
      const session = beginRequestSession();
      appendMessage("user", content, "你");
      setStatus("生成回复...", chatStatus);
      chatInput.value = "";
      try {
        const payload = await fetchJson("/api/chat-stream", {
          method: "POST",
          signal: session.controller.signal,
          body: JSON.stringify({ text: content }),
        });
        if (isStaleRequest(session)) {
          return;
        }
        appendMessage(
          "assistant",
          payload.response_text || "",
          payload.ros2_error ? `ROS2: ${payload.ros2_error}` : "助手",
          payload.image_artifacts || []
        );
        handleConfirmationPayload(payload);
        await followProgressiveTurn(payload, session);
      } catch (error) {
        if (error.name === "AbortError" || isStaleRequest(session)) {
          return;
        }
        appendMessage("system", error.message, "错误");
        setStatus(error.message, chatStatus);
      } finally {
        endRequestSession(session);
      }
    }

    function wait(ms) {
      return new Promise((resolve) => window.setTimeout(resolve, ms));
    }

    function mergeUint8Arrays(a, b) {
      const merged = new Uint8Array(a.length + b.length);
      merged.set(a, 0);
      merged.set(b, a.length);
      return merged;
    }

    function pcmByteLengthForSeconds(seconds, sampleRate, channels) {
      const bytesPerFrame = channels * 2;
      const frames = Math.max(1, Math.floor(sampleRate * seconds));
      return frames * bytesPerFrame;
    }

    function scheduleBufferedPcm(audioContext, pendingPcm, minScheduleBytes, bytesPerFrame, sampleRate, channels, playbackState, flush = false) {
      const alignedLength = Math.floor(pendingPcm.byteLength / bytesPerFrame) * bytesPerFrame;
      if (alignedLength <= 0) {
        return pendingPcm;
      }
      if (!flush) {
        const underflowed = Boolean(
          playbackState.nextTime &&
          playbackState.nextTime < audioContext.currentTime + PCM_MIN_SCHEDULE_LEAD_SECONDS
        );
        const requiredSeconds = !playbackState.started
          ? PCM_TARGET_BUFFER_SECONDS
          : (underflowed ? PCM_REBUFFER_SECONDS : PCM_MIN_SCHEDULE_SECONDS);
        const requiredBytes = pcmByteLengthForSeconds(requiredSeconds, sampleRate, channels);
        if (alignedLength < requiredBytes) {
          return pendingPcm;
        }
      }
      schedulePcmChunk(audioContext, pendingPcm.subarray(0, alignedLength), sampleRate, channels, playbackState);
      return pendingPcm.subarray(alignedLength);
    }

    function schedulePcmChunk(audioContext, pcmChunk, sampleRate, channels, playbackState) {
      const bytesPerFrame = channels * 2;
      const totalFrames = Math.floor(pcmChunk.byteLength / bytesPerFrame);
      if (totalFrames <= 0) {
        return;
      }
      const audioBuffer = audioContext.createBuffer(channels, totalFrames, sampleRate);
      const view = new DataView(pcmChunk.buffer, pcmChunk.byteOffset, totalFrames * bytesPerFrame);
      for (let channelIndex = 0; channelIndex < channels; channelIndex += 1) {
        const channelData = audioBuffer.getChannelData(channelIndex);
        for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
          const byteOffset = (frameIndex * channels + channelIndex) * 2;
          channelData[frameIndex] = view.getInt16(byteOffset, true) / 32768.0;
        }
      }
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      const gainNode = audioContext.createGain();
      source.connect(gainNode);
      gainNode.connect(audioContext.destination);
      const now = audioContext.currentTime;
      const underflowed = Boolean(playbackState.nextTime && playbackState.nextTime < now + PCM_MIN_SCHEDULE_LEAD_SECONDS);
      const startAt = Math.max(
        playbackState.nextTime || (now + PCM_MIN_SCHEDULE_LEAD_SECONDS),
        now + PCM_MIN_SCHEDULE_LEAD_SECONDS
      );
      if (!playbackState.fadeStarted || underflowed) {
        const fadeDuration = Math.min(PCM_STREAM_FADE_SECONDS, audioBuffer.duration / 2);
        gainNode.gain.setValueAtTime(0.0, startAt);
        gainNode.gain.linearRampToValueAtTime(1.0, startAt + fadeDuration);
        playbackState.fadeStarted = true;
      } else {
        gainNode.gain.setValueAtTime(1.0, startAt);
      }
      source.start(startAt);
      playbackState.started = true;
      playbackState.nextTime = startAt + audioBuffer.duration;
      playbackState.lastGainNode = gainNode;
      playbackState.lastStartAt = startAt;
      playbackState.lastDuration = audioBuffer.duration;
    }

    function applyFinalPcmFade(audioContext, playbackState) {
      if (!playbackState.lastGainNode || !playbackState.lastDuration) {
        return;
      }
      const fadeDuration = Math.min(PCM_FINAL_FADE_SECONDS, playbackState.lastDuration / 2);
      if (fadeDuration <= 0) {
        return;
      }
      const endAt = playbackState.lastStartAt + playbackState.lastDuration;
      const now = audioContext.currentTime;
      if (endAt <= now) {
        return;
      }
      const fadeStart = Math.max(playbackState.lastStartAt, endAt - fadeDuration, now);
      playbackState.lastGainNode.gain.setValueAtTime(1.0, fadeStart);
      playbackState.lastGainNode.gain.linearRampToValueAtTime(0.0, endAt);
    }

    function stopActivePlayback() {
      resetPlaybackQueue();
      const playback = state.activePlayback;
      if (!playback) {
        return;
      }
      state.activePlayback = null;
      playback.stopped = true;
      try {
        playback.controller.abort();
      } catch (error) {
        /* aborting an already-finished fetch is fine */
      }
      if (playback.audioContext && playback.audioContext.state !== "closed") {
        playback.audioContext.close().catch(() => {});
      }
      if (state.freeVoice.active) {
        setFreeVoiceVisualState("interrupted", "已中断");
        window.setTimeout(() => {
          if (state.freeVoice.active && !state.freeVoice.submitting) {
            setFreeVoiceVisualState("listening", "待机");
          }
        }, 450);
      }
    }

    function resetPlaybackQueue() {
      state.playback = Promise.resolve();
    }

    async function followProgressiveTurn(initialPayload, requestSession = null) {
      if (requestSession && isStaleRequest(requestSession)) {
        return;
      }
      if (!initialPayload || !initialPayload.audio_url) {
        setStatus(initialPayload?.status_line || "已完成", chatStatus);
        if (state.freeVoice.active) {
          setFreeVoiceVisualState("listening", "待机");
        }
        return;
      }
      setStatus(initialPayload.status_line || "MOSS 实时语音流生成中...", chatStatus);
      const session = { controller: new AbortController(), audioContext: null, stopped: false };
      state.playback = state.playback.then(async () => {
        if (session.stopped || (requestSession && isStaleRequest(requestSession))) {
          return;
        }
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) {
          throw new Error("当前浏览器不支持 Web Audio 实时播放。");
        }
        state.activePlayback = session;
        if (state.freeVoice.active) {
          setFreeVoiceVisualState("playback", "播放中");
        }
        const sampleRate = Number(initialPayload.sample_rate || 48000);
        const channels = Number(initialPayload.channels || 2);
        const audioContext = new AudioContextCtor({ sampleRate });
        session.audioContext = audioContext;
        await audioContext.resume();
        const response = await fetch(initialPayload.audio_url, {
          credentials: "include",
          signal: session.controller.signal,
        });
        if (!response.ok) {
          const text = await response.text().catch(() => "");
          throw new Error(text || `语音流请求失败：HTTP ${response.status}`);
        }
        if (!response.body) {
          throw new Error("当前浏览器不支持 ReadableStream 音频播放。");
        }

        const reader = response.body.getReader();
        const bytesPerFrame = channels * 2;
        const minScheduleBytes = pcmByteLengthForSeconds(PCM_MIN_SCHEDULE_SECONDS, sampleRate, channels);
        const playbackState = { nextTime: null, started: false };
        let pendingPcm = new Uint8Array(0);
        while (!session.stopped && !(requestSession && isStaleRequest(requestSession))) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          if (!value || value.length === 0) {
            continue;
          }
          pendingPcm = mergeUint8Arrays(pendingPcm, value);
          pendingPcm = scheduleBufferedPcm(
            audioContext,
            pendingPcm,
            minScheduleBytes,
            bytesPerFrame,
            sampleRate,
            channels,
            playbackState
          );
        }
        if (!session.stopped && !(requestSession && isStaleRequest(requestSession))) {
          pendingPcm = scheduleBufferedPcm(
            audioContext,
            pendingPcm,
            minScheduleBytes,
            bytesPerFrame,
            sampleRate,
            channels,
            playbackState,
            true
          );
          applyFinalPcmFade(audioContext, playbackState);
          while (!session.stopped && !(requestSession && isStaleRequest(requestSession))) {
            const remainingSeconds = playbackState.nextTime - audioContext.currentTime;
            if (remainingSeconds <= 0) {
              break;
            }
            await wait(Math.min(remainingSeconds + 0.05, 0.2) * 1000);
          }
        }
        if (audioContext.state !== "closed") {
          await audioContext.close();
        }
        if (!session.stopped && !(requestSession && isStaleRequest(requestSession))) {
          setStatus("已完成", chatStatus);
          if (state.freeVoice.active) {
            setFreeVoiceVisualState("listening", "待机");
          }
        }
      }).catch((error) => {
        if (session.stopped) {
          return;
        }
        appendMessage("system", error.message, "播放");
        setStatus(error.message, chatStatus);
        if (state.freeVoice.active) {
          setFreeVoiceVisualState("error", "错误");
        }
      }).finally(() => {
        if (state.activePlayback === session) {
          state.activePlayback = null;
        }
      });
      return state.playback;
    }

    async function sendVoice(blob) {
      const supersessionEpoch = await cancelActiveWebWork("voice_superseded");
      if (supersessionEpoch !== state.requestEpoch) {
        return;
      }
      setStatus("识别中...", chatStatus);
      const session = beginRequestSession();
      try {
        const response = await fetch("/api/voice-stream", {
          method: "POST",
          credentials: "include",
          headers: blob.type ? { "Content-Type": blob.type } : {},
          body: blob,
          signal: session.controller.signal,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.error || response.statusText || "语音请求失败");
        }
        if (isStaleRequest(session)) {
          return;
        }
        if (payload.metadata && payload.metadata.no_answer) {
          setStatus(payload.status_line || "已忽略无效语音", chatStatus);
          await followProgressiveTurn(payload, session);
          return;
        }
        if (payload.input_text) {
          appendMessage("user", payload.input_text, "你 / 语音");
        }
        appendMessage(
          "assistant",
          payload.response_text || "",
          payload.ros2_error ? `ROS2: ${payload.ros2_error}` : "助手",
          payload.image_artifacts || []
        );
        handleConfirmationPayload(payload);
        await followProgressiveTurn(payload, session);
      } catch (error) {
        if (error.name === "AbortError" || isStaleRequest(session)) {
          return;
        }
        throw error;
      } finally {
        endRequestSession(session);
      }
    }

    function setFreeVoiceVisualState(mode, statusTextValue) {
      voiceSphere.classList.remove("listening", "speaking", "submitting", "playback", "interrupted", "error");
      voiceSphere.classList.add(mode);
      freeVoiceStatus.textContent = statusTextValue || "";
    }

    function stopFreeVoiceTracks() {
      const voice = state.freeVoice;
      if (voice.vadTimer !== null) {
        window.clearInterval(voice.vadTimer);
      }
      voice.vadTimer = null;
      if (voice.recorder && voice.recorder.state !== "inactive") {
        voice.recorder.stop();
      }
      voice.recorder = null;
      voice.chunks = [];
      voice.preRollChunks = [];
      voice.preRollHeaderChunk = null;
      if (voice.stream) {
        stopMediaTracks(voice.stream);
      }
      voice.stream = null;
      if (voice.audioContext && voice.audioContext.state !== "closed") {
        voice.audioContext.close().catch(() => {});
      }
      voice.audioContext = null;
      voice.analyser = null;
      voice.speaking = false;
      voice.finalizing = false;
      voice.speechStartedAt = 0;
      voice.silenceStartedAt = 0;
      voice.submitting = false;
    }

    function getFreeVoiceMimeType() {
      return [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/mp4",
      ].find((type) => MediaRecorder.isTypeSupported(type));
    }

    function trimFreeVoicePreRoll(now) {
      const voice = state.freeVoice;
      const cutoff = now - FREE_VOICE_PRE_ROLL_MS;
      while (voice.preRollChunks.length > 0 && voice.preRollChunks[0].capturedAt < cutoff) {
        voice.preRollChunks.shift();
      }
    }

    function handleFreeVoiceRecorderData(event) {
      const voice = state.freeVoice;
      if (!event.data || event.data.size <= 0) {
        return;
      }
      if (!voice.preRollHeaderChunk) {
        voice.preRollHeaderChunk = event.data;
      }
      if (voice.speaking) {
        voice.chunks.push(event.data);
        return;
      }
      voice.preRollChunks.push({ blob: event.data, capturedAt: performance.now() });
      trimFreeVoicePreRoll(performance.now());
    }

    function startFreeVoicePreRollRecorder() {
      const voice = state.freeVoice;
      if (!voice.active || !voice.stream || voice.submitting || voice.finalizing || state.activePlayback) {
        return;
      }
      if (voice.recorder && voice.recorder.state !== "inactive") {
        return;
      }
      const preferred = getFreeVoiceMimeType();
      const recorder = new MediaRecorder(voice.stream, preferred ? { mimeType: preferred } : undefined);
      voice.recorder = recorder;
      voice.chunks = [];
      voice.preRollChunks = [];
      voice.preRollHeaderChunk = null;
      recorder.ondataavailable = handleFreeVoiceRecorderData;
      recorder.onerror = (event) => {
        const message = event.error?.message || "自由语音录制失败。";
        setFreeVoiceVisualState("error", "错误");
        appendMessage("system", message, "错误");
      };
      recorder.onstop = async () => {
        voice.recorder = null;
        if (!voice.active) {
          voice.speaking = false;
          voice.finalizing = false;
          voice.chunks = [];
          voice.preRollChunks = [];
          voice.preRollHeaderChunk = null;
          return;
        }
        if (!voice.speaking) {
          return;
        }
        await submitFreeVoiceUtterance(recorder.mimeType || preferred || "audio/webm");
      };
      recorder.start(FREE_VOICE_RECORDER_TIMESLICE_MS);
    }

    async function toggleFreeVoiceMode() {
      if (state.freeVoice.active) {
        await stopFreeVoiceMode();
        return;
      }
      if (!window.isSecureContext) {
        appendMessage("system", "当前页面不是安全上下文，浏览器会禁止麦克风；请用 localhost 或 HTTPS 打开。", "错误");
        return;
      }
      if (!navigator.mediaDevices || !window.MediaRecorder) {
        appendMessage("system", "当前浏览器不支持自由语音。", "错误");
        return;
      }
      try {
        await cancelActiveWebWork("free_voice_start");
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) {
          stopMediaTracks(stream);
          throw new Error("当前浏览器不支持 Web Audio。");
        }
        const audioContext = new AudioContextCtor();
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 1024;
        source.connect(analyser);
        state.freeVoice.active = true;
        state.freeVoice.stream = stream;
        state.freeVoice.audioContext = audioContext;
        state.freeVoice.analyser = analyser;
        freeVoicePanel.classList.remove("hidden");
        setFreeVoiceButtonLabel("退出语音");
        setFreeVoiceVisualState("listening", "待机");
        startFreeVoicePreRollRecorder();
        state.freeVoice.vadTimer = window.setInterval(sampleFreeVoiceVad, FREE_VOICE_VAD_INTERVAL_MS);
      } catch (error) {
        stopFreeVoiceTracks();
        state.freeVoice.active = false;
        freeVoicePanel.classList.add("hidden");
        setFreeVoiceButtonLabel("自由语音");
        appendMessage("system", error.message || "自由语音启动失败。", "错误");
        setStatus(error.message || "自由语音启动失败。", chatStatus);
      }
    }

    async function stopFreeVoiceMode() {
      let cancellationError = null;
      try {
        await cancelActiveWebWork("free_voice_stop");
      } catch (error) {
        cancellationError = error;
      }
      state.freeVoice.active = false;
      stopFreeVoiceTracks();
      freeVoicePanel.classList.add("hidden");
      setFreeVoiceButtonLabel("自由语音");
      if (cancellationError) {
        appendMessage("system", cancellationError.message || "后端取消请求失败", "错误");
        setStatus(cancellationError.message || "后端取消请求失败", chatStatus);
      } else {
        setStatus("", chatStatus);
      }
    }

    function sampleFreeVoiceVad() {
      const voice = state.freeVoice;
      if (!voice.active || !voice.analyser || voice.submitting || voice.finalizing || state.activePlayback) {
        return;
      }
      const data = new Uint8Array(voice.analyser.fftSize);
      voice.analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let index = 0; index < data.length; index += 1) {
        const centered = (data[index] - 128) / 128;
        sum += centered * centered;
      }
      const rms = Math.sqrt(sum / data.length);
      const now = performance.now();
      if (rms >= FREE_VOICE_SPEECH_THRESHOLD) {
        if (!voice.speaking) {
          startFreeVoiceUtterance(now);
        }
        voice.silenceStartedAt = 0;
        setFreeVoiceVisualState("speaking", "收音中");
        return;
      }
      if (!voice.speaking) {
        return;
      }
      if (!voice.silenceStartedAt) {
        voice.silenceStartedAt = now;
      }
      if (now - voice.silenceStartedAt >= FREE_VOICE_TRAILING_SILENCE_MS) {
        finishFreeVoiceUtterance();
      }
    }

    function startFreeVoiceUtterance(now) {
      const voice = state.freeVoice;
      if (!voice.stream) {
        return;
      }
      if (!voice.recorder || voice.recorder.state === "inactive") {
        startFreeVoicePreRollRecorder();
      }
      if (!voice.recorder || voice.recorder.state === "inactive") {
        return;
      }
      voice.chunks = buildFreeVoiceUtteranceChunks();
      voice.preRollChunks = [];
      voice.speaking = true;
      voice.finalizing = false;
      voice.speechStartedAt = now;
      voice.silenceStartedAt = 0;
    }

    function buildFreeVoiceUtteranceChunks() {
      const voice = state.freeVoice;
      const chunks = voice.preRollChunks.map((chunk) => chunk.blob);
      if (voice.preRollHeaderChunk && chunks[0] !== voice.preRollHeaderChunk) {
        chunks.unshift(voice.preRollHeaderChunk);
      }
      return chunks;
    }

    async function submitFreeVoiceUtterance(mimeType) {
      const voice = state.freeVoice;
      const chunks = voice.chunks;
      voice.chunks = [];
      voice.finalizing = false;
      if (!voice.active) {
        voice.speaking = false;
        return;
      }
      if (performance.now() - voice.speechStartedAt < FREE_VOICE_MIN_SPEECH_MS) {
        voice.speaking = false;
        setFreeVoiceVisualState("listening", "待机");
        startFreeVoicePreRollRecorder();
        return;
      }
      const blob = new Blob(chunks, { type: mimeType });
      if (blob.size === 0) {
        voice.speaking = false;
        setFreeVoiceVisualState("listening", "待机");
        startFreeVoicePreRollRecorder();
        return;
      }
      voice.submitting = true;
      voice.speaking = false;
      setFreeVoiceVisualState("submitting", "处理中");
      try {
        await sendVoice(blob);
      } catch (error) {
        appendMessage("system", error.message, "错误");
        setStatus(error.message, chatStatus);
        setFreeVoiceVisualState("error", "错误");
      } finally {
        voice.submitting = false;
        if (voice.active && !state.activePlayback) {
          startFreeVoicePreRollRecorder();
          setFreeVoiceVisualState("listening", "待机");
        }
      }
    }

    function finishFreeVoiceUtterance() {
      const voice = state.freeVoice;
      if (voice.finalizing) {
        return;
      }
      if (!voice.recorder || voice.recorder.state === "inactive") {
        voice.speaking = false;
        voice.finalizing = false;
        return;
      }
      voice.finalizing = true;
      setFreeVoiceVisualState("submitting", "处理中");
      voice.recorder.stop();
    }

    function stopMediaTracks(stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    async function sendEstop() {
      if (!state.authenticated) {
        return;
      }
      if (!window.confirm("确认急停？")) {
        return;
      }
      try {
        const payload = await fetchJson("/api/estop", { method: "POST", body: JSON.stringify({ active: true }) });
        appendMessage("system", payload.message || "急停已发送", payload.ros2_error ? `ROS2: ${payload.ros2_error}` : "急停");
        setStatus(payload.status_line || "急停已发送", chatStatus);
      } catch (error) {
        appendMessage("system", error.message, "错误");
        setStatus(error.message, chatStatus);
      }
    }

    loginBtn.addEventListener("click", login);
    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      await sendChat(chatInput.value);
    });
    freeVoiceBtn.addEventListener("click", toggleFreeVoiceMode);
    voiceSphere.addEventListener("click", () => {
      if (state.activePlayback) {
        cancelActiveWebWork("playback_interrupt").catch(() => {});
      }
    });
    confirmationConfirmBtn.addEventListener("click", confirmPending);
    confirmationCancelBtn.addEventListener("click", cancelPending);
    emergencyAlertDismissBtn.addEventListener("click", () => {
      hideEmergencyAlert();
      appendMessage("system", "已关闭急停告警提示；急停状态未解除，仍由 ROS2 停止源控制。", "急停告警");
    });
    clearBtn.addEventListener("click", () => {
      messages.innerHTML = "";
      setStatus("", chatStatus);
    });
    stopBtn.addEventListener("click", sendEstop);

    passwordInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        login();
      }
    });
    chatInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
      }
    });

    refreshStatus().catch((error) => {
      setStatus(error.message, statusText);
      appendMessage("system", error.message, "错误");
    });
  </script>
</body>
</html>
"""
