const sessionId = `session-${Date.now()}`;

const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chatForm");
const chatInput = document.querySelector("#chatInput");
const resetBtn = document.querySelector("#resetBtn");
const healthStatus = document.querySelector("#healthStatus");
const modelSelect = document.querySelector("#modelSelect");
const formatSelect = document.querySelector("#formatSelect");
const textTab = document.querySelector("#textTab");
const docxTab = document.querySelector("#docxTab");
const textForm = document.querySelector("#textForm");
const docxForm = document.querySelector("#docxForm");
const fixIeeeTextBtn = document.querySelector("#fixIeeeTextBtn");
const fixIeeeDocxBtn = document.querySelector("#fixIeeeDocxBtn");
const draftText = document.querySelector("#draftText");
const docxFile = document.querySelector("#docxFile");
const fileLabel = document.querySelector("#fileLabel");
const resultState = document.querySelector("#resultState");
const thinkingStatus = document.querySelector("#thinkingStatus");
const thinkingStep = document.querySelector("#thinkingStep");
const thinkingDetail = document.querySelector("#thinkingDetail");
const results = document.querySelector("#results");

function selectedModel() {
  return modelSelect.value;
}

function selectedFormat() {
  return formatSelect.value;
}

function addMessage(role, text, isError = false) {
  const node = document.createElement("div");
  node.className = `message ${role}${isError ? " error" : ""}`;
  node.textContent = text;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
}

function setBusy(label) {
  startThinking(label, "Waiting for the backend to report progress.");
}

function startThinking(initialLabel = "Starting analysis", detail = "Waiting for the backend to report progress.") {
  resultState.classList.add("hidden");
  results.classList.add("hidden");
  thinkingStatus.classList.remove("hidden");
  thinkingStep.textContent = initialLabel;
  thinkingDetail.textContent = detail;
}

function updateThinking(step, detail) {
  thinkingStatus.classList.remove("hidden");
  thinkingStep.textContent = step || "Working";
  thinkingDetail.textContent = detail || "Processing request.";
}

function stopThinking(hide = true) {
  if (hide) {
    thinkingStatus.classList.add("hidden");
  }
}

function setReady(label, detail) {
  stopThinking();
  resultState.classList.remove("hidden");
  resultState.innerHTML = `<span class="mark">${label}</span><p>${detail}</p>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function apiJson(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text };
    }
  }
  if (!response.ok) {
    throw new Error(formatApiError(payload, response.status));
  }
  return payload;
}

function formatApiError(payload, status) {
  if (!payload || typeof payload !== "object") {
    return `Request failed: HTTP ${status}`;
  }
  const detail = Array.isArray(payload.detail)
    ? payload.detail.map((item) => item.msg || JSON.stringify(item)).join("; ")
    : payload.detail;
  const errorType = payload.error_type ? ` (${payload.error_type})` : "";
  const error = payload.error ? `: ${payload.error}` : "";
  return `${detail || `Request failed: HTTP ${status}`}${errorType}${error}`;
}

async function apiStream(path, options = {}, handlers = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    let payload = {};
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text };
    }
    throw new Error(formatApiError(payload, response.status));
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      let event;
      try {
        event = JSON.parse(line);
      } catch (error) {
        throw new Error(`Backend sent invalid stream event: ${line}`);
      }
      if (event.event === "status" && handlers.status) {
        handlers.status(event);
      }
      if (event.event === "analysis" && handlers.analysis) {
        handlers.analysis(event.analysis);
      }
      if (event.event === "error") {
        const suffix = event.error_type ? ` (${event.error_type})` : "";
        throw new Error(`${event.detail || "Streaming analysis failed."}${suffix}`);
      }
    }
  }
}

async function checkHealth() {
  try {
    const health = await apiJson("/health");
    if (selectedModel().startsWith("groq:")) {
      healthStatus.className = "status-pill ok";
      healthStatus.textContent = "Groq hosted model selected";
      return;
    }
    healthStatus.className = `status-pill ${health.ollama_reachable ? "ok" : "bad"}`;
    healthStatus.textContent = health.ollama_reachable
      ? `Ollama ready: ${selectedModel()}`
      : "API ready, Ollama fallback active";
  } catch (error) {
    healthStatus.className = "status-pill bad";
    healthStatus.textContent = "API unavailable";
  }
}

async function sendChat(message) {
  addMessage("user", message);
  const likelyDraft = message.trim().split(/\s+/).length >= 60 || message.length >= 350;
  if (likelyDraft) {
    startThinking("Sending draft", "The backend is receiving the chat text.");
    await apiStream(
      "/analyze_text_stream",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: message,
          model: selectedModel(),
          format_mode: selectedFormat(),
          metadata: { source_type: "chat_message", format_mode: selectedFormat() },
        }),
      },
      {
        status: (event) => updateThinking(event.step, event.detail),
        analysis: (analysis) => {
          addMessage("agent", "I analyzed the draft and organized the feedback into the review panel.");
          renderResults(analysis);
        },
      },
    );
    return;
  }
  const data = await apiJson("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      model: selectedModel(),
      format_mode: selectedFormat(),
    }),
  });
  addMessage("agent", data.next_prompt);
  if (data.analysis) {
    stopThinking();
    renderResults(data.analysis);
  } else if (likelyDraft) {
    stopThinking();
  }
}

function issueHtml(item, mode = "issue") {
  const severity = String(item.severity || item.priority || "medium").toLowerCase();
  const title = item.issue || item.suggestion || "Suggestion";
  const recommendation = item.recommendation || item.expected_impact || "";
  const evidence = item.evidence || item.rationale || "";
  return `
    <div class="issue ${escapeHtml(severity)}">
      <div class="issue-title">
        <span>${escapeHtml(title)}</span>
        <span class="badge">${escapeHtml(severity)}</span>
      </div>
      ${evidence ? `<p><strong>${mode === "suggestion" ? "Rationale" : "Evidence"}:</strong> ${escapeHtml(evidence)}</p>` : ""}
      ${recommendation ? `<p><strong>${mode === "suggestion" ? "Impact" : "Recommendation"}:</strong> ${escapeHtml(recommendation)}</p>` : ""}
    </div>
  `;
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function renderHighlightedText(text, highlights = []) {
  if (!text || !highlights.length) {
    return "";
  }

  const matches = [];
  highlights.forEach((item, index) => {
    const excerpt = String(item.excerpt || "").trim();
    if (!excerpt) return;
    const pattern = new RegExp(escapeRegExp(excerpt), "i");
    const match = pattern.exec(text);
    if (!match) return;
    matches.push({
      start: match.index,
      end: match.index + match[0].length,
      text: match[0],
      message: item.message || "Review this statement.",
      severity: String(item.severity || "medium").toLowerCase(),
      index,
    });
  });

  matches.sort((a, b) => a.start - b.start || b.end - a.end);
  const filtered = [];
  let cursor = -1;
  for (const match of matches) {
    if (match.start < cursor) continue;
    filtered.push(match);
    cursor = match.end;
  }

  if (!filtered.length) {
    return "";
  }

  let html = "";
  let pos = 0;
  filtered.forEach((match) => {
    html += escapeHtml(text.slice(pos, match.start));
    html += `<mark class="text-highlight ${escapeHtml(match.severity)}" tabindex="0" data-message="${escapeHtml(match.message)}">${escapeHtml(match.text)}</mark>`;
    pos = match.end;
  });
  html += escapeHtml(text.slice(pos));

  return `
    <article class="result-card">
      <h3>Highlighted Draft</h3>
      <p>Hover or focus highlighted phrases to see why they were flagged.</p>
      <div class="highlighted-text">${html}</div>
    </article>
  `;
}

function renderSection(title, items, emptyText, mode = "issue") {
  const body = items?.length
    ? `<div class="issue-list">${items.map((item) => issueHtml(item, mode)).join("")}</div>`
    : `<p>${emptyText}</p>`;
  return `<article class="result-card"><h3>${escapeHtml(title)}</h3>${body}</article>`;
}

function renderRewrites(items) {
  if (!items?.length) {
    return `<article class="result-card"><h3>Optional Rewrite Suggestions</h3><p>No rewrite suggestions returned.</p></article>`;
  }
  const body = items
    .map(
      (item) => `
        <div class="issue">
          <div class="issue-title"><span>${escapeHtml(item.reason || "Rewrite suggestion")}</span></div>
          <p><strong>Original:</strong> ${escapeHtml(item.original_excerpt)}</p>
          <p><strong>Rewrite:</strong> ${escapeHtml(item.rewritten_excerpt)}</p>
        </div>
      `,
    )
    .join("");
  return `<article class="result-card"><h3>Optional Rewrite Suggestions</h3><div class="issue-list">${body}</div></article>`;
}

function renderFormatResults(data) {
  stopThinking();
  resultState.classList.add("hidden");
  results.classList.remove("hidden");
  const previewTitle = data.transformations_applied ? "IEEE Format Preview" : "Original Text Preview";
  const previewNote = data.transformations_applied
    ? `${data.transformation_count} deterministic fix(es) were applied. Review the remaining issues below.`
    : "No deterministic text transformations were applied. Review the findings below.";
  const grouped = {};
  (data.changes || []).forEach((item) => {
    const category = item.category || "IEEE formatting";
    grouped[category] = grouped[category] || [];
    grouped[category].push(item);
  });
  const changes = Object.keys(grouped).length
    ? Object.entries(grouped)
        .map(
          ([category, items]) => `
            <article class="result-card">
              <h3>${escapeHtml(category)}</h3>
              <div class="issue-list">${items
                .map(
                  (item) => `
                    <div class="issue ${escapeHtml(String(item.severity || "medium").toLowerCase())}">
                      <div class="issue-title">
                        <span>${escapeHtml(item.issue)}</span>
                        <span class="badge">${escapeHtml(item.severity)}</span>
                      </div>
                      ${item.original ? `<p><strong>Original:</strong> ${escapeHtml(item.original)}</p>` : ""}
                      ${item.replacement ? `<p><strong>Replacement:</strong> ${escapeHtml(item.replacement)}</p>` : ""}
                      <p><strong>Note:</strong> ${escapeHtml(item.note)}</p>
                      ${item.sub_issues?.length ? `<p><strong>Triggered checks:</strong> ${escapeHtml(item.sub_issues.join("; "))}</p>` : ""}
                    </div>
                  `,
                )
                .join("")}</div>
            </article>
          `,
        )
        .join("")
    : '<article class="result-card"><h3>IEEE Format Changes</h3><p>No IEEE formatting issues detected by the deterministic checker.</p></article>';

  results.innerHTML = `
    <article class="result-card">
      <h3>IEEE Format Fix Summary</h3>
      <p>${escapeHtml(data.summary)}</p>
      <button class="secondary-action" id="downloadIeeeDocxBtn" type="button">Download DOCX</button>
    </article>
    <article class="result-card">
      <h3>${escapeHtml(previewTitle)}</h3>
      <p>${escapeHtml(previewNote)} Verify source metadata and final Word styling manually.</p>
      <div class="fixed-text">${escapeHtml(data.fixed_text)}</div>
    </article>
    ${changes}
  `;
  document.querySelector("#downloadIeeeDocxBtn").addEventListener("click", () => {
    downloadIeeeDocx(data.fixed_text);
  });
}

async function downloadIeeeDocx(text) {
  try {
    const response = await fetch("/export_ieee_docx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, filename: "ieee_formatted.docx" }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(formatApiError(payload, response.status));
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "ieee_formatted.docx";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    setReady("Export Failed", error.message);
  }
}

function renderResults(data) {
  stopThinking();
  resultState.classList.add("hidden");
  results.classList.remove("hidden");
  const fallbackNote = data.fallback_used
    ? `<p><strong>Fallback used:</strong> Ollama was unavailable or returned invalid output.</p>`
    : "";
  results.innerHTML = `
    <article class="result-card">
      <h3>Summary</h3>
      <p>${escapeHtml(data.summary)}</p>
      ${fallbackNote}
    </article>
    ${renderHighlightedText(data.reviewed_text, data.highlights)}
    ${renderSection("Structure / Format Issues", data.structure_format_issues, "No structure issues returned.")}
    ${renderSection("Academic Quality Issues", data.academic_quality_issues, "No academic quality issues returned.")}
    ${renderSection("Citation / Consistency Issues", data.citation_consistency_issues, "No citation issues returned.")}
    ${renderSection("Prioritized Suggestions", data.prioritized_suggestions, "No prioritized suggestions returned.", "suggestion")}
    ${renderRewrites(data.optional_rewrite_suggestions)}
  `;
}

function switchMode(mode) {
  const textMode = mode === "text";
  textTab.classList.toggle("active", textMode);
  docxTab.classList.toggle("active", !textMode);
  textForm.classList.toggle("hidden", !textMode);
  docxForm.classList.toggle("hidden", textMode);
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  chatInput.value = "";
  try {
    await sendChat(message);
  } catch (error) {
    stopThinking();
    addMessage("agent", error.message, true);
  }
});

resetBtn.addEventListener("click", () => {
  messages.innerHTML = "";
  addMessage("agent", "Paste your academic text here, or ask me to review a draft.");
  setReady("Ready", "Paste text into chat or use the review panel.");
});

textTab.addEventListener("click", () => switchMode("text"));
docxTab.addEventListener("click", () => switchMode("docx"));

docxFile.addEventListener("change", () => {
  fileLabel.textContent = docxFile.files[0]?.name || "Choose a .docx file";
});

modelSelect.addEventListener("change", checkHealth);
formatSelect.addEventListener("change", () => {
  setReady("Format Mode Updated", `Reviews will now check against ${formatSelect.options[formatSelect.selectedIndex].text}.`);
});

textForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = draftText.value.trim();
  if (text.length < 20) {
    setReady("Need More Text", "Paste at least 20 characters before analysis.");
    return;
  }
  setBusy("Analyzing Text");
  try {
    await apiStream(
      "/analyze_text_stream",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: selectedModel(), format_mode: selectedFormat() }),
      },
      {
        status: (event) => updateThinking(event.step, event.detail),
        analysis: renderResults,
      },
    );
  } catch (error) {
    setReady("Analysis Failed", error.message);
  }
});

docxForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = docxFile.files[0];
  if (!file) {
    setReady("Missing File", "Choose a .docx file first.");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model", selectedModel());
  formData.append("format_mode", selectedFormat());
  setBusy("Analyzing DOCX");
  try {
    await apiStream(
      "/analyze_docx_stream",
      {
        method: "POST",
        body: formData,
      },
      {
        status: (event) => updateThinking(event.step, event.detail),
        analysis: renderResults,
      },
    );
  } catch (error) {
    setReady("Analysis Failed", error.message);
  }
});

fixIeeeTextBtn.addEventListener("click", async () => {
  const text = draftText.value.trim();
  if (text.length < 20) {
    setReady("Need More Text", "Paste at least 20 characters before formatting.");
    return;
  }
  startThinking("Fixing IEEE format", "Running deterministic citation and reference formatting checks.");
  try {
    const data = await apiJson("/format_ieee_text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    renderFormatResults(data);
  } catch (error) {
    setReady("IEEE Format Failed", error.message);
  }
});

fixIeeeDocxBtn.addEventListener("click", async () => {
  const file = docxFile.files[0];
  if (!file) {
    setReady("Missing File", "Choose a .docx file first.");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  startThinking("Fixing IEEE format", "Extracting DOCX text and running deterministic IEEE checks.");
  try {
    const data = await apiJson("/format_ieee_docx", {
      method: "POST",
      body: formData,
    });
    renderFormatResults(data);
  } catch (error) {
    setReady("IEEE Format Failed", error.message);
  }
});

addMessage("agent", "Paste your academic text here, or ask me to review a draft.");
checkHealth();
