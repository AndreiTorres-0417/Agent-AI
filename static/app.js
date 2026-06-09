const storageKeys = {
  sessionId: "academicReview.sessionId",
  messages: "academicReview.messages",
  model: "academicReview.model",
  reviewMode: "academicReview.reviewMode",
  formatMode: "academicReview.formatMode",
  draftText: "academicReview.draftText",
  inputMode: "academicReview.inputMode",
  fileName: "academicReview.fileName",
};

let sessionId = localStorage.getItem(storageKeys.sessionId) || `session-${Date.now()}`;
localStorage.setItem(storageKeys.sessionId, sessionId);
let messageHistory = [];
let latestAnalysis = null;
let chatFocusHighlight = null;
let chatThinkingNode = null;

const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chatForm");
const chatInput = document.querySelector("#chatInput");
const resetBtn = document.querySelector("#resetBtn");
const healthStatus = document.querySelector("#healthStatus");
const modelSelect = document.querySelector("#modelSelect");
const textTab = document.querySelector("#textTab");
const docxTab = document.querySelector("#docxTab");
const textForm = document.querySelector("#textForm");
const docxForm = document.querySelector("#docxForm");
const draftText = document.querySelector("#draftText");
const docxFile = document.querySelector("#docxFile");
const fileLabel = document.querySelector("#fileLabel");
const resultState = document.querySelector("#resultState");
const thinkingStatus = document.querySelector("#thinkingStatus");
const thinkingStep = document.querySelector("#thinkingStep");
const thinkingDetail = document.querySelector("#thinkingDetail");
const results = document.querySelector("#results");
const welcomeMessage = "Analyze a draft on the right, then ask me about the summary, issues, citations, or how to revise it.";

function saveMessages() {
  localStorage.setItem(storageKeys.messages, JSON.stringify(messageHistory.slice(-80)));
}

function saveSettings() {
  localStorage.setItem(storageKeys.model, selectedModel());
}

function selectedModel() {
  return modelSelect.value;
}

function selectedFormat() {
  return "ieee";
}

function selectedReviewMode() {
  return "academic";
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+?)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
}

function renderBasicMarkdown(value) {
  const normalized = String(value ?? "").replace(/\s+(?=\d+\.\s+\*\*)/g, "\n");
  const lines = normalized.split(/\r?\n/);
  const html = [];
  const lists = [];

  function closeLists(targetIndent = -1) {
    while (lists.length && lists[lists.length - 1].indent >= targetIndent) {
      html.push(`</${lists.pop().tag}>`);
    }
  }

  lines.forEach((line) => {
    const match = line.match(/^(\s*)(?:(\d+)\.|[-*])\s+(.+)$/);
    if (match) {
      const indent = match[1].replace(/\t/g, "  ").length;
      const tag = match[2] ? "ol" : "ul";
      while (lists.length && indent < lists[lists.length - 1].indent) {
        html.push(`</${lists.pop().tag}>`);
      }
      if (!lists.length || indent > lists[lists.length - 1].indent || lists[lists.length - 1].tag !== tag) {
        if (lists.length && indent === lists[lists.length - 1].indent) {
          html.push(`</${lists.pop().tag}>`);
        }
        html.push(`<${tag} class="chat-list chat-list-depth-${lists.length}">`);
        lists.push({ indent, tag });
      }
      html.push(`<li>${renderInlineMarkdown(match[3])}</li>`);
      return;
    }

    closeLists();
    if (line.trim()) {
      html.push(`<p>${renderInlineMarkdown(line.trim())}</p>`);
    }
  });
  closeLists();
  return html.join("");
}

function addMessage(role, text, isError = false, persist = true) {
  const node = document.createElement("div");
  node.className = `message ${role}${isError ? " error" : ""}`;
  if (!isError) {
    node.innerHTML = renderBasicMarkdown(text);
  } else {
    node.textContent = text;
  }
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  if (persist) {
    messageHistory.push({ role, text, isError });
    saveMessages();
  }
}

function setChatThinking(step, detail) {
  if (!chatThinkingNode) {
    chatThinkingNode = document.createElement("div");
    chatThinkingNode.className = "message agent chat-thinking";
    messages.appendChild(chatThinkingNode);
  }
  chatThinkingNode.innerHTML = `
    <span class="chat-thinking-step">${escapeHtml(step || "Working")}</span>
    <p>${escapeHtml(detail || "Processing your question.")}</p>
  `;
  messages.scrollTop = messages.scrollHeight;
}

function clearChatThinking() {
  if (chatThinkingNode) {
    chatThinkingNode.remove();
    chatThinkingNode = null;
  }
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

function resizeChatInput() {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 120)}px`;
}

function setReady(label, detail) {
  stopThinking();
  resultState.classList.remove("hidden");
  resultState.innerHTML = `<span class="mark">${label}</span><p>${detail}</p>`;
}

function clearLocalState() {
  [
    storageKeys.sessionId,
    storageKeys.messages,
    storageKeys.draftText,
    storageKeys.inputMode,
    storageKeys.fileName,
  ].forEach((key) => localStorage.removeItem(key));
  sessionId = `session-${Date.now()}`;
  localStorage.setItem(storageKeys.sessionId, sessionId);
  messageHistory = [];
  messages.innerHTML = "";
  draftText.value = "";
  docxFile.value = "";
  fileLabel.textContent = "Choose a .docx file";
  results.innerHTML = "";
  results.classList.add("hidden");
  setReady("Cleared", "Chat history and saved draft text were cleared.");
  addMessage("agent", welcomeMessage);
}

function restoreState() {
  modelSelect.value = localStorage.getItem(storageKeys.model) || modelSelect.value;
  draftText.value = localStorage.getItem(storageKeys.draftText) || "";
  fileLabel.textContent = localStorage.getItem(storageKeys.fileName) || "Choose a .docx file";

  const savedMode = localStorage.getItem(storageKeys.inputMode);
  if (savedMode === "docx") {
    switchMode("docx");
  } else {
    switchMode("text");
  }

  try {
    messageHistory = JSON.parse(localStorage.getItem(storageKeys.messages) || "[]");
  } catch {
    messageHistory = [];
  }
  messages.innerHTML = "";
  if (messageHistory.length) {
    messageHistory.forEach((item) => addMessage(item.role, item.text, item.isError, false));
  } else {
    addMessage("agent", welcomeMessage);
  }
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
  let lastAnalysis = null;

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
        lastAnalysis = event.analysis;
        handlers.analysis(event.analysis);
      }
      if (event.event === "error") {
        const suffix = event.error_type ? ` (${event.error_type})` : "";
        throw new Error(`${event.detail || "Streaming analysis failed."}${suffix}`);
      }
    }
  }
  return lastAnalysis;
}

async function checkHealth() {
  try {
    const health = await apiJson("/health");
    const localGemma = selectedModel() === "gemma3:1b";
    const ready = localGemma ? health.ollama_reachable : health.openai_configured;
    healthStatus.className = `status-pill ${ready ? "ok" : "bad"}`;
    healthStatus.textContent = localGemma
      ? ready ? "Local Gemma ready" : "Start Ollama for Gemma"
      : ready ? "OpenAI ready" : "Add OPENAI_API_KEY";
  } catch (error) {
    healthStatus.className = "status-pill bad";
    healthStatus.textContent = "API unavailable";
  }
}

async function sendChat(message) {
  if (message.trim().toLowerCase() === "/clear") {
    clearLocalState();
    return;
  }
  addMessage("user", message);
  setChatThinking("Reading your question", "Checking whether you asked about a specific paper section.");
  await new Promise((resolve) => setTimeout(resolve, 120));
  const lowerMessage = message.toLowerCase();
  const sectionHint = [
    "abstract",
    "introduction",
    "background",
    "statement of the problem",
    "sop",
    "objective",
    "method",
    "finding",
    "result",
    "discussion",
    "conclusion",
    "reference",
  ].some((term) => lowerMessage.includes(term));
  setChatThinking(
    sectionHint ? "Finding the matching section" : "Loading review context",
    sectionHint
      ? "Looking through the analyzed draft so the answer can focus on that section."
      : "Using the latest analysis, stored issues, and paper text for context."
  );
  await new Promise((resolve) => setTimeout(resolve, 120));
  setChatThinking("Asking the model", `Sending the focused request to ${selectedModel()}.`);
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
  if (data.context?.focused_section_found) {
    setChatThinking(
      "Section found",
      `The agent is answering with focus on ${data.context.focused_section || "the selected section"}.`
    );
    await new Promise((resolve) => setTimeout(resolve, 160));
  } else if (data.context?.focused_section) {
    setChatThinking(
      "Section not found",
      `The agent looked for ${data.context.focused_section}, but it was not found in the analyzed draft.`
    );
    await new Promise((resolve) => setTimeout(resolve, 160));
  }
  clearChatThinking();
  addMessage("agent", data.next_prompt);
  if (data.context?.focused_section_found && data.context.focused_section_text) {
    chatFocusHighlight = {
      excerpt: data.context.focused_section_text,
      message: `Chat is analyzing: ${data.context.focused_section || "selected section"}`,
      severity: "focus",
    };
    if (latestAnalysis) {
      renderResults(latestAnalysis);
    }
  }
  if (data.analysis) {
    stopThinking();
    renderResults(data.analysis);
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
  const merged = [];
  for (const match of matches) {
    const last = merged[merged.length - 1];
    if (last && match.start <= last.end) {
      last.end = Math.max(last.end, match.end);
      last.text = text.slice(last.start, last.end);
      last.messages.push(match.message);
      last.severities.push(match.severity);
      continue;
    }
    merged.push({
      ...match,
      messages: [match.message],
      severities: [match.severity],
    });
  }

  if (!merged.length) {
    return "";
  }

  const severityRank = { focus: 4, high: 3, medium: 2, low: 1 };

  let html = "";
  let pos = 0;
  merged.forEach((match) => {
    const severity = match.severities.reduce((best, current) => (
      (severityRank[current] || 0) > (severityRank[best] || 0) ? current : best
    ), "low");
    const message = [...new Set(match.messages)].join(" | ");
    html += escapeHtml(text.slice(pos, match.start));
    html += `<mark class="text-highlight ${escapeHtml(severity)}" tabindex="0" data-message="${escapeHtml(message)}">${escapeHtml(match.text)}</mark>`;
    pos = match.end;
  });
  html += escapeHtml(text.slice(pos));

  return `
    <article class="result-card">
      <h3>Highlighted Draft</h3>
      <p>Hover or focus highlighted phrases to see why they were flagged or what the chat is referencing.</p>
      <div class="highlighted-text">${html}</div>
    </article>
  `;
}

function renderSection(title, items, emptyText, mode = "issue") {
  const meaningfulItems = (items || []).filter((item) => {
    const title = String(item?.issue || item?.suggestion || "").trim();
    const detail = String(item?.evidence || item?.rationale || item?.recommendation || item?.expected_impact || "").trim();
    return title && detail;
  });
  const body = meaningfulItems.length
    ? `<div class="issue-list">${meaningfulItems.map((item) => issueHtml(item, mode)).join("")}</div>`
    : `<p>${emptyText}</p>`;
  return `<article class="result-card"><h3>${escapeHtml(title)}</h3>${body}</article>`;
}

function renderRewrites(items) {
  const meaningfulItems = (items || []).filter(
    (item) => String(item?.original_excerpt || "").trim() && String(item?.rewritten_excerpt || "").trim(),
  );
  if (!meaningfulItems.length) return "";
  const body = meaningfulItems
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
  return `<article class="result-card"><h3>Rewrite Examples</h3><div class="issue-list">${body}</div></article>`;
}

function firstIssueTitle(items) {
  const item = (items || []).find((entry) => String(entry?.issue || entry?.suggestion || "").trim());
  return item ? String(item.issue || item.suggestion).trim() : "";
}

function buildReviewDiscussionPrompt(data) {
  const mode = selectedReviewMode() === "format" ? "format check" : "academic review";
  const focus =
    firstIssueTitle(data.academic_quality_issues) ||
    firstIssueTitle(data.structure_format_issues) ||
    firstIssueTitle(data.citation_consistency_issues) ||
    firstIssueTitle(data.prioritized_suggestions);
  const focusLine = focus ? `The first thing I would discuss is: **${focus}**.` : "No major issue stood out in the structured results.";
  return `I finished the ${mode}. **Summary:** ${data.summary}\n\n${focusLine}\n\nAsk me what to fix first, how to rewrite a section, or why a citation/wording issue matters.`;
}

function handleAnalysisResult(data) {
  if (!data || data._discussionPosted) return;
  data._discussionPosted = true;
  latestAnalysis = data;
  chatFocusHighlight = null;
  renderResults(data);
  addMessage("agent", buildReviewDiscussionPrompt(data));
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
  latestAnalysis = data;
  stopThinking();
  resultState.classList.add("hidden");
  results.classList.remove("hidden");
  const fallbackNote = data.fallback_used
    ? `<p><strong>Fallback used:</strong> The selected model was unavailable or returned invalid output.</p>`
    : "";
  results.innerHTML = `
    <article class="result-card">
      <h3>Summary</h3>
      <p>${escapeHtml(data.summary)}</p>
      ${fallbackNote}
    </article>
    ${renderHighlightedText(data.reviewed_text, [...(data.highlights || []), ...(chatFocusHighlight ? [chatFocusHighlight] : [])])}
    ${renderSection(selectedReviewMode() === "format" ? "Formatting Violations" : "Structure / Writing Issues", data.structure_format_issues, "No issues detected.")}
    ${renderSection("Academic Quality Issues", data.academic_quality_issues, "No academic quality issues detected.")}
    ${renderSection(selectedReviewMode() === "format" ? "Citation / Reference Violations" : "Citation Needs", data.citation_consistency_issues, "No citation issues detected.")}
    ${renderSection("Priority Fixes", data.prioritized_suggestions, "No priority fixes required.", "suggestion")}
    ${selectedReviewMode() === "academic" ? renderRewrites(data.optional_rewrite_suggestions) : ""}
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
  resizeChatInput();
  try {
    await sendChat(message);
  } catch (error) {
    clearChatThinking();
    stopThinking();
    addMessage("agent", error.message, true);
  }
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

chatInput.addEventListener("input", resizeChatInput);

resetBtn.addEventListener("click", () => {
  clearLocalState();
});

textTab.addEventListener("click", () => {
  switchMode("text");
  localStorage.setItem(storageKeys.inputMode, "text");
});
docxTab.addEventListener("click", () => {
  switchMode("docx");
  localStorage.setItem(storageKeys.inputMode, "docx");
});

docxFile.addEventListener("change", () => {
  fileLabel.textContent = docxFile.files[0]?.name || "Choose a .docx file";
  localStorage.setItem(storageKeys.fileName, fileLabel.textContent);
});

modelSelect.addEventListener("change", () => {
  saveSettings();
  checkHealth();
});

draftText.addEventListener("input", () => {
  localStorage.setItem(storageKeys.draftText, draftText.value);
});

textForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = draftText.value.trim();
  if (text.length < 20) {
    setReady("Need More Text", "Paste at least 20 characters before analysis.");
    return;
  }
  setBusy("Analyzing Text");
  saveSettings();
  localStorage.setItem(storageKeys.draftText, draftText.value);
  try {
    const analysis = await apiStream(
      "/analyze_text_stream",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          session_id: sessionId,
          model: selectedModel(),
          review_mode: selectedReviewMode(),
          format_mode: selectedFormat(),
        }),
      },
      {
        status: (event) => updateThinking(event.step, event.detail),
        analysis: handleAnalysisResult,
      },
    );
    handleAnalysisResult(analysis);
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
  formData.append("session_id", sessionId);
  formData.append("model", selectedModel());
  formData.append("review_mode", selectedReviewMode());
  formData.append("format_mode", selectedFormat());
  setBusy("Analyzing DOCX");
  saveSettings();
  try {
    const analysis = await apiStream(
      "/analyze_docx_stream",
      {
        method: "POST",
        body: formData,
      },
      {
        status: (event) => updateThinking(event.step, event.detail),
        analysis: handleAnalysisResult,
      },
    );
    handleAnalysisResult(analysis);
  } catch (error) {
    setReady("Analysis Failed", error.message);
  }
});

restoreState();
checkHealth();
