/* DiffusionGemma local chat UI */

const $ = (id) => document.getElementById(id);

const state = {
  conversations: [],
  activeId: null,
  jobId: null,
  source: null,
  streaming: false,
  filter: "",
  attachment: null, // {file, url} while an image is staged for extraction
  schemas: [],
  backends: [],     // every configured backend, including unusable ones
  backendId: null,  // the one this conversation is pointed at
  defaultBackend: null,
  uiVersion: null,  // asset stamp this tab booted with
  uiStale: false,
};

/* ------------------------------------------------------------------ auth */

async function api(path, opts = {}) {
  const res = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (res.status === 401) {
    showAuthGate();
    throw new Error("unauthorized");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.status === 204 ? null : res.json();
}

function showAuthGate(message) {
  $("app").classList.add("hidden");
  $("auth-gate").classList.remove("hidden");
  if (message) $("auth-error").textContent = message;
  $("auth-input").focus();
}

async function submitToken(token) {
  const res = await fetch("/api/auth", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  if (!res.ok) throw new Error("토큰이 올바르지 않습니다.");
}

async function boot() {
  // A token in the URL is exchanged for a cookie once, then removed from the
  // address bar so it does not linger in history or get re-sent every request.
  const url = new URL(location.href);
  const urlToken = url.searchParams.get("token");
  if (urlToken) {
    try {
      await submitToken(urlToken);
    } catch (_) {
      /* fall through to the gate */
    }
    url.searchParams.delete("token");
    history.replaceState({}, "", url.pathname + url.search);
  }

  try {
    await refreshStatus();
    $("auth-gate").classList.add("hidden");
    $("app").classList.remove("hidden");
    await loadBackends();
    await refreshStatus();  // now that the picker knows which backend is live
    await loadSchemas();
    await loadConversations();
  } catch (_) {
    showAuthGate();
  }
}

$("auth-submit").onclick = async () => {
  $("auth-error").textContent = "";
  try {
    await submitToken($("auth-input").value.trim());
    location.reload();
  } catch (e) {
    $("auth-error").textContent = e.message;
  }
};
$("auth-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("auth-submit").click();
});

/* ---------------------------------------------------------------- status */

function checkUiVersion(version) {
  if (!version) return;
  if (state.uiVersion === null) {
    state.uiVersion = version; // first status call after boot
    return;
  }
  if (version === state.uiVersion || state.uiStale) return;
  state.uiStale = true;

  // Reloading is safe only when nothing would be lost. Otherwise say so and
  // let the user pick the moment -- but never keep quiet, because stale code
  // renders confidently wrong results.
  const busy =
    state.streaming || state.attachment || $("input").value.trim().length > 0;
  if (!busy) {
    location.reload();
    return;
  }
  const bar = document.createElement("div");
  bar.className = "update-bar";
  bar.innerHTML = "<span>UI가 업데이트되었습니다. 지금 화면은 이전 버전입니다.</span>";
  const btn = document.createElement("button");
  btn.textContent = "새로고침";
  btn.onclick = () => location.reload();
  bar.append(btn);
  document.querySelector(".main").prepend(bar);
}

async function refreshStatus() {
  const s = await api("/api/status");
  checkUiVersion(s.ui_version);
  state.defaultBackend = s.default_backend;
  const b = currentBackend();
  if (b && b.remote) {
    // Nothing runs on this machine for a remote backend, so peak memory and
    // the local context budget would both be describing the wrong model.
    $("engine-sub").textContent = `${b.label} · 원격`;
  } else if (s.max_context) {
    const peak = s.last_peak_gb ? `, 최근 피크 ${s.last_peak_gb}GB` : "";
    $("engine-sub").textContent =
      `컨텍스트 ${(s.max_context / 1000).toFixed(0)}K${peak}`;
  } else {
    $("engine-sub").textContent = b ? b.label : "연결됨";
  }
  updateQueueChip(s);
  return s;
}

/* -------------------------------------------------------------- backends */

function currentBackend() {
  const id = state.backendId || state.defaultBackend;
  return state.backends.find((b) => b.id === id) || null;
}

function knownBackend(id) {
  return id && state.backends.some((b) => b.id === id) ? id : null;
}

async function loadBackends() {
  state.backends = await api("/api/backends");
  const def = state.backends.find((b) => b.default);
  if (def) state.defaultBackend = def.id;
  renderBackendPicker();
}

function renderBackendPicker() {
  const sel = $("backend-select");
  sel.innerHTML = "";
  for (const b of state.backends) {
    const opt = document.createElement("option");
    opt.value = b.id;
    opt.textContent = b.available ? b.label : `${b.label} (사용 불가)`;
    // Kept in the list rather than hidden: a missing API key is worth seeing
    // and explaining, not silently pretending the backend does not exist.
    opt.disabled = !b.available;
    if (b.reason) opt.title = b.reason;
    sel.append(opt);
  }
  sel.value = state.backendId || state.defaultBackend || "";
  applyBackendCapabilities();
}

function applyBackendCapabilities() {
  const b = currentBackend();
  const canSee = !b || b.capabilities.vision;
  const attach = $("attach-btn");
  attach.disabled = !canSee;
  attach.title = canSee ? "이미지 첨부" : `${b.label}는 이미지를 읽을 수 없습니다`;
  if (!canSee && state.attachment) clearAttachment();
  const sub = $("empty-sub");
  if (sub && b) {
    sub.textContent = b.remote
      ? `${b.label}에 연결되어 있습니다.`
      : `${b.label}가 이 기기에서 실행 중입니다.`;
  }
  if (typeof fitPlaceholder === "function") fitPlaceholder();
}

async function setBackend(id, { persist = true } = {}) {
  state.backendId = id;
  $("backend-select").value = id;
  applyBackendCapabilities();
  if (persist && state.activeId) {
    await api(`/api/conversations/${state.activeId}`, {
      method: "PATCH",
      body: JSON.stringify({ backend: id }),
    }).catch(() => {});
  }
  refreshStatus().catch(() => {});
}

function updateQueueChip({ waiting, busy }) {
  const chip = $("queue-chip");
  if (state.streaming) return; // the stream drives the chip while active
  if (busy || waiting) {
    chip.classList.remove("hidden");
    chip.textContent = busy && !waiting ? "모델 사용 중" : `대기 ${waiting}명`;
  } else {
    chip.classList.add("hidden");
  }
}

/* --------------------------------------------------------- conversations */

async function loadConversations() {
  state.conversations = await api("/api/conversations");
  renderSidebar();
  if (!state.activeId && state.conversations.length) {
    await openConversation(state.conversations[0].id, { fromBoot: true });
  }
}

function renderSidebar() {
  const list = $("conv-list");
  list.innerHTML = "";
  const filtered = state.conversations.filter((c) =>
    c.title.toLowerCase().includes(state.filter)
  );
  for (const c of filtered) {
    const row = document.createElement("div");
    row.className = "conv" + (c.id === state.activeId ? " active" : "");
    row.onclick = () => openConversation(c.id);

    const title = document.createElement("div");
    title.className = "conv-title";
    title.textContent = c.title;

    const del = document.createElement("button");
    del.className = "conv-del";
    del.textContent = "×";
    del.title = "삭제";
    del.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`"${c.title}" 대화를 삭제할까요?`)) return;
      await api(`/api/conversations/${c.id}`, { method: "DELETE" });
      if (state.activeId === c.id) {
        state.activeId = null;
        $("thread").innerHTML = "";
        $("topbar-title").textContent = "새 채팅";
        showEmptyState();
      }
      await loadConversations();
    };

    row.append(title, del);
    list.append(row);
  }
}

async function newConversation() {
  const conv = await api("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ backend: state.backendId || state.defaultBackend }),
  });
  state.conversations.unshift(conv);
  state.activeId = conv.id;
  $("thread").innerHTML = "";
  $("topbar-title").textContent = conv.title;
  showEmptyState();
  renderSidebar();
  if (isNarrow()) setSidebar(false);
  $("input").focus();
  return conv;
}

const isNarrow = () => window.matchMedia("(max-width: 720px)").matches;

// Must match READ_MODE_SENTINEL in server.py.
const READ_MODE = "__read__";

async function openConversation(id, { fromBoot = false } = {}) {
  state.activeId = id;
  const conv = state.conversations.find((c) => c.id === id);
  $("topbar-title").textContent = conv ? conv.title : "채팅";
  // A conversation remembers its backend; null means the server default. So
  // does an id that models.json no longer lists -- the server resolves that
  // the same way, and leaving the picker blank would misreport what will run.
  state.backendId = knownBackend(conv && conv.backend) || state.defaultBackend;
  $("backend-select").value = state.backendId || "";
  applyBackendCapabilities();
  renderSidebar();

  const messages = await api(`/api/conversations/${id}/messages`);
  const thread = $("thread");
  thread.innerHTML = "";
  if (!messages.length) {
    showEmptyState();
    return;
  }
  for (const m of messages) {
    if (m.role === "user") {
      appendUser(m.content, m.attachment);
    } else if (m.extraction) {
      const turn = document.createElement("div");
      turn.className = "turn";
      turn.append(renderExtraction(m.extraction));
      thread.append(turn);
    } else {
      appendAssistant(m.content, m.thinking);
    }
  }
  scrollToBottom();
  // Picking a conversation on a phone should reveal it, not leave the drawer
  // covering the thread.
  if (!fromBoot && isNarrow()) setSidebar(false);
}

function showEmptyState() {
  const el = document.createElement("div");
  el.className = "empty-state";
  const b = currentBackend();
  const sub = !b
    ? "모델을 확인하는 중…"
    : b.remote
      ? `${b.label}에 연결되어 있습니다.`
      : `${b.label}가 이 기기에서 실행 중입니다.`;
  const h = document.createElement("h2");
  h.textContent = "무엇을 도와드릴까요?";
  const p = document.createElement("p");
  p.id = "empty-sub";
  p.textContent = sub;
  el.append(h, p);
  $("thread").append(el);
}

function backendLabel(id) {
  if (!id) return "";
  const b = state.backends.find((x) => x.id === id);
  return b ? b.label : id;
}

/* -------------------------------------------------------------- markdown */

// `>` is deliberately left alone: with `<` escaped it can never open a tag,
// and keeping it literal lets the block parser see blockquote markers.
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
}

// The model writes arithmetic in LaTeX. Rather than ship a math typesetter,
// unwrap the handful of constructs it actually uses into readable plain text.
const MATH_SYMBOLS = [
  [/\\times/g, "×"], [/\\div/g, "÷"], [/\\cdot/g, "·"],
  [/\\pm/g, "±"], [/\\leq/g, "≤"], [/\\geq/g, "≥"], [/\\neq/g, "≠"],
  [/\\approx/g, "≈"], [/\\rightarrow/g, "→"], [/\\to/g, "→"],
  [/\\alpha/g, "α"], [/\\beta/g, "β"], [/\\pi/g, "π"], [/\\Delta/g, "Δ"],
];

function stripLatex(s) {
  let m = s.replace(/\\(?:text|mathrm|mathbf|textbf)\{([^{}]*)\}/g, "$1");
  m = m.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, "($1)/($2)");
  m = m.replace(/\\sqrt\{([^{}]*)\}/g, "√($1)");
  for (const [re, sym] of MATH_SYMBOLS) m = m.replace(re, sym);
  return m.replace(/\\left|\\right|\\!|\\,|\\;/g, "");
}

/* --------------------------------------------------------- inline markup */

// Finished HTML is parked behind an index between two private-use characters
// while the remaining inline rules run, so emphasis can never chew through a
// link or a code span. Model output cannot contain these.
const MARK_L = String.fromCharCode(0xe000);
const MARK_R = String.fromCharCode(0xe001);
const PIPE = String.fromCharCode(0xe002);
const HOLE = new RegExp(MARK_L + "(\\d+)" + MARK_R, "g");

function stash(held, html) {
  held.push(html);
  return MARK_L + (held.length - 1) + MARK_R;
}

// Only schemes that cannot execute script; anything else stays plain text.
function safeUrl(url) {
  const u = url.trim();
  if (!/^(?:https?:\/\/|mailto:)/i.test(u)) return null;
  return u.replace(/"/g, "%22").replace(/\s/g, "%20");
}

function link(held, href, label) {
  return stash(
    held,
    `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`
  );
}

function renderInline(src) {
  const held = [];
  let t = src;

  // Code spans outrank every other inline rule, so they go first.
  t = t.replace(/(`+)([^\n]*?)\1/g, (_, __, code) =>
    stash(held, `<code>${code.trim()}</code>`)
  );

  t = t.replace(/\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g, (_, block, inline) =>
    stash(
      held,
      `<code>${stripLatex(block ?? inline).replace(/\\\\/g, " ").trim().replace(/\s+/g, " ")}</code>`
    )
  );
  // The model also writes bare \times, \text{...} and friends without any $
  // delimiters, so clean those up outside math mode too.
  t = stripLatex(t);

  // A backslash escape has to be neutralised before the link and emphasis
  // rules can see the character it was protecting.
  t = t.replace(/\\([\\`*_{}[\]()#+\-.!|~>])/g, (_, ch) => stash(held, ch));

  t = t.replace(/!?\[([^\]\n]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (whole, label, url) => {
    const href = safeUrl(url);
    return href ? link(held, href, label || url) : whole;
  });
  t = t.replace(/(^|[\s(])(https?:\/\/[^\s<>()]*[^\s<>().,;:!?])/g, (_, lead, url) =>
    lead + link(held, safeUrl(url), url)
  );

  t = t
    .replace(/\*\*\*([^*]+)\*\*\*/g, "<strong><em>$1</em></strong>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    // Intra-word underscores (snake_case names) must not become emphasis.
    .replace(/(^|[^_\w])_([^_\n]+)_(?!\w)/g, "$1<em>$2</em>")
    .replace(/~~([^~]+)~~/g, "<del>$1</del>");

  return t.replace(HOLE, (whole, i) => held[i] ?? whole);
}

/* ---------------------------------------------------------- block markup */

const indentOf = (line) => line.match(/^[ \t]*/)[0].replace(/\t/g, "    ").length;

const HR_RE = /^ {0,3}([-*_])[ \t]*(?:\1[ \t]*){2,}$/;
const HEADING_RE = /^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*$/;
const FENCE_RE = /^ {0,3}(```+|~~~+)[ \t]*(\S*)/;
const QUOTE_RE = /^ {0,3}>[ \t]?/;

function listMarker(line) {
  const m = line.match(/^([ \t]*)([-*+]|\d{1,9}[.)])[ \t]+(.*)$/);
  if (!m || HR_RE.test(line)) return null;
  return { indent: indentOf(line), bullet: m[2], text: m[3] };
}

function splitRow(line) {
  return line
    .trim()
    .replace(/\\\|/g, PIPE)
    .replace(/^\||\|$/g, "")
    .split("|")
    .map((c) => c.split(PIPE).join("|").trim());
}

function isTableDelimiter(line) {
  if (!line.includes("-") || !line.includes("|")) return false;
  const cells = splitRow(line);
  return cells.length > 0 && cells.every((c) => /^:?-+:?$/.test(c));
}

function startsBlock(lines, i) {
  const line = lines[i];
  return (
    HR_RE.test(line) ||
    HEADING_RE.test(line) ||
    FENCE_RE.test(line) ||
    QUOTE_RE.test(line) ||
    listMarker(line) !== null ||
    (line.includes("|") && i + 1 < lines.length && isTableDelimiter(lines[i + 1]))
  );
}

function renderBlocks(lines) {
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i++;
      continue;
    }

    const fence = line.match(FENCE_RE);
    if (fence) {
      const close = new RegExp(`^ {0,3}\\${fence[1][0]}{${fence[1].length},}[ \\t]*$`);
      const body = [];
      i++;
      // An unclosed fence still renders as code: while streaming, the closing
      // one simply has not arrived yet.
      while (i < lines.length && !close.test(lines[i])) body.push(lines[i++]);
      if (i < lines.length) i++;
      const lang = fence[2].replace(/[^\w+#-]/g, "");
      out.push(
        `<pre><code${lang ? ` data-lang="${lang}"` : ""}>${body.join("\n")}</code></pre>`
      );
      continue;
    }

    if (HR_RE.test(line)) {
      out.push("<hr>");
      i++;
      continue;
    }

    const h = line.match(HEADING_RE);
    if (h) {
      const level = h[1].length;
      out.push(`<h${level}>${renderInline(h[2])}</h${level}>`);
      i++;
      continue;
    }

    if (line.includes("|") && i + 1 < lines.length && isTableDelimiter(lines[i + 1])) {
      const [html, next] = renderTable(lines, i);
      out.push(html);
      i = next;
      continue;
    }

    if (QUOTE_RE.test(line)) {
      const body = [];
      while (i < lines.length && QUOTE_RE.test(lines[i])) {
        body.push(lines[i].replace(QUOTE_RE, ""));
        i++;
      }
      out.push(`<blockquote>${renderBlocks(body)}</blockquote>`);
      continue;
    }

    if (listMarker(line)) {
      const [html, next] = renderList(lines, i);
      out.push(html);
      i = next;
      continue;
    }

    // A run of plain lines is one paragraph; the single newlines inside it are
    // kept as breaks, because the model uses them to lay out its answer.
    const para = [];
    do {
      para.push(lines[i++]);
    } while (i < lines.length && lines[i].trim() && !startsBlock(lines, i));
    out.push(`<p>${para.map(renderInline).join("<br>")}</p>`);
  }

  return out.join("\n");
}

function renderTable(lines, start) {
  const head = splitRow(lines[start]);
  const aligns = splitRow(lines[start + 1]).map((c) => {
    const left = c.startsWith(":");
    const right = c.endsWith(":");
    if (left && right) return ' style="text-align:center"';
    if (right) return ' style="text-align:right"';
    return "";
  });

  let i = start + 2;
  const rows = [];
  while (i < lines.length && lines[i].trim() && lines[i].includes("|")) {
    rows.push(splitRow(lines[i]));
    i++;
  }

  // Ragged rows are padded out to the header width so columns stay aligned.
  const cells = (row, tag) =>
    head
      .map((_, n) => `<${tag}${aligns[n] || ""}>${renderInline(row[n] ?? "")}</${tag}>`)
      .join("");

  const body = rows.map((r) => `<tr>${cells(r, "td")}</tr>`).join("");
  return [
    `<div class="table-wrap"><table><thead><tr>${cells(head, "th")}</tr></thead>` +
      `<tbody>${body}</tbody></table></div>`,
    i,
  ];
}

function renderList(lines, start) {
  const first = listMarker(lines[start]);
  const ordered = /\d/.test(first.bullet);
  const base = first.indent;
  const items = [];
  let i = start;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      // A blank line only ends the list if what follows sits outside it.
      const next = lines[i + 1];
      const marker = next ? listMarker(next) : null;
      if (!next || !next.trim()) break;
      if (!marker && indentOf(next) <= base) break;
      if (marker && marker.indent <= base && /\d/.test(marker.bullet) !== ordered) break;
      if (items.length) items[items.length - 1].children.push("");
      i++;
      continue;
    }

    const m = listMarker(line);
    if (m && m.indent <= base + 1) {
      // A bullet list interrupting a numbered one (or vice versa) is a new list.
      if (/\d/.test(m.bullet) !== ordered) break;
      items.push({ text: m.text, children: [] });
      i++;
      continue;
    }

    // Anything indented past the marker is nested content of the open item.
    if (items.length && indentOf(line) > base) {
      items[items.length - 1].children.push(line);
      i++;
      continue;
    }
    break;
  }

  const html = items
    .map((item) => {
      const task = item.text.match(/^\[([ xX])\][ \t]+([\s\S]*)$/);
      const text = task
        ? `<input type="checkbox" disabled${task[1] === " " ? "" : " checked"}> ` +
          renderInline(task[2])
        : renderInline(item.text);
      const nested = item.children.length ? renderBlocks(dedent(item.children)) : "";
      return `<li${task ? ' class="task"' : ""}>${text}${nested}</li>`;
    })
    .join("");

  const tag = ordered ? "ol" : "ul";
  const from = ordered ? parseInt(first.bullet, 10) : 1;
  return [`<${tag}${from !== 1 ? ` start="${from}"` : ""}>${html}</${tag}>`, i];
}

// Nested content is re-parsed from column zero, so a code block inside a list
// item keeps only the indentation it actually meant to have.
function dedent(lines) {
  const filled = lines.filter((l) => l.trim());
  if (!filled.length) return lines;
  const pad = Math.min(...filled.map(indentOf));
  return lines.map((l) =>
    l.replace(/^[ \t]*/, (w) =>
      " ".repeat(Math.max(0, w.replace(/\t/g, "    ").length - pad))
    )
  );
}

function renderMarkdown(src) {
  // Escape first, so nothing below can inject markup.
  return renderBlocks(escapeHtml(src).split("\n"));
}

/* --------------------------------------------------------- message nodes */

function appendUser(text, attachment) {
  clearEmptyState();
  const turn = document.createElement("div");
  turn.className = "turn msg-user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (attachment) {
    const img = document.createElement("img");
    img.className = "bubble-thumb";
    img.src = attachment.startsWith("blob:") || attachment.startsWith("data:")
      ? attachment
      : `/api/attachments/${attachment}`;
    img.alt = "첨부 이미지";
    bubble.append(img);
  }
  const label = document.createElement("div");
  label.textContent = text;
  bubble.append(label);
  turn.append(bubble);
  $("thread").append(turn);
  return turn;
}

/* ------------------------------------------------------ extraction result */

function buildDisclosure(openLabel, closeLabel, text) {
  const toggle = document.createElement("button");
  toggle.className = "thinking-toggle";
  toggle.innerHTML = `<span>${openLabel}</span><span class="caret">⌄</span>`;
  const body = document.createElement("div");
  body.className = "thinking-body hidden";
  body.textContent = text;
  toggle.onclick = () => {
    const open = body.classList.toggle("hidden") === false;
    toggle.classList.toggle("open", open);
    toggle.firstChild.textContent = open ? closeLabel : openLabel;
  };
  return { toggle, body };
}

const STATUS_LABEL = {
  ok: "3패스 일치",
  majority: "다수결로 채택 — 확인 권장",
  resolved: "재판독으로 확정",
  conflict: "패스마다 다름",
  unresolved: "확정 실패 — 직접 확인 필요",
  invalid: "형식 규칙 위반",
  missing: "값 없음",
};

function renderExtraction(result) {
  const wrap = document.createElement("div");
  wrap.className = "extraction";

  const s = result.stats || {};
  const flagged = (result.flagged_rows || []).length;

  const head = document.createElement("div");
  head.className = "extract-head";
  head.innerHTML =
    `<strong>${s.rows ?? 0}행</strong> 추출 · 타일 ${s.tiles ?? "?"}개 × ${s.passes ?? "?"}패스 · ` +
    `모델 호출 ${s.model_calls ?? "?"}회` +
    (s.rereads ? ` · 재판독 ${s.rereads}회` : "") +
    ` · ${s.wall_seconds ?? "?"}s`;
  wrap.append(head);

  // What the model worked out about the page before reading any of it. Shown
  // above the table because it is the premise every value below rests on: if
  // the column mapping is wrong here, every row is wrong in the same way.
  const a = result.analysis;
  if (a && (a.doc_type || (a.columns || []).length)) {
    const box = document.createElement("div");
    box.className = "analysis";
    const line = document.createElement("div");
    line.className = "analysis-line";
    const bits = [];
    if (a.doc_type) bits.push(`<strong>${escapeHtml(a.doc_type)}</strong>`);
    if ((a.columns || []).length)
      bits.push(`열: ${escapeHtml(a.columns.join(", "))}`);
    line.innerHTML = "먼저 파악한 것 — " + bits.join(" · ");
    box.append(line);

    if (a.notes) {
      const note = document.createElement("div");
      note.className = "analysis-note";
      note.textContent = `주의: ${a.notes}`;
      box.append(note);
    }
    if (a.thinking) {
      const { toggle, body } = buildDisclosure("판단 근거 보기", "판단 근거 숨기기", a.thinking);
      box.append(toggle, body);
    }
    wrap.append(box);
  }

  const rowCount = (result.records || []).length;
  const verdict = document.createElement("div");
  if (rowCount === 0) {
    // Nothing was extracted, so there is nothing that "passed". Saying so in
    // green would be the exact silent-success failure this tool exists to
    // avoid, one level up.
    verdict.className = "extract-verdict empty";
    verdict.textContent =
      result.empty_reason ||
      "이 이미지에서 스키마와 맞는 행을 찾지 못했습니다.";
  } else {
    verdict.className = "extract-verdict " + (flagged ? "warn" : "clean");
    verdict.textContent = flagged
      ? `${flagged}행은 사람이 확인해야 합니다. 표시된 셀을 눌러 근거를 보세요.`
      : `${rowCount}행 모두 ${s.passes ?? 3}패스 일치 및 형식 검증을 통과했습니다.`;
  }
  wrap.append(verdict);

  if (rowCount === 0) {
    const help = document.createElement("div");
    help.className = "extract-empty-help";
    const fields = (result.fields || []).join(", ");
    help.innerHTML = result.aborted
      ? `<p>표를 읽기 전에 멈췄습니다. 모델 호출 ` +
        `${s.model_calls ?? 1}회로 끝났으므로 헛돈 시간은 없습니다.</p>`
      : `<p>선택한 스키마 <code>${escapeHtml(result.schema_name || "")}</code>는 ` +
        `<code>${escapeHtml(fields)}</code> 열을 가진 표를 찾습니다. ` +
        `이미지에 그런 표가 없으면 아무 행도 나오지 않습니다.</p>`;
    wrap.append(help);

    const retry = document.createElement("button");
    retry.className = "retry-read";
    retry.textContent = "이미지 읽기 모드로 다시 시도";
    retry.onclick = () => retryAsRead(result);
    wrap.append(retry);

    if (result.samples && result.samples.length) {
      const { toggle, body } = buildDisclosure(
        "모델이 실제로 답한 내용 보기",
        "모델 답변 숨기기",
        result.samples.join("\n---\n")
      );
      wrap.append(toggle, body);
    }
    return wrap; // an empty table adds nothing
  }

  const scroll = document.createElement("div");
  scroll.className = "extract-table-wrap";
  const table = document.createElement("table");
  table.className = "extract-table";

  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  hr.append(document.createElement("th")); // row number
  for (const name of result.fields) {
    const th = document.createElement("th");
    th.textContent = name;
    hr.append(th);
  }
  thead.append(hr);
  table.append(thead);

  const tbody = document.createElement("tbody");
  (result.records || []).forEach((rec, i) => {
    const tr = document.createElement("tr");
    const num = document.createElement("td");
    num.className = "row-num";
    num.textContent = i + 1;
    if (rec.seen < rec.passes) {
      num.classList.add("partial");
      num.title = `${rec.passes}회 중 ${rec.seen}회만 관측됨`;
    }
    tr.append(num);

    for (const name of result.fields) {
      const cell = rec.cells[name] || {};
      const td = document.createElement("td");
      td.className = `cell status-${cell.status || "ok"}`;
      td.textContent = cell.value || "";
      if (cell.status && cell.status !== "ok") {
        const parts = [STATUS_LABEL[cell.status] || cell.status];
        if (cell.votes) parts.push(`득표 ${cell.votes}`);
        if (cell.reason) parts.push(cell.reason);
        if (cell.candidates && cell.candidates.length)
          parts.push(`후보: ${cell.candidates.join(" / ")}`);
        td.title = parts.join(" · ");
        td.tabIndex = 0;
        td.onclick = () => showCellDetail(td, name, cell);
      }
      tr.append(td);
    }
    tbody.append(tr);
  });
  table.append(tbody);
  scroll.append(table);
  wrap.append(scroll);

  // Set aside, never hidden: a discarded row is still shown on request so the
  // count can be checked against the source document.
  if (result.discarded && result.discarded.length) {
    const toggle = document.createElement("button");
    toggle.className = "thinking-toggle";
    toggle.innerHTML =
      `<span>판독 실패로 제외한 ${result.discarded.length}행 보기</span>` +
      `<span class="caret">⌄</span>`;
    const body = document.createElement("div");
    body.className = "thinking-body hidden";
    body.textContent = result.discarded
      .map((r, i) =>
        `${i + 1}. ` +
        result.fields.map((f) => `${f}=${r.cells[f]?.value || ""}`).join("  ")
      )
      .join("\n");
    toggle.onclick = () => {
      const open = body.classList.toggle("hidden") === false;
      toggle.classList.toggle("open", open);
      toggle.firstChild.textContent = open
        ? `제외한 ${result.discarded.length}행 숨기기`
        : `판독 실패로 제외한 ${result.discarded.length}행 보기`;
    };
    wrap.append(toggle, body);
  }

  if (result.problems && result.problems.length) {
    const list = document.createElement("ul");
    list.className = "extract-problems";
    for (const p of result.problems) {
      const li = document.createElement("li");
      li.textContent = p;
      list.append(li);
    }
    wrap.append(list);
  }

  wrap.append(buildExtractActions(result));
  return wrap;
}

function showCellDetail(td, name, cell) {
  const existing = td.querySelector(".cell-pop");
  if (existing) {
    existing.remove();
    return;
  }
  document.querySelectorAll(".cell-pop").forEach((p) => p.remove());
  const pop = document.createElement("div");
  pop.className = "cell-pop";
  const lines = [`${name}: ${STATUS_LABEL[cell.status] || cell.status}`];
  if (cell.votes) lines.push(`득표 ${cell.votes}`);
  if (cell.reason) lines.push(cell.reason);
  if (cell.candidates && cell.candidates.length)
    lines.push(`패스별 값: ${cell.candidates.join(" / ")}`);
  pop.textContent = lines.join("\n");
  pop.onclick = (e) => {
    e.stopPropagation();
    pop.remove();
  };
  td.append(pop);
}

function buildExtractActions(result) {
  const row = document.createElement("div");
  row.className = "msg-actions";

  const csv = document.createElement("button");
  csv.className = "icon-btn";
  csv.textContent = "⤓";
  csv.title = "CSV 복사 (검토 필요 셀은 ? 표시)";
  csv.onclick = async () => {
    const lines = [result.fields.join(",")];
    for (const rec of result.records) {
      lines.push(
        result.fields
          .map((f) => {
            const c = rec.cells[f] || {};
            const v = (c.value || "").replace(/"/g, '""');
            // Never hand out a flagged value as if it were clean.
            const mark = c.status && c.status !== "ok" ? "?" : "";
            return `"${mark}${v}"`;
          })
          .join(",")
      );
    }
    await navigator.clipboard.writeText(lines.join("\n"));
    csv.textContent = "✓";
    setTimeout(() => (csv.textContent = "⤓"), 1200);
  };
  row.append(csv);
  return row;
}

function appendAssistant(answer, thinking) {
  clearEmptyState();
  const turn = document.createElement("div");
  turn.className = "turn";

  const node = { turn };

  if (thinking) {
    const { toggle, body } = buildThinking(thinking);
    turn.append(toggle, body);
    node.thinkingBody = body;
  }

  const body = document.createElement("div");
  body.className = "msg-assistant";
  body.innerHTML = renderMarkdown(answer || "");
  turn.append(body);
  node.body = body;

  if (answer) turn.append(buildActions(answer));

  $("thread").append(turn);
  return node;
}

function buildThinking(text) {
  const toggle = document.createElement("button");
  toggle.className = "thinking-toggle";
  toggle.innerHTML = `<span>생각하는 과정 표시</span><span class="caret">⌄</span>`;
  const body = document.createElement("div");
  body.className = "thinking-body hidden";
  body.textContent = text;
  toggle.onclick = () => {
    const open = body.classList.toggle("hidden") === false;
    toggle.classList.toggle("open", open);
    toggle.firstChild.textContent = open ? "생각하는 과정 숨기기" : "생각하는 과정 표시";
  };
  return { toggle, body };
}

function buildActions(text) {
  const row = document.createElement("div");
  row.className = "msg-actions";
  const copy = document.createElement("button");
  copy.className = "icon-btn";
  copy.textContent = "⧉";
  copy.title = "복사";
  copy.onclick = async () => {
    await navigator.clipboard.writeText(text);
    copy.textContent = "✓";
    setTimeout(() => (copy.textContent = "⧉"), 1200);
  };
  row.append(copy);
  return row;
}

function clearEmptyState() {
  const el = $("thread").querySelector(".empty-state");
  if (el) el.remove();
}

function notice(text, isError) {
  const el = document.createElement("div");
  el.className = "notice" + (isError ? " error" : "");
  el.textContent = text;
  $("thread").append(el);
  scrollToBottom();
  return el;
}

function scrollToBottom() {
  const t = $("thread");
  t.scrollTop = t.scrollHeight;
}

/* ---------------------------------------------------------------- sending */

/* ----------------------------------------------------------- attachments */

async function loadSchemas() {
  try {
    state.schemas = await api("/api/schemas");
  } catch (_) {
    state.schemas = [];
  }
  const sel = $("schema-select");
  sel.innerHTML = "";
  // Default to plain reading: an arbitrary image forced through a table schema
  // yields nothing, which reads as a broken tool rather than a wrong choice.
  const read = document.createElement("option");
  read.value = READ_MODE;
  read.textContent = "이미지 읽기 (표 아님, 검증 없음)";
  sel.append(read);
  for (const s of state.schemas) {
    const opt = document.createElement("option");
    opt.value = s.file;
    opt.textContent = `표 추출: ${s.name} (${s.fields.join(", ")})`;
    sel.append(opt);
  }
  syncSchemaHint();
}

function syncSchemaHint() {
  const isRead = $("schema-select").value === READ_MODE;
  $("passes-select").disabled = isRead;
  $("passes-select").title = isRead
    ? "읽기 모드에서는 다중 패스 검증을 쓰지 않습니다"
    : "";
  $("attach-hint").textContent = isRead
    ? "이미지 내용을 그대로 읽습니다. 교차 검증은 하지 않습니다."
    : "표를 찾아 스키마대로 추출하고, 패스 간 불일치를 표시합니다.";
}

function setAttachment(file) {
  clearAttachment();
  if (!file) return;
  state.attachment = { file, url: URL.createObjectURL(file) };
  $("attach-thumb").src = state.attachment.url;
  $("attach-name").textContent =
    `${file.name} · ${(file.size / 1024).toFixed(0)}KB`;
  $("attach-bar").classList.remove("hidden");
  if (!state.schemas.length) loadSchemas();
}

function clearAttachment({ keepUrl = false } = {}) {
  // After sending, the blob URL is still the src of the thumbnail in the sent
  // message; revoking it there would blank the image the user just posted.
  // The bubble owns it from then on until the page reloads.
  if (state.attachment && !keepUrl) URL.revokeObjectURL(state.attachment.url);
  state.attachment = null;
  $("file-input").value = "";
  $("attach-bar").classList.add("hidden");
}

async function sendExtraction() {
  const { file, url } = state.attachment;
  const schemaFile = $("schema-select").value;
  if (!schemaFile) {
    notice("스키마가 없습니다. schemas/ 폴더에 JSON을 추가하세요.", true);
    return;
  }
  if (!state.activeId) await newConversation();

  const input = $("input");
  const question = input.value.trim();
  const isRead = schemaFile === READ_MODE;

  const form = new FormData();
  form.append("file", file);
  form.append("conversation_id", state.activeId);
  if (state.backendId) form.append("backend_id", state.backendId);
  form.append("schema_file", schemaFile);
  form.append("passes", $("passes-select").value);
  if (isRead && question) form.append("question", question);

  // Kept so a rejected extraction can be retried as a plain read without
  // asking the user to find and attach the same file again.
  state.lastImage = file;

  const label = isRead
    ? question || "[이미지 읽기]"
    : `[이미지 추출] ${schemaFile.replace(/\.json$/, "")}`;
  input.value = "";
  input.style.height = "auto";
  appendUser(label, url);
  scrollToBottom();
  const pending = notice("업로드 중…");
  clearAttachment({ keepUrl: true });

  let jobId;
  try {
    // Not api(): that helper always sets a JSON content type, and multipart
    // needs the browser to set its own boundary.
    const res = await fetch("/api/extract", {
      method: "POST",
      credentials: "same-origin",
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `${res.status} ${res.statusText}`);
    }
    jobId = (await res.json()).job_id;
  } catch (e) {
    pending.remove();
    notice(`추출 실패: ${e.message}`, true);
    return;
  }

  state.jobId = jobId;
  setStreaming(true);
  streamJob(jobId, pending);
}

function retryAsRead() {
  if (state.streaming) return;
  if (!state.lastImage) {
    notice("원본 이미지가 없습니다. 다시 첨부해 주세요.", true);
    return;
  }
  setAttachment(state.lastImage);
  $("schema-select").value = READ_MODE;
  syncSchemaHint();
  sendExtraction();
}

async function send() {
  if (state.streaming) return;
  if (state.attachment) {
    await sendExtraction();
    return;
  }

  const input = $("input");
  const content = input.value.trim();
  if (!content) return;

  if (!state.activeId) await newConversation();

  input.value = "";
  input.style.height = "auto";
  appendUser(content);
  scrollToBottom();

  const pending = notice("전송 중…");

  let jobId;
  try {
    const r = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        conversation_id: state.activeId,
        content,
        backend: state.backendId || undefined,
      }),
    });
    jobId = r.job_id;
  } catch (e) {
    pending.remove();
    notice(`요청 실패: ${e.message}`, true);
    return;
  }

  state.jobId = jobId;
  setStreaming(true);
  streamJob(jobId, pending);
}

function streamJob(jobId, pendingNotice) {
  const src = new EventSource(`/api/stream/${jobId}`, { withCredentials: true });
  state.source = src;

  let node = null;
  let answerRaw = "";
  let thinkingRaw = "";
  let draftRaw = "";
  let frame = null;

  const ensureNode = () => {
    if (!node) {
      pendingNotice.remove();
      node = appendAssistant("", null);
    }
    return node;
  };

  // The canvas the model is denoising right now: a snapshot that is replaced
  // whole on every frame, not text that accumulates. It sits below the settled
  // answer and disappears once that canvas is committed as real tokens.
  const renderDraft = () => {
    if (!draftRaw) {
      clearDraft();
      return;
    }
    if (!node.draft) {
      node.draft = document.createElement("div");
      node.draft.className = "draft-canvas";
      node.turn.append(node.draft);
    }
    node.draft.innerHTML = escapeHtml(draftRaw).replaceAll(
      "[Mask]",
      `<span class="mask">░</span>`
    );
  };

  const clearDraft = () => {
    draftRaw = "";
    if (node && node.draft) {
      node.draft.remove();
      node.draft = null;
    }
  };

  const scheduleRender = () => {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = null;
      if (!node) return;
      node.body.innerHTML = renderMarkdown(answerRaw);
      if (node.thinkingBody) node.thinkingBody.textContent = thinkingRaw;
      renderDraft();
      scrollToBottom();
    });
  };

  src.addEventListener("queued", (e) => {
    const { position } = JSON.parse(e.data);
    const chip = $("queue-chip");
    chip.classList.remove("hidden");
    if (position <= 0) {
      chip.textContent = "곧 시작합니다";
      pendingNotice.textContent = "곧 시작합니다…";
    } else {
      chip.textContent = `대기열 ${position}번째`;
      pendingNotice.textContent = `앞에 ${position}개의 요청이 있습니다. 대기 중…`;
    }
  });

  src.addEventListener("start", (e) => {
    const d = JSON.parse(e.data);
    $("queue-chip").textContent = d.mode === "extract" ? "추출 중" : "생성 중";
    pendingNotice.innerHTML =
      `<span class="dots"><span></span><span></span><span></span></span>`;
    if (d.dropped_turns > 0) {
      const budget = d.prompt_tokens != null
        ? `컨텍스트 예산(${d.prompt_tokens.toLocaleString()} 토큰)`
        : "컨텍스트 예산";
      notice(
        `${budget}에 맞추기 위해 오래된 대화 ${d.dropped_turns}턴을 제외했습니다.`
      );
    }
  });

  src.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    if (d.phase === "probe") {
      pendingNotice.textContent = "먼저 이미지 구조를 파악하는 중…";
      $("queue-chip").textContent = "구조 파악";
    } else if (d.phase === "tile") {
      pendingNotice.textContent =
        `읽는 중 — ${d.pass_index}/${d.passes}패스, 타일 ${d.tile}/${d.tiles}`;
      $("queue-chip").textContent = `${d.pass_index}/${d.passes}패스`;
    } else if (d.phase === "reread") {
      pendingNotice.textContent = `${d.row}행 재판독 중 (패스 간 불일치)`;
    }
  });

  src.addEventListener("draft", (e) => {
    const { text } = JSON.parse(e.data);
    ensureNode();
    draftRaw = text;
    scheduleRender();
  });

  src.addEventListener("token", (e) => {
    const { kind, text } = JSON.parse(e.data);
    const n = ensureNode();
    // Settled text supersedes the canvas it came out of.
    draftRaw = "";
    if (kind === "thinking") {
      if (!n.thinkingBody) {
        const { toggle, body } = buildThinking("");
        n.turn.insertBefore(toggle, n.body);
        n.turn.insertBefore(body, n.body);
        n.thinkingBody = body;
      }
      thinkingRaw += text;
    } else {
      answerRaw += text;
    }
    scheduleRender();
  });

  src.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);

    if (d.mode === "extract") {
      pendingNotice.remove();
      clearEmptyState();
      const turn = document.createElement("div");
      turn.className = "turn";
      turn.append(renderExtraction(d.extraction));
      const s = d.stats || {};
      const line = document.createElement("div");
      line.className = "stats-line";
      line.textContent =
        `호출 ${s.model_calls ?? "?"}회 · ${s.wall_seconds ?? "?"}s` +
        (s.peak_gb ? ` · 피크 ${s.peak_gb}GB` : "");
      turn.append(line);
      $("thread").append(turn);
      finish(src, pendingNotice);
      loadConversations();
      return;
    }

    const n = ensureNode();
    clearDraft();
    n.body.innerHTML = renderMarkdown(answerRaw);
    if (n.thinkingBody) n.thinkingBody.textContent = thinkingRaw;
    n.turn.append(buildActions(answerRaw));

    const s = d.stats || {};
    const line = document.createElement("div");
    line.className = "stats-line";
    const bits = [];
    if (s.prompt_tokens != null) bits.push(`ctx ${s.prompt_tokens.toLocaleString()} tok`);
    bits.push(`생성 ${s.generation_tokens ?? "?"} tok`);
    if (s.generation_tps != null) bits.push(`${s.generation_tps} tok/s`);
    bits.push(`${s.wall_seconds}s`);
    if (s.peak_gb) bits.push(`피크 ${s.peak_gb}GB`);
    const label = backendLabel(d.backend);
    if (label) bits.push(label);
    line.textContent = bits.join(" · ");
    n.turn.append(line);

    finish(src, pendingNotice);
    loadConversations();
  });

  src.addEventListener("cancelled", () => {
    clearDraft();
    notice("중지되었습니다.");
    finish(src, pendingNotice);
  });

  src.addEventListener("error", (e) => {
    clearDraft();
    // A server-sent 'error' event carries data; a transport drop does not.
    if (e.data) {
      const d = JSON.parse(e.data);
      notice(`오류: ${d.message}`, true);
    } else if (state.streaming) {
      notice("서버와의 연결이 끊겼습니다.", true);
    }
    finish(src, pendingNotice);
  });
}

function finish(src, pendingNotice) {
  src.close();
  pendingNotice.remove();
  state.source = null;
  state.jobId = null;
  setStreaming(false);
  refreshStatus().catch(() => {});
  scrollToBottom();
}

function setStreaming(on) {
  state.streaming = on;
  $("send").classList.toggle("hidden", on);
  $("stop").classList.toggle("hidden", !on);
  if (!on) $("queue-chip").classList.add("hidden");
}

/* ----------------------------------------------------------------- events */

$("send").onclick = send;
$("backend-select").onchange = (e) => setBackend(e.target.value);
$("attach-btn").onclick = () => $("file-input").click();
$("file-input").onchange = (e) => setAttachment(e.target.files[0]);
$("attach-clear").onclick = clearAttachment;
$("schema-select").onchange = syncSchemaHint;

// Dropping an image anywhere on the thread stages it, which is how people
// actually move a screenshot into a page.
const thread = $("thread");
["dragover", "drop"].forEach((ev) =>
  thread.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === "drop") {
      const file = [...(e.dataTransfer?.files || [])].find((f) =>
        f.type.startsWith("image/")
      );
      if (file) setAttachment(file);
    }
  })
);
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) =>
    i.type.startsWith("image/")
  );
  if (item) setAttachment(item.getAsFile());
});
$("stop").onclick = async () => {
  if (!state.jobId) return;
  try {
    await api(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
  } catch (_) {
    /* the job may have finished already */
  }
};
$("new-chat").onclick = newConversation;
$("search").oninput = (e) => {
  state.filter = e.target.value.toLowerCase();
  renderSidebar();
};
function setSidebar(show) {
  // Narrow screens use `open` (hidden by default, overlays the thread); wide
  // screens use `collapsed` (visible by default, sits beside the thread).
  const sb = $("sidebar");
  if (isNarrow()) {
    sb.classList.toggle("open", show);
    $("backdrop").classList.toggle("show", show);
  } else {
    sb.classList.toggle("collapsed", !show);
    $("backdrop").classList.remove("show");
  }
}

$("sidebar-toggle").onclick = () => setSidebar(false);
$("sidebar-open").onclick = () => setSidebar(true);
$("backdrop").onclick = () => setSidebar(false);

// Crossing the breakpoint clears the overlay state so a mobile-mode toggle
// never leaves the desktop layout in a stuck position.
window.addEventListener("resize", () => {
  if (!isNarrow()) {
    $("sidebar").classList.remove("open");
    $("backdrop").classList.remove("show");
  }
});

const input = $("input");

// A one-row textarea wraps a too-long placeholder and clips it mid-line, so
// pick the wording to fit the width rather than relying on CSS ellipsis
// (which browsers do not apply to textarea placeholders).
function fitPlaceholder() {
  const b = currentBackend();
  input.placeholder =
    input.clientWidth < 320 ? "메시지 입력"
      : b ? `${b.label}에게 물어보기` : "무엇이든 물어보기";
}
fitPlaceholder();
window.addEventListener("resize", fitPlaceholder);

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
});

setInterval(() => {
  if (!state.streaming) refreshStatus().catch(() => {});
}, 5000);

boot();
