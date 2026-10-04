/* open-medi web UI — vanilla JS, no dependencies. */

const $ = (id) => document.getElementById(id);

/* ---------- health chip ---------- */

async function loadHealth() {
  const chip = $("health");
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    const kb = h.knowledge_base;
    const kbText = kb.ready
      ? `KB: ${kb.diseases} diseases / ${kb.chunks} chunks`
      : "KB: not built (run scripts/build_db.py)";
    chip.textContent = `${kbText} · LLM: ${h.llm.model}` +
      (h.icd11.configured ? " · ICD-11 ✓" : "");
    chip.classList.toggle("ok", kb.ready);
    chip.classList.toggle("warn", !kb.ready);
  } catch {
    chip.textContent = "API unreachable";
    chip.classList.add("warn");
  }
}

/* ---------- status helper ---------- */

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text || "";
  el.classList.toggle("error", isError);
}

/* ---------- audio recording ---------- */

let recorder = null;
let chunks = [];

async function toggleRecording() {
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("This browser does not support microphone capture.", true);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : undefined;
    recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    chunks = [];
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      document.body.classList.remove("recording");
      $("recordBtn").classList.remove("recording");
      $("recordHint").textContent = "Tap the mic and describe your symptoms";
      sendAudio(new Blob(chunks, { type: recorder.mimeType || "audio/webm" }));
    };
    recorder.start();
    document.body.classList.add("recording");
    $("recordBtn").classList.add("recording");
    $("recordHint").textContent = "Recording… tap again to stop";
    setStatus("");
  } catch (err) {
    setStatus("Microphone access denied: " + err.message, true);
  }
}

async function sendAudio(blob) {
  setStatus("Transcribing…");
  const form = new FormData();
  const ext = blob.type.includes("ogg") ? "ogg" : "webm";
  form.append("file", blob, `recording.${ext}`);
  let res;
  try {
    res = await fetch("/api/pipeline", { method: "POST", body: form });
  } catch {
    setStatus("Could not reach the API. Is the server running?", true);
    return;
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    setStatus(body.detail || `Request failed (${res.status})`, true);
    return;
  }
  setStatus("");
  const data = await res.json();
  renderTranscript(data.transcript);
  if (data.diagnosis) {
    renderDiagnosis(data.diagnosis);
  } else {
    setStatus("No speech detected in the recording.", true);
    $("results").hidden = true;
  }
}

/* ---------- typed symptoms ---------- */

async function submitTyped(event) {
  event.preventDefault();
  const text = $("typedText").value.trim();
  if (text.length < 2) return;
  setStatus("Analyzing…");
  const res = await fetch("/api/diagnose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }).catch(() => null);
  if (!res || !res.ok) {
    const body = res ? await res.json().catch(() => ({})) : {};
    setStatus(body.detail || "Request failed.", true);
    return;
  }
  setStatus("");
  $("transcriptPanel").hidden = true;
  renderDiagnosis(await res.json());
}

/* ---------- rendering ---------- */

function renderTranscript(transcript) {
  $("transcriptPanel").hidden = false;
  $("transcriptText").textContent = transcript.text || "(nothing audible)";
  const meta = [];
  if (transcript.language) meta.push(`language: ${transcript.language}`);
  if (transcript.duration) meta.push(`${transcript.duration.toFixed(1)}s`);
  $("transcriptMeta").textContent = meta.join(" · ");
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function confidenceClass(score) {
  return score >= 75 ? "high" : score >= 40 ? "medium" : "low";
}

function renderDiagnosis(diagnosis) {
  $("results").hidden = false;
  $("summary").textContent = diagnosis.summary || "";

  const flags = $("redFlagList");
  flags.replaceChildren();
  $("redFlags").hidden = !(diagnosis.red_flags || []).length;
  (diagnosis.red_flags || []).forEach((flag) =>
    flags.append(el("li", null, flag))
  );

  const wrap = $("conditions");
  wrap.replaceChildren();
  (diagnosis.conditions || []).forEach((cond) => {
    const card = el("article", "panel condition");
    const head = el("div", "cond-head");
    head.append(el("h3", null, cond.name));
    if (cond.icd11_code) {
      head.append(el("span", "icd-badge", `ICD-11 ${cond.icd11_code}`));
    }
    card.append(head);

    const barWrap = el("div", "bar");
    const bar = el("div", `bar-fill ${confidenceClass(cond.confidence)}`);
    bar.style.width = `${Math.max(4, cond.confidence)}%`;
    barWrap.append(bar);
    card.append(barWrap);
    card.append(
      el("p", "muted", `estimated likelihood: ${cond.confidence}%`)
    );

    if ((cond.matched_symptoms || []).length) {
      const chips = el("div", "chips");
      cond.matched_symptoms.forEach((m) => chips.append(el("span", "chip", m)));
      card.append(chips);
    }
    if (cond.rationale) card.append(el("p", "rationale", cond.rationale));
    wrap.append(card);
  });

  $("advice").textContent = diagnosis.advice || "";
  $("results").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------- ICD-11 search ---------- */

async function submitIcd(event) {
  event.preventDefault();
  const q = $("icdQuery").value.trim();
  if (!q) return;
  const list = $("icdResults");
  list.replaceChildren(el("li", "muted", "Searching…"));
  const res = await fetch(`/api/icd/search?q=${encodeURIComponent(q)}`).catch(() => null);
  list.replaceChildren();
  if (!res || !res.ok) {
    const body = res ? await res.json().catch(() => ({})) : {};
    list.append(el("li", "error-text", body.detail || "Search failed."));
    return;
  }
  const data = await res.json();
  if (!data.results.length) {
    list.append(el("li", "muted", "No matches."));
    return;
  }
  data.results.forEach((r) => {
    const li = el("li", "icd-hit");
    li.append(el("span", "icd-badge", r.code || "—"));
    li.append(el("span", null, r.title || ""));
    list.append(li);
  });
}

/* ---------- wire up ---------- */

$("recordBtn").addEventListener("click", toggleRecording);
$("typedForm").addEventListener("submit", submitTyped);
$("icdForm").addEventListener("submit", submitIcd);
loadHealth();
