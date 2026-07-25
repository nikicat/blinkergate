"use strict";

const state = {
  samples: [],
  activeGeneration: null,
  selected: new Set(),
  lastIndex: null,   // anchor for shift-click ranges
  shown: [],
};

const $ = (sel) => document.querySelector(sel);
const grid = $("#grid");

function toast(message, ms = 2600) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, ms);
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

// --- dataset ---------------------------------------------------------------

function conflicts(s) {
  return s.p_closed !== null && s.label !== "skip" &&
         (s.p_closed > 0.5) !== (s.label === "closed");
}

function sortKey(mode) {
  switch (mode) {
    // Nearest the decision boundary: where a corrected label buys the most.
    case "uncertainty":
      return (a, b) => uncertainty(a) - uncertainty(b);
    case "disagreement":
      return (a, b) => Number(conflicts(b)) - Number(conflicts(a)) ||
                       uncertainty(a) - uncertainty(b);
    case "burst":
      return (a, b) => a.burst.localeCompare(b.burst) || a.id.localeCompare(b.id);
    // Groups each eye together: one eye is often harder than the other, and
    // that is invisible while they are interleaved.
    case "eye":
      return (a, b) => a.side.localeCompare(b.side) ||
                       a.label.localeCompare(b.label) ||
                       a.id.localeCompare(b.id);
    case "newest":
      return (a, b) => b.frame.localeCompare(a.frame);
    default:
      return (a, b) => a.label.localeCompare(b.label) || a.id.localeCompare(b.id);
  }
}

function uncertainty(s) {
  return s.p_closed === null ? 1 : Math.abs(s.p_closed - 0.5);
}

function visible() {
  const f = $("#filter").value;
  const eye = $("#eye").value;
  let rows = state.samples;
  if (eye !== "all") rows = rows.filter((s) => s.side === eye);
  if (f === "trainable") rows = rows.filter((s) => s.label !== "skip");
  else if (f === "conflict") rows = rows.filter(conflicts);
  else if (f !== "all") rows = rows.filter((s) => s.label === f);
  return rows.slice().sort(sortKey($("#sort").value));
}

function render() {
  state.shown = visible();
  grid.replaceChildren(...state.shown.map((s, i) => {
    const cell = document.createElement("div");
    cell.className = `cell ${s.label}` + (conflicts(s) ? " conflict" : "") +
                     (state.selected.has(s.id) ? " selected" : "");
    cell.dataset.index = i;

    const img = document.createElement("img");
    img.src = s.thumb;
    img.alt = `${s.side} ${s.label}`;
    img.loading = "lazy";

    const meta = document.createElement("div");
    meta.className = "meta";
    const left = document.createElement("span");
    // "img_left" is the subject's left eye - on the RIGHT of the picture.
    left.textContent = s.side === "img_left" ? "your L" : "your R";
    const right = document.createElement("span");
    right.textContent = s.p_closed === null ? "-" : `p${s.p_closed.toFixed(2)}`;
    meta.append(left, right);

    cell.append(img, meta);
    return cell;
  }));
  updateCounts();
}

function tally(rows) {
  const c = { open: 0, closed: 0, skip: 0 };
  rows.forEach((s) => { c[s.label] = (c[s.label] || 0) + 1; });
  return c;
}

function updateCounts() {
  const c = tally(state.samples);
  const bad = state.samples.filter(conflicts).length;
  // Per eye as well as overall: an imbalance between the two is worth seeing,
  // because a model can be fine on one eye and poor on the other.
  const perEye = ["img_left", "img_right"].map((side) => {
    const t = tally(state.samples.filter((s) => s.side === side));
    return `${side === "img_left" ? "L" : "R"} ${t.open}/${t.closed}`;
  }).join(" · ");
  $("#summary").textContent =
    `${state.samples.length} samples · ${c.open} open · ${c.closed} closed · ` +
    `${c.skip} skip · ${bad} disagree · open/closed per eye ${perEye}` +
    (state.activeGeneration ? ` · active ${state.activeGeneration}` : " · no model");
  $("#selcount").textContent = `${state.selected.size} selected`;
}

grid.addEventListener("click", (event) => {
  const cell = event.target.closest(".cell");
  if (!cell) return;
  const index = Number(cell.dataset.index);
  if (event.shiftKey && state.lastIndex !== null) {
    const [a, b] = [state.lastIndex, index].sort((x, y) => x - y);
    for (let i = a; i <= b; i++) state.selected.add(state.shown[i].id);
  } else {
    const id = state.shown[index].id;
    state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id);
    state.lastIndex = index;
  }
  render();
});

async function applyLabel(label) {
  if (!state.selected.size) return toast("nothing selected");
  const ids = [...state.selected];
  const res = await api("/api/label", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids, label }),
  });
  // Update in place: no reason to refetch every thumbnail for a label change.
  const wanted = new Set(ids);
  state.samples.forEach((s) => { if (wanted.has(s.id)) s.label = label; });
  state.selected.clear();
  state.lastIndex = null;
  render();
  toast(`${res.updated} → ${label}`);
}

async function loadSamples() {
  const data = await api("/api/samples");
  state.samples = data.samples;
  state.activeGeneration = data.active_generation;
  render();
}

// --- models ----------------------------------------------------------------

async function loadModels() {
  const data = await api("/api/models");
  const body = $("#models tbody");
  if (!data.generations.length) {
    body.replaceChildren(Object.assign(document.createElement("tr"), {
      innerHTML: '<td colspan="7" class="muted">No generations yet — label some ' +
                 "samples and press Train.</td>",
    }));
    return;
  }
  const pct = (v) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
  body.replaceChildren(...data.generations.slice().reverse().map((m) => {
    const tr = document.createElement("tr");
    if (m.active) tr.className = "active-gen";
    const counts = m.counts || {};
    const cells = [
      m.active ? "●" : "",
      m.id,
      (m.created_at || "").replace("T", " ").replace("+00:00", ""),
      pct(m.accuracy),
      // Not scored rather than scored wrongly: a generation fitted on an older
      // crop convention cannot be compared on crops cut to this one.
      m.stale_convention ? "old crop convention"
        : m.n_unseen ? `${pct(m.unseen_accuracy)} (${m.n_unseen})` : "—",
      `${m.n_train} in ${m.n_bursts} bursts`,
      `${counts.open || 0}/${counts.closed || 0}/${counts.skip || 0}`,
    ];
    cells.forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      tr.append(td);
    });
    const td = document.createElement("td");
    if (!m.active) {
      const button = document.createElement("button");
      button.textContent = "Use";
      button.onclick = async () => {
        await api(`/api/models/${m.id}/activate`, { method: "POST" });
        toast(`${m.id} active`);
        await Promise.all([loadModels(), loadSamples()]);
      };
      td.append(button);
    }
    tr.append(td);
    return tr;
  }));
}

// --- wiring ----------------------------------------------------------------

document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const models = tab.dataset.view === "models";
    $("#models-view").hidden = !models;
    $("#dataset-view").hidden = models;
    if (models) loadModels();
  };
});

document.querySelectorAll(".label-btn").forEach((b) => {
  b.onclick = () => applyLabel(b.dataset.label);
});
$("#sort").onchange = render;
$("#filter").onchange = () => { state.lastIndex = null; render(); };
$("#eye").onchange = () => { state.lastIndex = null; render(); };
$("#sel-all").onclick = () => {
  state.shown.forEach((s) => state.selected.add(s.id));
  render();
};
$("#sel-none").onclick = () => { state.selected.clear(); state.lastIndex = null; render(); };

$("#train").onclick = async (event) => {
  const button = event.target;
  button.disabled = true;
  button.textContent = "Training…";
  try {
    const res = await api("/api/models", { method: "POST" });
    const acc = res.accuracy === null ? "n/a" : `${(res.accuracy * 100).toFixed(1)}%`;
    toast(`${res.generation}: ${acc} held-out on ${res.n_train} samples`, 5000);
    await loadSamples();
  } catch (err) {
    toast(err.message, 5000);
  } finally {
    button.disabled = false;
    button.textContent = "Train new generation";
  }
};

document.addEventListener("keydown", (event) => {
  if (event.target.matches("select, input")) return;
  const keys = { o: "open", c: "closed", s: "skip" };
  if (keys[event.key]) { event.preventDefault(); applyLabel(keys[event.key]); }
  else if (event.key === "a" && !event.ctrlKey) $("#sel-all").click();
  else if (event.key === "Escape") $("#sel-none").click();
});

loadSamples().catch((err) => toast(err.message, 6000));
