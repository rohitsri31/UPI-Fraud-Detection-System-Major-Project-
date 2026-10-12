const $ = (id) => document.getElementById(id);
const RUPEE = "₹";

const PRESETS = {
  normal: {
    amount: 450, typical_amount: 500, hour: 14, txn_type: "merchant",
    is_new_payee: false, is_new_device: false, is_new_city: false, txns_last_hour: 0,
  },
  suspicious: {
    amount: 24000, typical_amount: 600, hour: 3, txn_type: "p2p",
    is_new_payee: true, is_new_device: true, is_new_city: true, txns_last_hour: 3,
  },
};

const RISK_STYLES = {
  high: { alert: "alert-danger", bar: "bg-danger", title: "Flagged for review" },
  medium: { alert: "alert-warning", bar: "bg-warning", title: "Needs a closer look" },
  low: { alert: "alert-success", bar: "bg-success", title: "Looks normal" },
};

let filter = "all";

async function request(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function fillForm(values) {
  for (const [key, value] of Object.entries(values)) {
    const el = $(key);
    if (!el) continue;
    if (el.type === "checkbox") el.checked = Boolean(value);
    else el.value = value;
  }
}

function readForm() {
  return {
    amount: $("amount").value,
    typical_amount: $("typical_amount").value,
    hour: $("hour").value,
    txn_type: $("txn_type").value,
    txns_last_hour: $("txns_last_hour").value,
    is_new_payee: $("is_new_payee").checked,
    is_new_device: $("is_new_device").checked,
    is_new_city: $("is_new_city").checked,
  };
}

function showError(message) {
  const box = $("form-error");
  box.textContent = message;
  box.hidden = false;
}

function showResult(result) {
  const style = RISK_STYLES[result.risk_level];
  $("result").hidden = false;
  $("result-banner").className = `alert ${style.alert} mb-3`;
  $("result-title").textContent = style.title;
  $("result-prob").textContent =
    `Fraud probability ${(result.fraud_probability * 100).toFixed(1)}% ` +
    `(flag threshold ${(result.threshold * 100).toFixed(1)}%)`;
  const bar = $("result-bar");
  bar.className = `progress-bar ${style.bar}`;
  bar.style.width = `${Math.max(result.fraud_probability * 100, 2)}%`;

  const list = $("result-indicators");
  list.replaceChildren();
  const reasons = result.indicators.length ? result.indicators : ["Nothing unusual in the inputs"];
  for (const reason of reasons) {
    const li = document.createElement("li");
    li.textContent = reason;
    list.appendChild(li);
  }
}

function renderTable(rows) {
  const body = $("txn-body");
  body.replaceChildren();
  $("txn-empty").hidden = rows.length > 0;
  for (const row of rows) {
    const tr = document.createElement("tr");
    const cells = [
      new Date(row.created_at).toLocaleTimeString(),
      RUPEE + Number(row.amount).toLocaleString("en-IN"),
      row.txn_type.replace("_", " "),
    ];
    for (const text of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }

    const riskCell = document.createElement("td");
    const badge = document.createElement("span");
    const badgeClass = { high: "text-bg-danger", medium: "text-bg-warning", low: "text-bg-success" };
    badge.className = `badge ${badgeClass[row.risk_level]}`;
    badge.textContent = `${(row.probability * 100).toFixed(1)}%`;
    riskCell.appendChild(badge);
    tr.appendChild(riskCell);

    const reasonCell = document.createElement("td");
    reasonCell.className = "small text-secondary";
    reasonCell.textContent = row.indicators.join(", ") || "-";
    tr.appendChild(reasonCell);

    body.appendChild(tr);
  }
}

async function refresh() {
  const flaggedOnly = filter === "flagged" ? "&flagged=1" : "";
  const [stats, rows] = await Promise.all([
    request("/api/stats"),
    request(`/api/transactions?limit=25${flaggedOnly}`),
  ]);
  $("stat-total").textContent = stats.total.toLocaleString();
  $("stat-flagged").textContent = stats.flagged.toLocaleString();
  $("stat-rate").textContent = `${(stats.flag_rate * 100).toFixed(1)}%`;
  renderTable(rows);
}

async function loadModelInfo() {
  const badge = $("model-badge");
  try {
    const info = await request("/api/health");
    if (info.model_loaded) {
      badge.className = "badge text-bg-success";
      badge.textContent = `${info.model_name} + ${info.strategy}`;
      $("stat-threshold").textContent = `${(info.threshold * 100).toFixed(1)}%`;
    } else {
      badge.className = "badge text-bg-danger";
      badge.textContent = "No model: run python -m src.train";
    }
  } catch (err) {
    badge.className = "badge text-bg-danger";
    badge.textContent = "Server not reachable";
  }
}

$("score-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("form-error").hidden = true;
  try {
    const result = await request("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    showResult(result);
    await refresh();
  } catch (err) {
    $("result").hidden = true;
    showError(err.message);
  }
});

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => {
    fillForm(PRESETS[button.dataset.preset]);
    $("form-error").hidden = true;
  });
});

document.querySelectorAll("[data-filter]").forEach((button) => {
  button.addEventListener("click", async () => {
    filter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((b) => b.classList.toggle("active", b === button));
    await refresh();
  });
});

fillForm({ hour: new Date().getHours(), txns_last_hour: 0, txn_type: "p2p" });
loadModelInfo();
refresh().catch(() => {});
