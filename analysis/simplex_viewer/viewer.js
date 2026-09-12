/* global Plotly */
"use strict";

const colors = {target: "#168379"};
const checkpointColors = ["#bd7336", "#7257ba", "#356a9c", "#943947"];
const checkpointSymbols = ["diamond", "circle", "square", "cross"];
const vertexColors = [[229, 83, 94], [38, 158, 115], [60, 111, 214]];
let ids = [];
const el = id => document.getElementById(id);
const initialCamera = {eye: {x: 1.55, y: 1.55, z: 1.25}};
let camera = initialCamera;
let syncingCamera = false;
let timer = null;
let rendering = false;
let pendingRender = false;
let loadedRun = null;
let loadedSequence = null;
let linked = false;

function block(row, component) {
  return component.indices.map(index => row[index]);
}

const escapeHtml = text => text.replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[char]);

function mass(row) {
  return row.reduce((a, b) => a + b, 0);
}

function beliefColor(point) {
  const positive = point.map(value => Math.max(0, value));
  const total = mass(positive);
  const depth = Math.sqrt(Math.max(0, Math.min(1, mass(point))));
  const rgb = [0, 1, 2].map(channel => {
    const hue = total > 0
      ? positive.reduce((sum, value, state) => sum + value / total * vertexColors[state][channel], 0)
      : 40;
    return Math.round(40 + depth * (hue - 40));
  });
  return `rgb(${rgb.join(",")})`;
}

function axes(points) {
  return {x: points.map(p => p[0]), y: points.map(p => p[1]), z: points.map(p => p[2])};
}

function plane(weight, color = null, opacity = 0.24) {
  return {
    type: "mesh3d", x: [weight, 0, 0], y: [0, weight, 0], z: [0, 0, weight],
    i: [0], j: [1], k: [2],
    ...(color ? {color} : {vertexcolor: [
      beliefColor([weight, 0, 0]), beliefColor([0, weight, 0]), beliefColor([0, 0, weight]),
    ]}),
    lighting: {ambient: 1, diffuse: 0, specular: 0, fresnel: 0},
    opacity, hoverinfo: "skip", showscale: false,
  };
}

function outline(weight, color, width = 2) {
  return {
    type: "scatter3d", mode: "lines",
    x: [weight, 0, 0, weight], y: [0, weight, 0, 0], z: [0, 0, weight, 0],
    line: {color, width}, hoverinfo: "skip", showlegend: false,
  };
}

function envelope() {
  return [
    plane(1, "#879b8e", 0.035), outline(1, "#b6c5ba", 1),
    {
      type: "scatter3d", mode: "lines", x: [1, 0, null, 0, 0, null, 0, 0],
      y: [0, 0, null, 1, 0, null, 0, 0], z: [0, 0, null, 0, 0, null, 1, 0],
      line: {color: "#ccd7ce", width: 1}, hoverinfo: "skip", showlegend: false,
    },
  ];
}

function pointTrace(points, name, color, {sequence = false, current = false, labels = [], symbol = "circle", stateLabels = ["local 0", "local 1", "local 2"]} = {}) {
  return {
    type: "scatter3d", mode: sequence ? "lines+markers" : "markers", ...axes(points),
    name, customdata: points.map((p, i) => [mass(p), labels[i] ?? i]),
    marker: {
      color: points.map(beliefColor), symbol,
      size: current ? 7 : sequence ? 2.5 : 2, opacity: current ? 1 : 0.75,
      line: {color, width: current ? 2 : 0},
    },
    line: {color, width: 2}, showlegend: false,
    hovertemplate: `${escapeHtml(name)}<br>row / t = %{customdata[1]}<br>${escapeHtml(stateLabels[0])}=%{x:.4f}<br>${escapeHtml(stateLabels[1])}=%{y:.4f}<br>${escapeHtml(stateLabels[2])}=%{z:.4f}<br>mass=%{customdata[0]:.4f}<extra></extra>`,
  };
}

function rangeFor(run, layer) {
  let low = 0, high = 1;
  for (const checkpoint of run.checkpoints) {
    const predictions = run.predictions[checkpoint][layer];
    for (const row of [...predictions.cloud, ...predictions.sequences.flat()]) {
      for (const value of row) {
        low = Math.min(low, value); high = Math.max(high, value);
      }
      for (const component of run.components) {
        const weight = mass(block(row, component));
        low = Math.min(low, weight); high = Math.max(high, weight);
      }
    }
  }
  return [low - 0.06, high + 0.06];
}

function layout(range, component) {
  const labels = component.state_labels || component.indices.map(index => `State ${index}`);
  const axis = title => ({
    title: {text: title, font: {size: 11}}, range, autorange: false,
    tickfont: {size: 9, color: "#718278"}, nticks: 5, showbackground: false,
    gridcolor: "#e7ede8", zerolinecolor: "#c6d2c9",
  });
  return {
    margin: {l: 0, r: 0, t: 0, b: 0}, paper_bgcolor: "white",
    showlegend: false, uirevision: "linked-camera",
    scene: {
      xaxis: axis(`${escapeHtml(labels[0])} mass`),
      yaxis: axis(`${escapeHtml(labels[1])} mass`),
      zaxis: axis(`${escapeHtml(labels[2])} mass`),
      camera, aspectmode: "cube", dragmode: "orbit",
    },
  };
}

const number = value => value == null ? "undefined" : value.toFixed(3);
const percent = value => `${(100 * value).toFixed(1)}%`;

function node(tag, text = "", id = "") {
  const result = document.createElement(tag);
  result.textContent = text;
  if (id) result.id = id;
  return result;
}

function options(id, values, selected) {
  el(id).replaceChildren(...values.map(([value, label]) => {
    const option = node("option", label);
    option.value = String(value);
    return option;
  }));
  el(id).value = String(selected);
}

function fillRun(run) {
  for (const id of ids) Plotly.purge(id);
  ids = [];
  linked = false;
  loadedSequence = null;
  options("layer", run.sites.map(site => [site, site === run.primary_site ? `${site} · primary` : site]), run.primary_site);
  options("checkpoint", [[-1, "All checkpoints"], ...run.checkpoints.map((name, i) => [i, name])], -1);
  options("sequence", run.sequences.map((sequence, i) => [i, sequence.label]), 0);
  const panels = [];
  for (const [kind, title] of [["target", "Bayesian"], ["probe", "Activation probe"]]) {
    run.components.forEach((component, i) => {
      const id = `${kind}-${i}`;
      ids.push(id);
      const article = node("article");
      const heading = node("h2", title);
      heading.append(node("span", component.name));
      const plot = node("div", "", id);
      plot.className = "plot";
      article.append(heading, plot, node("p", "", `${id}-caption`));
      panels.push(article);
    });
  }
  el("plots").replaceChildren(...panels);
  el("plots").style.gridTemplateColumns = `repeat(${Math.min(3, run.components.length)}, minmax(0, 1fr))`;
  el("metrics").replaceChildren(...run.checkpoints.map((name, i) => {
    const card = node("div");
    card.className = "metric";
    card.style.borderColor = checkpointColors[i % checkpointColors.length];
    card.append(node("span", `${name} · full-belief R²`), node("strong", "", `score-${i}`),
      node("small", "", `error-${i}`), node("small", "", `mass-${i}`));
    return card;
  }));
  const legend = node("span", "Bayesian outline");
  legend.style.color = colors.target;
  el("legend").replaceChildren(legend, ...run.checkpoints.map((name, i) => {
    const item = node("span", `${name} · ${checkpointSymbols[i % checkpointSymbols.length]}`);
    item.style.color = checkpointColors[i % checkpointColors.length];
    return item;
  }), node("span", "Drag to rotate · scroll to zoom · cameras linked"));
}

function fillTokens(sequence) {
  el("time").value = 0;
  el("time").max = sequence.targets.length - 1;
  el("tokens").replaceChildren(...sequence.tokens.map((token, t) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = token;
    button.title = `t=${t}: ${token}`;
    button.setAttribute("aria-label", button.title);
    button.onclick = () => { el("time").value = t; stop(); requestRender(); };
    return button;
  }));
}

function stop() {
  clearInterval(timer); timer = null;
  el("play").textContent = "Play";
  el("play").setAttribute("aria-label", "Play sequence");
}

async function render() {
  const run = window.SIMPLEX_DATA.runs[Number(el("run").value)];
  if (loadedRun !== run) { fillRun(run); loadedRun = run; }
  const layer = el("layer").value;
  const sequenceMode = el("mode").value === "sequence";
  const sequenceIndex = Number(el("sequence").value);
  const sequence = run.sequences[sequenceIndex];
  if (loadedSequence !== sequence) { fillTokens(sequence); loadedSequence = sequence; }
  const time = Number(el("time").value);
  const choice = Number(el("checkpoint").value);
  const checkpoints = choice === -1 ? run.checkpoints : [run.checkpoints[choice]];
  const range = rangeFor(run, layer);
  el("sequence-controls").hidden = !sequenceMode;
  el("run-meta").textContent = run.description;
  el("position").textContent = `${time} / ${sequence.targets.length - 1}`;
  el("time").setAttribute("aria-valuetext", `Position ${time} of ${sequence.targets.length - 1}`);
  if (sequenceMode) {
    el("step-meta").textContent = `t=${time} · ${sequence.tokens[time]}` +
      (sequence.notes[time] ? ` · ${sequence.notes[time]}` : "");
    [...el("tokens").children].forEach((button, t) => {
      button.className = t === time ? "active" : t > time ? "future" : "";
      button.setAttribute("aria-pressed", String(t === time));
    });
  }
  run.checkpoints.forEach((checkpoint, i) => {
    const metrics = run.metrics[checkpoint][layer];
    el(`score-${i}`).textContent = number(metrics.r_squared);
    el(`error-${i}`).textContent = `MSE ${metrics.mse.toExponential(2)} · ${percent(metrics.outside_simplex_fraction)} outside simplex`;
    el(`mass-${i}`).textContent = `Component posterior R² ${number(metrics.component_posterior.r_squared)}`;
  });
  el("plot-note").textContent = sequenceMode
    ? "Planes use current component mass. Trails include only positions up to t. Raw probe coordinates and overshoots remain visible."
    : `${run.cloud.targets.length.toLocaleString()} held-out points per component. Faint outlines mark the unit simplex and its sweep to the origin. Checkpoints share histories and axes.`;
  const promises = [];
  for (const [suffix, component] of run.components.entries()) {
    const stateLabels = component.state_labels;
    const targetPoints = (sequenceMode ? sequence.targets : run.cloud.targets).map(row => block(row, component));
    const targetTraces = envelope();
    const probeTraces = envelope();
    if (sequenceMode) {
      const point = targetPoints[time], weight = mass(point);
      targetTraces.push(plane(weight), outline(weight, colors.target));
      targetTraces.push(pointTrace(targetPoints.slice(0, time + 1), "Bayesian", colors.target, {sequence: true, stateLabels}));
      targetTraces.push(pointTrace([point], "Bayesian · current", colors.target, {current: true, labels: [time], stateLabels}));
      el(`target-${suffix}-caption`).textContent = `Posterior weight = ${weight.toFixed(4)}`;
    } else {
      targetTraces.push(pointTrace(targetPoints, "Bayesian", colors.target, {labels: run.cloud.rows, stateLabels}));
      el(`target-${suffix}-caption`).textContent = "Coordinates = posterior weight × local state belief";
    }
    const captions = [];
    for (const checkpoint of checkpoints) {
      const prediction = run.predictions[checkpoint][layer];
      const points = (sequenceMode ? prediction.sequences[sequenceIndex] : prediction.cloud).map(row => block(row, component));
      const name = checkpoint;
      const checkpointIndex = run.checkpoints.indexOf(checkpoint);
      const symbol = checkpointSymbols[checkpointIndex % checkpointSymbols.length];
      const color = checkpointColors[checkpointIndex % checkpointColors.length];
      if (sequenceMode) {
        const point = points[time], weight = mass(point);
        probeTraces.push(plane(weight, null, 0.16), outline(weight, color));
        probeTraces.push(pointTrace(points.slice(0, time + 1), name, color, {sequence: true, symbol, stateLabels}));
        probeTraces.push(pointTrace([point], `${name} · current`, color, {current: true, labels: [time], symbol, stateLabels}));
        captions.push(`${name} mass = ${weight.toFixed(4)}`);
      } else {
        probeTraces.push(pointTrace(points, name, color, {labels: run.cloud.rows, symbol, stateLabels}));
      }
    }
    el(`probe-${suffix}-caption`).textContent = captions.join(" · ") || "Raw affine predictions · no projection onto the simplex";
    for (const [id, traces] of [[`target-${suffix}`, targetTraces], [`probe-${suffix}`, probeTraces]]) {
      promises.push(Plotly.react(id, traces, layout(range, component), {
        responsive: true, displaylogo: false, scrollZoom: true,
        modeBarButtonsToRemove: ["toImage"],
      }));
    }
  }
  await Promise.all(promises);
  if (!linked) {
    for (const id of ids) {
      el(id).on("plotly_relayout", async event => {
        if (!event["scene.camera"] || syncingCamera) return;
        syncingCamera = true;
        camera = event["scene.camera"];
        try {
          await Promise.all(ids.filter(other => other !== id).map(other => Plotly.relayout(other, {"scene.camera": camera})));
        } finally { syncingCamera = false; }
      });
    }
    linked = true;
  }
}

async function requestRender() {
  if (rendering) { pendingRender = true; return; }
  rendering = true;
  try {
    do { pendingRender = false; await render(); } while (pendingRender);
    el("error").hidden = true;
    el("error").textContent = "";
  } catch (error) {
    stop();
    el("error").hidden = false;
    el("error").textContent = `Could not render: ${error.message}`;
  } finally {
    rendering = false;
  }
}

async function start() {
  const data = window.SIMPLEX_DATA;
  if (!data || data.schema !== 2 || !data.runs.length) throw new Error("missing or unsupported viewer data");
  document.title = data.title;
  el("title").textContent = data.title;
  el("study-method").textContent = data.description;
  el("report-links").replaceChildren(...data.reports.map(report => {
    const link = node("a", report.name);
    link.href = report.path;
    link.download = "";
    return link;
  }));
  options("run", data.runs.map((run, i) => [i, run.name]), 0);
  await requestRender();
  for (const id of ["run", "mode", "layer", "checkpoint", "sequence"]) {
    el(id).addEventListener("change", () => {
      stop();
      requestRender();
    });
  }
  el("time").addEventListener("input", () => { stop(); requestRender(); });
  el("play").onclick = () => {
    if (timer) { stop(); return; }
    const last = Number(el("time").max);
    if (Number(el("time").value) === last) el("time").value = 0;
    el("play").textContent = "Pause";
    el("play").setAttribute("aria-label", "Pause sequence");
    timer = setInterval(() => {
      if (rendering) return;
      const next = Number(el("time").value) + 1;
      el("time").value = Math.min(last, next);
      requestRender();
      if (next >= last) stop();
    }, 350);
  };
  el("reset-camera").onclick = async () => {
    camera = initialCamera;
    syncingCamera = true;
    try { await Promise.all(ids.map(id => Plotly.relayout(id, {"scene.camera": camera}))); }
    finally { syncingCamera = false; }
  };
}

start().catch(error => {
  el("error").hidden = false;
  el("error").textContent = `Could not start: ${error.message}. Keep index.html, data.js and plotly.min.js together.`;
});
