const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {setImmediate} = require("node:timers/promises");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname,
  "../analysis/simplex_viewer/viewer.js"), "utf8");

async function viewer() {
  const elements = new Map();
  const element = () => ({
    value: "", children: [], textContent: "", hidden: false, handlers: {}, style: {},
    set id(value) { elements.set(value, this); },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    setAttribute() {},
    addEventListener(name, callback) { this.handlers[name] = callback; },
    on(name, callback) { this.handlers[name] = callback; },
  });
  const get = id => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  for (const [id, value] of Object.entries({
    run: "0", mode: "sequence", sequence: "0", time: "0",
  })) get(id).value = value;
  const targets = Array.from({length: 128}, (_, t) => {
    const weight = 0.5 * 0.6 ** t;
    return [(1 - weight) / 3, (1 - weight) / 3, (1 - weight) / 3, weight / 2, weight / 3, weight / 6];
  });
  const prediction = {cloud: targets, sequences: [targets]};
  const metrics = {r_squared: 0.9, mse: 0.01, outside_simplex_fraction: 0.1, component_posterior: {r_squared: 0.8}};
  const run = {
    name: "First fixture", description: "Caller-defined timing",
    components: [{name: "Left", indices: [0, 1, 2]}, {name: "Right", indices: [3, 4, 5]}],
    sites: ["post_final_norm"], primary_site: "post_final_norm", checkpoints: ["Initialization", "Final"],
    sequences: [{id: 2, label: "Example 2", targets, tokens: targets.map((_, t) => t ? "a" : "BOS"), notes: targets.map(() => "")}],
    cloud: {targets, rows: targets.map((_, i) => i)},
    predictions: {Initialization: {post_final_norm: prediction}, Final: {post_final_norm: prediction}},
    metrics: {Initialization: {post_final_norm: metrics}, Final: {post_final_norm: metrics}},
  };
  const shortTargets = targets.slice(0, 3).map(row => [...row, 0, 0, 0]);
  const passive = {
    name: "Passive fixture", description: "Filtered source belief · passive observations",
    components: [
      {name: "Hot", indices: [2, 0, 1]}, {name: "Cold", indices: [3, 4, 5]}, {name: "Third", indices: [6, 7, 8]},
    ],
    sites: ["embedding"], primary_site: "embedding", checkpoints: ["Epoch 5"],
    sequences: [
      {id: 0, label: "Short history", targets: shortTargets, tokens: ["reset", "red", "blue"], notes: ["", "", "end"]},
      {id: 1, label: "Other history", targets: shortTargets, tokens: ["reset", "green", "gold"], notes: ["", "", ""]},
    ],
    cloud: {targets: shortTargets, rows: [0, 1, 2]},
    predictions: {"Epoch 5": {embedding: {cloud: shortTargets, sequences: [shortTargets, shortTargets]}}},
    metrics: {"Epoch 5": {embedding: {...metrics, r_squared: null}}},
  };
  const plots = new Map();
  const relayouts = [];
  let tick;
  const context = vm.createContext({
    window: {SIMPLEX_DATA: {schema: 2, title: "Fixtures", description: "", reports: [], runs: [run, passive]}},
    document: {getElementById: get, createElement: element},
    Plotly: {
      async react(id, traces, layout) { plots.set(id, {traces, layout}); },
      async relayout(id, update) { relayouts.push({id, update}); },
      purge(id) { plots.delete(id); },
    },
    clearInterval() {},
    setInterval(callback) { tick = callback; return 1; },
  });
  vm.runInContext(source, context);
  await setImmediate();
  assert.equal(get("error").textContent, "");
  return {context, get, plots, targets, relayouts, tick: () => tick()};
}

test("hue distinguishes vertices and blends local states; weight controls depth", async () => {
  const {context} = await viewer();
  const color = point => context.beliefColor(point);
  assert.equal(color([1, 0, 0]), "rgb(229,83,94)");
  assert.equal(color([0, 1, 0]), "rgb(38,158,115)");
  assert.equal(color([0, 0, 1]), "rgb(60,111,214)");
  assert.equal(color([0.5, 0.5, 0]), "rgb(134,121,105)");
  assert.equal(color([0, 0, 0]), "rgb(40,40,40)");
  assert.notEqual(color([1, 0, 0]), color([0.25, 0, 0]));
  assert.notEqual(color([0, 0, 0]), color([0.25, 0, 0]));
});

test("bounded colors preserve signed, out-of-simplex geometry", async () => {
  const {context} = await viewer();
  const point = [1.4, -0.2, -0.1];
  const before = [...point];
  const trace = context.pointTrace([point], "raw", "#7257ba");
  assert.deepEqual(point, before);
  assert.equal(trace.x[0], 1.4);
  assert.equal(trace.y[0], -0.2);
  assert.equal(trace.marker.color[0], "rgb(229,83,94)");
  assert.equal(context.beliefColor([-1, 0, 0]), "rgb(40,40,40)");
  const plane = context.plane(-0.5);
  assert.deepEqual([...plane.x], [-0.5, 0, 0]);
  assert.equal(plane.vertexcolor.length, 3);
});

test("tiny masses remain nonzero in geometry while labels use fixed decimals", async () => {
  const {get, plots, targets} = await viewer();
  let previous;
  for (const t of [25, 30, 100, 127]) {
    get("time").value = String(t);
    get("time").handlers.input();
    await setImmediate();
    assert.equal(get("error").textContent, "");
    const {traces, layout} = plots.get("target-1");
    const current = traces.at(-1);
    const expected = targets[t].slice(3);
    assert.deepEqual([current.x[0], current.y[0], current.z[0]], expected);
    assert.ok(current.x[0] > 0);
    assert.notEqual(current.x[0], previous);
    previous = current.x[0];
    assert.equal(traces[3].x[0], expected.reduce((a, b) => a + b, 0));
    assert.equal(get("target-1-caption").textContent, "Posterior weight = 0.0000");
    assert.match(current.hovertemplate, /customdata\[0\]:\.4f/);
    assert.ok(layout.scene.xaxis.range[1] >= 1);
  }
});

test("checkpoint markers and plane colors agree across modes", async () => {
  const {get, plots, context} = await viewer();
  get("checkpoint").value = "-1";
  for (const mode of ["sequence", "cloud"]) {
    get("mode").value = mode;
    await vm.runInContext("requestRender()", context);
    const traces = plots.get("probe-0").traces;
    const init = traces.find(t => t.name === "Initialization");
    const final = traces.find(t => t.name === "Final");
    assert.equal(init.marker.symbol, "diamond");
    assert.equal(final.marker.symbol, "circle");
    assert.deepEqual([...init.marker.color], [...final.marker.color]);
    if (mode === "sequence") {
      const targetPlane = plots.get("target-0").traces[3];
      assert.equal(new Set(targetPlane.vertexcolor).size, 3);
      assert.deepEqual([...targetPlane.vertexcolor], [...traces[3].vertexcolor]);
    }
  }
});

test("run changes rebuild components, sites, checkpoint choices and passive timing", async () => {
  const {get, plots} = await viewer();
  get("time").value = "127";
  get("run").value = "1";
  get("run").handlers.change();
  await setImmediate();
  assert.equal(get("error").textContent, "");
  assert.equal(plots.size, 6);
  assert.equal(get("layer").value, "embedding");
  assert.deepEqual(get("checkpoint").children.map(option => option.textContent), ["All checkpoints", "Epoch 5"]);
  assert.equal(get("time").max, 2);
  assert.equal(get("time").value, 0);
  assert.equal(get("score-0").textContent, "undefined");
  assert.equal(plots.get("target-0").layout.scene.xaxis.title.text, "State 2 mass");
  get("tokens").children[2].onclick();
  await setImmediate();
  assert.equal(get("position").textContent, "2 / 2");
  assert.equal(get("step-meta").textContent, "t=2 · blue · end");
  assert.doesNotMatch(get("step-meta").textContent, /action|terminal observation/);
  get("sequence").value = "1";
  get("sequence").handlers.change();
  await setImmediate();
  assert.equal(get("time").value, 0);
  assert.equal(get("tokens").children[2].textContent, "gold");
  get("run").value = "0";
  get("run").handlers.change();
  await setImmediate();
  assert.equal(plots.size, 4);
  assert.equal(get("time").max, 127);
});

test("camera links and playback survive switching to shorter histories", async () => {
  const {get, relayouts, tick} = await viewer();
  get("run").value = "1";
  get("run").handlers.change();
  await setImmediate();
  const camera = {eye: {x: 1, y: 2, z: 3}};
  await get("target-2").handlers.plotly_relayout({"scene.camera": camera});
  assert.equal(relayouts.length, 5);
  assert.ok(relayouts.every(item => item.update["scene.camera"] === camera));
  await get("reset-camera").onclick();
  assert.equal(relayouts.length, 11);
  get("play").onclick();
  tick();
  await setImmediate();
  tick();
  await setImmediate();
  assert.equal(get("position").textContent, "2 / 2");
  assert.equal(get("play").textContent, "Play");
});
