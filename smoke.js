// Load index.html in jsdom, point fetch at the live server, and drive the UI.
const fs = require("fs");
const path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");

// Resolved relative to this file so the test works from a clone, whatever the
// repository is called or where it lives.
const HTML_PATH = process.env.SPIKENET_HTML ||
  path.resolve(__dirname, "index.html");
const HTML = fs.readFileSync(HTML_PATH, "utf8");

const BASE = process.env.SPIKENET_URL || "http://127.0.0.1:8000";

const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", (e) => errors.push("jsdomError: " + (e.stack || e.message)));
vc.on("error", (...a) => errors.push("console.error: " + a.join(" ")));

const dom = new JSDOM(HTML, {
  runScripts: "dangerously",
  pretendToBeVisual: true,
  url: BASE + "/",
  virtualConsole: vc,
  beforeParse(window) {
    // boot() calls fetch during parse and jsdom does not ship one.
    window.fetch = (input, init) =>
      fetch(new URL(typeof input === "string" ? input : input.url, BASE), init);
    // jsdom has no layout, so every rect is 0x0 and plots would draw into a
    // zero-size viewBox. Give the drawing code plausible numbers.
    window.Element.prototype.getBoundingClientRect = function () {
      return { width: 380, height: 210, top: 0, left: 0,
               right: 380, bottom: 210, x: 0, y: 0 };
    };
    // Bare jsdom has no canvas 2d context; stub what the neuron grid touches.
    window.HTMLCanvasElement.prototype.getContext = function () {
      return { setTransform(){}, clearRect(){}, fillRect(){}, fillText(){},
               fillStyle:"", font:"" };
    };
  },
});
const win = dom.window;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const q = (s) => win.document.querySelector(s);
const qa = (s) => Array.from(win.document.querySelectorAll(s));
let pass = 0, fail = 0;
const check = (name, cond, extra = "") => {
  cond ? pass++ : fail++;
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};

(async () => {
  await sleep(4000);

  console.log("\n[boot]");
  check("population + input nodes rendered",
    qa("#nodeLayer .node").length === 4, `got ${qa("#nodeLayer .node").length}`);
  check("synapse edges rendered",
    qa("#edgeLayer .edge").length === 4, `got ${qa("#edgeLayer .edge").length}`);
  check("four panels created", qa(".pnl").length === 4);
  check("metric catalog loaded from server",
    win.eval("metricCatalog.length") >= 10, `${win.eval("metricCatalog.length")} metrics`);
  check("graph validates", (q("#status").textContent || "").includes("valid"));
  check("transport shows the time mapping",
    /1 s real = 100 ms model/.test(q("#tmap").textContent), JSON.stringify(q("#tmap").textContent));

  console.log("\n[#2 role derived from edges, not declared variables]");
  const excRole = win.eval("popRole(popById('exc')).role");
  const inhRole = win.eval("popRole(popById('inh')).role");
  check("exc reads as excitatory", excRole === "excitatory", excRole);
  check("inh reads as inhibitory", inhRole === "inhibitory", inhRole);
  win.eval("select('population','inh')");
  await sleep(150);
  check("inspector explains the s_e / s_i convention",
    q("#inspector").textContent.includes("inhibitory cells still receive excitation"));

  console.log("\n[#3 connect an input to any population]");
  win.eval("select(null); setConnectMode(true)");
  await sleep(80);
  const before = win.eval("findItem('stimulus','drive_e').target");
  win.eval("connectFrom = {kind:'stimulus', id:'drive_e'}");
  const inhNode = qa('#nodeLayer .node[data-kind="population"]').find(n => n.dataset.id === "inh");
  inhNode.dispatchEvent(new win.PointerEvent("pointerdown", {bubbles:true, clientX:400, clientY:150}));
  win.dispatchEvent(new win.PointerEvent("pointerup", {bubbles:true}));
  await sleep(150);
  const after = win.eval("findItem('stimulus','drive_e').target");
  check("input retargeted by clicking a population",
    before === "exc" && after === "inh", `${before} -> ${after}`);
  win.eval("retargetStimulus('drive_e','exc')");
  await sleep(80);

  console.log("\n[#3b third population appears in the target list]");
  win.eval("addPopulation()");
  await sleep(150);
  win.eval("select('stimulus','drive_e')");
  await sleep(200);
  const opts = qa('#inspector select[data-act="stim:target"] option').length;
  check("target dropdown lists all three populations", opts === 3, `${opts} options`);
  win.eval("select('population', graph.populations[2].id); deleteSelected()");
  await sleep(150);

  console.log("\n[#4 slider range]");
  win.eval("select(null)");
  await sleep(200);
  const gSlider = qa('#inspector input[type=range]').find(s => s.dataset.key === "g");
  check("g slider spans well past its current value",
    gSlider && parseFloat(gSlider.max) >= 40, gSlider ? `max=${gSlider.max}` : "not found");
  const gNum = qa('#inspector input[type=number]').find(s => s.dataset.key === "g");
  gNum.value = "250";
  gNum.dispatchEvent(new win.Event("input", {bubbles:true}));
  await sleep(150);
  check("typing beyond the range widens the slider",
    parseFloat(gSlider.max) > 250, `max=${gSlider.max}`);
  check("graph took the typed value",
    win.eval("graph.globals.g") === "250", win.eval("graph.globals.g"));
  win.eval("graph.globals.g='4.5'");

  console.log("\n[#5 axis formatting with tiny numbers]");
  const suffix = win.eval("scaleFor(0,0.0012).suffix");
  check("small ranges get a x10^n factor instead of 0.00",
    suffix.includes("10"), JSON.stringify(suffix));
  const tick = win.eval("fmtTick(0.0008/scaleFor(0,0.0012).k, 0.0012/scaleFor(0,0.0012).k)");
  check("tick text is non-zero after rescaling", tick !== "0.00" && tick !== "0", tick);

  console.log("\n[run]");
  win.eval("graph.run.duration='0.6*second'; graph.run.transient='0.2*second'; doRun()");
  await sleep(32000);
  check("simulation returned", win.eval("!!lastResult"));
  check("animation data precomputed", win.eval("!!anim && anim.nb > 0"),
    `${win.eval("anim ? anim.nb : 0")} bins`);
  check("raster drawn",
    !!q('[data-panel="pa"] svg') && q('[data-panel="pa"] svg').innerHTML.includes("<rect"));
  check("playhead present on raster", q('[data-panel="pa"] .playhead') !== null);
  check("neuron grid canvas created", q('[data-panel="pb"] canvas') !== null);
  check("spectrum drawn", q('[data-panel="pc"] svg').innerHTML.includes("<path"));
  check("spectrum labelled normalised",
    q('[data-panel="pc"] svg').textContent.includes("normalised"));
  check("metrics table populated",
    q('[data-panel="pd"]').textContent.includes("resonant frequency"));

  console.log("\n[#1 playback]");
  const t0 = win.eval("play.t");
  win.eval("startPlay()");
  await sleep(1300);
  const t1 = win.eval("play.t");
  check("clock advances while playing", t1 > t0, `${t0.toFixed(3)} -> ${t1.toFixed(3)} s`);
  check("advance matches 0.1x speed", t1 > 0.02 && t1 < 0.40,
    `${t1.toFixed(3)} s model in ~1.3 s real`);
  win.eval("stopPlay()");
  check("pause stops the clock", win.eval("play.on") === false);
  win.eval("seek(0.3)");
  await sleep(150);
  check("scrubbing moves the playhead", Math.abs(win.eval("play.t") - 0.3) < 1e-6);
  check("population halo reflects live rate", win.eval("liveRate('exc')") > 0,
    `liveRate=${win.eval("liveRate('exc')").toFixed(3)}`);
  const loopT = win.eval("(function(){play.t=anim.T-0.001;play.last=performance.now()-500;play.on=true;tickPlay(performance.now());var t=play.t;stopPlay();return t})()");
  check("playback loops back to zero", loopT < 0.05, `wrapped to ${loopT.toFixed(3)} s`);

  console.log("\n[#6 swappable panels]");
  const typeSel = q('[data-panel-type="pd"]');
  check("panel type list offers 9 analyses", typeSel.options.length === 9,
    `${typeSel.options.length} options`);
  typeSel.value = "isi";
  typeSel.dispatchEvent(new win.Event("change", {bubbles:true}));
  await sleep(500);
  check("panel swapped to ISI histogram",
    !!q('[data-panel="pd"] svg') && q('[data-panel="pd"] svg').textContent.includes("inter-spike"));
  const t2 = q('[data-panel-type="pd"]');
  t2.value = "ratedist"; t2.dispatchEvent(new win.Event("change", {bubbles:true}));
  await sleep(500);
  check("panel swapped to rate distribution",
    q('[data-panel="pd"] svg').textContent.includes("per-neuron rate"));

  console.log("\n[#7 custom sweep]");
  const t3 = q('[data-panel-type="pd"]');
  t3.value = "sweep"; t3.dispatchEvent(new win.Event("change", {bubbles:true}));
  await sleep(500);
  check("sweep controls rendered", q('[data-panel="pd"] [data-sw="go"]') !== null);
  const keySel = q('[data-panel="pd"] [data-sw="key"]');
  check("sweepable constants discovered", keySel && keySel.options.length >= 6,
    keySel ? `${keySel.options.length} constants` : "none");
  check("global g is offered",
    Array.from(keySel.options).some(o => o.textContent.includes("global")));

  win.eval(`(function(){
    var p = panels.find(function(x){return x.id === 'pd';});
    p.opts.scope='globals'; p.opts.node_id=null; p.opts.key='g'; p.opts.unit='';
    p.opts.metric='amplitude_at'; p.opts.at='20';
    p.opts.from=2; p.opts.to=8; p.opts.steps=4; p.target='exc';
  })()`);
  win.eval("runSweep(panels.find(function(x){return x.id==='pd';}))");
  await sleep(70000);
  const swept = win.eval("(function(){var p=panels.find(function(x){return x.id==='pd';});return p._sweep?JSON.stringify(p._sweep.points.map(function(x){return x.metric;})):'none';})()");
  check("sweep returned points", swept !== "none", swept);
  const out = q('[data-panel="pd"] [data-sw="out"] svg');
  check("sweep plotted a curve", !!out && out.innerHTML.includes("<circle"));
  check("sweep axis labelled with the metric", !!out && out.textContent.includes("Amplitude"));

  console.log("\n[errors]");
  if (errors.length) errors.slice(0,6).forEach(e => console.log("   " + e.slice(0,200)));
  check("no uncaught JS errors", errors.length === 0, `${errors.length}`);

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
