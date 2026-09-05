/* ==========================================================================
   Handwritten Character Recognition — front end
   Small composable controllers, no framework. Every prediction and metric
   comes from the Flask API, which calls the real trained model.
   ========================================================================== */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const API = {
    predict: "/api/predict",
    modelInfo: "/api/model-info",
  };

  /* ---------------- Theme ---------------- */
  const Theme = {
    key: "hcr-theme",
    init() {
      const stored = safeGet(this.key);
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      this.apply(stored || (prefersDark ? "dark" : "light"));
      $("theme-toggle").addEventListener("click", () => {
        const next =
          document.documentElement.dataset.theme === "dark" ? "light" : "dark";
        this.apply(next);
        safeSet(this.key, next);
      });
    },
    apply(theme) {
      document.documentElement.dataset.theme = theme;
      const dark = theme === "dark";
      const btn = $("theme-toggle");
      btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
      btn.querySelector(".icon-sun").hidden = dark;
      btn.querySelector(".icon-moon").hidden = !dark;
    },
  };

  function safeGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
  function safeSet(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } }

  /* ---------------- Tabs ---------------- */
  const Tabs = {
    init() {
      this.tabs = [$("tab-draw"), $("tab-upload")];
      this.tabs.forEach((tab, i) => {
        tab.addEventListener("click", () => this.select(i));
        tab.addEventListener("keydown", (e) => {
          // Roving tabindex: arrow keys move between tabs (WAI-ARIA pattern).
          if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
          e.preventDefault();
          const next = (i + (e.key === "ArrowRight" ? 1 : this.tabs.length - 1)) % this.tabs.length;
          this.select(next);
          this.tabs[next].focus();
        });
      });
    },
    select(index) {
      this.tabs.forEach((tab, i) => {
        const active = i === index;
        tab.setAttribute("aria-selected", String(active));
        tab.tabIndex = active ? 0 : -1;
        $(tab.getAttribute("aria-controls")).hidden = !active;
      });
      InputError.clear();
    },
  };

  /* ---------------- Inline input error ---------------- */
  const InputError = {
    show(message) {
      $("input-error-text").textContent = message;
      $("input-error").hidden = false;
    },
    clear() { $("input-error").hidden = true; },
  };

  /* ---------------- Drawing canvas ---------------- */
  const Canvas = {
    init() {
      this.el = $("pad");
      this.ctx = this.el.getContext("2d", { willReadFrequently: true });
      this.wrap = $("canvas-wrap");
      this.hint = $("canvas-hint");
      this.strokes = [];      // committed strokes, for undo
      this.current = null;
      this.drawing = false;

      // Keep the backing store square and crisp on high-DPI screens.
      this.resize();
      window.addEventListener("resize", debounce(() => this.resize(), 150));

      const pos = (e) => {
        const r = this.el.getBoundingClientRect();
        const p = e.touches ? e.touches[0] : e;
        return [
          ((p.clientX - r.left) / r.width) * this.el.width,
          ((p.clientY - r.top) / r.height) * this.el.height,
        ];
      };

      const start = (e) => {
        if (e.button !== undefined && e.button !== 0) return;
        e.preventDefault();
        this.drawing = true;
        this.wrap.classList.add("is-drawing");
        this.hint.hidden = true;
        this.current = [pos(e)];
        this.redraw();
      };
      const move = (e) => {
        if (!this.drawing) return;
        e.preventDefault();
        this.current.push(pos(e));
        this.redraw();
      };
      const end = () => {
        if (!this.drawing) return;
        this.drawing = false;
        this.wrap.classList.remove("is-drawing");
        if (this.current && this.current.length) this.strokes.push(this.current);
        this.current = null;
        this.redraw();
      };

      this.el.addEventListener("pointerdown", start);
      this.el.addEventListener("pointermove", move);
      window.addEventListener("pointerup", end);
      window.addEventListener("pointercancel", end);
      // Fallback for browsers/automation without pointer events.
      this.el.addEventListener("mousedown", start);
      this.el.addEventListener("mousemove", move);
      window.addEventListener("mouseup", end);
      this.el.addEventListener("touchstart", start, { passive: false });
      this.el.addEventListener("touchmove", move, { passive: false });
      window.addEventListener("touchend", end);

      this.clear();
    },

    resize() {
      const width = Math.round(this.el.getBoundingClientRect().width);
      if (!width) return;
      const side = Math.max(280, Math.min(560, width * (window.devicePixelRatio > 1 ? 2 : 1)));
      if (side === this.el.width) return;
      this.el.width = side;
      this.el.height = side;
      this.redraw();
    },

    redraw() {
      const ctx = this.ctx;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, this.el.width, this.el.height);
      ctx.strokeStyle = "#111111";
      ctx.lineWidth = this.el.width * 0.062;   // scales with canvas size
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      const all = this.current ? this.strokes.concat([this.current]) : this.strokes;
      for (const stroke of all) {
        if (!stroke.length) continue;
        ctx.beginPath();
        ctx.moveTo(stroke[0][0], stroke[0][1]);
        for (const [x, y] of stroke.slice(1)) ctx.lineTo(x, y);
        if (stroke.length === 1) ctx.lineTo(stroke[0][0] + 0.1, stroke[0][1]);
        ctx.stroke();
      }
      this.hint.hidden = all.length > 0;
      this.syncButtons();
    },

    syncButtons() {
      const empty = this.isEmpty();
      $("btn-undo").disabled = empty;
      $("btn-clear").disabled = empty;
    },

    isEmpty() { return this.strokes.length === 0 && !this.current; },

    undo() { this.strokes.pop(); this.redraw(); },

    clear() { this.strokes = []; this.current = null; this.redraw(); },

    toDataURL() { return this.el.toDataURL("image/png"); },
  };

  function debounce(fn, ms) {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  }

  /* ---------------- Upload ---------------- */
  const Uploader = {
    MAX_BYTES: 8 * 1024 * 1024,
    ALLOWED: ["image/png", "image/jpeg", "image/jpg"],
    file: null,

    init() {
      this.zone = $("dropzone");
      this.input = $("file-input");
      this.preview = $("file-preview");

      this.zone.addEventListener("click", () => this.input.click());
      this.zone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); this.input.click(); }
      });
      this.input.addEventListener("change", () => {
        if (this.input.files.length) this.accept(this.input.files[0]);
      });

      ["dragenter", "dragover"].forEach((type) =>
        this.zone.addEventListener(type, (e) => {
          e.preventDefault();
          this.zone.classList.add("is-over");
        })
      );
      ["dragleave", "drop"].forEach((type) =>
        this.zone.addEventListener(type, (e) => {
          e.preventDefault();
          this.zone.classList.remove("is-over");
        })
      );
      this.zone.addEventListener("drop", (e) => {
        const file = e.dataTransfer && e.dataTransfer.files[0];
        if (file) this.accept(file);
      });

      $("btn-remove-file").addEventListener("click", () => this.reset());
    },

    accept(file) {
      InputError.clear();
      const type = (file.type || "").toLowerCase();
      if (!this.ALLOWED.includes(type)) {
        this.reset();
        InputError.show("That file type isn't supported. Please choose a PNG or JPG image.");
        return;
      }
      if (file.size > this.MAX_BYTES) {
        this.reset();
        InputError.show("That image is larger than 8 MB. Please choose a smaller file.");
        return;
      }
      if (file.size === 0) {
        this.reset();
        InputError.show("That file is empty. Please choose a different image.");
        return;
      }

      this.file = file;
      $("file-name").textContent = file.name;
      const reader = new FileReader();
      reader.onload = () => {
        $("preview-img").src = reader.result;
        this.zone.hidden = true;
        this.preview.hidden = false;
      };
      reader.onerror = () => {
        this.reset();
        InputError.show("That image couldn't be read. Please try a different file.");
      };
      reader.readAsDataURL(file);
    },

    reset() {
      this.file = null;
      this.input.value = "";
      $("preview-img").removeAttribute("src");
      this.preview.hidden = true;
      this.zone.hidden = false;
    },
  };

  /* ---------------- Result panel ---------------- */
  const Result = {
    states: ["state-empty", "state-loading", "state-error", "state-result"],

    show(which) {
      this.states.forEach((id) => { $(id).hidden = id !== which; });
    },

    announce(message) { $("live-region").textContent = message; },

    loading() {
      this.show("state-loading");
      this.announce("Analysing handwriting.");
    },

    error(message) {
      $("result-error-text").textContent = message;
      this.show("state-error");
      this.announce("Recognition failed. " + message);
    },

    empty() { this.show("state-empty"); },

    render(data) {
      $("result-glyph").textContent = data.predicted_character;
      // Round *down* so a 99.96% prediction never displays as a flat "100%".
      const pct = confidenceText(data.confidence);
      $("result-conf").textContent = pct + " confidence";

      const badge = $("result-badge");
      const note = $("result-note");
      if (data.is_uncertain) {
        badge.className = "badge badge--warning";
        badge.textContent = "Low confidence";
        badge.hidden = false;
        note.textContent =
          "The model is uncertain about this prediction. Check the alternatives below, " +
          "or redraw the character larger and more clearly.";
      } else {
        badge.className = "badge badge--success";
        badge.textContent = "Confident";
        badge.hidden = false;
        note.textContent = "Most confident prediction: " + data.predicted_character + ".";
      }

      const list = $("topk");
      list.innerHTML = "";
      data.top_k.forEach((entry, i) => {
        const row = document.createElement("div");
        row.className = "topk__row";

        const ch = document.createElement("span");
        ch.className = "topk__char";
        ch.textContent = entry.character;

        const track = document.createElement("span");
        track.className = "topk__track";
        const fill = document.createElement("span");
        fill.className = "topk__fill";
        track.appendChild(fill);

        const pctEl = document.createElement("span");
        pctEl.className = "topk__pct";
        const value = (entry.probability * 100).toFixed(2) + "%";
        pctEl.textContent = value;

        // Text carries the value too — never colour/length alone.
        row.setAttribute("aria-label", "Rank " + (i + 1) + ": " + entry.character + ", " + value);
        row.append(ch, track, pctEl);
        list.appendChild(row);

        // Commit the 0-width starting state with a forced reflow, then set the
        // final width so the CSS transition still runs. requestAnimationFrame
        // is deliberately avoided here: it is throttled in background tabs, so
        // the bars would silently never fill.
        void fill.offsetWidth;
        fill.style.width = Math.max(entry.probability * 100, 0.8) + "%";
      });

      $("result-thumb").src = data.preprocessed_png;
      this.show("state-result");
      this.announce(
        "Predicted character " + data.predicted_character + " with " + pct + " confidence."
      );
    },
  };

  /* ---------------- Prediction ---------------- */
  const Recognizer = {
    busy: false,

    async run(getBody, button) {
      if (this.busy) return;
      this.busy = true;
      setButtonLoading(button, true);
      InputError.clear();
      Result.loading();

      try {
        const response = await fetch(API.predict, await getBody());
        let payload;
        try {
          payload = await response.json();
        } catch {
          throw new Error("The server returned an unexpected response.");
        }
        if (!response.ok) {
          throw new Error(
            (payload && payload.error && payload.error.message) ||
              "The server could not process that image."
          );
        }
        Result.render(payload);
      } catch (err) {
        const offline = typeof navigator !== "undefined" && navigator.onLine === false;
        Result.error(
          offline
            ? "You appear to be offline. Reconnect and try again."
            : err.message || "Something went wrong. Please try again."
        );
      } finally {
        this.busy = false;
        setButtonLoading(button, false);
      }
    },
  };

  function setButtonLoading(button, loading) {
    if (!button) return;
    button.disabled = loading;
    button.setAttribute("aria-busy", String(loading));
    const label = button.querySelector(".btn__label");
    if (!label) return;
    if (loading) {
      label.dataset.original = label.textContent;
      label.textContent = "Analysing…";
    } else if (label.dataset.original) {
      label.textContent = label.dataset.original;
    }
  }

  /* ---------------- Model info ---------------- */
  const ModelInfo = {
    async init() {
      let info;
      try {
        const response = await fetch(API.modelInfo);
        if (!response.ok) throw new Error("request failed");
        info = await response.json();
      } catch {
        this.unavailable("Model information could not be loaded. Is the server running?");
        return;
      }

      if (info.model_file) $("footer-model").textContent = info.model_file;

      if (!info.metrics) {
        this.unavailable(
          "Test metrics have not been generated yet. Run src/finalize.py to evaluate " +
            "the model and this section will populate with real results."
        );
      } else {
        this.renderMetrics(info.metrics);
      }

      this.renderFacts(info);
      this.renderConfusions(info.top_confusions);
      this.renderExperiments(info.experiments);
      this.renderFigures(info.figures);
    },

    unavailable(message) {
      $("metrics-error-text").textContent = message;
      $("metrics-error").hidden = false;
      $("metrics").hidden = true;
      $("metrics").setAttribute("aria-busy", "false");
    },

    renderMetrics(m) {
      const cards = [
        { label: "Accuracy", value: pct(m.accuracy), hint: m.n_test.toLocaleString() + " test images" },
        { label: "Macro F1", value: pct(m.f1_macro), hint: "all classes weighted equally" },
        { label: "Precision", value: pct(m.precision_macro), hint: "macro average" },
        { label: "Recall", value: pct(m.recall_macro), hint: "macro average" },
      ];
      const host = $("metrics");
      host.innerHTML = "";
      for (const card of cards) {
        const el = document.createElement("div");
        el.className = "card metric";
        el.innerHTML =
          '<div class="metric__label"></div><div class="metric__value"></div><div class="metric__hint"></div>';
        el.querySelector(".metric__label").textContent = card.label;
        el.querySelector(".metric__value").textContent = card.value;
        el.querySelector(".metric__hint").textContent = card.hint;
        host.appendChild(el);
      }
      host.setAttribute("aria-busy", "false");
    },

    renderFacts(info) {
      const rows = [];
      if (info.dataset) {
        rows.push(["Dataset", info.dataset.name]);
        rows.push(["Training images", info.dataset.n_train.toLocaleString()]);
        rows.push(["Test images", info.dataset.n_test.toLocaleString()]);
        rows.push(["Classes", String(info.dataset.n_classes)]);
      }
      rows.push(["Input", "28 × 28 × 1 grayscale"]);
      if (info.architecture) {
        rows.push(["Parameters", info.architecture.parameters.toLocaleString()]);
        rows.push(["Optimiser", "Adam · lr " + info.architecture.learning_rate]);
        rows.push(["Batch size", String(info.architecture.batch_size)]);
        rows.push(["Augmentation", info.architecture.augmented ? "Enabled" : "None"]);
      }
      if (info.model_file) rows.push(["Model file", info.model_file]);

      const host = $("facts");
      host.innerHTML = "";
      for (const [key, value] of rows) {
        const el = document.createElement("div");
        el.className = "fact";
        el.innerHTML = '<span class="fact__key"></span><span class="fact__val"></span>';
        el.querySelector(".fact__key").textContent = key;
        el.querySelector(".fact__val").textContent = value;
        host.appendChild(el);
      }
    },

    renderConfusions(pairs) {
      const host = $("confusions");
      host.innerHTML = "";
      if (!pairs || !pairs.length) {
        host.innerHTML =
          '<p style="color:var(--fg-subtle);font-size:var(--text-sm)">' +
          "Available once the evaluation has been run.</p>";
        return;
      }
      for (const pair of pairs.slice(0, 8)) {
        const el = document.createElement("div");
        el.className = "confusion";
        el.innerHTML =
          '<span class="confusion__pair"></span><span class="confusion__arrow">→</span>' +
          '<span class="confusion__pair"></span><span class="confusion__count"></span>';
        const spans = el.querySelectorAll(".confusion__pair");
        spans[0].textContent = pair.true;
        spans[1].textContent = pair.predicted;
        el.querySelector(".confusion__count").textContent =
          pair.count + " of " + pair.true_class_support;
        host.appendChild(el);
      }
    },

    renderExperiments(rows) {
      const body = document.querySelector("#experiments-table tbody");
      if (!body) return;
      body.innerHTML = "";
      if (!rows || !rows.length) {
        body.innerHTML = '<tr><td colspan="4">Not generated yet.</td></tr>';
        return;
      }
      const best = rows.reduce((a, b) => (b.val_accuracy > a.val_accuracy ? b : a));
      for (const row of rows) {
        const tr = document.createElement("tr");
        if (row === best) tr.className = "is-best";
        tr.innerHTML = "<td></td><td class='num'></td><td class='num'></td><td class='num'></td>";
        const cells = tr.querySelectorAll("td");
        cells[0].textContent = row.experiment;
        cells[1].textContent = row.params.toLocaleString();
        cells[2].textContent = pct(row.val_accuracy);
        cells[3].textContent = (row.train_val_gap >= 0 ? "+" : "") + (row.train_val_gap * 100).toFixed(2) + "%";
        body.appendChild(tr);
      }
    },

    renderFigures(figures) {
      const map = { confusion_matrix: "fig-cm", training_curves: "fig-curves" };
      for (const [key, id] of Object.entries(map)) {
        const el = $(id);
        if (!el) continue;
        if (figures && figures[key]) {
          el.src = figures[key];
        } else {
          const details = el.closest("details");
          if (details) details.hidden = true;
        }
      }
    },
  };

  const pct = (v) => (v * 100).toFixed(2) + "%";

  /** Confidence for display, floored to one decimal so we never round up to 100%. */
  function confidenceText(value) {
    const percent = value * 100;
    if (percent >= 99.95 && value < 1) return "99.9%";
    return percent.toFixed(1) + "%";
  }

  /* ---------------- Wiring ---------------- */
  function init() {
    Theme.init();
    Tabs.init();
    Canvas.init();
    Uploader.init();
    Result.empty();
    ModelInfo.init();

    $("btn-clear").addEventListener("click", () => {
      Canvas.clear();
      InputError.clear();
      Result.empty();
    });
    $("btn-undo").addEventListener("click", () => Canvas.undo());

    $("btn-predict-draw").addEventListener("click", () => {
      if (Canvas.isEmpty()) {
        InputError.show("Please draw a character before recognising it.");
        $("pad").focus();
        return;
      }
      Recognizer.run(
        async () => ({
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: Canvas.toDataURL() }),
        }),
        $("btn-predict-draw")
      );
    });

    $("btn-predict-upload").addEventListener("click", () => {
      if (!Uploader.file) {
        InputError.show("Please choose an image before recognising it.");
        return;
      }
      Recognizer.run(
        async () => {
          const form = new FormData();
          form.append("file", Uploader.file);
          return { method: "POST", body: form };
        },
        $("btn-predict-upload")
      );
    });

    $("btn-retry").addEventListener("click", () => Result.empty());

    $("btn-again").addEventListener("click", () => {
      const drawing = $("tab-draw").getAttribute("aria-selected") === "true";
      if (drawing) {
        Canvas.clear();
        $("pad").focus();
      } else {
        Uploader.reset();
      }
      Result.empty();
      document.getElementById("recognize").scrollIntoView({ block: "start" });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
