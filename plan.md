DataMender: Smart Cleaning for Large CSV/Parquet

---

## Week-by-Week Roadmap

### **Week ### 🎯 **Final Deliverables**

1. Working interactive data cleaning tool:

   * Profiles large file
   * Suggests constraints with AI
   * Applies vectorized fixes
2. YAML of discovered rules.
3. Short video: run through the same "ride-sharing CSV" example.
4. Brief report with metrics & lessons.f & Dataset**

* Finalize dataset (e.g., 5–10 GB ride-sharing or taxi CSV/Parquet).
* Draft clear **problem statement** and objectives.
* Split roles inside group (profiling, LLM prompts, UI).
* Set up repo & dev environment (Python + high-performance libraries + interactive interface).

---

### **Week 2 – Core Profiler**

* Implement fast scan with optimized data processing:

  * Row count, column types, % missing, min/max, histograms.
* Make profiler return a JSON summary for each column.
* Test on a small sample to ensure fast processing performance.

---

### **Week 3 – Rule Discovery**

* Create LLM prompt template:

  > “Given profile X, guess valid ranges, uniqueness, monotonicity.”
* Build a library of **universal sanity checks** (negatives, nulls, date order).
* Combine: profile → heuristics → LLM suggestions.

---

### **Week 4 – Human-in-the-Loop Validation**

* Design simple review UI:

  * Show proposed rule per column.
  * Checkbox to accept / edit / reject.
* Save accepted rules as YAML/JSON for later reuse.

---

### **Week 5 – Batch Fix Engine**

* Implement vectorized fixes for:

  * Range clipping
  * Null filling / dropping
  * Absolute value for negatives
  * Cross-column checks (start\_time < end\_time)
* Apply on whole file; measure time.

---

### **Week 6 – Re-Profiling & Metrics**

* After fixes, re-run profiler → verify issues are resolved.
* Collect metrics:

  * % anomalies removed
  * Processing time vs. file size
  * User steps saved (baseline: manual pandas).

---

### **Week 7 – Polish & Recording**

* Add nice touches:

  * “Preview Clean Data” button
  * Export to new Parquet
  * Clear logs of applied rules
* Script the 15-min recorded demo:

  * Load messy file → profile → confirm rules → clean → show before/after.

---

### **Week 8 – Proposal Write-Up & Final Report**

* Prepare:

  * 2-page written summary (problem, method, results, future work).
  * Final video upload.
* Buffer for debugging or late dataset issues.

---

### 🎯 **Final Deliverables**

1. Working Streamlit/CLI tool:

   * Profiles large file
   * Suggests constraints with AI
   * Applies vectorized fixes
2. YAML of discovered rules.
3. Short video: run through the same “ride-sharing CSV” example.
4. Brief report with metrics & lessons.

---

> **Key success factors:**
>
> * Keep the dataset & scope fixed early.
> * Use AI only for rule generation, not per-row edits.
> * Validate everything on a few sample rows before cleaning 5 M+.

Would you like me to adapt this roadmap into a 3-slide outline (Title / Problem & Motivation / Timeline & Deliverables) for your proposal presentation?
