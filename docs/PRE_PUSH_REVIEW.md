# Pre-Push Review: Report Weaknesses & Code Exposure

> Checklist before pushing to GitHub. Two categories: (1) Report content weaknesses, (2) Code/content that should not be exposed.

---

## 1. Report Content — Potential Weaknesses

### 1.1 PROJECT_REPORT.md

| Item | Location | Assessment |
|------|----------|------------|
| **Alpha not significant (t=0.15)** | §1.4, §1.5, §6 | Honest disclosure. Framed as "value lies in tail-risk efficiency" — acceptable. |
| **Block1 warm-up 2005–2013** | §1.2 | Explains inactive period. Transparent. |
| **5y outperformed 15y in Block1** | §6 | Design choice, not a flaw. |
| **Crisis false positive / mid-range rigidity** | §3 | Trade-off acknowledged. Professional. |
| **Reference to dev_logs/experimental_details.md** | §3 | File is gitignored — link will 404 for clone. Consider removing or noting "internal only." |

**Recommendation:** No critical weaknesses. The "alpha not significant" is framed positively. Optional: Remove or soften the reference to `dev_logs/experimental_details.md` since that file won't exist in the repo.

### 1.2 experiments/outputs/ (if committed)

| File | Content | Assessment |
|------|---------|------------|
| stress_test_report.md | "SPY ≤-10% N=1 – not statistically meaningful" | Honest limitation. Acceptable. |
| stress_test_report.md | "Do not overstate diversification benefits" | Cautious. Good. |
| benchmark_metrics.json | IR = -0.83, excess return negative | Raw performance. Transparent. |

**Recommendation:** Acceptable. Shows transparency and rigor.

---

## 2. Code & Content — Should Not Be Exposed

### 2.1 ✅ Already Protected (gitignore)

| Item | Status |
|------|--------|
| **.env** | In `.gitignore` — API key not exposed |
| **data/, data_refresh/** | gitignore |
| **outputs/, outputs_refresh/** | gitignore |
| **dev_logs/** | gitignore — internal logs, personal notes |
| **experiments/outputs_refresh/** | gitignore — live outputs |
| **docs/GITHUB_READINESS_CHECKLIST.md** | gitignore |

### 2.2 ⚠️ Review Recommended

| Item | Location | Risk | Recommendation |
|------|----------|------|----------------|
| **PRINCIPAL = 3000, SGD** | `scripts/write_dashboard_state.py` L22–24 | Low — could be inferred as portfolio size | Optional: Change to `PRINCIPAL = 10000` or similar |
| **LIVE_START = "2025-12-15"** | Same file | Low — deployment date | OK to keep |
| **Korean in generated reports** | `performance_attribution.py` L197–203 | "미미", "유의", "총 비용", "비용 격차" — mixed language | Consider translating to English for consistency |
| **Korean in README** | README.md L128–132 | Dashboard access instructions | Translate to English for consistency |

### 2.3 ✅ No Exposure Risk

| Item | Status |
|------|--------|
| **API keys** | `os.getenv("FRED_API_KEY")` — no hardcoding |
| **Absolute paths** | None found |
| **Passwords** | None |
| **Personal identifiers** | None |

---

## 3. Summary of Actions

### Must fix before push (none)

No critical issues that would block a push.

### Optional before push

| # | Action | Priority |
|---|--------|----------|
| 1 | Remove or rephrase reference to `dev_logs/experimental_details.md` in PROJECT_REPORT.md | Low |
| 2 | Translate Korean in README (Dashboard section) to English | Medium |
| 3 | Translate Korean in `performance_attribution.py` report output to English | Low |
| 4 | Anonymize PRINCIPAL (3000 → 10000) if desired | Low |

---

## 4. Conclusion

**Safe to push.** No security or sensitive-data exposure. Report content is transparent and professionally framed. Optional improvements are for consistency and polish.
