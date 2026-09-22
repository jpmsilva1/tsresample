# Research verification protocol

Rules for any claim that is not code. Adapted from the previous kit, which got this part
right — the rules below are what caused the IRonPy claim to be re-checked, and re-checking
it is what saved the project from building on a broken oracle.

---

**1 — Fetch before you claim.** If you are describing what a repository, package, paper or
file contains, open it in this session first. Pattern-matching against similar things you
have seen is not reading. The IRonPy claim ("MIT-licensed, usable as our φ oracle")
survived one full spec revision because nobody opened the repository.

**2 — One citation per claim.** Every factual statement carries the source that supports
it: a file path with line numbers, a URL, a quoted sentence, or a command with its output.
A paragraph with one citation at the end is a paragraph of unsupported claims with a
decoration.

**3 — Flag confidence explicitly.** Every claim is one of:

- **Confirmed** — you read the primary source in this session and can quote it.
- **Inferred** — it follows from something confirmed, and you state the inference step.
- **Unconfirmed** — plausible, unverified. Say so in the same sentence, not in a footnote.

Never let Unconfirmed drift into Confirmed through repetition across documents. That is the
specific failure mode that produced the two wrong formulas in spec v0.6.0.

**4 — Specificity is a scrutiny trigger, not a credibility signal.** A claim that names
exact filenames, version numbers, or line ranges is *more* likely to be fabricated, not
less — specific detail is what a confident wrong answer looks like. When a claim is
unusually specific, that is the moment to open the file, not the moment to relax.

**5 — Numbers beat prose.** Where a claim can be settled by computing something, compute
it. "The reference uses the plain boxplot" backed by a 20-row comparison table is
settled. The same sentence backed by a citation to a DESCRIPTION file is suggestive.
Both are better than reasoning, and the first is what an ADR needs.

**6 — Self-adversarial pass, before you report.** Ask: what would make this wrong? Then go
check that specific thing. For a formula, the question is almost always "did I test this on
more than one dataset, spanning more than one regime?" One dataset can be curve-fitted by
a three-parameter function without effort.

**7 — Corrections propagate to the permanent record.** When a claim turns out wrong, do not
silently edit it. Show old claim → corrected claim → source, so the correction is auditable
and the next reader can see which way the error ran. SPEC §0's table is exactly this, and
it is the most useful page in the repository.

**8 — A confirmed-correct re-check is worth reporting.** A verification pass that finds
three claims already right and one wrong is more useful than one that lists only the
change, because it tells the reader how much of the document was checked.

---

## Applying this to a licence question

Licence claims get the strictest treatment, because they are the only claims here whose
failure cannot be fixed later by editing code.

- A `LICENSE` file at the repository root describes the repository author's intent. It does
  **not** describe vendored subdirectories.
- Check `src/`, `vendor/`, and any compiled artifact separately. ADR-0003 exists because
  IRonPy is labelled MIT at the root and vendors GPL-2 robustbase C and LGPL R headers
  underneath.
- `file` on a binary and a grep for copyright headers answer more than the README does.
- If the answer is "the label and the contents disagree", the contents win and the
  dependency is rejected.
