# Plan: Split migration guide into Migration Guide + What's New in 1.0

## Steps

- [x] **1. Create GitHub issue**  
  Propose splitting the current migration guide into two documents: a short "need to know" migration guide and a user-focused "What's new in ArviZ 1.0?" document. Link to this plan and the outline below.

- [x] **2. Create PR (after issue is agreed)**  
  Use **jupytext** for the split and revisions so we keep editable notebooks in the repo while doing the heavy editing in markdown:

  1. **Convert existing doc to markdown:** Use jupytext to convert `migration_guide.ipynb` to `.md` (e.g. `migration_guide.md`).
  2. **Split the markdown:** Create two markdown files from the converted content—one for the **migration guide** (need-to-know only) and one for **"What's new in ArviZ 1.0?"** (informative)—following the outline below.
  3. **Revise both docs:** Edit the two `.md` files (trim migration guide, expand/organize What's new, add cross-links).
  4. **Convert back to notebooks:** Use jupytext to convert both revised `.md` files to `.ipynb` for the built docs (e.g. `migration_guide.ipynb` and `whats_new_1_0.ipynb`).
  5. **Finalize:** Cross-link the two docs in the text, update `index.rst` / nav so both are discoverable, and remove or archive the temporary `.md` files if desired (or keep them as jupytext pair if the project prefers).

- [ ] **3. (Optional)**  
  Add a short "For developers" or "Technical details" section or doc if any content is still too implementation-focused for the main user docs.

---

## Address PR #2528 review (OriolAbril)

Steps to address colleague feedback on the migration guide and doc format:

- [x] **4. Reorder migration guide sections by expected impact**  
  Change section order so the highest-impact breaking changes come first:
  1. **Plot names and signatures** (1st — renames affect many users)
  2. **Plot return type and kwargs** (return type + visuals/stats/pc_kwargs)
  3. **DataTree (replaces InferenceData)** (order of 2 vs 3 can be debated)
  4. **Credible intervals and rcParams** and/or **Model comparison (WAIC, compare default)** (ci_prob/ci_kind, WAIC removal, stacking)
  5. **I/O (netcdf/zarr)** last — behavior stays similar; main impact is installing dependencies.

- [x] **5. Revisit dim / sample_dims in the migration guide**  
  Colleague notes that dim/sample_dims is "basically all new functionality"; in 0.x only `hdi` (via kwarg with xarray.apply_ufunc) and a couple of plots with `skip_dims` could reduce over something other than chain+draw. Consider **removing or greatly shortening** the dim/sample_dims entry on this short, breaking-changes-only page (keep full treatment in What's new).

- [x] **6. Consolidate rcParams into one section (last)**  
  Merge the scattered rcParams-related bullets into a **single general "rcParams and defaults" section** at the end of the migration guide. Include:
  - Ways to control behavior: **user-level or project templates**, **`rcParams` at the start of your code**, or **per-call arguments**.
  - Clarify that **we do not consider changing rcParam defaults a breaking change**; users who care strongly about defaults should use a user-level template.
  - Note: the `hdi_prob` → `ci_prob` rename was done in preparation (existing arvizrc templates already use `ci_prob`); the default weighting method was changed to stacking a while ago, so that is unchanged. Link to 0.x defaults: [arvizrc.template (v0.x)](https://github.com/arviz-devs/arviz/blob/v0.x/arvizrc.template).

- [x] **7. Use MyST format for both docs**  
  Use **MyST** (not plain `.md`) for `migration_guide` and `whats_new_1_0` so that **myst-nb** can process them natively; cell tags and more advanced features then work out of the box. Otherwise we would need to configure myst-nb to use jupytext to parse these as notebooks, which is not worth it. These docs should be quick to run; if we add a zarr showcase, double-check the libraries installed for the docs build.

- [x] **8. Fix Sphinx header hierarchy**  
  Even though we use markdown-style text, Sphinx requires headers to **decrease by one level** (no skipping). A `####` level header cannot come directly after a `##` level header — use `##` → `###` → `####` as needed. Audit both docs and fix any broken hierarchy.

- [x] **9. Review and update URLs that link to the migration guide**  
  There are URLs around the repo (and possibly external) that link to the migration guide and to specific sections. Before merge:
  - **Update** links where the target has moved or the wording is wrong.
  - **Decide per link** whether it should point to the **migration guide** (short, need-to-know) or **What's new in 1.0** (full narrative). Document or fix each so we merge with consistent pointers.

  **Done:** Updated `issue.md` to link to `migration_guide.md` (was `migration_guide.ipynb`). **Left as-is:** `README.md` points to built docs `migration_guide.html` (correct); `getting_started.md` uses `{ref}\`migration_guide\`` (correct); `index.rst` includes both docs (correct). All point to the short migration guide where appropriate; What's new is linked from within the migration guide and from the toctree.

---

## Target audience and goals

### Migration guide (need to know)

- **Audience:** Users upgrading from pre-1.0 ArviZ (and anyone who must fix breakage quickly).  
- **Goal:** Get people unblocked. Answer: *What must I change to run my code on 1.0?*  
- **Tone:** Short, actionable. Focus on breaking changes, one-line or few-line before/after where helpful, and pointers to the right place for details (including "What's new in 1.0?").  
- **Length:** Aim for a document that can be read in a few minutes; detailed explanations live in "What's new" or in API docs.

**Migration guide entry schema:** For each breaking change (or related group of changes), use this structure so entries are parallel and easy to scan:

1. **What changed and why** — State the change in one or two sentences and give a brief rationale (e.g. why the new default, why the rename). Keep the tone positive (the change is an improvement; migration is minimal).
2. **What functions are affected (if applicable)** — List the functions that take the new parameter, are renamed, or otherwise affected. Omit this block if the change is global (e.g. rcParams only) or if the affected surface is already obvious from the change description.
3. **What to do** — Spell out required changes and user options: what they must change if they used the old API, and what they can choose (e.g. "If you're fine with the new default, no change needed. If you want the old behavior, do X or Y.").
4. **Examples (when appropriate)** — Add short code examples that show the new API: e.g. no-arg usage (new defaults), explicit new args, and optionally how to restore old behavior. For structural changes (e.g. DataTree), show the new access pattern or method replacement in a few lines.

See the "Credible intervals and rcParams" section in `migration_guide.md` for the worked example (three parallel items with examples: `ci_kind`, `hdi_prob`→`ci_prob`, default 0.94→0.89). The "DataTree" section shows the same schema with an example for group access and method replacements.

### What's new in ArviZ 1.0? (informative, still user-focused)

- **Audience:** Users (and curious contributors) who want to understand what changed and what’s new—not necessarily "I need to fix my script right now."  
- **Goal:** Inform and excite. Answer: *What’s different in 1.0 and how can I use it?*  
- **Tone:** Explanatory, with context and examples. Can include rationale (e.g. why DataTree, why new defaults). Still written for users (concepts, workflows, options), not for core developers (internal architecture, unless it directly affects usage).  
- **Length:** Can be longer; it’s the place for "the full story" at a user level.

---

## Outline of current document and suggested split

Below is the structure of the current `migration_guide.ipynb`, with notes on where each part should live. **MG** = Migration guide (need to know). **WN** = What's new in 1.0 (informative, user-focused). **Both** = appear in both, with different depth (MG = brief + link, WN = full explanation/examples).

| Section | Migration Guide (MG) | What's New (WN) | Notes |
|--------|----------------------|------------------|--------|
| **Title / intro** | | | |
| ArviZ migration guide (intro paragraph) | ✓ Short "why 1.0 / why two docs" | ✓ Same intro, then "below is the full picture" | MG: 1–2 sentences + link to WN. WN: keep modularity story (arviz-base, arviz-stats, arviz-plots, optional deps). |
| Check libraries exposed (`az.info`) | Optional 1-liner | ✓ | WN: keep as "how to check your install." MG: only if we say "first step is to confirm arviz loads." |
| **## arviz-base** | | | |
| ### Credible intervals and rcParams | ✓ | ✓ | **MG:** "Defaults changed: ci_prob 0.94→0.89, ci_kind now 'eti' (was 'hdi'). Override with rcParams or per-call." **WN:** Why we changed (stability, reminder), full rcParams list, when to use eti vs hdi. |
| ### DataTree | ✓ | ✓ | **MG:** "InferenceData is gone; we use xarray.DataTree. Same groups (posterior, etc.). dt['posterior'] is a DataTree; use .dataset or .to_dataset() when you need a Dataset." **WN:** What DataTree is, benefits (I/O, nesting), compatibility note (important callout), example of structure. |
| #### What about my existing netcdf/zarr files? | ✓ | ✓ | **MG:** "Old files still work. Read with from_netcdf/from_zarr. Write with .to_netcdf() / .to_zarr() on the tree (not top-level to_netcdf)." **WN:** Table of legacy vs new API, footnote on from_netcdf/from_zarr and engines. |
| #### Other key differences | ✓ | ✓ | **MG:** Short bullets: extend→update (with order note); map→map_over_datasets + filter/match; groups→different (link to WN). **WN:** Full explanation and code for extend/update, map/map_over_datasets, group access. |
| ##### InferenceData.extend | ✓ (minimal) | ✓ | MG: "Use DataTree.update(); order is reversed for default left-like merge." WN: Full equivalence, how=left/right. |
| ##### InferenceData.map | ✓ (minimal) | ✓ | MG: "Use map_over_datasets; use filter/match for group selection." WN: Full example, filter vs match. |
| ##### InferenceData.groups | ✓ (minimal) | ✓ | MG: "No .groups property; use DataTree structure (e.g. .children, .ds or .to_dataset())." WN: How to list/inspect groups. |
| ### Enhanced converter flexibility | Optional | ✓ | MG: Only if there’s a breaking change in from_* APIs. WN: New flexibility (e.g. from_dict), examples. |
| ### New data wrangling features | No | ✓ | Purely "new stuff"; WN only. |
| **## arviz-stats** | | | |
| ### Model comparison | ✓ (if defaults/API changed) | ✓ | MG: "If you used compare/loo/waic, check new method/stacking default (ic_compare_method)." WN: stacking vs other methods, when to use what. |
| ### dim and sample_dims | ✓ | ✓ | **MG:** "sample_dims default is still (chain, draw); if you pass dim= explicitly, note it’s the same idea. Any renames here: one-line." **WN:** Full explanation of dim/sample_dims, rcParams, custom dims. |
| ### Accessors on xarray objects | Optional | ✓ | MG: Only if accessor names or behavior changed. WN: What accessors exist, how to use them. |
| ### Computational backends | No | ✓ | WN: Backend options, performance. |
| ### Array interface | No | ✓ | WN: For users who care about array protocols; can be short. |
| **## arviz-plots** | | | |
| ### More and better supported backends! | No | ✓ | WN: matplotlib, bokeh, plotly, how to choose. |
| ### Plotting function inventory | Optional | ✓ | MG: Only "some plot names/signatures changed; see What's new." WN: Full inventory / mapping. |
| ### What to expect from the new plotting functions | ✓ | ✓ | **MG:** "Plot functions may return PlotCollection; kwargs passed to backend. See What's new for details." **WN:** kwarg forwarding, PlotCollection return type, how to use. |
| #### kwarg forwarding | ✓ (one line) | ✓ | WN: Full explanation. |
| #### New return type: PlotCollection | ✓ (one line) | ✓ | WN: What it is, how to get figures. |
| ### Plotting manager classes | No | ✓ | WN: For users who want programmatic control. |
| ### Other arviz-plots features | No | ✓ | WN only. |
| **## Other nice features** | | | |
| ### Citation helper | No | ✓ | WN: How to cite, az.citation(). |
| ### Extended documentation | No | ✓ | WN: New docs, EABM, navigation. |

---

## Summary

- **Migration guide:** Short, scannable list of breaking changes and minimal code edits (DataTree vs InferenceData, rcParams/ci defaults, I/O, extend/map/groups, stats/plot API tweaks). Link to "What's new in 1.0?" for rationale and depth.  
- **What's new in 1.0?:** Full narrative by area (base, stats, plots, other), with context, examples, and "nice to have" features. Same topics can appear in both with different depth (MG = need to know; WN = full story for users).

This keeps the migration guide truly "need to know" while giving users a single, user-focused place to read "what’s new" without digging through developer-level detail.

---

## Run the migration guide (or What's new) from the command line

To confirm the code examples in the markdown run correctly:

1. **Environment:** Use the `arviz` conda env with ArviZ from the repo and optional deps:
   ```bash
   mamba activate arviz
   pip install -e ".[doc]"   # ArviZ 1.0 from repo
   pip install jupytext nbconvert ipykernel numba plotly webcolors bokeh  # optional deps used in the notebooks
   python -m ipykernel install --user --name arviz --display-name "Python (arviz)"  # once per env
   ```

2. **Convert md → notebook and execute** (from `docs/source/user_guide`):
   ```bash
   cd docs/source/user_guide
   jupytext --to ipynb migration_guide.md -o migration_guide_run.ipynb
   jupyter nbconvert --to notebook --execute migration_guide_run.ipynb \
     --ExecutePreprocessor.kernel_name=arviz --ExecutePreprocessor.timeout=600
   ```
   For the What's new doc, replace `migration_guide` with `whats_new_1_0` in the filenames.

3. **Output:** The executed notebook is written to `migration_guide_run.nbconvert.ipynb` (or `whats_new_1_0_run.nbconvert.ipynb`). You can delete the `*_run*.ipynb` files or add them to `.gitignore` if you don't want to commit them.
