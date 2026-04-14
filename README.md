# resume-generator

Generic markdown-driven resume generation workspace.

This public repo intentionally excludes private resume corpus files, user profile content, and generated outputs. Use the sample files in `examples/` plus the directory notes in `data/` as the starting point for your own corpus.

## Goals

- Public-safe and GitHub-uploadable
- Uses markdown project records as the source of truth
- Supports multiple role-family resume variants
- Emits modern LaTeX output
- Preserves the layered `selected.json -> resume-plan.json -> tex/pdf` flow

## Expected input

Project records live under `data/projects/` and follow the md schema used by the portfolio corpus.

Optional personal/context overlays live under `data/context/` and explain why a project existed, what triggered it, and how you want it framed.

Optional family overlays live under `data/families/` and group multiple repos into one project family.

Sample inputs live under `examples/` and the repo ships empty tracked placeholders for `data/projects/`, `data/context/`, and `output/` so you can populate your own corpus without publishing private material.

Minimum useful fields:

- `cosine_title`
- `cosine_summary`
- `resume_keywords`
- `role_family_targets`
- `## Resume Bullet Candidates`
- `## Caveats`

Useful context overlay fields:

- `motivation_summary`
- `problem_trigger`
- `personal_connection`
- `why_now`
- `context_keywords`

Useful optional project frontmatter fields:

- `resume_display_name` for resume/prompt title overrides without changing the source display name
- `technical_impressiveness` for small data-driven ranking bias (defaults to `1.0`)

## Suggested flow

1. Collect repo-local `resume-project.md` files
2. Normalize/copy selected records into `data/projects/`
3. Rank records against a role target or job description
4. Build `selected.json` as the broad ranked candidate bundle
5. Let the LLM choose and optionally refine the final subset into `resume-plan.json`
6. Render LaTeX with the template in `templates/`

## Included pipeline stub

`rank_projects.py` now uses a hybrid ranker: lexical TF-IDF, JD chunking, keyword overlap, existing heuristics, optional SentenceTransformers embeddings, and optional cross-encoder reranking.

Install semantic extras only if you want dense retrieval/reranking locally:

```bash
uv pip install -r requirements-semantic.txt
```

Optional local JD keyword expansion can be enabled with Ollama:

```bash
export SEMANTHA_ENABLE_OLLAMA_EXPANSION=1
export SEMANTHA_OLLAMA_URL=http://localhost:11434
export SEMANTHA_OLLAMA_MODEL=llama3.2
```

`sync_projects.py` copies repo-local `resume-project.md` files into the generator's central `data/projects/` directory.

`sync_context.py` copies manual user-context overlays into `data/context/`.

`build_resume_prompt.py` builds the broad ranked `selected.json` candidate bundle, writes a sibling `*-match-report.json` score breakdown artifact, and emits an LLM-ready prompt bundle containing the target, profile, selected project summaries, and LaTeX template.

`draft_resume_tex.py` renders a deterministic `.tex` draft from either a `resume-plan.json` editorial plan or a legacy `selected.json` bundle, using synthesis and suggested bullet points where available.

`semantha_tailor.py` is the convenience one-shot CLI. It preserves the internal layered flow, but runs the default bundle -> plan -> render -> compile path in one command.

## SEmantha MCP server

`semantha_server.py` exposes the resume workspace as an MCP stdio server without adding any external dependency.

SEmantha works best when this repo is paired with a populated corpus under `data/` and, optionally, a sibling `portfolio/` workspace for the original local authoring flow. The public repo is still usable without that private sibling workspace if you provide explicit target text and populate your own `data/projects/`, `data/context/`, and profile inputs.

Name origin:

- **SE**mantic search + resume tailoring = **SEmantha**

It wraps the existing workflow rather than replacing it.

SEmantha is intentionally **not** a full workflow engine. The narrowed MCP scope is retrieval, packaging, rendering, and compilation. The LLM is expected to decide what actually goes on the resume; SEmantha packages and renders those decisions.

### Run

```bash
python3 semantha_server.py
```

### What it exposes

- **Tools** for:
  - semantic project search
  - project inspection
  - resume bundle creation (`selected.json`)
  - resume plan creation (`resume-plan.json`)
  - deterministic TeX rendering
  - PDF compilation

- **Resources** for:
  - `semantha://profile`
  - `semantha://projects/index`
  - `semantha://projects/{project_id}`
  - `semantha://context/{project_id}`
  - `semantha://families/index`
  - `semantha://families/{family_id}`
  - `semantha://targets/index`
  - `semantha://outputs/index`
  - `semantha://outputs/{label}/{kind}` where `kind` is `selected`, `report`, `plan`, `prompt`, `tex`, or `pdf`

- **Prompts** for:
  - end-to-end resume tailoring
  - existing resume refinement

### Intended MCP workflow

1. Search relevant projects for a target
2. Build `selected.json` as the broad ranked retrieval bundle
3. Let the LLM choose the final ordered subset and optionally prepare text overrides
4. Call `create_resume_plan` to package that editorial decision into `resume-plan.json`
5. Render TeX from the resume plan
6. Compile PDF

If subset selection is skipped, `create_resume_plan` can auto-pick the top-ranked projects.

Family overlays let the ranker diversify across repo families instead of treating every repo as unrelated.

It:

- parses markdown project records with simple YAML frontmatter
- builds a weighted TF-IDF representation over title, summary, keywords, and bullet sections
- chunks long job descriptions for more stable matching
- optionally adds SentenceTransformers embeddings and cross-encoder reranking when those dependencies are installed
- optionally folds in context overlays so ranking can reflect why a project mattered, not just what was built
- carries full context overlays into the LLM prompt bundle so drafting can use motive, trigger, and stakes
- computes lexical + semantic similarity against a query or target file
- applies heuristic boosts/penalties for role-family fit, authorship confidence, and evidence strength
- emits score decomposition and match-report artifacts
- estimates project-section line pressure and warns about likely bullet or summary wraps

Example:

```bash
python3 sync_projects.py \
  --source-root ../portfolio/repos \
  --output-dir data/projects

python3 sync_context.py \
  --source-root ../portfolio/context \
  --output-dir data/context

python3 rank_projects.py \
  --projects-dir data/projects \
  --context-dir data/context \
  --role-family linux-systems \
  --query "Rust Linux daemon PAM systemd GPU inference" \
  --top 5 \
  --label linux-systems

python3 build_resume_prompt.py \
  --projects-dir data/projects \
  --context-dir data/context \
  --family-dir data/families \
  --profile-file ../portfolio/resume/profile.md \
  --template-file templates/modern-onepage.tex \
  --target-file ../portfolio/resume/linux-systems-target.md \
  --role-family linux-systems \
  --top 6 \
  --label linux-systems

python3 - <<'PY'
from pathlib import Path
from semantha_core import SemanthaWorkspace

workspace = SemanthaWorkspace(Path('.').resolve())
workspace.create_resume_plan(label='linux-systems', top_n=3)
PY

python3 draft_resume_tex.py \
  --plan-file output/linux-systems-resume-plan.json \
  --profile-file ../portfolio/resume/profile.md \
  --template-file templates/modern-onepage.tex \
  --output-tex output/linux-systems.tex \
  --max-projects 3 \
  --max-bullets-per-project 1

python3 semantha_tailor.py \
  --label linux-systems-quick \
  --target-file ../portfolio/resume/linux-systems-target.md \
  --role-family linux-systems \
  --top 6 \
  --plan-top-n 3 \
  --max-projects 3 \
  --max-bullets-per-project 1
```

## Repository layout

- `templates/` LaTeX templates
- `data/projects/` markdown project database
- `data/context/` optional user-story overlays
- `data/families/` optional project-family overlays
- `examples/` sample records and sample target inputs
- `output/` generated artifacts

## Notes

- This scaffold is intentionally generator-agnostic.
- The current handoff model is: rank first, package a broad `selected.json`, let the LLM produce a narrower `resume-plan.json`, then render/compile.
- `selected.json` is the retrieval bundle, `resume-plan.json` is the editorial intermediary, and TeX/PDF are presentation artifacts.
- A deterministic `.tex` renderer is also included so the project can emit a baseline draft without requiring an external LLM call.
