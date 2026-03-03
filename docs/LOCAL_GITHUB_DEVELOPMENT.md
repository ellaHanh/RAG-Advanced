# Local–GitHub Development Best Practices

Guidance for working with RAG-Advanced on your machine and syncing to GitHub, especially in the initial stage.

---

## 1. Branch strategy (initial-stage project)

| Approach | Pros | Cons |
|---------|------|------|
| **Single `main`** | Simple; everything on one branch; easy to push/pull | No protection for broken states; risk of force-push mistakes |
| **`main` + short-lived feature branches** | Clear history; PRs later; reversible work | Slightly more workflow; need to remember to merge/delete branches |
| **`main` + `develop`** | Classic flow; `develop` for integration, `main` for releases | Heavier for a very small team or solo early stage |
| **Trunk-based (main only, small commits)** | Fast feedback; encourages small, reviewable changes | Requires discipline and good tests |

**Recommendation for “very initial stage”:** Use **`main`** only at first. Create short-lived branches (e.g. `feature/xyz`, `fix/abc`) when you want to experiment or isolate work; merge back to `main` and push. Add `develop` later if you introduce a release cycle.

---

## 2. Commit discipline

- **Atomic commits:** One logical change per commit (one fix, one feature slice, one refactor).
- **Clear messages:** Use present tense, start with a verb, e.g. `Add cost tracking to executor`, `Fix rate limiter for missing Redis`.
- **Avoid “kitchen sink” commits:** Don’t mix unrelated edits (e.g. formatting + behavior change) in one commit.

---

## 3. Sync rhythm (local ↔ GitHub)

- **Push often:** After every few logical commits or at least daily, so GitHub is a reliable backup and collaboration point.
- **Pull before starting work:** `git pull origin main` (or your default branch) to avoid push rejections and merge mess.
- **First-time setup:** One-time init → add remote → first commit → push (see “First-time push” below).

---

## 4. Remote and default branch

- **Single remote:** `origin` → `https://github.com/EllaHanh/RAG-Advanced.git`.
- **Default branch:** `main`. Create the repo on GitHub with “main” as default, or rename later in GitHub settings.
- **Track explicitly:** First push uses `git push -u origin main` so local `main` tracks `origin/main`; later a plain `git push` is enough.

---

## 5. What never goes to GitHub

- **Secrets:** `.env`, `api_keys.json`, `credentials.json`, `secrets.json` (already in `.gitignore`).
- **Local only:** `.venv/`, `venv/`, `__pycache__/`, `.DS_Store`, IDE/project files under `.idea/`, `.vscode/` (as in your `.gitignore`).
- **Large/binary blobs:** Documents, media, model weights, DB dumps—keep off Git or use Git LFS only if you decide to version them.

Keep using `.env.example` (no secrets) so others know which variables to set.

---

## 6. When you grow: protection and automation

- **Branch protection:** Require PRs and status checks for `main` when you care about “nothing broken”.
- **CI:** Run tests and lint on push/PR (e.g. GitHub Actions from your `tests/` and tooling).
- **Dependabot:** Turn on for `pyproject.toml` and lockfiles to get dependency update PRs.

---

## 7. First-time push (what is set up)

Already done in this repo:

1. **Init and remote**
   - `git init`
   - `git remote add origin https://github.com/EllaHanh/RAG-Advanced.git`
   - `git branch -M main`

2. **First commit**
   - `git add -A` (respecting `.gitignore`)
   - `git commit -m "Initial commit: RAG-Advanced orchestration, API, evaluation"`

**You still need to push from your machine** (Git needs your credentials):

```bash
cd /Users/heekyung/Documents/Github/RAG-Advanced
git push -u origin main
```

If the repo does not exist on GitHub yet: create **EllaHanh/RAG-Advanced** at https://github.com/new (no need to add a README or .gitignore). Then run the command above.

Use **HTTPS** with a personal access token, or **SSH** (e.g. `git@github.com:EllaHanh/RAG-Advanced.git`) if you use SSH keys. To switch to SSH:

```bash
git remote set-url origin git@github.com:EllaHanh/RAG-Advanced.git
git push -u origin main
```

After the first push, normal loop: **pull → change → commit → push**.

---

## 8. Quick reference

```bash
# Daily start
git pull origin main

# After changes
git status
git add <files>   # or git add -A with care
git commit -m "Descriptive message"
git push

# Optional: work on a branch
git checkout -b feature/my-feature
# ... work, commit ...
git checkout main
git merge feature/my-feature
git push
git branch -d feature/my-feature
```

This file is the source of truth for this project’s local–GitHub workflow. Update it when you adopt a different branch model or CI steps.
