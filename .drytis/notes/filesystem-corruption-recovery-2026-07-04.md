# Filesystem Corruption & Git Recovery — 2026-07-04

## Incident
The `/workspace/.git` directory suffered catastrophic filesystem-level corruption:
- `.git/refs/heads/main` was overwritten with JPEG image data (not a SHA)
- `.git/objects/` was flattened — subdirectories (`0a/`, `0d/`...) became 149KB garbage files
- `.git/index` had bad signature `0x2e302030`
- Additional victims: `yolov8n.pt`, `models/.gitkeep` (became 273KB JPEG), `app/static/{results,uploads}/.gitkeep` — all unstatable / "Structure needs cleaning"

## Root cause
Unknown — likely XFS/filesystem-level event that wrote JPEG data over random inodes. Not a git operation.

## Recovery (performed by git_manager)
1. Backed up corrupt `.git` → `.git.corrupt-backup/`
2. `git init` fresh on `main`, re-added `origin`, set identity (Rohit Singh Rathor)
3. `git fetch origin` + `git reset --soft c93914f` (recovered commit lineage, kept working tree)
4. Reconciled: local tree had the v2 multi-scale/WBF work from 2026-07-03 — all preserved
5. Created `.gitignore` (was missing) excluding corrupted `yolov8n.pt`, base weights, generated datasets (`data_v2-4/`), training artifacts, caches, `*.png`, `*.pt`, `.env`, `*.log`
6. Committed as `bb31637` — "Recover repository after filesystem corruption; commit v2 multi-scale + WBF detection improvements"
7. Pushed fast-forward (`c93914f..bb31637`) — NO force-push needed
8. Removed `.git.corrupt-backup/` after verification

## Verification
- `git fsck --full`: clean
- `git status`: up to date with origin/main, ahead/behind = 0/0
- `git log`: bb31637 → c93914f (clean lineage)
- Test suite: 119 passed
- Preview URL: HTTP 200

## Key lessons
- Remote `origin` saved the commit lineage; working tree on disk was the source of truth for uncommitted work
- `yolov8n.pt` at workspace root remains filesystem-corrupted and CANNOT be deleted/renamed/stat'd — it is in `.gitignore` permanently
- `.gitignore` now exists (was missing entirely before this incident)
