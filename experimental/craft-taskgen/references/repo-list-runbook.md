# Repo List Runbook

How to add repos (excluding excluded-repos.csv), filter by license, and maintain the repo list.

All commands assume you are running from the repo root (`craft-taskgen/`).

---

## Managed pipeline: add / remove repos (`manage_repo_list.py`)

**This is the recommended way to update the repo list.** Both `add` and `remove` always read from the latest `repo_list_v{N}.csv` and write to a new `repo_list_v{N+1}.csv` — the source is never modified. Source and output paths are always printed explicitly.

**Add a single repo:**
```bash
python3 scripts/manage_repo_list.py add {https://github.com/owner/repo}
```

**Add from a URL list file** (most file types supported including `.txt`, `.csv` — detected by content: if the first line starts with `http` it's treated as a URL list, blank lines and non-`http` lines are skipped):
```bash
python3 scripts/manage_repo_list.py add {urls.txt}
```

**Add from a pre-fetched CSV** (file whose first line is a header, not a URL — skips the `gh` fetch step):
```bash
python3 scripts/manage_repo_list.py add {repos.csv}
```

**Remove by short_name or repo slug:**
```bash
python3 scripts/manage_repo_list.py remove {short_name e.g. jinja}
python3 scripts/manage_repo_list.py remove {owner/repo e.g. pallets/jinja}
```

**Remove a batch from a text file** (one short_name or slug per line):
```bash
python3 scripts/manage_repo_list.py remove {remove.txt}
```

**Use a specific version as the source** (reads from `v{N}`, writes to `v{N+1}`):
```bash
python3 scripts/manage_repo_list.py --list references/{repo_list_vN.csv} add {https://github.com/owner/repo}
```

**Write output to a specific path** (default: `references/repo_list_v{N+1}.csv`):
```bash
python3 scripts/manage_repo_list.py --out {output.csv} add {https://github.com/owner/repo}
```

**Override the excluded repos list** (default: `references/excluded-repos.csv`):
```bash
python3 scripts/manage_repo_list.py --exclude {excluded.csv} add {https://github.com/owner/repo}
```

---

## Standalone dedup + exclude check

Remove duplicates and any repos in `references/excluded-repos.csv` from any CSV:

```bash
# In-place (overwrites input)
python3 scripts/dedup_repos.py references/{source_file.csv}

# Write to a new file
python3 scripts/dedup_repos.py references/{source_file.csv} --out references/{output_file.csv}

# Override the excluded list
python3 scripts/dedup_repos.py references/{source_file.csv} --exclude {excluded.csv}
```

---

## Filter by approved licenses

Produce a filtered copy that only includes permissive/approved licenses:

```bash
python3 scripts/filter_by_license.py references/{source_file.csv}
# Output: references/{source_file}-filtered.csv
# Stdout shows: how many kept, and any excluded license names
```

To write to a specific path:

```bash
python3 scripts/filter_by_license.py references/{source_file.csv} --out references/{output_file.csv}
```

To also see which individual repos were excluded:

```bash
python3 scripts/filter_by_license.py references/{source_file.csv} --list-excluded
```

**Approved licenses:** MIT, Apache-2.0, BSD-2-Clause, BSD-2, BSD-3-Clause, BSD-3, ISC, Zlib, CC0, CC0-1.0, LGPL-2.1, LGPL-3.0, MPL-2.0, EPL-2.0, BSL-1.0, GPL-3.0

---

## Typical workflow: add repos → filter

```bash
# 1. Add (fetches, dedups, exclude-checks automatically) — writes to references/repo_list_v{N+1}.csv
python3 scripts/manage_repo_list.py add {urls.txt}

# 2. Filter to approved licenses
python3 scripts/filter_by_license.py references/{source_file.csv} --out references/{output_file.csv}
```

---

## CSV format

All repo list CSVs share the same columns:

| Column | Example |
|---|---|
| `short_name` | `jinja` |
| `github_repo` | `pallets/jinja` |
| `github_url` | `https://github.com/pallets/jinja` |
| `stars` | `10500` |
| `license` | `BSD-3-Clause` |
| `domain` | `Web/API` |
| `description` | `A very fast and expressive template engine` |

> **Note:** `fetch_repo_details.py` and `manage_repo_list.py` auto-classify `domain` using keyword rules. Check and correct manually if the classification looks wrong (e.g. a templating library classified as `Data/DB`).
