"""
Download all training history and media from W&B projects
using the GraphQL + REST API directly.
Saves to mrbert/analysis/wandb_plots/<project>/<run_name>/
"""

import requests
import json
import os
import csv
from pathlib import Path

KEY = os.environ["WANDB_API_KEY"]

ENTITY = "aronima7-stanford-university"
OUTPUT_DIR = Path("mrbert/analysis/wandb_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
}

def gql(query, variables=None):
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    r = requests.post("https://api.wandb.ai/graphql",
                      data=json.dumps(payload), headers=HEADERS, timeout=30)
    return r.json()

# ── 1. List all projects ──────────────────────────────────────────
print(f"Fetching projects for {ENTITY}...")
data = gql(f'{{ projects(entityName: "{ENTITY}", first: 50) {{ edges {{ node {{ name id }} }} }} }}')
projects = [e["node"] for e in data["data"]["projects"]["edges"]]
print(f"Found {len(projects)} projects: {[p['name'] for p in projects]}")

all_summary_rows = []

for proj in projects:
    proj_name = proj["name"]
    print(f"\n{'='*60}\nProject: {proj_name}\n{'='*60}")
    proj_dir = OUTPUT_DIR / proj_name
    proj_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. List runs ─────────────────────────────────────────────
    runs_query = """
    query($entity: String!, $project: String!, $cursor: String) {
      project(name: $project, entityName: $entity) {
        runs(first: 100, after: $cursor) {
          edges {
            node {
              id name displayName state createdAt
              config summaryMetrics
              tags
            }
          }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """
    cursor = None
    all_runs = []
    while True:
        result = gql(runs_query, {"entity": ENTITY, "project": proj_name, "cursor": cursor})
        proj_data = result.get("data", {}).get("project")
        if not proj_data:
            break
        runs_page = proj_data["runs"]
        all_runs.extend([e["node"] for e in runs_page["edges"]])
        if runs_page["pageInfo"]["hasNextPage"]:
            cursor = runs_page["pageInfo"]["endCursor"]
        else:
            break

    print(f"  {len(all_runs)} runs")

    for run in all_runs:
        run_id = run["id"]
        run_name = run.get("displayName") or run.get("name") or run_id
        safe_name = run_name.replace("/", "_").replace(" ", "_")
        run_dir = proj_dir / f"{safe_name}_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Run: {run_name} ({run_id}) state={run.get('state')}")

        # Parse config and summary
        try:
            config = json.loads(run.get("config") or "{}")
            config = {k: v.get("value", v) if isinstance(v, dict) else v
                      for k, v in config.items()}
        except Exception:
            config = {}
        try:
            summary = json.loads(run.get("summaryMetrics") or "{}")
        except Exception:
            summary = {}

        # Save meta
        meta = {
            "id": run_id, "name": run_name, "state": run.get("state"),
            "project": proj_name, "entity": ENTITY,
            "url": f"https://wandb.ai/{ENTITY}/{proj_name}/runs/{run_id}",
            "config": config,
            "summary": {k: v for k, v in summary.items() if not k.startswith("_")},
        }
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        row = {"project": proj_name, "run_name": run_name, "run_id": run_id,
               "state": run.get("state"), "url": meta["url"]}
        row.update({k: v for k, v in summary.items()
                    if not k.startswith("_") and not isinstance(v, (dict, list))})
        all_summary_rows.append(row)

        # ── 3. Download history (training curves) ────────────────
        history_query = """
        query($entity: String!, $project: String!, $run: String!, $samples: Int) {
          project(name: $project, entityName: $entity) {
            run(name: $run) {
              history(samples: $samples)
            }
          }
        }
        """
        try:
            hr = gql(history_query, {
                "entity": ENTITY, "project": proj_name,
                "run": run_id, "samples": 2000
            })
            history_raw = hr.get("data", {}).get("project", {}).get("run", {}).get("history", "[]")
            history = json.loads(history_raw) if isinstance(history_raw, str) else history_raw
            if history:
                with open(run_dir / "history.json", "w") as f:
                    json.dump(history, f)
                print(f"    history: {len(history)} steps, keys: {list(history[0].keys())[:6]}")
        except Exception as e:
            print(f"    history error: {e}")

        # ── 4. Download files (media/images) ─────────────────────
        files_query = """
        query($entity: String!, $project: String!, $run: String!) {
          project(name: $project, entityName: $entity) {
            run(name: $run) {
              files(first: 100) {
                edges { node { name url sizeBytes } }
              }
            }
          }
        }
        """
        try:
            fr = gql(files_query, {"entity": ENTITY, "project": proj_name, "run": run_id})
            files = [e["node"] for e in
                     fr.get("data", {}).get("project", {}).get("run", {})
                       .get("files", {}).get("edges", [])]
            media_exts = {".png", ".jpg", ".jpeg", ".svg", ".html", ".pdf", ".csv"}
            media_files = [f for f in files
                           if any(f["name"].endswith(ext) for ext in media_exts)
                           or "media/" in f["name"]]
            if media_files:
                media_dir = run_dir / "media"
                media_dir.mkdir(exist_ok=True)
                downloaded = 0
                for finfo in media_files:
                    try:
                        fname = Path(finfo["name"]).name
                        file_resp = requests.get(finfo["url"],
                                                 headers={"Authorization": f"Bearer {KEY}"},
                                                 timeout=30)
                        if file_resp.status_code == 200:
                            with open(media_dir / fname, "wb") as fout:
                                fout.write(file_resp.content)
                            downloaded += 1
                    except Exception:
                        pass
                if downloaded:
                    print(f"    media: {downloaded} files downloaded")
        except Exception as e:
            print(f"    files error: {e}")

# ── 5. Write global summary CSV ──────────────────────────────────
all_keys = set()
for row in all_summary_rows:
    all_keys.update(row.keys())
sorted_keys = sorted(all_keys)

csv_path = OUTPUT_DIR / "all_runs_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sorted_keys)
    writer.writeheader()
    for row in all_summary_rows:
        writer.writerow({k: row.get(k, "") for k in sorted_keys})

print(f"\n\nDone! {len(all_summary_rows)} runs across {len(projects)} projects.")
print(f"Summary CSV: {csv_path}")
print(f"Plots saved to: {OUTPUT_DIR}/")
