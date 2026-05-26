"""
B6 — Absolute click/selection rate percentages (CIKM 2026 resubmission).

Parses the raw user-study SQL dumps to compute:
  - total OPEN_DOC events (= clicks)
  - total PASSAGE_SELECTION events (= selections)
  - distinct (user, query) screens shown
  - cell-opportunities = #screens × 9

We filter to *valid* participants only — those who pass BOT_DETECTION (pass_flag=1).
Both studies have exactly 54 valid participants, matching the paper.

Inputs (final per-study SQL backups, identified by event-count match to paper):
  Study 1: railway_database_backup_v1.sql   (7088 selections / 7879 clicks ✓)
  Study 2: railway_database_backup_v2.sql   (7872 selections / 8848 clicks ✓)

Outputs:
  - b6_per_study_rates.csv

Reports two rate variants (the paper can pick whichever it prefers):
  (i)  raw event rate          = #events / cell-opps  (a single cell can be opened twice)
  (ii) unique-cell engagement  = #distinct (user, qid, docno) cells touched / cell-opps
                                  (bounded 0–100%)

For B6's purpose (absolute context for selection percentages), the raw event rate
is the most consistent with the existing paper text on line 544 which reports raw
event counts ("7,088 passage selections", "7,879 clicks").
"""

import os
import re
import pandas as pd

SQL_PATHS = {
    "Study 1": "/mnt/primary/Trec-llm/utils/case_study_rankings/railway_database_backup_v1.sql",
    "Study 2": "/mnt/primary/Trec-llm/utils/case_study_rankings/railway_database_backup_v2.sql",
}
OUT_DIR = "/mnt/primary/cikm 2026/experiments"
CELLS_PER_SCREEN = 9


LOGS_BLOCK_RE = re.compile(r"INSERT INTO `logs` VALUES (.*?);", re.DOTALL)
LOGS_ROW_RE = re.compile(
    r"\(\s*"
    r"(\d+),"                     # id
    r"'([^']*)',"                 # user_id
    r"(\d+),"                     # qid
    r"'([^']*)',"                 # docno
    r"'([^']*)',"                 # event_type
    r"(-?\d+),"                   # start_idx
    r"(-?\d+),"                   # end_idx
    r"(\d+),"                     # duration
    r"(\d+),"                     # pass_flag
    r"'([^']*)'"                  # timestamp
    r"\)"
)


def parse_logs(sql_path):
    with open(sql_path, "r", encoding="utf-8", errors="replace") as f:
        sql = f.read()
    rows = []
    for block in LOGS_BLOCK_RE.findall(sql):
        for m in LOGS_ROW_RE.finditer(block):
            rows.append(m.groups())
    df = pd.DataFrame(rows, columns=[
        "id", "user_id", "qid", "docno", "event_type",
        "start_idx", "end_idx", "duration", "pass_flag", "timestamp",
    ])
    df["qid"] = df["qid"].astype(int)
    df["pass_flag"] = df["pass_flag"].astype(int)
    return df


def main():
    rows_summary = []
    for study, path in SQL_PATHS.items():
        print(f"\n=== {study} : {os.path.basename(path)} ===")
        logs = parse_logs(path)

        # Valid participants = those with a successful BOT_DETECTION (pass_flag=1)
        bot = logs[(logs["event_type"] == "BOT_DETECTION") & (logs["pass_flag"] == 1)]
        valid_users = set(bot["user_id"])
        n_users = len(valid_users)
        print(f"  Valid participants (BOT_DETECTION pass_flag=1): {n_users}")

        df_v = logs[logs["user_id"].isin(valid_users)]

        # Distinct (user, query) screens — each screen = 9 cells
        screens = df_v[["user_id", "qid"]].drop_duplicates()
        n_screens = len(screens)
        n_screens_per_user = screens.groupby("user_id").size()
        n_unique_queries = df_v["qid"].nunique()
        cell_opps = n_screens * CELLS_PER_SCREEN

        # Event counts
        n_open = int((df_v["event_type"] == "OPEN_DOC").sum())
        n_sel = int((df_v["event_type"] == "PASSAGE_SELECTION").sum())

        # Unique cells touched
        unique_clicked = df_v.loc[
            df_v["event_type"] == "OPEN_DOC", ["user_id", "qid", "docno"]
        ].drop_duplicates()
        unique_selected = df_v.loc[
            df_v["event_type"] == "PASSAGE_SELECTION", ["user_id", "qid", "docno"]
        ].drop_duplicates()

        rec = {
            "study": study,
            "n_valid_users": n_users,
            "n_unique_queries": n_unique_queries,
            "n_screens": n_screens,
            "screens_per_user_mean": float(n_screens_per_user.mean()),
            "screens_per_user_min": int(n_screens_per_user.min()),
            "screens_per_user_max": int(n_screens_per_user.max()),
            "cells_per_screen": CELLS_PER_SCREEN,
            "cell_opportunities": cell_opps,
            "n_open_doc_events": n_open,
            "n_passage_selection_events": n_sel,
            "n_unique_clicked_cells": int(len(unique_clicked)),
            "n_unique_selected_cells": int(len(unique_selected)),
            "click_rate_raw_events_pct": 100.0 * n_open / cell_opps,
            "click_rate_unique_pct": 100.0 * len(unique_clicked) / cell_opps,
            "selection_rate_raw_events_pct": 100.0 * n_sel / cell_opps,
            "selection_rate_unique_pct": 100.0 * len(unique_selected) / cell_opps,
        }
        rows_summary.append(rec)

        print(f"  Distinct queries seen: {n_unique_queries}")
        print(f"  Distinct (user, query) screens: {n_screens}")
        print(f"  Screens per user: mean={rec['screens_per_user_mean']:.2f}, "
              f"min={rec['screens_per_user_min']}, max={rec['screens_per_user_max']}")
        print(f"  Cell-opportunities = {n_screens} × {CELLS_PER_SCREEN} = {cell_opps}")
        print(f"  OPEN_DOC events: {n_open} (unique cells: {len(unique_clicked)})")
        print(f"  PASSAGE_SELECTION events: {n_sel} (unique cells: {len(unique_selected)})")
        print(f"  Click rate (raw events):     {rec['click_rate_raw_events_pct']:.2f}%")
        print(f"  Click rate (unique cells):   {rec['click_rate_unique_pct']:.2f}%")
        print(f"  Selection rate (raw events): {rec['selection_rate_raw_events_pct']:.2f}%")
        print(f"  Selection rate (unique cells): {rec['selection_rate_unique_pct']:.2f}%")

    df = pd.DataFrame(rows_summary)
    out_path = os.path.join(OUT_DIR, "b6_per_study_rates.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Pretty summary
    print("\n" + "=" * 78)
    print("FINAL B6 RATES (summary)")
    print("=" * 78)
    cols = ["study", "n_valid_users", "n_screens", "cell_opportunities",
            "n_open_doc_events", "n_passage_selection_events",
            "click_rate_raw_events_pct", "click_rate_unique_pct",
            "selection_rate_raw_events_pct", "selection_rate_unique_pct"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
