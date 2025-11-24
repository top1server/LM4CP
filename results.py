from collections import defaultdict
from datasets import load_dataset

RESULTS_DATASET_ID = "tiendung6b/qwen3-4b-instruct-2507-ioi_2024_8k-results"
RESULTS_SPLIT = "train"

def build_results_by_seed():
    print(f"Loading results dataset: {RESULTS_DATASET_ID} (split={RESULTS_SPLIT})")
    ds = load_dataset(RESULTS_DATASET_ID, split=RESULTS_SPLIT)
    problems = defaultdict(lambda: defaultdict(dict))

    for row in ds:
        year = row.get("year")
        pid = row.get("problem_id")
        pname = row.get("problem_name")
        subtask = row.get("target_subtask") or row.get("subtask")
        seed = row.get("seed", 0)
        key = (year, pid, pname)
        score = row.get("target_subtask_score")
        status = row.get("target_subtask_status")
        subtask_points = None
        subtask_max_points = None
        all_results = row.get("all_subtasks_results") or []
        for info in all_results:
            if info.get("subtask") == subtask:
                subtask_points = info.get("points")
                s = info.get("score")
                if s not in (None, 0):
                    try:
                        subtask_max_points = subtask_points / s
                    except TypeError:
                        subtask_max_points = None
                break

        problems[key][subtask][seed] = {
            "points": subtask_points,
            "max_points": subtask_max_points,
            "score": score,
            "status": status,
        }

    return problems


def print_results_by_seed(problems):
    for (year, pid, pname), subtask_map in sorted(problems.items()):
        print("=" * 60)
        print(f"IOI {year} – {pname}")
        print()

        for subtask_name in sorted(subtask_map.keys()):
            seeds_map = subtask_map[subtask_name]

            print(f"  [{subtask_name}]")
            print("    seed    pts  /  max    score%   status")
            print("    --------------------------------------")

            for seed in sorted(seeds_map.keys()):
                v = seeds_map[seed]
                pts = v["points"]
                mpts = v["max_points"]
                score = v["score"]
                status = v["status"]

                pts_str = f"{pts:.1f}" if isinstance(pts, (int, float)) else "  ?  "
                mpts_str = f"{mpts:.1f}" if isinstance(mpts, (int, float)) else "0.0"
                score_pct = (
                    f"{(score * 100):5.1f}%" if isinstance(score, (int, float)) else "  N/A "
                )

                print(
                    f"    {seed:4d}  {pts_str:>5}  /  {mpts_str:<5}  {score_pct}     {status}"
                )

            best_seed = None
            best_score = -1.0
            for seed, v in seeds_map.items():
                s = v["score"]
                s_val = float(s) if isinstance(s, (int, float)) else 0.0
                if s_val > best_score:
                    best_score = s_val
                    best_seed = seed

            if best_seed is not None:
                print(
                    f"  --> best seed for [{subtask_name}]: "
                    f"{best_seed} (score = {best_score * 100:.1f}%)"
                )

            print()

        print()

if __name__ == "__main__":
    problems = build_results_by_seed()
    print_results_by_seed(problems)
