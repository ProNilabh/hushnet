"""S3 versus C6 benchmark table from the on-device replay test.

    python benchmark.py COM5 COM6            (one or more boards running hushnet_poc)

For every port it sends INFO and INFER, then writes benchmark.csv and prints a Markdown table
with accuracy on the replay clips, agreement with the PC, feature time, inference time and memory.
"""
import argparse
import csv
import sys

from hn_serial import Board, parse_line

COLUMNS = [
    ("board", "Board"), ("chip", "Chip"), ("cores", "Cores"), ("cpu_mhz", "CPU MHz"), ("clips", "Clips"),
    ("correct", "Correct"), ("same_as_pc", "Same class as PC"), ("feat_maxdiff", "Max feature diff"),
    ("score_maxdiff", "Max score diff"), ("feat_ms_avg", "Features ms"), ("infer_ms_avg", "Inference ms"),
    ("total_ms_avg", "Total ms"), ("model_bytes", "Model bytes"), ("arena_used", "Arena bytes"),
    ("sketch_kb", "Sketch KB"), ("heap_free", "Free heap"), ("heap_min", "Min free heap"),
]


def run(port):
    b = Board(port)
    info = b.hello()
    b.drain()
    b.send("INFER")
    row = dict(info)
    feat_max, score_max = 0, 0
    while True:
        item = b.read(30.0)
        if item is None:
            b.close()
            sys.exit(f"{port}: no INFER_SUMMARY within 30 s. Was the sketch built with ENABLE_ML 1?")
        tag, f = parse_line(item[1])
        print(f"[{port}] {item[1]}")
        if tag == "ERR":
            b.close()
            sys.exit(f"{port}: {item[1]}")
        if tag == "INFER":
            feat_max = max(feat_max, int(f.get("feat_maxdiff", 0)))
            score_max = max(score_max, int(f.get("score_maxdiff", 0)))
        if tag == "INFER_SUMMARY":
            row.update(f)
            break
    row["feat_maxdiff"], row["score_maxdiff"] = feat_max, score_max
    row["board"] = info.get("board", port)
    b.close()
    return row


def main():
    ap = argparse.ArgumentParser(description="Replay benchmark on one or more boards")
    ap.add_argument("ports", nargs="+")
    ap.add_argument("--out", default="benchmark.csv")
    a = ap.parse_args()
    rows = [run(p) for p in a.ports]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k, _ in COLUMNS], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print()
    print("| Metric | " + " | ".join(str(r.get("board", "")) for r in rows) + " |")
    print("|---|" + "---|" * len(rows))
    for key, label in COLUMNS[1:]:
        print(f"| {label} | " + " | ".join(str(r.get(key, "")) for r in rows) + " |")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
