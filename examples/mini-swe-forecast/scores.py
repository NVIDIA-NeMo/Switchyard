# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Langfuse numeric scores; predictions never write actual cost fields."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def score_rows(record):
    rows = []
    for target, result in record["predictions"].items():
        values = {k: v for k, v in result.items() if isinstance(v, (int, float))}
        if "probabilities" in result:
            values.update(
                {f"probability_class_{i}": p for i, p in enumerate(result["probabilities"])}
            )
            values["upper_unbounded"] = float(result["upper_exclusive"] is None)
        for field, value in values.items():
            name = f"prediction.{target}.{field}"
            rows.append(
                {
                    "id": hashlib.sha256((record["trace_id"] + "/" + name).encode()).hexdigest(),
                    "traceId": record["trace_id"],
                    "observationId": record["forecast_span_id"],
                    "name": name,
                    "value": float(value),
                    "dataType": "NUMERIC",
                    "comment": "LightGBM forecast, not actual usage or billed cost.",
                    "metadata": {
                        "run_id": record["run_id"],
                        "call_index": record["call_index"],
                        "checkpoint": record["checkpoint"],
                        "source_phase": record["source_ir"]["phase"],
                    },
                }
            )
    return rows


class ScoreClient:
    def __init__(self):
        self.url = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
        credential = os.environ["LANGFUSE_PUBLIC_KEY"] + ":" + os.environ["LANGFUSE_SECRET_KEY"]
        self.headers = {
            "Authorization": "Basic " + base64.b64encode(credential.encode()).decode(),
            "Content-Type": "application/json",
        }

    def send(self, record):
        rows = score_rows(record)
        request = Request(
            self.url + "/api/public/scores",
            data=json.dumps(rows).encode(),
            headers=self.headers,
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
        return {"trace_id": record["trace_id"], "submitted_scores": len(rows), "response": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, required=True)
    args = parser.parse_args()
    client = ScoreClient()
    receipts = []
    for line in (args.run_folder / "forecasts.jsonl").read_text().splitlines():
        record = json.loads(line)
        receipts.append(client.send(record))
    (args.run_folder / "score-export-receipt.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "calls": len(receipts),
                "submitted_scores": sum(r["submitted_scores"] for r in receipts),
            }
        )
    )


if __name__ == "__main__":
    main()
