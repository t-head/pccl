#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare per-rank uploaded plans and channel batches from PCCL work dumps.

Use one communicator per rank and separate log directories for each build.
Descriptor offsets intentionally do not participate in the comparison: shared
work changes offsets without necessarily changing the channel execution plan.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


HEADER = re.compile(
    r"channel: (\d+), blkIdx (\d+), rank: (\d+), kernelIdx: (\d+), "
    r"channelMask ([0-9a-f]+) channelWorks (\d+) totalPreChannelWorks \d+ "
    r"batchIx (\d+) workType:(\d+), funcId:(\d+)\([^)]*\), "
    r"nextJump:(\d+), nextExtends:(\d+) offsetBase:([0-9a-f]+),? "
    r"offsetBitset:([0-9a-f]+)"
)


def read_plans(directory):
    batches = defaultdict(dict)
    for path in sorted(directory.glob("*.log")):
        for match in HEADER.finditer(path.read_text(errors="strict")):
            values = match.groups()
            channel, block, rank, kernel = map(int, values[:4])
            mask, works = int(values[4], 16), int(values[5])
            batch, kind, func, jump, extends = map(int, values[6:11])
            key = (rank, kernel, channel)
            if batch in batches[key]:
                raise ValueError(f"duplicate batch {key}/{batch}; use separate per-rank logs")
            batches[key][batch] = {
                "block": block, "mask": mask, "works": works, "kind": kind,
                "func": func, "jump": jump, "extends": extends,
                "count": bin(int(values[12], 16)).count("1"),
            }
    if not batches:
        raise ValueError(f"no work dumps in {directory}")

    plans = defaultdict(lambda: defaultdict(dict))
    for (rank, kernel, channel), entries in sorted(batches.items()):
        first = next(iter(entries.values()))
        index, pending, seen, groups = first["block"], 0, set(), []
        group_type = None
        while True:
            if index in seen or index not in entries:
                raise ValueError(f"broken batch chain {(rank, kernel, channel)}/{index}")
            seen.add(index)
            entry = entries[index]
            if any(entry[field] != first[field] for field in ("block", "mask", "works")):
                raise ValueError("inconsistent channel header")
            current_type = (entry["kind"], entry["func"])
            if pending and current_type != group_type:
                raise ValueError("extension changes work type/function")
            group_type = current_type
            pending += entry["count"]
            if not entry["extends"]:
                groups.append({"work_type": group_type[0], "func_id": group_type[1], "nWorks": pending})
                pending = 0
            if entry["jump"] == 0:
                if entry["extends"]:
                    raise ValueError("unterminated extension")
                break
            index += entry["jump"]
        if seen != set(entries) or sum(g["nWorks"] for g in groups) != first["works"]:
            raise ValueError("incomplete channel work dump")
        plans[rank][kernel][channel] = {
            "channel_mask": first["mask"], "channelWorks": first["works"], "batches": groups,
        }
    return {str(rank): [channels for _, channels in sorted(kernels.items())]
            for rank, kernels in sorted(plans.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--ranks", type=int, required=True)
    args = parser.parse_args()
    before, after = read_plans(args.before), read_plans(args.after)
    expected = {str(rank) for rank in range(args.ranks)}
    if set(before) != expected or set(after) != expected:
        raise ValueError("missing or unexpected ranks")
    result = {
        "same_execution_structure": before == after,
        "before_uploaded_plans_per_rank": {rank: len(plans) for rank, plans in before.items()},
        "after_uploaded_plans_per_rank": {rank: len(plans) for rank, plans in after.items()},
        "before": before,
        "after": after,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if before == after else 1


if __name__ == "__main__":
    raise SystemExit(main())
