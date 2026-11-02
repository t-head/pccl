# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
import json


def final_response(output, *, allow_tools=False):
    # CLI 1.0.100 can mis-escape the user.message echo; it is never a response.
    events = [json.loads(line) for line in output.splitlines()
              if line.strip() and not line.lstrip().startswith('{"type":"user.message",')]
    if not allow_tools and any(event.get("type") == "tool.execution_start" or
                               event.get("data", {}).get("toolRequests") for event in events):
        raise ValueError("Copilot must not execute tools")
    replies = [event["data"]["content"] for event in events
               if event.get("type") == "assistant.message" and
               not event.get("data", {}).get("toolRequests") and
               event.get("data", {}).get("content", "").strip()]
    if not replies:
        raise ValueError("Copilot did not return a final response")
    return replies[-1]
