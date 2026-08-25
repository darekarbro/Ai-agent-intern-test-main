"""
Order lookup tool.

This is the only way the agent can learn anything about an order. The full
`orders.json` is never placed in a prompt; this module loads it once and
exposes a single callable, allow-listed function.

Field allow-list and status rules come from `data/orders-data-dictionary.md`:
  - Never return customer.name, customer.email, customer.shipping_address,
    or anything under `internal`.
  - status is authoritative; suppress stale delivery-estimate fields when
    status is cancelled or returned.
  - status == exception -> flag for human handoff, no invented explanation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

ALLOWED_ORDER_FIELDS = {
    "order_id",
    "membership_tier",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
}
ALLOWED_ITEM_FIELDS = {"name", "quantity", "final_sale"}

STALE_ON_TERMINAL_STATUS = {"carrier", "tracking_number", "estimated_delivery"}
TERMINAL_STATUSES_SUPPRESS_ETA = {"cancelled", "returned"}

_ORDER_ID_RE = re.compile(r"^ORD-\d+$")


@dataclass
class OrderLookupResult:
    found: bool
    order_id_queried: str
    normalized_id: Optional[str] = None
    data: Optional[dict] = None
    error: Optional[str] = None  # "not_found" | "malformed"
    needs_handoff: bool = False


def normalize_order_id(raw: str) -> Optional[str]:
    """Strip whitespace/punctuation noise and uppercase. Returns None if the
    result doesn't look like a plausible order ID at all (caller treats that
    as malformed rather than guessing)."""
    if not raw:
        return None
    candidate = raw.strip().upper()
    candidate = re.sub(r"\s+", "", candidate)
    # Tolerate a stray trailing/leading punctuation like "ORD-1007." or "(ORD-1007)"
    candidate = candidate.strip(".,;:()[]{}!?")
    if not candidate.startswith("ORD"):
        return None
    # Normalize "ORD1007" / "ORD_1007" -> "ORD-1007"
    m = re.match(r"^ORD[-_]?(\d+)$", candidate)
    if not m:
        return None
    return f"ORD-{m.group(1)}"


class OrderLookupTool:
    def __init__(self, orders_path: str):
        with open(orders_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.snapshot_at = raw.get("snapshot_at")
        self._orders_by_id = {o["order_id"]: o for o in raw.get("orders", [])}

    def lookup(self, order_id_raw: str) -> OrderLookupResult:
        normalized = normalize_order_id(order_id_raw)
        if normalized is None or not _ORDER_ID_RE.match(normalized):
            return OrderLookupResult(
                found=False,
                order_id_queried=order_id_raw,
                error="malformed",
                needs_handoff=False,
            )

        order = self._orders_by_id.get(normalized)
        if order is None:
            return OrderLookupResult(
                found=False,
                order_id_queried=order_id_raw,
                normalized_id=normalized,
                error="not_found",
                needs_handoff=True,
            )

        sanitized = self._sanitize(order)
        needs_handoff = sanitized["status"] == "exception"
        return OrderLookupResult(
            found=True,
            order_id_queried=order_id_raw,
            normalized_id=normalized,
            data=sanitized,
            needs_handoff=needs_handoff,
        )

    def _sanitize(self, order: dict) -> dict:
        out = {k: v for k, v in order.items() if k in ALLOWED_ORDER_FIELDS}

        items = []
        for item in order.get("items", []):
            items.append({k: v for k, v in item.items() if k in ALLOWED_ITEM_FIELDS})
        out["items"] = items

        if out.get("status") in TERMINAL_STATUSES_SUPPRESS_ETA:
            for f in STALE_ON_TERMINAL_STATUS:
                out.pop(f, None)
            out["_stale_fields_suppressed"] = sorted(STALE_ON_TERMINAL_STATUS)

        # Defense in depth: guarantee no internal/customer-identifying key can
        # ever leak even if the allow-list above is edited incorrectly later.
        forbidden_leaked = {"customer", "internal"} & set(out.keys())
        assert not forbidden_leaked, f"forbidden fields leaked: {forbidden_leaked}"

        return out


TOOL_SCHEMA = {
    "name": "order_lookup",
    "description": (
        "Look up the current status of a customer order by order ID. "
        "Returns only customer-safe fields. Use this whenever the user asks "
        "about the status, tracking, or delivery of a specific order."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID as the customer provided it, e.g. 'ORD-1007' or 'ord 1007'.",
            }
        },
        "required": ["order_id"],
    },
}


if __name__ == "__main__":
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "data", "orders.json")
    tool = OrderLookupTool(path)
    for q in ["ORD-1007", "  ord-1004 ", "ORD-9999", "not-an-id", "ord1011"]:
        r = tool.lookup(q)
        print(q, "->", r)
