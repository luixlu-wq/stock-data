"""
Utilities to build daily trade-operation strategy payloads from predictions.

The payload structure is designed for persistence in Qdrant and later
verification against realized next-day returns.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


K = 38
LONG_EXPOSURE = 0.65
SHORT_EXPOSURE = 0.35


def _safe_float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    return out.dropna(subset=["date"])


def build_recommendations_for_date(
    date_str: str,
    pred_df: pd.DataFrame,
    k: int = K,
    long_exposure: float = LONG_EXPOSURE,
    short_exposure: float = SHORT_EXPOSURE,
) -> list[dict]:
    """
    Apply S2_FilterNegative selection for one date.

    Returns per-ticker records aligned with stock_recommendations payload schema.
    """
    day = pred_df[pred_df["date"] == date_str].copy()
    if day.empty:
        return []

    day = day.sort_values("y_pred_reg", ascending=False)

    longs = day.head(k)
    long_weight = long_exposure / len(longs)

    short_candidates = day.tail(k)
    shorts = short_candidates[short_candidates["y_pred_reg"] < 0]
    short_weight = -short_exposure / len(shorts) if len(shorts) > 0 else 0.0

    records: list[dict] = []
    for _, row in longs.iterrows():
        records.append(
            {
                "ticker": str(row["ticker"]),
                "position": "LONG",
                "weight": float(long_weight),
                "prediction": float(row["y_pred_reg"]),
                "actual_return": _safe_float_or_none(row.get("y_true_reg")),
                "close_price": _safe_float_or_none(row.get("close")),
                "date": date_str,
            }
        )

    for _, row in shorts.iterrows():
        records.append(
            {
                "ticker": str(row["ticker"]),
                "position": "SHORT",
                "weight": float(short_weight),
                "prediction": float(row["y_pred_reg"]),
                "actual_return": _safe_float_or_none(row.get("y_true_reg")),
                "close_price": _safe_float_or_none(row.get("close")),
                "date": date_str,
            }
        )
    return records


def _compute_direction_correct(position: str, actual_return: float | None) -> bool | None:
    if actual_return is None:
        return None
    if position == "LONG":
        return actual_return >= 0
    return actual_return <= 0


def _compute_realized_proxy(position: str, weight: float | None, actual_return: float | None) -> float | None:
    if weight is None or actual_return is None:
        return None
    w = abs(float(weight))
    if position == "LONG":
        return w * float(actual_return)
    return w * (-float(actual_return))


def _build_action_from_rec(action: str, rec: dict, note: str | None = None) -> dict:
    position = str(rec["position"])
    actual = _safe_float_or_none(rec.get("actual_return"))
    weight = _safe_float_or_none(rec.get("weight"))
    direction_correct = _compute_direction_correct(position, actual)
    realized_proxy = _compute_realized_proxy(position, weight, actual)
    return {
        "ticker": rec["ticker"],
        "action": action,
        "position": position,
        "prediction": _safe_float_or_none(rec.get("prediction")),
        "weight": weight,
        "actual_return": actual,
        "close_price": _safe_float_or_none(rec.get("close_price")),
        "direction_correct": direction_correct,
        "realized_pnl_proxy": realized_proxy,
        "note": note,
    }


def _build_action_exit(action: str, rec: dict) -> dict:
    return {
        "ticker": rec["ticker"],
        "action": action,
        "position": rec["position"],
        "prediction": _safe_float_or_none(rec.get("prediction")),
        "weight": _safe_float_or_none(rec.get("weight")),
        "actual_return": None,
        "close_price": _safe_float_or_none(rec.get("close_price")),
        "direction_correct": None,
        "realized_pnl_proxy": None,
        "note": "Removed from current recommendations",
    }


def _previous_date(date_str: str, all_dates_sorted: list[str]) -> str | None:
    for i, d in enumerate(all_dates_sorted):
        if d == date_str:
            return all_dates_sorted[i - 1] if i > 0 else None
    return None


def build_trade_strategy_payload(
    date_str: str,
    pred_df: pd.DataFrame,
    previous_date: str | None = None,
    k: int = K,
    long_exposure: float = LONG_EXPOSURE,
    short_exposure: float = SHORT_EXPOSURE,
) -> dict | None:
    """
    Build one per-date strategy payload with action list and evaluation metrics.
    """
    df = _normalize_date_column(pred_df)
    curr_records = build_recommendations_for_date(date_str, df, k, long_exposure, short_exposure)
    if not curr_records:
        return None

    all_dates = sorted(df["date"].unique().tolist())
    prev_date = previous_date if previous_date else _previous_date(date_str, all_dates)
    prev_records = build_recommendations_for_date(prev_date, df, k, long_exposure, short_exposure) if prev_date else []

    curr_long = {r["ticker"]: r for r in curr_records if r["position"] == "LONG"}
    curr_short = {r["ticker"]: r for r in curr_records if r["position"] == "SHORT"}
    prev_long = {r["ticker"]: r for r in prev_records if r["position"] == "LONG"}
    prev_short = {r["ticker"]: r for r in prev_records if r["position"] == "SHORT"}

    buy: list[dict] = []
    sell: list[dict] = []
    open_short: list[dict] = []
    cover_short: list[dict] = []
    hold_long: list[dict] = []
    hold_short: list[dict] = []

    for ticker, rec in curr_long.items():
        if ticker in prev_long:
            hold_long.append(_build_action_from_rec("HOLD_LONG", rec))
        elif ticker in prev_short:
            buy.append(_build_action_from_rec("BUY", rec, note="Flip SHORT->LONG"))
        else:
            buy.append(_build_action_from_rec("BUY", rec))

    for ticker, rec in curr_short.items():
        if ticker in prev_short:
            hold_short.append(_build_action_from_rec("HOLD_SHORT", rec))
        elif ticker in prev_long:
            open_short.append(_build_action_from_rec("OPEN_SHORT", rec, note="Flip LONG->SHORT"))
        else:
            open_short.append(_build_action_from_rec("OPEN_SHORT", rec))

    for ticker, rec in prev_long.items():
        if ticker not in curr_long and ticker not in curr_short:
            sell.append(_build_action_exit("SELL", rec))

    for ticker, rec in prev_short.items():
        if ticker not in curr_short and ticker not in curr_long:
            cover_short.append(_build_action_exit("COVER_SHORT", rec))

    actions = buy + sell + open_short + cover_short + hold_long + hold_short

    evaluated = [a for a in actions if a["direction_correct"] is not None]
    correct = [a for a in evaluated if a["direction_correct"]]
    realized_vals = [float(a["realized_pnl_proxy"]) for a in evaluated if a["realized_pnl_proxy"] is not None]

    evaluated_count = len(evaluated)
    hit_rate = (len(correct) / evaluated_count) if evaluated_count > 0 else None
    realized_proxy_sum = sum(realized_vals) if realized_vals else None

    return {
        "date": date_str,
        "compare_to_date": prev_date,
        "buy_count": len(buy),
        "sell_count": len(sell),
        "open_short_count": len(open_short),
        "cover_short_count": len(cover_short),
        "hold_long_count": len(hold_long),
        "hold_short_count": len(hold_short),
        "total_actions": len(actions),
        "evaluated_actions": evaluated_count,
        "correct_actions": len(correct),
        "hit_rate": hit_rate,
        "realized_pnl_proxy_sum": realized_proxy_sum,
        "actions": actions,
    }


def strategy_payload_to_vector(payload: dict) -> list[float]:
    """
    Compact numeric vector for Qdrant storage/query.
    """
    hit_rate = payload.get("hit_rate")
    realized = payload.get("realized_pnl_proxy_sum")
    return [
        float(payload.get("buy_count", 0)),
        float(payload.get("sell_count", 0)),
        float(payload.get("open_short_count", 0)),
        float(payload.get("cover_short_count", 0)),
        float(payload.get("hold_long_count", 0)),
        float(payload.get("hold_short_count", 0)),
        float(hit_rate) if hit_rate is not None else 0.0,
        float(realized) if realized is not None else 0.0,
    ]

