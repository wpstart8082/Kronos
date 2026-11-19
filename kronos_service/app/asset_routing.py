"""Symbol → asset-type ("index" / "stock") inference for dual-model routing.

A股的指数与深市股票共享 ``000xxx`` 代码段（000001 = 上证指数 vs 平安银行），
只有带交易所前缀/后缀的写法才能无歧义识别；裸 6 位 ``000xxx`` 返回 None，
由调用方按默认模型处理。识别规则覆盖 finetune 指数清单的全部写法
（sh000300 / sz399001 / csi000922 / 000300.SH …）与常见股票段。
"""
from __future__ import annotations


def infer_asset_type(symbol: str | None) -> str | None:
    """Return ``"index"``, ``"stock"`` or ``None`` (unknown / ambiguous)."""
    if not symbol:
        return None
    core, _, suffix = symbol.strip().upper().partition(".")
    if not core:
        return None
    has_suffix = suffix in ("SH", "SZ", "CSI")

    # ---- index ----
    # 带前缀: sh000xxx(上证/中证) / sh999xxx(上证指数别名) / sz399xxx(深证) / csi*
    if core.startswith(("SH000", "SH999")) or core.startswith("CSI"):
        return "index"
    if core.startswith("SZ399"):
        return "index"
    # 带后缀: 000xxx.SH / 399xxx.SZ / *.CSI
    if has_suffix:
        if suffix == "SH" and core.startswith("000"):
            return "index"
        if suffix == "SZ" and core.startswith("399"):
            return "index"
        if suffix == "CSI":
            return "index"
    # 裸 399xxx: 深市股票没有 399 开头 -> 指数; 999999 是上证指数行情别名
    if len(core) == 6 and core.isdigit() and core.startswith("399"):
        return "index"
    if core == "999999":
        return "index"

    # ---- stock ----
    # 带前缀: sh60*/sh68*(沪主板/科创) / sz00*/sz30*(深主板/创业板, 含 sz000xxx)
    if core.startswith(("SH60", "SH68", "SZ00", "SZ30")):
        return "stock"
    if has_suffix:
        if suffix == "SH" and core.startswith(("60", "68")):
            return "stock"
        if suffix == "SZ" and core.startswith(("00", "30")):
            return "stock"
    # 裸 6 位: 60/68/30 开头是股票; 00 开头中 002/003 是深市股票,
    # 000xxx 与指数代码段冲突 -> 返回 None 交由默认模型兜底。
    if len(core) == 6 and core.isdigit():
        if core.startswith(("60", "68", "30")):
            return "stock"
        if core.startswith("00") and not core.startswith("000"):
            return "stock"
    return None
