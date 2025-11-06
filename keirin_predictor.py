"""Keirin trifecta formation predictor CLI."""
from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Referer": "https://www.chariloto.com/",
}
REQUEST_TIMEOUT = 15
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 1.5

CHARILOTO_BASE = "https://www.chariloto.com"
VENUE_MAP_RAW: Dict[str, int] = {
    # 北日本
    "函館": 11,
    "青森": 12,
    "いわき平": 13,
    # 関東
    "弥彦": 21,
    "前橋": 22,
    "取手": 23,
    "宇都宮": 24,
    "大宮": 25,
    "西武園": 26,
    "京王閣": 27,
    "立川": 28,
    # 南関東
    "松戸": 31,
    "千葉": 32,
    "川崎": 34,
    "平塚": 35,
    "小田原": 36,
    "伊東": 37,
    "静岡": 38,
    # 中部
    "名古屋": 42,
    "岐阜": 43,
    "大垣": 44,
    "豊橋": 45,
    "富山": 46,
    "松阪": 47,
    "四日市": 48,
    # 近畿
    "福井": 51,
    "奈良": 53,
    "向日町": 54,
    "和歌山": 55,
    "岸和田": 56,
    # 中国
    "玉野": 61,
    "広島": 62,
    "防府": 63,
    # 四国
    "高松": 71,
    "小松島": 73,
    "高知": 74,
    "松山": 75,
    # 九州
    "小倉": 81,
    "久留米": 83,
    "武雄": 84,
    "佐世保": 85,
    "別府": 86,
    "熊本": 87,
}


def _normalize_text(value: str) -> str:
    """Return NFKC-normalized lowercase text."""

    return unicodedata.normalize("NFKC", value).lower()


VENUE_MAP: Dict[str, int] = {
    _normalize_text(name): code for name, code in VENUE_MAP_RAW.items()
}
LINE_PATTERN = re.compile(r"\d(?:-\d+)+(?:\s*/\s*\d(?:-\d+)+)*")

THETA_WEIGHTS = {
    "score": 0.40,
    "recent": 0.20,
    "self": 0.15,
    "position": 0.10,
    "local": 0.10,
    "start": 0.05,
}
LINE_WEIGHTS = {
    "lead": 0.45,
    "second": 0.20,
    "third": 0.10,
    "score": 0.15,
    "history": 0.10,
}
ROLE_DISTRIBUTION = {
    "lead": 0.25,
    "second": 0.55,
    "support": 0.20,
}
CONDITION_WEIGHTS = {
    "track": 0.10,
    "wind_resist": 0.05,
    "wind_penalty": -0.05,
}
PL_ADJ_COEFF = {
    "same_line_ab": 0.20,
    "same_line_bc": 0.10,
    "line_conflict": -0.12,
}
LONG_PENALTY_BETA = 0.18
RHO = 0.65
GAMMA = 0.10
OMEGA_BASE = 0.25
OMEGA_SCALE = 0.35
OMEGA_MAX = 0.60
ETA_MIN = 0.40

TAU_MIN = {
    "low": 0.008,
    "mid": 0.006,
    "high": 0.005,
}
COVERAGE_MIN = {
    "low": 0.22,
    "mid": 0.18,
    "high": 0.15,
}

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RiderRaw:
    """Raw rider information container."""

    number: int
    name: str
    score: Optional[float] = None
    recent_results: Optional[List[int]] = None
    recent_speed: Optional[List[float]] = None
    recent_top3_rate: Optional[float] = None
    style: Optional[str] = None
    start_response: Optional[float] = None
    position_score: Optional[float] = None
    local_factor: Optional[float] = None
    self_strength: Optional[float] = None
    lead_power: Optional[float] = None
    second_power: Optional[float] = None
    third_stability: Optional[float] = None
    bank_suitability: Optional[float] = None
    wind_resistance: Optional[float] = None
    wind_penalty: Optional[float] = None


@dataclass
class LineInfo:
    """Line (formation) information."""

    members: List[int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request_with_retry(
    url: str, *, params: Optional[Dict[str, str]] = None, referer: Optional[str] = None
) -> requests.Response:
    """HTTP GET with retry policy."""

    headers = dict(DEFAULT_HEADERS)
    if referer:
        headers["Referer"] = referer
    last_exc: Optional[Exception] = None
    for attempt in range(REQUEST_RETRIES):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == REQUEST_RETRIES - 1:
                break
            time.sleep(REQUEST_BACKOFF ** attempt)
    assert last_exc is not None
    raise last_exc


def _zscore(series: pd.Series) -> pd.Series:
    """Return z-score standardized series (zero for constant series)."""

    if series.empty:
        return series
    std = float(series.std(ddof=0))
    if math.isclose(std, 0.0):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def _parse_float(value: str) -> Optional[float]:
    """Parse float from string, returning None on failure."""

    if value is None:
        return None
    text = value.strip().replace(",", "")
    if not text:
        return None
    text = text.replace("秒", "").replace("m", "").replace("倍", "")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_percentage(value: str) -> Optional[float]:
    """Parse percentage string into ratio (0-1)."""

    if value is None:
        return None
    text = value.strip().replace(",", "")
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text) / 100.0
    except ValueError:
        return None


def _parse_recent_results(value: str) -> List[int]:
    """Parse recent results string into a list of finishes."""

    if not value:
        return []
    digits = re.findall(r"\d", value)
    return [int(d) for d in digits[:3]]


def _resolve_venue_id(race_name: str) -> Tuple[int, str]:
    """Resolve venue identifier from race name."""

    normalized = _normalize_text(race_name)
    for venue_name, venue_id in VENUE_MAP.items():
        if venue_name in normalized:
            return venue_id, venue_name
    raise ValueError("unknown venue")


def _extract_lines(soup: BeautifulSoup) -> List[LineInfo]:
    """Extract line formation information from soup."""

    seen: set[str] = set()
    for text in soup.stripped_strings:
        match = LINE_PATTERN.search(text)
        if not match:
            continue
        candidate = match.group()
        if candidate in seen:
            continue
        seen.add(candidate)
        groups = []
        for block in candidate.split("/"):
            members = [int(part) for part in re.findall(r"\d+", block)]
            if members:
                groups.append(LineInfo(members=members))
        if groups:
            return groups
    return []


def _ensure_rider(riders: Dict[int, RiderRaw], number: int) -> RiderRaw:
    """Ensure a RiderRaw entry exists for given number."""

    if number not in riders:
        riders[number] = RiderRaw(number=number, name="")
    return riders[number]


def _extract_riders(soup: BeautifulSoup) -> Dict[int, RiderRaw]:
    """Extract rider information from race detail page."""

    riders: Dict[int, RiderRaw] = {}
    for table in soup.find_all("table"):
        header_row = None
        for row in table.find_all("tr"):
            th_cells = row.find_all("th")
            if th_cells:
                header_row = th_cells
                break
        if header_row is None:
            continue
        headers = [th.get_text(strip=True) for th in header_row]
        normalized_headers = [_normalize_text(h) for h in headers]
        if not any("車" in h or "枠" in h for h in normalized_headers):
            continue
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            raw_text = [cell.get_text(" ", strip=True) for cell in cells]
            numbers = [int(n) for n in re.findall(r"\d+", raw_text[0])]
            if not numbers:
                continue
            number = numbers[0]
            rider = _ensure_rider(riders, number)
            for header, value in zip(normalized_headers, raw_text):
                if not value:
                    continue
                if "選手" in header or "氏名" in header:
                    rider.name = value.split()[0]
                elif "競走得点" in header or "得点" in header:
                    parsed = _parse_float(value)
                    rider.score = parsed if parsed is not None else rider.score
                elif "直近" in header or "最近" in header:
                    rider.recent_results = _parse_recent_results(value)
                elif "3連対" in header:
                    rider.recent_top3_rate = _parse_percentage(value)
                elif "脚質" in header:
                    rider.style = value
                elif "スタート" in header or "ダッシュ" in header:
                    parsed = _parse_float(value)
                    rider.start_response = parsed if parsed is not None else rider.start_response
                elif "位置" in header:
                    parsed = _parse_float(value)
                    rider.position_score = parsed if parsed is not None else rider.position_score
                elif "地元" in header or "相性" in header:
                    parsed = _parse_float(value)
                    rider.local_factor = parsed if parsed is not None else rider.local_factor
                elif "自力" in header:
                    parsed = _parse_float(value)
                    rider.self_strength = parsed if parsed is not None else rider.self_strength
                elif "先行力" in header:
                    parsed = _parse_float(value)
                    rider.lead_power = parsed if parsed is not None else rider.lead_power
                elif "番手" in header:
                    parsed = _parse_float(value)
                    rider.second_power = parsed if parsed is not None else rider.second_power
                elif "三番手" in header:
                    parsed = _parse_float(value)
                    rider.third_stability = parsed if parsed is not None else rider.third_stability
                elif "バンク" in header:
                    parsed = _parse_float(value)
                    rider.bank_suitability = parsed if parsed is not None else rider.bank_suitability
                elif "風" in header and "耐" in header:
                    parsed = _parse_float(value)
                    rider.wind_resistance = parsed if parsed is not None else rider.wind_resistance
                elif "風" in header and "影響" in header:
                    parsed = _parse_float(value)
                    rider.wind_penalty = parsed if parsed is not None else rider.wind_penalty
    return riders


def _extract_trifecta_odds(html: str) -> Dict[Tuple[int, int, int], float]:
    """Extract trifecta odds from odds page HTML."""

    soup = BeautifulSoup(html, "lxml")
    odds: Dict[Tuple[int, int, int], float] = {}
    target_table = None
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "div"]):
        if "3連単" in heading.get_text(strip=True):
            target_table = heading.find_next("table")
            if target_table:
                break
    if target_table is None:
        tables = soup.find_all("table")
        for table in tables:
            header_text = " ".join(th.get_text(strip=True) for th in table.find_all("th"))
            if "3連単" in header_text or "組番" in header_text:
                target_table = table
                break
    if target_table:
        for row in target_table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            if len(cells) < 2:
                continue
            combo_text = None
            for cell in cells:
                match = re.search(r"\d(?:-\d+){2}", cell)
                if match:
                    combo_text = match.group()
                    break
            if not combo_text:
                continue
            odds_value = None
            for cell in cells[1:]:
                parsed = _parse_float(cell)
                if parsed is not None:
                    odds_value = parsed
                    break
            if odds_value is None:
                continue
            numbers = tuple(int(part) for part in combo_text.split("-"))
            if len(numbers) == 3:
                odds[numbers] = odds_value
    if odds:
        return odds
    matches = re.finditer(r"(\d(?:-\d+){2})"r"\D+([0-9]+\.?[0-9]*)", soup.get_text(" ", strip=True))
    for match in matches:
        combo_text = match.group(1)
        odds_text = match.group(2)
        numbers = tuple(int(part) for part in combo_text.split("-"))
        if len(numbers) != 3:
            continue
        odds_value = _parse_float(odds_text)
        if odds_value is None:
            continue
        odds.setdefault(numbers, odds_value)
    return odds


def _extract_bank_and_wind(soup: BeautifulSoup) -> Tuple[Optional[int], Optional[Dict[str, object]]]:
    """Extract bank length and wind information if present."""

    text = soup.get_text(" ", strip=True)
    bank_length: Optional[int] = None
    bank_match = re.search(r"(33[3]?|400|500)\s*m", text)
    if bank_match:
        try:
            bank_length = int(bank_match.group(1))
        except ValueError:
            bank_length = None
    wind_speed = None
    wind_dir = None
    speed_match = re.search(r"風速[：: ]*([0-9]+\.?[0-9]*)", text)
    if speed_match:
        wind_speed = _parse_float(speed_match.group(1))
    dir_match = re.search(r"風向[：: ]*([東西南北]+)", text)
    if dir_match:
        wind_dir = dir_match.group(1)
    wind_info = None
    if wind_speed is not None or wind_dir:
        wind_info = {"direction": wind_dir, "speed": float(wind_speed or 0.0)}
    return bank_length, wind_info
# ---------------------------------------------------------------------------
# Fetching and parsing
# ---------------------------------------------------------------------------


def fetch_race_data(date_str: str, race_name: str) -> Dict:
    """Fetch race metadata, riders, lineups, and odds."""

    try:
        date_digits = date_str.replace("-", "")
        if len(date_digits) != 8:
            raise ValueError
        int(date_digits)
    except ValueError as exc:
        raise ValueError("Invalid date format; expected YYYY-MM-DD") from exc

    venue_id, venue_key = _resolve_venue_id(race_name)
    race_label = unicodedata.normalize("NFKC", race_name).upper()
    match = re.search(r"(\d{1,2})\s*[RＲ]", race_label)
    if not match:
        raise ValueError("race number not found in race name")
    race_no = int(match.group(1))

    meta: Dict[str, object] = {
        "date": date_str,
        "race_name": race_name,
        "venue_id": venue_id,
        "venue_key": venue_key,
        "race_no": race_no,
    }
    for original, code in VENUE_MAP_RAW.items():
        if code == venue_id:
            meta["venue"] = original
            break
    else:
        meta["venue"] = venue_key

    detail_url = f"{CHARILOTO_BASE}/keirin/athletes/{date_str}/{venue_id}/{race_no}"
    detail_resp = _request_with_retry(detail_url, referer=CHARILOTO_BASE + "/")
    detail_resp.encoding = detail_resp.apparent_encoding or detail_resp.encoding or "utf-8"
    detail_html = detail_resp.text
    detail_soup = BeautifulSoup(detail_html, "lxml")

    riders = _extract_riders(detail_soup)
    if not riders:
        raise ValueError("failed to parse riders from detail page")

    lines = _extract_lines(detail_soup)
    bank_length, wind_info = _extract_bank_and_wind(detail_soup)

    odds_url = f"{detail_url}?odds=1"
    odds_resp = _request_with_retry(odds_url, referer=detail_url)
    odds_resp.encoding = odds_resp.apparent_encoding or odds_resp.encoding or "utf-8"
    odds_html = odds_resp.text
    odds = _extract_trifecta_odds(odds_html)
    if not odds:
        raise ValueError("Threefold odds not available")
    rider_count = len(riders)
    if rider_count >= 3:
        expected_combos = rider_count * (rider_count - 1) * (rider_count - 2)
        if len(odds) < expected_combos:
            raise ValueError("Threefold odds not available")

    meta["detail_url"] = detail_url
    meta["odds_url"] = odds_url

    return {
        "riders": riders,
        "lines": lines,
        "odds": odds,
        "bank_length": bank_length,
        "wind": wind_info,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_features(raw: Dict) -> Dict:
    """Build feature matrices from raw race data."""

    riders: Dict[int, RiderRaw] = raw["riders"]
    numbers = sorted(riders)
    df = pd.DataFrame(
        {
            "number": numbers,
            "score": [riders[n].score for n in numbers],
            "position": [riders[n].position_score for n in numbers],
            "local": [riders[n].local_factor for n in numbers],
            "start": [riders[n].start_response for n in numbers],
            "self_strength": [riders[n].self_strength for n in numbers],
            "lead_power": [riders[n].lead_power for n in numbers],
            "second_power": [riders[n].second_power for n in numbers],
            "third_stability": [riders[n].third_stability for n in numbers],
            "bank": [riders[n].bank_suitability for n in numbers],
            "wind_resist": [riders[n].wind_resistance for n in numbers],
            "wind_penalty": [riders[n].wind_penalty for n in numbers],
        }
    ).set_index("number")

    def recent_index(rider: RiderRaw) -> float:
        mapping = {1: 1.0, 2: 0.8, 3: 0.65, 4: 0.45, 5: 0.30}
        weights = [0.5, 0.3, 0.2]
        results = rider.recent_results or []
        total = 0.0
        for finish, weight in zip(results, weights):
            total += weight * mapping.get(finish, 0.15)
        if not results and rider.recent_top3_rate is not None:
            total = rider.recent_top3_rate
        return total

    def style_strength(rider: RiderRaw) -> float:
        if rider.style is None:
            return rider.self_strength or 0.0
        style = rider.style
        components = {"先行": 1.0, "捲り": 0.8, "追込": -0.3, "自在": 0.4}
        value = sum(weight for key, weight in components.items() if key in style)
        return value if value else rider.self_strength or 0.0

    df["recent"] = [recent_index(riders[n]) for n in numbers]
    df["self_index"] = [style_strength(riders[n]) for n in numbers]
    df = df.fillna(0.0)

    z_cols = {col: _zscore(df[col]) for col in df.columns}

    return {
        "numbers": numbers,
        "z": z_cols,
        "raw": df,
        "lines": raw["lines"],
        "odds": raw["odds"],
        "bank_length": raw.get("bank_length"),
        "wind": raw.get("wind"),
        "meta": raw.get("meta", {}),
    }


# ---------------------------------------------------------------------------
# Probability estimation
# ---------------------------------------------------------------------------


def _build_line_lookup(lines: Sequence[LineInfo]) -> Dict[int, int]:
    lookup: Dict[int, int] = {}
    for idx, line in enumerate(lines):
        for member in line.members:
            lookup[member] = idx
    return lookup


def pl_triple_probs(lambdas: np.ndarray, adj_meta: Dict[str, object]) -> pd.DataFrame:
    """Compute Plackett–Luce probabilities for all permutations."""

    numbers: List[int] = adj_meta["numbers"]
    line_lookup: Dict[int, int] = adj_meta["line_lookup"]
    conflict_pairs: set[frozenset[int]] = adj_meta.get("conflicts", set())

    lambda_map = {num: float(val) for num, val in zip(numbers, lambdas)}
    combos: List[Tuple[int, int, int]] = []
    probs: List[float] = []

    total_lambda = sum(lambda_map.values())
    if total_lambda <= 0:
        raise ValueError("Invalid lambda values")

    for a, b, c in itertools.permutations(numbers, 3):
        denom1 = total_lambda
        denom2 = denom1 - lambda_map[a]
        denom3 = denom2 - lambda_map[b]
        if denom2 <= 0 or denom3 <= 0:
            continue
        base = (lambda_map[a] / denom1) * (lambda_map[b] / denom2) * (lambda_map[c] / denom3)
        same_ab = int(line_lookup.get(a) is not None and line_lookup.get(a) == line_lookup.get(b))
        same_bc = int(line_lookup.get(b) is not None and line_lookup.get(b) == line_lookup.get(c))
        conflict_ab = int(frozenset({a, b}) in conflict_pairs)
        adj = math.exp(
            PL_ADJ_COEFF["same_line_ab"] * same_ab
            + PL_ADJ_COEFF["same_line_bc"] * same_bc
            + PL_ADJ_COEFF["line_conflict"] * conflict_ab
        )
        combos.append((a, b, c))
        probs.append(base * adj)

    series = pd.Series(probs, index=pd.MultiIndex.from_tuples(combos, names=["a", "b", "c"]))
    series /= series.sum()
    return series.to_frame("prob")


# ---------------------------------------------------------------------------
# Formation construction
# ---------------------------------------------------------------------------


def _market_probabilities(odds: Dict[Tuple[int, int, int], float]) -> pd.Series:
    inv = {k: (1.0 / v) if v else 0.0 for k, v in odds.items()}
    total = sum(inv.values())
    if total <= 0:
        raise ValueError("Invalid odds distribution")
    return pd.Series({k: val / total for k, val in inv.items()})


def _chaos_level(p1: pd.Series, line_scores: List[float]) -> Tuple[str, float]:
    entropy = -float(np.sum([p * math.log(p) for p in p1 if p > 0]))
    n = len(p1)
    h_norm = entropy / math.log(n) if n > 1 else 0.0
    sorted_p = sorted(p1, reverse=True)
    second = sorted_p[1] if len(sorted_p) > 1 else 0.0
    mix = 1.0 - (sorted_p[0] + second)
    var_lines = float(np.var(line_scores)) if line_scores else 0.0
    chaos = 0.35 * h_norm + 0.45 * mix + 0.20 * var_lines
    if chaos < 0.45:
        return "low", chaos
    if chaos <= 0.65:
        return "mid", chaos
    return "high", chaos


def _select_anchor(p1: pd.Series, edge_avg: Dict[int, float], chaos_label: str) -> List[int]:
    anchors: List[int]
    if chaos_label in {"low", "mid"}:
        if len(p1) > 1:
            ratio = p1.max() / max(p1.nlargest(2).iloc[1], 1e-6)
        else:
            ratio = float("inf")
        if p1.max() >= 0.22 or ratio >= 1.20:
            anchors = [int(p1.idxmax())]
        else:
            anchors = list(p1.nlargest(2).index.astype(int))
    else:
        anchors = [int(idx) for idx, value in p1.items() if value >= 0.12]
        if not anchors:
            anchors = list(p1.nlargest(2).index.astype(int))
    scores = {i: p1.loc[i] * (0.7 + 0.3 * edge_avg.get(i, 0.0)) for i in anchors}
    anchors_sorted = sorted(scores, key=lambda k: scores[k], reverse=True)
    return anchors_sorted[:2] if chaos_label == "high" else anchors_sorted[:1]


def _select_groups(
    anchors: List[int],
    valid_combos: List[Tuple[int, int, int]],
    prob_eff: pd.Series,
    positive_edge: pd.Series,
    line_lookup: Dict[int, int],
    chaos_label: str,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Select second and third place sets according to spec."""

    if not valid_combos:
        raise ValueError("No valid combinations after filtering")

    seconds_all = sorted({b for a, b, _ in valid_combos if a in anchors and b not in anchors})
    thirds_all = sorted({c for a, _, c in valid_combos if a in anchors and c not in anchors})
    if not seconds_all or not thirds_all:
        raise ValueError("Insufficient candidates for formation")

    def score_second(candidate: int) -> float:
        total = 0.0
        for a in anchors:
            if a == candidate:
                continue
            for combo in valid_combos:
                if combo[0] == a and combo[1] == candidate and combo[2] not in {a, candidate}:
                    total += prob_eff[combo]
            if line_lookup.get(a) is not None and line_lookup.get(a) == line_lookup.get(candidate):
                total += 0.03
        return total

    def score_third(candidate: int) -> float:
        combos = [combo for combo in valid_combos if combo[2] == candidate and combo[0] in anchors]
        base = sum(prob_eff[combo] for combo in combos)
        edges = [positive_edge[combo] for combo in combos if positive_edge[combo] > 0]
        bonus = 0.5 * (sum(edges) / len(edges)) if edges else 0.0
        return base + bonus

    ordered_seconds = sorted(seconds_all, key=score_second, reverse=True)
    ordered_thirds = sorted(thirds_all, key=score_third, reverse=True)

    max_second = 3 if chaos_label == "high" else 2
    max_third = 3 if chaos_label != "low" else 2

    selected_second = ordered_seconds[:max_second] or ordered_seconds[:1]
    selected_third = ordered_thirds[:max_third] or ordered_thirds[:1]

    return selected_second, selected_third, ordered_seconds, ordered_thirds


def construct_formation(P: pd.DataFrame, meta: Dict) -> Tuple[str, float]:
    """Construct formation string and hit rate."""

    market: pd.Series = meta["market"]
    chaos_label: str = meta["chaos_label"]
    chaos_value: float = meta["chaos_value"]
    omega: float = meta["omega"]

    p_series = P["prob"]
    market_matrix = market.reindex(p_series.index).fillna(0.0)
    positive_edge = (p_series - market_matrix).clip(lower=0.0)

    replacement = market_matrix.replace(0, np.nan)
    with np.errstate(divide="ignore"):
        long_term = -np.log(replacement)
    mu = float(long_term.mean()) if not math.isnan(float(long_term.mean())) else 0.0
    sigma = float(long_term.std(ddof=0))
    if math.isnan(sigma) or math.isclose(sigma, 0.0):
        sigma = 1.0
    long_z = ((long_term - mu) / sigma).fillna(0.0).clip(lower=0.0)

    prob_eff = (p_series * np.exp(-LONG_PENALTY_BETA * long_z)).fillna(0.0)
    prob_eff /= prob_eff.sum()

    tau = TAU_MIN[chaos_label]
    coverage_min = COVERAGE_MIN[chaos_label]

    valid_combos = [
        combo
        for combo in p_series.index
        if len({combo[0], combo[1], combo[2]}) == 3
        and prob_eff[combo] >= tau
        and market_matrix.get(combo, 0.0) >= 0.002
    ]
    if not valid_combos:
        raise ValueError("No combinations satisfy safety constraints")

    p1 = p_series.groupby(level=0).sum()
    edge_avg = positive_edge.groupby(level=0).mean().to_dict()
    anchors = _select_anchor(p1, edge_avg, chaos_label)

    line_lookup = _build_line_lookup(meta["lines"])
    selected_second, selected_third, ordered_seconds, ordered_thirds = _select_groups(
        anchors, valid_combos, prob_eff, positive_edge, line_lookup, chaos_label
    )

    def score_combo(combo: Tuple[int, int, int]) -> float:
        return (
            RHO * prob_eff[combo]
            + (1 - RHO) * positive_edge[combo]
            - GAMMA * long_z[combo]
        )

    ranked_combos = sorted(valid_combos, key=score_combo, reverse=True)

    def build_combos(seconds: Sequence[int], thirds: Sequence[int]) -> List[Tuple[int, int, int]]:
        result = []
        seconds_set = set(seconds)
        thirds_set = set(thirds)
        for combo in ranked_combos:
            if combo[0] in anchors and combo[1] in seconds_set and combo[2] in thirds_set:
                result.append(combo)
        return result

    seconds = list(dict.fromkeys(selected_second))
    thirds = list(dict.fromkeys(selected_third))
    combos_selected = build_combos(seconds, thirds)
    if not combos_selected:
        raise ValueError("Unable to assemble valid formation")

    def coverage_value(combos: Sequence[Tuple[int, int, int]]) -> float:
        return float(sum(prob_eff[c] for c in combos))

    coverage = coverage_value(combos_selected)
    idx_second = len(seconds)
    idx_third = len(thirds)

    while coverage < coverage_min:
        expanded = False
        if idx_third < len(ordered_thirds):
            candidate = ordered_thirds[idx_third]
            idx_third += 1
            if candidate not in thirds and candidate not in anchors:
                thirds.append(candidate)
                expanded = True
        elif idx_second < len(ordered_seconds):
            candidate = ordered_seconds[idx_second]
            idx_second += 1
            if candidate not in seconds and candidate not in anchors:
                seconds.append(candidate)
                expanded = True
        if not expanded:
            break
        combos_selected = build_combos(seconds, thirds)
        if not combos_selected:
            break
        coverage = coverage_value(combos_selected)

    combos_selected = sorted(combos_selected, key=score_combo, reverse=True)

    def adjust_sets() -> None:
        nonlocal seconds, thirds, combos_selected
        combos_selected = build_combos(seconds, thirds)

    while len(combos_selected) > 6 and (len(thirds) > 2 or len(seconds) > 2):
        if len(thirds) > 2:
            thirds.pop()
        elif len(seconds) > 2:
            seconds.pop()
        adjust_sets()
        if not combos_selected:
            break

    if len(combos_selected) > 6:
        combos_selected = combos_selected[:6]

    candidate_seconds = [s for s in ordered_seconds if s not in seconds and s not in anchors]
    candidate_thirds = [t for t in ordered_thirds if t not in thirds and t not in anchors]
    idx_second_extra = 0
    idx_third_extra = 0
    while len(combos_selected) < 4:
        added = False
        if idx_third_extra < len(candidate_thirds):
            thirds.append(candidate_thirds[idx_third_extra])
            idx_third_extra += 1
            added = True
        elif idx_second_extra < len(candidate_seconds):
            seconds.append(candidate_seconds[idx_second_extra])
            idx_second_extra += 1
            added = True
        else:
            break
        adjust_sets()
        if len(combos_selected) >= 4:
            break
    if len(combos_selected) < 4:
        for combo in ranked_combos:
            if combo not in combos_selected:
                combos_selected.append(combo)
            if len(combos_selected) >= 4:
                break

    combos_selected = combos_selected[:6]

    eta = max(ETA_MIN, 1 - omega)
    hit_prob = (eta * p_series + (1 - eta) * market_matrix).loc[combos_selected].sum()
    hit_rate = round(float(hit_prob) * 100, 1)

    first_part = "".join(str(a) for a in sorted(set(anchors)))
    second_part = "".join(str(b) for b in sorted(set(seconds)))
    third_part = "".join(str(c) for c in sorted(set(thirds)))
    formation = f"{first_part}-{second_part}-{third_part}"

    return formation, hit_rate


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(description="Keirin trifecta formation predictor")
    parser.add_argument("--date", required=True, help="Race date YYYY-MM-DD")
    parser.add_argument("--race", required=True, help="Race name keyword")
    args = parser.parse_args()

    try:
        raw = fetch_race_data(args.date, args.race)
        features = build_features(raw)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    numbers = features["numbers"]
    z = features["z"]

    lambdas_base = []
    lines = features["lines"]
    line_lookup = _build_line_lookup(lines)
    for num in numbers:
        theta0 = (
            THETA_WEIGHTS["score"] * z["score"].loc[num]
            + THETA_WEIGHTS["recent"] * z["recent"].loc[num]
            + THETA_WEIGHTS["self"] * z["self_index"].loc[num]
            + THETA_WEIGHTS["position"] * z["position"].loc[num]
            + THETA_WEIGHTS["local"] * z["local"].loc[num]
            + THETA_WEIGHTS["start"] * z["start"].loc[num]
        )

        role = "support"
        for line in lines:
            if num in line.members:
                if line.members[0] == num:
                    role = "lead"
                elif len(line.members) > 1 and line.members[1] == num:
                    role = "second"
                break

        lp_components = {
            "lead": z["lead_power"].loc[num],
            "second": z["second_power"].loc[num],
            "third": z["third_stability"].loc[num],
            "score": z["score"].loc[num],
            "history": z["local"].loc[num],
        }
        line_power = sum(LINE_WEIGHTS[key] * lp_components[key] for key in LINE_WEIGHTS)
        theta_l = ROLE_DISTRIBUTION[role] * line_power

        theta_c = (
            CONDITION_WEIGHTS["track"] * z["bank"].loc[num]
            + CONDITION_WEIGHTS["wind_resist"] * z["wind_resist"].loc[num]
            + CONDITION_WEIGHTS["wind_penalty"] * z["wind_penalty"].loc[num]
        )

        phi = theta0 + theta_l + theta_c
        lambdas_base.append(math.exp(phi))

    lambdas_base = np.array(lambdas_base)
    adj_info = {"numbers": numbers, "line_lookup": line_lookup, "conflicts": set()}

    P_base = pl_triple_probs(lambdas_base, adj_info)
    market_series = _market_probabilities(features["odds"])

    line_scores_base = []
    lambda_map_base = {num: val for num, val in zip(numbers, lambdas_base)}
    for line in lines:
        line_scores_base.append(sum(lambda_map_base.get(member, 0.0) for member in line.members))

    p1_base = P_base["prob"].groupby(level=0).sum()
    chaos_label, chaos_value = _chaos_level(p1_base, line_scores_base)
    omega = min(OMEGA_MAX, OMEGA_BASE + OMEGA_SCALE * chaos_value)

    market_p1 = market_series.groupby(level=0).sum()
    market_lambda = market_p1.reindex(numbers).fillna(market_p1.mean())
    if (market_lambda <= 0).all():
        market_lambda = pd.Series([1.0] * len(numbers), index=numbers)
    market_lambda = market_lambda.replace(0, market_lambda.mean() or 1.0)

    lambdas_post = lambdas_base ** (1 - omega) * market_lambda.values ** omega
    P = pl_triple_probs(lambdas_post, adj_info)

    meta = {
        "odds": features["odds"],
        "lines": lines,
        "market": market_series,
        "chaos_label": chaos_label,
        "chaos_value": chaos_value,
        "omega": omega,
        "lambda_map": {num: val for num, val in zip(numbers, lambdas_post)},
    }

    try:
        formation, hit_rate = construct_formation(P, meta)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(formation)
    print(f"的中率: {hit_rate:.1f}%")


if __name__ == "__main__":
    main()
