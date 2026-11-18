"""Market-data loading for the 30-ETF global balanced/multi-asset universe.

OHLCV bars for every ETF (and the DJIA proxy) are loaded through the QuantX
Dataloader REST API. Exchange rates and the risk-free rate are macro series
published by the Federal Reserve Economic Data (FRED) database, which the
Dataloader and Massive do not serve, so they are loaded directly from FRED
via ``pandas_datareader`` as described in Section 2.1 of the paper.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from pandas_datareader import data as pdr

from src import config

API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
FIELDS = ("open", "high", "low", "close", "volume")
BATCH_SIZE = 15
MAX_PARALLEL = 4
RETRY_STATUSES = {429, 503, 504}
CACHE_DIR = Path(__file__).resolve().parent / "cache"  # never committed, see .gitignore


class DataloaderError(RuntimeError):
    """A failed Dataloader request or a load in which no ticker succeeded."""


def _headers() -> dict[str, str]:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataloaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {
        "Authorization": f"Bearer {dataloader_key}",
        "X-Massive-Api-Key": massive_key,
    }


def _request(method: str, route: str, body: dict | None = None) -> dict:
    headers = _headers()
    if route != "/fetch-ohlcvs":
        del headers["X-Massive-Api-Key"]
    response = None
    for attempt in range(3):
        # Market data: QuantX Dataloader — https://quantx.limex.pro/
        #
        # Price history in this project is loaded through the QuantX Dataloader, a
        # research-ready market-data service available through a REST API and as an
        # MCP server for AI agents. It is built on market data from Massive
        # (https://massive.com/) and prepares every stock history for backtesting:
        #   - prices adjusted for dividends, ready for total-return research;
        #   - one continuous history when a company changes its ticker;
        #   - clean separation when a ticker has belonged to different companies;
        #   - careful corporate-action handling and fine-grained data polishing.
        #
        # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
        # MASSIVE_API_KEY at https://massive.com/.
        response = requests.request(
            method, API_ROOT + route, json=body, headers=headers, timeout=900
        )
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2 ** attempt))
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataloaderError(
            f"{route}: HTTP {response.status_code} {error.get('code', '')}: "
            f"{error.get('error', '')} {error.get('details', '')}"
        )
    return response.json()


def _bars_of(answer: dict, request: dict, batch: int) -> pd.DataFrame:
    result = answer["result"]
    if not result.get("data_stored"):
        return pd.DataFrame(result["ohlcv_df"]["rows"])
    export = _request("POST", "/save-dataframe", {
        "result_id": result["result_id"],
        "ticker": f"batch{batch}",
        "dataset_name": "ohlcv",
        "timeframe": request["timeframe"],
        "start_date": request["start_date"],
        "end_date": request["end_date"],
        "file_format": "csv",
        "data_kind": "ohlcv",
    })
    link = _request("GET", f"/artifacts/{export['result']['artifact_id']}/download")
    csv = requests.get(link["result"]["url"], timeout=600)
    csv.raise_for_status()
    return pd.read_csv(io.BytesIO(csv.content))


def _tidy(wide: pd.DataFrame, details: list[dict]) -> pd.DataFrame:
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    frames = []
    for detail in details:
        if detail.get("error"):
            continue
        ticker, alias = detail["ticker"], detail.get("alias") or ""
        columns = {}
        for field in FIELDS:
            for name in (f"{ticker}_{field}", f"{alias}_{field}", field):
                if name in wide.columns:
                    columns[name] = field
                    break
            else:
                raise DataloaderError(f"{ticker}: no {field} column in the answer")
        bars = wide[list(columns)].rename(columns=columns)
        bars.insert(0, "ticker", ticker)
        bars.insert(0, "timestamp", stamps)
        frames.append(bars.dropna(subset=["close"]))
    bars = pd.concat(frames, ignore_index=True)
    local = bars["timestamp"].dt.tz_convert("America/New_York")
    bars.insert(1, "date", local.dt.normalize().dt.tz_localize(None))
    return bars.sort_values(["ticker", "timestamp"], ignore_index=True)


def _load_batch(request: dict, batch: int) -> tuple[pd.DataFrame, list[dict], dict]:
    answer = _request("POST", "/fetch-ohlcvs", request)
    details = answer["result"].get("ticker_details") or []
    loaded = [d for d in details if not d.get("error")]
    wide = _bars_of(answer, request, batch) if loaded else pd.DataFrame()
    bars = _tidy(wide, details) if loaded else pd.DataFrame()
    notes = {
        "result_id": answer["result"].get("result_id") or answer.get("result_id"),
        "meta": answer.get("meta"),
    }
    return bars, details, notes


def load_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    *,
    adjust_dividends: bool = True,
    identity_date: str | None = None,
    yahoo_backstop: bool = True,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Daily OHLCV bars for ``tickers`` over the inclusive window ``[start_date,
    end_date]``, plus load metadata (``ticker_details``, ``failed``, ``warnings``,
    ``non_massive_source``).
    """
    request = {
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": adjust_dividends,
        "yahoo_backstop": yahoo_backstop,
    }
    if identity_date:
        request["identity_date"] = identity_date
    tickers = list(dict.fromkeys(tickers))
    key = hashlib.sha256(
        json.dumps({**request, "tickers": sorted(tickers)}, sort_keys=True).encode()
    ).hexdigest()[:16]
    bars_path, meta_path = CACHE_DIR / f"{key}.csv.gz", CACHE_DIR / f"{key}.json"
    if use_cache and bars_path.exists() and meta_path.exists():
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars, json.loads(meta_path.read_text())

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        results = list(pool.map(
            lambda item: _load_batch({**request, "tickers": item[1]}, item[0]),
            enumerate(batches),
        ))
    frames = [bars for bars, _, _ in results if not bars.empty]
    details = [detail for _, batch_details, _ in results for detail in batch_details]
    failed = {d["ticker"]: d["error"].get("message") for d in details if d.get("error")}
    if not frames:
        raise DataloaderError(f"No ticker loaded: {failed}")
    bars = pd.concat(frames, ignore_index=True)
    meta = {
        "request": request,
        "ticker_details": details,
        "failed": failed,
        "warnings": {d["ticker"]: d["warnings"] for d in details if d.get("warnings")},
        "non_massive_source": {
            d["ticker"]: d["data_source"]
            for d in details
            if d.get("data_source") not in (None, "massive")
        },
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_csv(bars_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return bars, meta


def load_fred_series(
    series: list[str], start_date: str, end_date: str, use_cache: bool = True
) -> pd.DataFrame:
    """Daily FRED series (e.g. DEXCAUS, DEXUSAL, DGS3MO), raw and un-forward-filled."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        json.dumps({"series": sorted(series), "s": start_date, "e": end_date}, sort_keys=True).encode()
    ).hexdigest()[:16]
    path = CACHE_DIR / f"fred_{key}.csv.gz"
    if use_cache and path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = pdr.DataReader(series, "fred", start=start_date, end=end_date)
    df.to_csv(path)
    return df


def load_etf_prices_usd(
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Daily adjusted close prices of the 30-ETF universe and the DJIA proxy,
    converted to USD, aligned to U.S. trading dates, and forward-filled.

    Returns ``(prices, risk_free_daily, meta)`` where ``prices`` has one column
    per ticker in :data:`src.config.ALL_TICKERS` plus
    :data:`src.config.DJIA_PROXY_TICKER`, and ``risk_free_daily`` is the daily
    simple-equivalent 3-month Treasury yield (Section 2.2, Eq. in 2.2).
    """
    tickers = config.ALL_TICKERS + [config.DJIA_PROXY_TICKER]
    bars, meta = load_ohlcv(
        tickers, config.SAMPLE_START, config.SAMPLE_END, "1d",
        adjust_dividends=True, identity_date=config.SAMPLE_END, use_cache=use_cache,
    )
    raw = bars.pivot(index="date", columns="ticker", values="close").sort_index()

    fred = load_fred_series(
        [config.FX_CAD_PER_USD_SERIES, config.FX_USD_PER_AUD_SERIES, config.RISK_FREE_SERIES],
        config.SAMPLE_START, config.SAMPLE_END, use_cache=use_cache,
    )
    fred = fred.reindex(raw.index).ffill()  # most recent preceding observation, Section 2.1/2.2

    usd_prices = raw.copy()
    for ticker in config.CA_TICKERS:
        if ticker in usd_prices.columns:
            usd_prices[ticker] = raw[ticker] / fred[config.FX_CAD_PER_USD_SERIES]
    for ticker in config.AU_TICKERS:
        if ticker in usd_prices.columns:
            usd_prices[ticker] = raw[ticker] * fred[config.FX_USD_PER_AUD_SERIES]

    n_before = usd_prices.isna().sum().sum()
    usd_prices = usd_prices.ffill()  # forward-fill missing ETF prices, Section 2.1
    meta["n_forward_filled"] = int(n_before - usd_prices.isna().sum().sum())
    meta["n_total_obs"] = int(usd_prices.shape[0] * usd_prices.shape[1])

    risk_free_annual_pct = fred[config.RISK_FREE_SERIES]
    risk_free_daily = (1 + risk_free_annual_pct / 100) ** (1 / config.TRADING_DAYS_PER_YEAR) - 1
    risk_free_daily.name = "risk_free_daily"

    return usd_prices, risk_free_daily, meta
