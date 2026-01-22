"""Utilities for downloading historical market data.

This module currently supports pulling time-series data from Yahoo Finance
via `yfinance` and saving it to Excel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def normalize_yahoo_ticker(ticker: str) -> str:
	t = str(ticker).strip()
	if len(t) >= 2 and ((t[0] == t[-1] == "'") or (t[0] == t[-1] == '"')):
		t = t[1:-1].strip()

	upper = t.upper().replace(" ", "")
	# Users often type the index ticker without the leading caret due to shell escaping.
	if upper == "GSPC":
		return "^GSPC"
	if upper in {"SP500", "S&P500", "SNP500", "SPX", "SPX500"}:
		return "^GSPC"

	return t


def download_yfinance_history(
	ticker: str,
	start: str | pd.Timestamp,
	end: str | pd.Timestamp,
	*,
	end_inclusive: bool = True,
	interval: str = "1d",
) -> pd.DataFrame:
	"""Download OHLCV history from Yahoo Finance.

	Notes:
		`yfinance.download` treats `end` as an exclusive bound in practice.
		When `end_inclusive=True`, we add one day to ensure the end date is
		included (useful for daily bars).
	"""

	start_ts = pd.Timestamp(start).normalize()
	end_ts = pd.Timestamp(end).normalize()
	if end_inclusive:
		end_ts = end_ts + pd.Timedelta(days=1)

	ticker = normalize_yahoo_ticker(ticker)

	start_str = start_ts.strftime("%Y-%m-%d")
	end_str = end_ts.strftime("%Y-%m-%d")

	def _download(t: str) -> pd.DataFrame:
		return yf.download(
			tickers=t,
			start=start_str,
			end=end_str,
			interval=interval,
			auto_adjust=False,
			progress=False,
			group_by="column",
			threads=False,
		)

	def _history(t: str) -> pd.DataFrame:
		# Some Yahoo index tickers can fail metadata/timezone resolution via download().
		return yf.Ticker(t).history(
			start=start_str,
			end=end_str,
			interval=interval,
			auto_adjust=False,
			actions=False,
		)

	df = _download(ticker)
	if df is None or df.empty:
		df = _history(ticker)

	# Yahoo index tickers often start with '^'. If the caret isn't encoded correctly upstream,
	# Yahoo may respond as if the symbol were missing. Try an explicit %5E-encoded ticker.
	if (df is None or df.empty) and ticker.startswith("^"):
		encoded = "%5E" + ticker[1:]
		df = _download(encoded)
		if df is None or df.empty:
			df = _history(encoded)

	if df is None or df.empty:
		end_inclusive_date = (end_ts - pd.Timedelta(days=1)).date() if end_inclusive else end_ts.date()
		raise RuntimeError(
			f"No data returned for ticker={ticker!r} in range {start_ts.date()}..{end_inclusive_date}. "
			"If you are trying an index like ^GSPC, ensure the caret is passed literally (PowerShell: --ticker '^GSPC')."
		)

	df.index.name = "Date"

	# Ensure a simple, flat column index.
	if isinstance(df.columns, pd.MultiIndex):
		df.columns = ["_".join([str(x) for x in col if x]) for col in df.columns.to_list()]

	return df


def save_history_to_excel(df: pd.DataFrame, output_path: str | Path, *, sheet_name: str = "data") -> Path:
	"""Save a history DataFrame to an Excel file."""

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_excel(output_path, sheet_name=sheet_name)
	return output_path


def download_sp500_to_excel(
	*,
	start: str = "2016-01-01",
	end: str = "2024-12-31",
	output_path: str | Path | None = None,
	interval: str = "1d",
) -> Path:
	"""Download S&P 500 (Yahoo ticker: ^GSPC) and save to an Excel file."""

	if output_path is None:
		output_path = Path(__file__).resolve().parent / "historical_sp500_2016_2024.xlsx"

	df = download_yfinance_history("^GSPC", start=start, end=end, end_inclusive=True, interval=interval)
	return save_history_to_excel(df, output_path, sheet_name="^GSPC")


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Download Yahoo Finance data and save it to Excel.")
	parser.add_argument(
		"--ticker",
		default="^GSPC",
		help="Yahoo Finance ticker (default: ^GSPC for S&P 500)",
	)
	parser.add_argument("--start", default="2016-01-01", help="Start date (YYYY-MM-DD)")
	parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD), inclusive")
	parser.add_argument("--interval", default="1d", help="Sampling interval, e.g. 1d, 1wk")
	parser.add_argument(
		"--out",
		default=None,
		help="Output Excel path. Default: data/historical_sp500_2016_2024.xlsx",
	)
	return parser


def main() -> int:
	parser = _build_arg_parser()
	args = parser.parse_args()

	ticker = normalize_yahoo_ticker(args.ticker)

	if args.out is None:
		out_path = Path(__file__).resolve().parent / "historical_sp500_2016_2024.xlsx"
	else:
		out_path = Path(args.out)

	df = download_yfinance_history(
		ticker,
		start=args.start,
		end=args.end,
		end_inclusive=True,
		interval=args.interval,
	)
	saved = save_history_to_excel(df, out_path, sheet_name=ticker)
	print(f"Saved {ticker} history to: {saved}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
