"""Stanford CoreNLP POS-tagging utilities for BESSTIE-derived CSV files."""

from __future__ import annotations

import csv
import http.client
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.parse import quote, urlparse

import pandas as pd


_UNIVERSAL_TAG_MAP = {
    "CC": "CONJ",
    "CD": "NUM",
    "DT": "DET",
    "EX": "PRON",
    "FW": "X",
    "IN": "ADP",
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    "LS": "X",
    "MD": "VERB",
    "NN": "NOUN",
    "NNS": "NOUN",
    "NNP": "PROPN",
    "NNPS": "PROPN",
    "PDT": "DET",
    "POS": "PRT",
    "PRP": "PRON",
    "PRP$": "PRON",
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
    "RP": "PRT",
    "SYM": "X",
    "TO": "PRT",
    "UH": "INTJ",
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    "WDT": "DET",
    "WP": "PRON",
    "WP$": "PRON",
    "WRB": "ADV",
    ".": "PUNCT",
    ",": "PUNCT",
    ":": "PUNCT",
    "(": "PUNCT",
    ")": "PUNCT",
    "\"": "PUNCT",
    "#": "SYM",
    "$": "SYM",
}

SUMMARY_TAGS = (
    "NOUN",
    "PROPN",
    "VERB",
    "ADJ",
    "ADV",
    "PRON",
    "ADP",
    "DET",
    "NUM",
    "PRT",
    "CONJ",
    "INTJ",
    "PUNCT",
    "X",
)


@dataclass(frozen=True)
class PosTagRecord:
    example_id: str
    source: str
    subset: str
    variety: str
    sentiment: int
    sarcasm: int
    text: str
    token_count: int
    tokens: list[str]
    ptb_tags: list[str]
    universal_tags: list[str]
    tag_counts: dict[str, int]


class CoreNlpPosTagger:
    """Small Stanford CoreNLP server client for tokenization and POS tagging."""

    def __init__(
        self,
        server_url: str = "http://localhost:9000",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def annotate(self, text: str) -> dict[str, object]:
        properties = {
            "annotators": "tokenize,ssplit,pos",
            "outputFormat": "json",
        }
        parsed = urlparse(self.server_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"Unsupported CoreNLP server URL scheme: {self.server_url!r}"
            )
        properties_json = json.dumps(properties, separators=(",", ":"))
        path = f"/?properties={quote(properties_json, safe='')}"
        body = text.encode("utf-8")
        headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Content-Length": str(len(body)),
            "Connection": "close",
            "Accept": "application/json",
        }
        connection_class = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_class(
            parsed.hostname,
            parsed.port,
            timeout=self.timeout_seconds,
        )
        try:
            connection.request("POST", path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read().decode("utf-8", errors="replace")
            if response.status >= 400:
                raise RuntimeError(
                    f"Stanford CoreNLP returned HTTP {response.status}: {payload}"
                )
            return json.loads(payload)
        except http.client.BadStatusLine as exc:
            raise RuntimeError(
                "Stanford CoreNLP returned an invalid HTTP response. "
                "This often means the local server rejected the request formatting "
                "or the target port is not the CoreNLP HTTP server."
            ) from exc
        except http.client.HTTPException as exc:
            raise RuntimeError(
                f"Stanford CoreNLP HTTP error: {exc}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                "Could not reach the Stanford CoreNLP server at "
                f"{self.server_url}. Start it first, for example with:\n"
                'java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer '
                "-port 9000 -timeout 30000"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                "Timed out waiting for Stanford CoreNLP. "
                f"Current timeout: {self.timeout_seconds} seconds."
            ) from exc
        finally:
            connection.close()


def _to_universal_tag(tag: str) -> str:
    return _UNIVERSAL_TAG_MAP.get(tag, "X")


def pos_tag_text(
    text: str,
    tagger: CoreNlpPosTagger,
) -> tuple[list[str], list[str], list[str]]:
    annotation = tagger.annotate(text)
    sentences = annotation.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError(
            "Stanford CoreNLP response did not contain a `sentences` list."
        )

    tokens: list[str] = []
    ptb_tags: list[str] = []
    for sentence in sentences:
        sentence_tokens = sentence.get("tokens", [])
        if not isinstance(sentence_tokens, list):
            continue
        for token in sentence_tokens:
            token_text = token.get("originalText", token.get("word", ""))
            tokens.append(str(token_text))
            ptb_tags.append(str(token.get("pos", "")))

    universal_tags = [_to_universal_tag(tag) for tag in ptb_tags]
    return tokens, ptb_tags, universal_tags


def build_pos_record(
    row: pd.Series,
    tagger: CoreNlpPosTagger,
) -> PosTagRecord:
    tokens, ptb_tags, universal_tags = pos_tag_text(str(row["text"]), tagger)
    counts = Counter(universal_tags)
    return PosTagRecord(
        example_id=str(row["example_id"]),
        source=str(row["source"]),
        subset=str(row["subset"]),
        variety=str(row["variety"]),
        sentiment=int(row["sentiment"]),
        sarcasm=int(row["sarcasm"]),
        text=str(row["text"]),
        token_count=len(tokens),
        tokens=tokens,
        ptb_tags=ptb_tags,
        universal_tags=universal_tags,
        tag_counts={tag: int(counts.get(tag, 0)) for tag in SUMMARY_TAGS},
    )


def _jsonl_records(path: Path) -> Iterable[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_processed_example_ids(jsonl_path: Path) -> set[str]:
    return {
        str(record["example_id"])
        for record in _jsonl_records(jsonl_path)
        if "example_id" in record
    }


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def summary_rows_from_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in _jsonl_records(path):
        tag_counts = {
            tag: int(record.get("tag_counts", {}).get(tag, 0)) for tag in SUMMARY_TAGS
        }
        rows.append(
            {
                "example_id": record["example_id"],
                "source": record["source"],
                "subset": record["subset"],
                "variety": record["variety"],
                "sentiment": int(record["sentiment"]),
                "sarcasm": int(record["sarcasm"]),
                "token_count": int(record["token_count"]),
                **tag_counts,
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "example_id",
        "source",
        "subset",
        "variety",
        "sentiment",
        "sarcasm",
        "token_count",
        *SUMMARY_TAGS,
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_csv_to_pos_artifacts(
    input_csv: Path,
    output_dir: Path,
    server_url: str = "http://localhost:9000",
    timeout_seconds: float = 30.0,
    limit: int | None = None,
) -> dict[str, object]:
    frame = pd.read_csv(input_csv)
    if limit is not None:
        frame = frame.iloc[:limit].copy()

    tagger = CoreNlpPosTagger(
        server_url=server_url,
        timeout_seconds=timeout_seconds,
    )
    detailed_path = output_dir / f"{input_csv.stem}.pos.jsonl"
    summary_path = output_dir / f"{input_csv.stem}.pos_summary.csv"
    processed_ids = load_processed_example_ids(detailed_path)

    for _, row in frame.iterrows():
        example_id = str(row["example_id"])
        if example_id in processed_ids:
            continue
        record = build_pos_record(row, tagger)
        append_jsonl(
            detailed_path,
            {
                "example_id": record.example_id,
                "source": record.source,
                "subset": record.subset,
                "variety": record.variety,
                "sentiment": record.sentiment,
                "sarcasm": record.sarcasm,
                "text": record.text,
                "token_count": record.token_count,
                "tokens": record.tokens,
                "ptb_tags": record.ptb_tags,
                "universal_tags": record.universal_tags,
                "tag_counts": record.tag_counts,
            },
        )

    summary_rows = summary_rows_from_jsonl(detailed_path)
    allowed_ids = {str(value) for value in frame["example_id"].tolist()}
    summary_rows = [row for row in summary_rows if str(row["example_id"]) in allowed_ids]
    write_summary_csv(summary_path, summary_rows)

    return {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "server_url": server_url,
        "rows_requested": int(len(frame)),
        "rows_summarized": int(len(summary_rows)),
        "detailed_jsonl": str(detailed_path),
        "summary_csv": str(summary_path),
    }
