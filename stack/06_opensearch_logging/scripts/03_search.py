"""Three queries you'll write a thousand times in your MLOps career."""

import json

from opensearchpy import OpenSearch

INDEX_PATTERN = "logs-mlops-*"
client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])


Q_ERRORS = {
    "query": {
        "bool": {
            "must": [{"term": {"level": "ERROR"}}],
            "filter": [{"range": {"@timestamp": {"gte": "now-24h"}}}],
        }
    },
    "size": 5,
    "sort": [{"@timestamp": "desc"}],
    "_source": ["@timestamp", "request_id", "model_name", "model_version", "exception"],
}


Q_SLOWEST = {
    "query": {"term": {"event": "inference_done"}},
    "sort": [{"latency_ms": "desc"}],
    "size": 5,
    "_source": ["@timestamp", "request_id", "model_version", "latency_ms"],
}


Q_ERROR_RATE_BY_VERSION = {
    "size": 0,
    "aggs": {
        "by_version": {
            "terms": {"field": "model_version", "size": 10},
            "aggs": {
                "errors": {
                    "filter": {"term": {"level": "ERROR"}}
                }
            }
        }
    }
}


def run(title: str, body: dict) -> None:
    print(f"\n=== {title} ===")
    res = client.search(index=INDEX_PATTERN, body=body)
    print(json.dumps(res.get("hits", {}).get("hits", []), indent=2, default=str)[:1500])
    if "aggregations" in res:
        print("\naggs:")
        print(json.dumps(res["aggregations"], indent=2))


def main() -> None:
    run("most recent errors (24h)", Q_ERRORS)
    run("slowest 5 inferences", Q_SLOWEST)
    run("docs per model_version + error count", Q_ERROR_RATE_BY_VERSION)

    print("\nTip: in real life, error_rate = errors.doc_count / by_version.doc_count")


if __name__ == "__main__":
    main()
