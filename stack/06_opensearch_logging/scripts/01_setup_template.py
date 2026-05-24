"""Create an index template for our app logs.

A template = mappings + settings applied automatically to any new index
matching the pattern. This is how `logs-mlops-2026.05.24` born today gets
the same schema as the one born next month.

Key mapping decisions:
- request_id, model_name, model_version: `keyword`  -> exact match + aggs
- message: `text`                                   -> full-text search
- latency_ms: `float`                               -> numeric aggs
- @timestamp: `date`                                -> Dashboards time filter
"""

from opensearchpy import OpenSearch

INDEX_PATTERN = "logs-mlops-*"
TEMPLATE_NAME = "logs-mlops-template"

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,
    use_ssl=False,
)

template_body = {
    "index_patterns": [INDEX_PATTERN],
    "priority": 100,
    "template": {
        "settings": {
            "number_of_shards": 1,    # tiny data -> 1 is plenty locally
            "number_of_replicas": 0,  # single-node cluster
            "refresh_interval": "1s",
        },
        "mappings": {
            "properties": {
                "@timestamp":    {"type": "date"},
                "level":         {"type": "keyword"},
                "service":       {"type": "keyword"},
                "request_id":    {"type": "keyword"},
                "user_id":       {"type": "long"},
                "model_name":    {"type": "keyword"},
                "model_version": {"type": "keyword"},
                "latency_ms":    {"type": "float"},
                "status_code":   {"type": "short"},
                "event":         {"type": "keyword"},
                "message":       {"type": "text"},   # analysed for FTS
                "exception":     {"type": "text"},
            },
            # Anything we forgot to map -> keep as keyword (no analysis).
            "dynamic_templates": [
                {"strings_as_keyword": {
                    "match_mapping_type": "string",
                    "mapping": {"type": "keyword"},
                }}
            ],
        },
    },
}


def main() -> None:
    resp = client.indices.put_index_template(name=TEMPLATE_NAME, body=template_body)
    print("template:", resp)
    print(f"any new index matching {INDEX_PATTERN!r} now uses this template.")


if __name__ == "__main__":
    main()
