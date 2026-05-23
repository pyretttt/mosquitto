# OpenSearch / Dashboards

After `make up-all`, open http://localhost:5601 and create a data view for
`mlops-logs-*`. From there:

- Filter on `service: "api"` to see request logs as JSON.
- Use the `top_class`, `latency_ms`, and `image_sha` fields directly thanks to
  the JSON parser in `infra/vector/vector.toml`.

To bootstrap the data view from the CLI once OpenSearch is up:

```bash
curl -X POST http://localhost:5601/api/saved_objects/index-pattern/mlops-logs \
  -H 'osd-xsrf: true' -H 'Content-Type: application/json' \
  -d '{"attributes":{"title":"mlops-logs-*","timeFieldName":"timestamp"}}'
```

The advanced track covers ILM rollover, ingest pipelines, and alerts.
