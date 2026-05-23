from apps.api.app.settings import Settings


def test_database_url_built_from_components():
    s = Settings(
        postgres_user="u",
        postgres_password="p",
        postgres_host="h",
        postgres_port=15432,
        postgres_db="d",
    )
    assert s.database_url == "postgresql+asyncpg://u:p@h:15432/d"
    assert s.sync_database_url == "postgresql+psycopg2://u:p@h:15432/d"


def test_defaults_have_sane_values():
    s = Settings()
    assert s.api_rate_limit_per_minute > 0
    assert s.redis_cache_ttl_seconds > 0
    assert s.kafka_prediction_topic
    assert s.triton_model_name
