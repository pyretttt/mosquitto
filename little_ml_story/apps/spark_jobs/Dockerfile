FROM bitnami/spark:3.5.3

USER root

# Postgres JDBC driver (Spark JDBC sink) and ClickHouse JDBC driver (source).
ENV PG_JDBC_VERSION=42.7.4 \
    CH_JDBC_VERSION=0.7.1

RUN mkdir -p /opt/spark/jars \
    && curl -fsSL -o /opt/spark/jars/postgresql.jar \
       "https://repo1.maven.org/maven2/org/postgresql/postgresql/${PG_JDBC_VERSION}/postgresql-${PG_JDBC_VERSION}.jar" \
    && curl -fsSL -o /opt/spark/jars/clickhouse-jdbc.jar \
       "https://repo1.maven.org/maven2/com/clickhouse/clickhouse-jdbc/${CH_JDBC_VERSION}/clickhouse-jdbc-${CH_JDBC_VERSION}-shaded-all.jar"

WORKDIR /app
COPY apps /app/apps

USER 1001

ENTRYPOINT []
CMD ["spark-submit", "--master", "local[2]", "/app/apps/spark_jobs/daily_aggregations.py"]
