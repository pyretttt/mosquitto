SHELL := /bin/bash
COMPOSE := docker compose -f docker-compose.yml -f docker-compose.observability.yml

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.PHONY: env
env: ## Copy .env.example to .env if missing
	@test -f .env || cp .env.example .env

.PHONY: fetch-model
fetch-model: ## Download ResNet50 ONNX into Triton model repository
	bash infra/triton/fetch_model.sh

.PHONY: up
up: env fetch-model ## Boot core stack (api, triton, postgres, redis, redpanda, clickhouse, consumer)
	docker compose -f docker-compose.yml up -d --build

.PHONY: up-all
up-all: env fetch-model ## Boot core + observability stacks
	$(COMPOSE) up -d --build

.PHONY: down
down: ## Stop all containers
	$(COMPOSE) down

.PHONY: nuke
nuke: ## Stop everything and remove volumes
	$(COMPOSE) down -v --remove-orphans

.PHONY: logs
logs: ## Tail logs (svc=<name> to filter)
	$(COMPOSE) logs -f $(svc)

.PHONY: ps
ps: ## Show container status
	$(COMPOSE) ps

.PHONY: migrate
migrate: ## Run alembic upgrade head
	docker compose -f docker-compose.yml run --rm api alembic -c apps/api/alembic.ini upgrade head

.PHONY: migration
migration: ## Autogenerate a new alembic revision: make migration m="msg"
	docker compose -f docker-compose.yml run --rm api alembic -c apps/api/alembic.ini revision --autogenerate -m "$(m)"

.PHONY: predict
predict: ## Send a sample image to /predict (set IMG=./path/to/img.jpg)
	@curl -s -F "file=@$${IMG:-tests/fixtures/cat.jpg}" http://localhost:8000/predict | jq .

.PHONY: test
test: ## Run unit tests inside the api container
	docker compose -f docker-compose.yml run --rm api pytest -q tests/unit

.PHONY: test-e2e
test-e2e: ## Run e2e tests against running compose
	pytest -q tests/e2e

.PHONY: spark-batch
spark-batch: ## Run yesterday's Spark batch rollup
	docker compose -f docker-compose.yml run --rm spark \
		spark-submit --master local[2] /app/apps/spark_jobs/daily_aggregations.py

.PHONY: load
load: ## Hit /predict in a hot loop (requires k6)
	k6 run infra/load/k6.js

.PHONY: kind-up
kind-up: ## Create a local kind cluster
	kind create cluster --config infra/k8s/kind-config.yaml --name little-ml-story

.PHONY: kind-down
kind-down: ## Tear down kind cluster
	kind delete cluster --name little-ml-story

.PHONY: helm-install
helm-install: ## Install api chart on current kube context
	kubectl apply -f infra/k8s/manifests/
	helm upgrade --install api infra/k8s/helm/api --namespace mlops --create-namespace

.PHONY: helm-uninstall
helm-uninstall: ## Uninstall api chart
	helm uninstall api -n mlops || true
	kubectl delete -f infra/k8s/manifests/ || true
