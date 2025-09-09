.PHONY: setup data preprocess splits train eval export up down logs api drift test lint pipeline clean

# Environment variables
SESSION_GAP_MIN ?= 45
MAX_SEQ_LEN ?= 50
VOCAB_MIN_FREQ ?= 5

# Setup development environment
setup:
	pip install -r api/requirements.txt
	pip install -e ./core

# Data pipeline
data:
	python -m core.data.ingest --out data/raw

preprocess:
	python -m core.data.preprocess --in data/raw --out data/processed --gap $(SESSION_GAP_MIN) --max_len $(MAX_SEQ_LEN) --min_freq $(VOCAB_MIN_FREQ)

splits:
	python -m core.data.splits --in data/processed --out data/processed

# Training
train:
	python -m core.train.ray_train --config core/train/config.yaml

eval:
	python -m core.train.eval --in data/processed --ckpt data/artifacts/model.pt

export:
	python -m core.train.export --ckpt data/artifacts/model.pt --out data/artifacts

# Docker operations
up:
	docker compose -f docker/docker-compose.yml up -d

down:
	docker compose -f docker/docker-compose.yml down

logs:
	docker compose -f docker/docker-compose.yml logs -f api

# Development
api:
	uvicorn api.app.main:app --reload --port 8088

# Monitoring
drift:
	python -m core.monitor.drift --baseline data/reports/baseline.parquet --current data/processed/stats.parquet

# Testing and quality
test:
	pytest -q

lint:
	ruff check . && black --check .

# Full pipeline
pipeline: data preprocess splits train export

# Cleanup
clean:
	rm -rf data/processed data/artifacts data/reports

# Kubernetes deployment
k8s-deploy:
	kubectl apply -f k8s/raycluster.yaml
	kubectl apply -f k8s/api-deployment.yaml
	kubectl apply -f k8s/api-service.yaml
	kubectl apply -f k8s/hpa.yaml

k8s-clean:
	kubectl delete -f k8s/hpa.yaml
	kubectl delete -f k8s/api-service.yaml
	kubectl delete -f k8s/api-deployment.yaml
	kubectl delete -f k8s/raycluster.yaml

