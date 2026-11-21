.PHONY: up down logs test

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

test:
	cd backend && LYFTR_TEST_MODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -v
