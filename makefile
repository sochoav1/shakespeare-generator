start-back:
	uvicorn backend.main:app --reload

start-front:
	cd frontend && npm start
	
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C,W1203,W0702 app.py

test:
	python -m pytest -vv --cov=app test_app.py

format:
	black *.py

all: install lint test format
