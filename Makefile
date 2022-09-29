all: venv

venv:
	python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

clean:
	rm -rf logs/fit/*
