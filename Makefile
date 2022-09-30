all: venv dogscats

venv:
	python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

dogscats:
	wget http://files.fast.ai/data/examples/dogscats.tgz
	tar xzf dogscats.tgz

clean:
	rm -rf logs/fit/*
