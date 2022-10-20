all: venv dogscats
# nbqa pylint --disable=C0413,C0103,W0621 mlp.ipynb

venv:
	python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

dogscats:
	wget http://files.fast.ai/data/examples/dogscats.tgz
	tar xzf dogscats.tgz

clean:
	rm -rf logs/fit/*
