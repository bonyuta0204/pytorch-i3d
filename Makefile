.PHONY: clean download test
PY_SOURCES := $(wildcard src/*.py)

clean:
	-rm -r *.pyc tags .pytest_cache __pycache__ nohup.out src/*.pyc experiment/*.pyc test/*.pyc src/__pycache__ test/__pycache__ experiment/__pycache__

tags: $(PY_SOURCES)
	ctags -a --languages=python $(PY_SOURCES)

download: models/googlenet/bvlc_googlenet.caffemodel

models/googlenet/bvlc_googlenet.caffemodel:
	wget -O $@ http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel

html: $(PY_SOURCES)
	make -C docs html

test:
	pytest -v test/
