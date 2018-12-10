.PHONY: clean download
PY_SOURCES := $(wildcard src/*.py)

clean:
	-rm -r *.pyc tags .pytest_cache

tags: $(PY_SOURCES)
	ctags -a --languages=python $(PY_SOURCES)

download: models/googlenet/bvlc_googlenet.caffemodel ;

models/googlenet/bvlc_googlenet.caffemodel:
	wget -O $@ http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel

html: $(PY_SOURCES)
	make -C docs html > /dev/null

