.DEFAULT_GOAL := all
all: build


tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl:
	wget https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
build: tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
	tox -edev --notest
clean: 
	rm -rf venv
