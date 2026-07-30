SUBDIRS = cuda
 
all: compression
	python3 setup.py install 
 
compression:
	$(MAKE) -C cuda
 
clean:
	rm -rf build dist ./*.egg-info
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir Makefile $@; done
 

