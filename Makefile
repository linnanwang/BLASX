# Overwritten in make.inc
BLASX_DIR = .
include ./Makefile.internal

all: lib testing

lib: libblasx_blas

libblasx_blas:
	@echo ======================================== libblasX:blas implementation
	( cd blas      && $(MAKE) )

lib: testing
	@echo ======================================== testing
	( cd testing     && $(MAKE) )

clean:
	(cd blas       	  && $(MAKE) clean )
	(cd testing       && $(MAKE) clean )

cleanall:
	(cd blas       && $(MAKE) clean )
	(cd testing       && $(MAKE) clean )
	(cd lib        && rm -f * )
