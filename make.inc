#Those variables have to be changed accordingly!
# Compilers, linker/loaders, the archiver, and their options.

CC        = gcc
FC        = gfortran
CPLUS     = g++

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib

CFLAGS    =

# Blas Library
LIBCPUBLAS = "/opt/local/lib/libopenblas.dylib"
#LIBCPUBLAS = "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Versions/Current/libBLAS.dylib"
LIBGPUBLAS = -L/usr/local/cuda/lib -lcudart -lcublas -lcuda
LIBBLAS   = $(LIBGPUBLAS)

#BLAS Include
INCBLAS   = -I/usr/local/cuda/include
