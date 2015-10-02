BLASX_DIR = ..
include ../Makefile.internal

1SRC := \
    asum.c   \
    axpy.c   \
    copy.c   \
    dot.c    \
    nrm2.c   \
    rot.c    \
    scal.c   \
    swap.c   \
    amax.c   \
    lsame.c  \

Z2SRC := \
    zgemv.c  \
    zgbmv.c  \
    zgerc.c  \
    zgeru.c  \
    zhbmv.c  \
    zhemv.c  \
    zher.c   \
    zher2.c  \
    zhpmv.c  \
    zhpr.c   \
    zhpr2.c  \
    ztbmv.c  \
    ztbsv.c  \
    ztpmv.c  \
    ztpsv.c  \
    ztrmv.c  \
    ztrsv.c  \

C2SRC := \
    cgemv.c  \
    cgbmv.c  \
    cgerc.c  \
    cgeru.c  \
    chbmv.c  \
    chemv.c  \
    cher.c   \
    cher2.c  \
    chpmv.c  \
    chpr.c   \
    chpr2.c  \
    ctbmv.c  \
    ctbsv.c  \
    ctpmv.c  \
    ctpsv.c  \
    ctrmv.c  \
    ctrsv.c  \

S2SRC := \
    sgemv.c  \
    sgbmv.c  \
    strmv.c  \
    stbmv.c  \
    stpmv.c  \
    strsv.c  \
    sger.c   \
    ssbmv.c  \
    sspmv.c  \
    sspr.c   \
    sspr2.c  \
    ssymv.c  \
    ssyr.c   \
    ssyr2.c  \
    stbsv.c  \
    stpsv.c  \

D2SRC := \
    dgemv.c  \
    dgbmv.c  \
    dtrmv.c  \
    dtbmv.c  \
    dtpmv.c  \
    dtrsv.c  \
    dger.c   \
    dsbmv.c  \
    dspmv.c  \
    dspr.c   \
    dspr2.c  \
    dsymv.c  \
    dsyr.c   \
    dsyr2.c  \
    dtbsv.c  \
    dtpsv.c  \

TEST := \
    sgemm.c  \
    dgemm.c  \
    cgemm.c  \
    zgemm.c  \
    ssyrk.c  \
    dsyrk.c  \
    csyrk.c  \
    zsyrk.c  \
    ssyr2k.c \
    dsyr2k.c \
    csyr2k.c \
    zsyr2k.c \
    strmm.c  \
    dtrmm.c  \
    ctrmm.c  \
    ztrmm.c  \
    strsm.c  \
    dtrsm.c  \
    ctrsm.c  \
    ztrsm.c  \
    ssymm.c  \
    dsymm.c  \
    csymm.c  \
    zsymm.c  \
    zhemm.c  \
    chemm.c  \
    zherk.c  \
    cherk.c  \
    zher2k.c \
    cher2k.c \


OTHER := \
    blasx_tile_resource.c  \
    blasx_affinity.c       \
    red_black_tree.c       \
    LRU.c                  \
    gpu_malloc.c           \
    blasx_mem_control.c    \
    blasx_task_queue.c     \
    error.c                \
    blasx_mutex.c          \
    blasx_internal.c       \
    blasx_globalpointers.c \
    blasx_sgemm.c          \
	blasx_dgemm.c          \

ALLSRC := $(OTHER) $(TEST) $(1SRC) $(Z2SRC) $(C2SRC) $(S2SRC) $(D2SRC)

ALLOBJ := $(ALLSRC:.c=.o)

.PHONY: all clean

all: $(LIBBLASX)

$(LIBBLASX): $(ALLOBJ)
	mkdir -p $(BLASX_DIR)/lib
	$(CC) $(CFLAGS) -shared -fPIC $(LIBBLAS) -g -o $(LIBBLASX_DYLIB) $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ) 
	$(RANLIB) $@

.c.o :
	$(CC) -w -g $(CFLAGS) $(INC) -DCPUBLAS='$(LIBCPUBLAS)' -c -fPIC $< -o $@

clean:
	rm -f *.o
