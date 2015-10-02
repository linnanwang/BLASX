BLASX_DIR = ..
include ../Makefile.internal
LIB = -L$(BLASX_DIR)/lib -lblasx $(LIBGPUBLAS)
INC = -I../include -I/opt/local/include
#LIB = -L/opt/local/lib -lopenblas
#LIB = -L/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Versions/Current -lBLAS
CPLUS=gcc

DSRC :=	gemm
	
ALLSRC = $(DSRC:%=%.c)
ALLOBJ = $(ALLSRC:.c=.o)
ALLEXE := $(DSRC)

all:$(ALLEXE)	

$(ALLOBJ): %.o: %.c
	$(CC) -c -g $(CFLAGS) $(INC) -I/ $< -o $@
	
$(ALLEXE): % : %.o
	$(CPLUS) -g $< $(LIB) -o $@
	
clean:
	rm -f *.o
	rm -f $(DSRC)
.PHONY: all clean

