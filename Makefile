CC=			g++
CFLAGS=		-g -mavx2 -mbmi -Wall -Wno-unused-function -Wsign-compare -O2 
AR=			ar
DFLAGS=		
LOBJS=		utils.o kstring.o ksw.o bwt.o bntseq.o bwa.o bwamem.o bwamem_pair.o bwamem_extra.o
AOBJS=		rope.o rle.o bwashm.o bwase.o bwaseqio.o bwtgap.o bwtaln.o bamlite.o \
			is.o bwtindex.o bwape.o kopen.o pemerge.o maxk.o \
			bwtsw2_core.o bwtsw2_main.o bwtsw2_aux.o bwt_lite.o \
			bwtsw2_chain.o fastmap.o bwtsw2_pair.o \
			bwaperf.o bwt_avx2.o ksw_align_avx2.o ksw_extend_avx2.o
PROG=		bwa
INCLUDES=	-Ivcpkg/installed/x64-linux/include
LIBS=		-Lvcpkg/installed/x64-linux/lib -lm -lz -ltbb -ltbbmalloc -lpthread
SUBDIRS=	.

ifeq ($(shell uname -s),Linux)
	LIBS += -lrt
endif

.SUFFIXES:.c .o .cc .cpp

.c.o:
		$(CC) -c $(CFLAGS) $(DFLAGS) $(INCLUDES) $< -o $@

.cpp.o:
		$(CC) -c $(CFLAGS) $(DFLAGS) $(INCLUDES) $< -o $@

all:$(PROG)

bwa:libbwa.a $(AOBJS) main.o
		$(CC) $(CFLAGS) $(DFLAGS) $(AOBJS) main.o -o $@ -L. -lbwa $(LIBS)

bwamem-lite:libbwa.a example.o
		$(CC) $(CFLAGS) $(DFLAGS) example.o -o $@ -L. -lbwa $(LIBS)

bwamem-snap:libbwa.a example-snap.o
		$(CC) $(CFLAGS) $(DFLAGS) example-snap.o -o $@ -L. -lbwa $(LIBS)

libbwa.a:$(LOBJS)
		$(AR) -csru $@ $(LOBJS)

clean:
		rm -f gmon.out *.o a.out $(PROG) *~ *.a

depend:
	( LC_ALL=C ; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c )

# DO NOT DELETE THIS LINE -- make depend depends on it.

QSufSort.o: QSufSort.h
bamlite.o: bamlite.h
bntseq.o: bntseq.h utils.h kseq.h khash.h
bwa.o: bntseq.h bwa.h bwt.h ksw.h utils.h kstring.h kvec.h
bwa.o: kseq.h
bwamem.o: kstring.h bwamem.h bwt.h bntseq.h bwa.h ksw.h kvec.h
bwamem.o: ksort.h utils.h kbtree.h
bwamem_extra.o: bwa.h bntseq.h bwt.h bwamem.h kstring.h
bwamem_pair.o: kstring.h bwamem.h bwt.h bntseq.h bwa.h kvec.h
bwamem_pair.o: utils.h ksw.h
bwape.o: bwtaln.h bwt.h kvec.h bntseq.h utils.h bwase.h bwa.h
bwape.o: ksw.h khash.h
bwase.o: bwase.h bntseq.h bwt.h bwtaln.h utils.h kstring.h
bwase.o: bwa.h ksw.h
bwaseqio.o: bwtaln.h bwt.h utils.h bamlite.h kseq.h
bwashm.o: bwa.h bntseq.h bwt.h
bwt.o: utils.h bwt.h kvec.h
bwt_avx2.o: utils.h bwt.h bwt_avx.h bwt_perf.h kvec.h
bwt_lite.o: bwt_lite.h
bwt_perf.o: bwt_perf.h
bwtaln.o: bwtaln.h bwt.h bwtgap.h utils.h bwa.h bntseq.h
bwtgap.o: bwtgap.h bwt.h bwtaln.h
bwtindex.o: bntseq.h bwa.h bwt.h utils.h rle.h rope.h
bwtsw2_aux.o: bntseq.h bwt_lite.h utils.h bwtsw2.h bwt.h kstring.h
bwtsw2_aux.o: bwa.h ksw.h kseq.h ksort.h
bwtsw2_chain.o: bwtsw2.h bntseq.h bwt_lite.h bwt.h ksort.h
bwtsw2_core.o: bwt_lite.h bwtsw2.h bntseq.h bwt.h kvec.h
bwtsw2_core.o: khash.h ksort.h
bwtsw2_main.o: bwt.h bwtsw2.h bntseq.h bwt_lite.h utils.h bwa.h
bwtsw2_pair.o: utils.h bwt.h bntseq.h bwtsw2.h bwt_lite.h kstring.h ksw.h
example.o: bwamem.h bwt.h bntseq.h bwa.h kseq.h
fastmap.o: bwa.h bntseq.h bwt.h bwamem.h kvec.h utils.h kseq.h
kstring.o: kstring.h
ksw.o: ksw.h
ksw_align_avx2.o: ksw.h ksw_avx.h ksw_perf.h
ksw_extend_avx2.o: ksw.h ksw_avx.h ksw_perf.h
main.o: kstring.h utils.h bwt_perf.h
maxk.o: bwa.h bntseq.h bwt.h bwamem.h kseq.h
pemerge.o: ksw.h kseq.h kstring.h bwa.h bntseq.h bwt.h utils.h
rle.o: rle.h
rope.o: rle.h rope.h
utils.o: utils.h ksort.h kseq.h
