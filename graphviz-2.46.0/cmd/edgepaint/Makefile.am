# $Id$ $Revision$
## Process this file with automake to produce Makefile.in

pdfdir = $(pkgdatadir)/doc/pdf

AM_CPPFLAGS = \
	-I$(top_srcdir)/lib \
	-I$(top_srcdir)/lib/common \
	-I$(top_srcdir)/lib/cgraph \
	-I$(top_srcdir)/lib/cdt

bin_PROGRAMS = edgepaint

man_MANS = edgepaint.1
if ENABLE_MAN_PDFS
pdf_DATA = edgepaint.1.pdf

if HAVE_PS2PDF
pdf = $(pdf_DATA)
else
pdf = ERROR_NO_PS2PDF
endif
else
pdf =
endif

edgepaint_SOURCES = edgepaintmain.c
edgepaint_CPPFLAGS = $(AM_CPPFLAGS)
edgepaint_LDADD = \
    $(top_builddir)/lib/edgepaint/libedgepaint_C.la \
    $(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/sfdpgen/libsfdpgen_C.la \
	$(top_builddir)/lib/neatogen/libneatogen_C.la \
	$(top_builddir)/lib/sparse/libsparse_C.la \
	$(top_builddir)/lib/rbtree/librbtree_C.la \
	$(top_builddir)/lib/common/libcommon_C.la \
	$(top_builddir)/lib/edgepaint/liblab_gamut.la \
	$(top_builddir)/lib/cgraph/libcgraph.la \
	$(top_builddir)/lib/cdt/libcdt.la \
	$(ANN_LIBS) -lm

# add a non-existent C++ source to force the C++ compiler to be used for
# linking, so the C++ standard library is included for our C++ dependencies
nodist_EXTRA_edgepaint_SOURCES = fake.cxx

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
edgepaint.1.pdf: edgepaint.1.ps
	@PS2PDF@ $<

edgepaint.1.ps: $(srcdir)/edgepaint.1
	@GROFF@ -e -Tps -man -t $< >$@
endif
endif

EXTRA_DIST = $(man_MANS) $(pdf) edgepaint.vcxproj*

DISTCLEANFILES = $(pdf) edgepaint.1.ps
