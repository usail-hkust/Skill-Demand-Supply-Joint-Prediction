# $Id$ $Revision$
## Process this file with automake to produce Makefile.in

pdfdir = $(pkgdatadir)/doc/pdf

AM_CPPFLAGS = \
	-I$(top_srcdir)/lib \
	-I$(top_srcdir)/lib/common \
	-I$(top_srcdir)/lib/cgraph \
	-I$(top_srcdir)/lib/cdt

if WITH_ANN
bin_PROGRAMS = mingle
endif

man_MANS = mingle.1
if ENABLE_MAN_PDFS
pdf_DATA = mingle.1.pdf

if HAVE_PS2PDF
pdf = $(pdf_DATA)
else
pdf = ERROR_NO_PS2PDF
endif
else
pdf =
endif

mingle_SOURCES = minglemain.c
mingle_CPPFLAGS = $(AM_CPPFLAGS)
mingle_LDADD = \
    $(top_builddir)/lib/mingle/libmingle_C.la \
    $(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/sfdpgen/libsfdpgen_C.la \
	$(top_builddir)/lib/neatogen/libneatogen_C.la \
	$(top_builddir)/lib/sparse/libsparse_C.la \
	$(top_builddir)/lib/rbtree/librbtree_C.la \
	$(top_builddir)/lib/common/libcommon_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la \
	$(top_builddir)/lib/cdt/libcdt.la \
	$(ANN_LIBS) -lm

# add a non-existent C++ source to force the C++ compiler to be used for
# linking, so the C++ standard library is included for our C++ dependencies
nodist_EXTRA_mingle_SOURCES = fake.cxx

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
mingle.1.pdf: mingle.1.ps
	@PS2PDF@ $<

mingle.1.ps: $(srcdir)/mingle.1
	@GROFF@ -e -Tps -man -t $< >$@
endif
endif

EXTRA_DIST = $(man_MANS) $(pdf) mingle.vcxproj*

DISTCLEANFILES = $(pdf) mingle.1.ps
