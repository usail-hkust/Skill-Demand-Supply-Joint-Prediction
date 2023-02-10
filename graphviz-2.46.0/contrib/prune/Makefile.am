# $Id$ $Revision$
## Process this file with automake to produce Makefile.in

pdfdir = $(pkgdatadir)/doc/pdf

AM_CPPFLAGS = \
        -I$(top_srcdir) \
        -I$(top_srcdir)/lib/ingraphs \
	-I$(top_srcdir)/lib/cgraph \
	-I$(top_srcdir)/lib/cdt

bin_PROGRAMS = prune
noinst_HEADERS = generic_list.h
man_MANS = prune.1
if ENABLE_MAN_PDFS
pdf_DATA = prune.1.pdf

if HAVE_PS2PDF
pdf = $(pdf_DATA)
else
pdf = ERROR_NO_PS2PDF
endif
else
pdf =
endif

prune_SOURCES = generic_list.c prune.c

prune_LDADD = $(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la \
	$(top_builddir)/lib/cdt/libcdt.la

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
prune.1.pdf: prune.1.ps
	@PS2PDF@ $<

prune.1.ps: prune.1
	@GROFF@ -Tps -man $< >$@
endif
endif

EXTRA_DIST = $(man_MANS) $(pdf) README test.gv prune.vcxproj*


DISTCLEANFILES = $(pdf) prune.1.ps
