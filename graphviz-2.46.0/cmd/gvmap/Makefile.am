# $Id$Revision$
## Process this file with automake to produce Makefile.in

pdfdir = $(pkgdatadir)/doc/pdf

AM_CPPFLAGS = \
	-I$(top_srcdir)/lib \
	-I$(top_srcdir)/lib/common \
	-I$(top_srcdir)/lib/cgraph \
	-I$(top_srcdir)/lib/cdt

if WITH_SFDP
bin_PROGRAMS = gvmap cluster 
dist_bin_SCRIPTS = gvmap.sh
endif

man_MANS = gvmap.1 cluster.1 gvmap.sh.1
if ENABLE_MAN_PDFS
pdf_DATA = gvmap.1.pdf cluster.1.pdf gvmap.sh.1.pdf

if HAVE_PS2PDF
pdf = $(pdf_DATA)
else
pdf = ERROR_NO_PS2PDF
endif
else
pdf =
endif

noinst_HEADERS = make_map.h power.h country_graph_coloring.h

gvmap_SOURCES = gvmap.c make_map.c power.c country_graph_coloring.c

cluster_SOURCES = cluster.c make_map.c power.c country_graph_coloring.c 

gvmap_CPPFLAGS = $(AM_CPPFLAGS)

cluster_CPPFLAGS = $(AM_CPPFLAGS) -g

gvmap_LDADD = \
	$(top_builddir)/lib/edgepaint/libedgepaint_C.la \
	$(top_builddir)/lib/sfdpgen/libsfdpgen_C.la \
	$(top_builddir)/lib/neatogen/libneatogen_C.la \
	$(top_builddir)/lib/sparse/libsparse_C.la \
	$(top_builddir)/lib/rbtree/librbtree_C.la \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/common/libcommon_C.la \
	$(top_builddir)/lib/edgepaint/liblab_gamut.la \
	$(top_builddir)/lib/cgraph/libcgraph.la \
	$(GTS_LIBS) -lm

cluster_LDADD = \
	$(top_builddir)/lib/edgepaint/libedgepaint_C.la \
	$(top_builddir)/lib/sfdpgen/libsfdpgen_C.la \
	$(top_builddir)/lib/neatogen/libneatogen_C.la \
	$(top_builddir)/lib/sparse/libsparse_C.la \
	$(top_builddir)/lib/rbtree/librbtree_C.la \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/common/libcommon_C.la \
	$(top_builddir)/lib/edgepaint/liblab_gamut.la \
	$(top_builddir)/lib/cgraph/libcgraph.la \
	$(GTS_LIBS) -lm

gvmap.sh :

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
gvmap.1.pdf: gvmap.1.ps
	@PS2PDF@ $<

gvmap.1.ps: $(srcdir)/gvmap.1
	@GROFF@ -Tps -man $< >$@

cluster.1.pdf: cluster.1.ps
	@PS2PDF@ $<

cluster.1.ps: $(srcdir)/cluster.1
	@GROFF@ -Tps -man $< >$@

gvmap.sh.1.pdf: gvmap.sh.1.ps
	@PS2PDF@ $<

gvmap.sh.1.ps: $(srcdir)/gvmap.sh.1
	@GROFF@ -Tps -man $< >$@
endif
endif

EXTRA_DIST = $(man_MANS) $(pdf) gvmap.vcxproj*

DISTCLEANFILES = $(pdf) gvmap.1.ps cluster.1.ps gvmap.sh.1.ps
