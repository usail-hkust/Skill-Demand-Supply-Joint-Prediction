# $Id$ $Revision$
## Process this file with automake to produce Makefile.in

AM_CPPFLAGS = \
	-I$(top_srcdir)/lib \
	-I$(top_srcdir)/lib/cgraph \
	-I$(top_srcdir)/lib/cdt \
	-I$(top_srcdir)/lib/pathplan \
	-I$(top_srcdir)/lib/pack \
	-I$(top_srcdir)/lib/gvc \
	-I$(top_srcdir)/lib/common \
	@EXPAT_INCLUDES@

pdfdir = $(pkgdatadir)/doc/pdf

noinst_HEADERS = colortbl.h convert.h mmio.h matrix_market.h \
	 graph_generator.h gml2gv.h gmlparse.h
if ENABLE_STATIC
bin_PROGRAMS = gc gvcolor gxl2gv acyclic nop ccomps sccmap tred \
	unflatten gvpack gvpack_static dijkstra bcomps mm2gv gvgen gml2gv gv2gml graphml2gv
else
bin_PROGRAMS = gc gvcolor gxl2gv acyclic nop ccomps sccmap tred \
	unflatten gvpack dijkstra bcomps mm2gv gvgen gml2gv gv2gml graphml2gv
endif

man_MANS = gc.1 gvcolor.1 gxl2gv.1 acyclic.1 nop.1 ccomps.1 sccmap.1 \
	tred.1 unflatten.1 gvpack.1 dijkstra.1 bcomps.1 mm2gv.1 gvgen.1 gml2gv.1 graphml2gv.1
if ENABLE_MAN_PDFS
pdf_DATA = gc.1.pdf gvcolor.1.pdf gxl2gv.1.pdf acyclic.1.pdf \
	   nop.1.pdf ccomps.1.pdf sccmap.1.pdf tred.1.pdf \
	   unflatten.1.pdf gvpack.1.pdf dijkstra.1.pdf \
	   bcomps.1.pdf mm2gv.1.pdf gvgen.1.pdf gml2gv.1.pdf graphml2gv.1.pdf

if HAVE_PS2PDF
pdf = $(pdf_DATA)
else
pdf = ERROR_NO_PS2PDF
endif
else
pdf =
endif

install-data-hook:
	(cd $(DESTDIR)$(man1dir); rm -f gv2gxl.1; $(LN_S) gxl2gv.1 gv2gxl.1;)
if ENABLE_MAN_PDFS
	(cd $(DESTDIR)$(pdfdir); rm -f gv2gxl.1.pdf; $(LN_S) gxl2gv.1.pdf gv2gxl.1.pdf;)
endif
	(cd $(DESTDIR)$(man1dir); rm -f gxl2dot.1; $(LN_S) gxl2gv.1 gxl2dot.1;)
if ENABLE_MAN_PDFS
	(cd $(DESTDIR)$(pdfdir); rm -f gxl2dot.1.pdf; $(LN_S) gxl2gv.1.pdf gxl2dot.1.pdf;)
endif
	(cd $(DESTDIR)$(man1dir); rm -f dot2gxl.1; $(LN_S) gv2gxl.1 dot2gxl.1;)
if ENABLE_MAN_PDFS
	(cd $(DESTDIR)$(pdfdir); rm -f dot2gxl.1.pdf; $(LN_S) gv2gxl.1.pdf dot2gxl.1.pdf;)
endif
	(cd $(DESTDIR)$(man1dir); rm -f gv2gml.1; $(LN_S) gml2gv.1 gv2gml.1;)
if ENABLE_MAN_PDFS
	(cd $(DESTDIR)$(pdfdir); rm -f gv2gml.1.pdf; $(LN_S) gml2gv.1.pdf gv2gml.1.pdf;)
endif

install-exec-hook:
	(cd $(DESTDIR)$(bindir); rm -f gv2gxl$(EXEEXT); $(LN_S) gxl2gv$(EXEEXT) gv2gxl$(EXEEXT);)
	(cd $(DESTDIR)$(bindir); rm -f dot2gxl$(EXEEXT); $(LN_S) gxl2gv$(EXEEXT) dot2gxl$(EXEEXT);)
	(cd $(DESTDIR)$(bindir); rm -f gxl2dot$(EXEEXT); $(LN_S) gxl2gv$(EXEEXT) gxl2dot$(EXEEXT);)

uninstall-hook:
	(cd $(DESTDIR)$(man1dir); rm -f gv2gxl.1;)
if ENABLE_MAN_PDFS
	(cd $(DESTDIR)$(pdfdir); rm -f gv2gxl.1.pdf;)
endif
	(cd $(DESTDIR)$(bindir); rm -f gv2gxl$(EXEEXT);)
	(cd $(DESTDIR)$(bindir); rm -f dot2gxl$(EXEEXT);)
	(cd $(DESTDIR)$(bindir); rm -f gxl2dot$(EXEEXT);)

gxl2gv_SOURCES = cvtgxl.c gv2gxl.c gxl2gv.c

gxl2gv_LDADD = \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la \
	$(top_builddir)/lib/cdt/libcdt.la @EXPAT_LIBS@ 

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
gxl2gv.1.pdf: gxl2gv.1.ps
	@PS2PDF@ $<

gxl2gv.1.ps: $(srcdir)/gxl2gv.1
	@GROFF@ -Tps -man $< >$@
endif
endif

sccmap_SOURCES = sccmap.c

sccmap_LDADD = \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la @EXPAT_LIBS@ 

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
sccmap.1.pdf: sccmap.1.ps
	@PS2PDF@ $<

sccmap.1.ps: $(srcdir)/sccmap.1
	@GROFF@ -Tps -man $< >$@
endif
endif

ccomps_SOURCES = ccomps.c

ccomps_LDADD = \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
ccomps.1.pdf: ccomps.1.ps
	@PS2PDF@ $<

ccomps.1.ps: $(srcdir)/ccomps.1
	@GROFF@ -Tps -man $< >$@
endif
endif

#tred2_SOURCES = tred2.c
#
#tred2_LDADD = \
#	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
#	$(top_builddir)/lib/cgraph/libcgraph.la \
#	$(top_builddir)/lib/gvc/libgvc.la

tred_SOURCES = tred.c

tred_LDADD = \
    $(top_builddir)/lib/common/libcommon_C.la \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
tred.1.pdf: tred.1.ps
	@PS2PDF@ $<

tred.1.ps: $(srcdir)/tred.1
	@GROFF@ -Tps -man $< >$@
endif
endif

unflatten_SOURCES = unflatten.c

unflatten_LDADD = \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
unflatten.1.pdf: unflatten.1.ps
	@PS2PDF@ $<

unflatten.1.ps: $(srcdir)/unflatten.1
	@GROFF@ -Tps -man $< >$@
endif
endif

acyclic_SOURCES = acyclic.c

acyclic_LDADD = \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
acyclic.1.pdf: acyclic.1.ps
	@PS2PDF@ $<

acyclic.1.ps: $(srcdir)/acyclic.1
	@GROFF@ -Tps -man $< >$@
endif
endif

nop_SOURCES = nop.c

nop_LDADD = \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
nop.1.pdf: nop.1.ps
	@PS2PDF@ $<

nop.1.ps: $(srcdir)/nop.1
	@GROFF@ -Tps -man $< >$@
endif
endif

gvcolor_SOURCES = gvcolor.c colxlate.c colortbl.h

gvcolor_LDADD = \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
gvcolor.1.pdf: gvcolor.1.ps
	@PS2PDF@ $<

gvcolor.1.ps: $(srcdir)/gvcolor.1
	@GROFF@ -Tps -man $< >$@
endif
endif

bcomps_SOURCES = bcomps.c

bcomps_LDADD = \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
bcomps.1.pdf: bcomps.1.ps
	@PS2PDF@ $<

bcomps.1.ps: $(srcdir)/bcomps.1
	@GROFF@ -Tps -man $< >$@
endif
endif

gc_SOURCES = gc.c

gc_LDADD = \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
gc.1.pdf: gc.1.ps
	@PS2PDF@ $<

gc.1.ps: $(srcdir)/gc.1
	@GROFF@ -Tps -man $< >$@
endif
endif

gvpack_SOURCES = gvpack.c
gvpack_CPPFLAGS = $(AM_CPPFLAGS) -DDEMAND_LOADING=0

gvpack_LDADD = \
	$(top_builddir)/lib/gvc/libgvc.la \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la \
	$(top_builddir)/lib/cdt/libcdt.la \
	$(top_builddir)/plugin/neato_layout/libgvplugin_neato_layout.la

# add a non-existent C++ source to force the C++ compiler to be used for
# linking, so the C++ standard library is included for our C++ dependencies
nodist_EXTRA_gvpack_SOURCES = fake.cxx

if ENABLE_STATIC
gvpack_static_SOURCES = gvpack.c
gvpack_static_CPPFLAGS = $(AM_CPPFLAGS) -DDEMAND_LOADING=0
gvpack_static_LDADD = \
	$(top_builddir)/plugin/neato_layout/libgvplugin_neato_layout_C.la \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/gvc/libgvc_C.la \
	$(top_builddir)/lib/pathplan/libpathplan_C.la \
	$(top_builddir)/lib/cgraph/libcgraph_C.la \
	$(top_builddir)/lib/cdt/libcdt_C.la \
		$(EXPAT_LIBS) $(Z_LIBS) $(GTS_LIBS) $(SOCKET_LIBS) $(IPSEPCOLA_LIBS) $(MATH_LIBS)
endif

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
gvpack.1.pdf: gvpack.1.ps
	@PS2PDF@ $<

gvpack.1.ps: $(srcdir)/gvpack.1
	@GROFF@ -Tps -man $< >$@
endif
endif

mm2gv_SOURCES = mm2gv.c matrix_market.c mmio.c

mm2gv_LDADD = \
	$(top_builddir)/lib/sparse/libsparse_C.la \
    $(top_builddir)/lib/common/libcommon_C.la \
    $(top_builddir)/lib/gvc/libgvc_C.la \
    $(top_builddir)/lib/pathplan/libpathplan_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la @MATH_LIBS@

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
mm2gv.1.pdf: mm2gv.1.ps
	@PS2PDF@ $<

mm2gv.1.ps: mm2gv.1
	@GROFF@ -Tps -man -e -t $< >$@
endif
endif

gv2gml_SOURCES = gv2gml.c

gv2gml_LDADD = \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la

gml2gv_SOURCES = gml2gv.c gmlparse.y gmlscan.l

gmlparse.c gmlparse.h: $(top_srcdir)/cmd/tools/gmlparse.y
	@YACC@ -Wno-yacc -dv $(top_srcdir)/cmd/tools/gmlparse.y -o gmlparse.c
 
gmlparse.o gmlparse.lo gmlscan.o gmlscan.lo : gmlscan.c gmlparse.h

gmlscan.c: $(top_srcdir)/cmd/tools/gmlscan.l
	@LEX@ --outfile=$@ -i $(top_srcdir)/cmd/tools/gmlscan.l

gml2gv_LDADD = \
	$(top_builddir)/lib/cgraph/libcgraph.la \
	$(top_builddir)/lib/cdt/libcdt.la @MATH_LIBS@

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
gml2gv.1.pdf: gml2gv.1.ps
	@PS2PDF@ $<

gml2gv.1.ps: gml2gv.1
	@GROFF@ -Tps -man $< >$@
endif
endif

graphml2gv_SOURCES = graphml2gv.c

graphml2gv_LDADD = \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la \
	$(top_builddir)/lib/cdt/libcdt.la @EXPAT_LIBS@ 

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
graphml2gv.1.pdf: graphml2gv.1.ps
	@PS2PDF@ $<

graphml2gv.1.ps: $(srcdir)/graphml2gv.1
	@GROFF@ -Tps -man $< >$@
endif
endif

dijkstra_SOURCES = dijkstra.c

dijkstra_LDADD = \
	$(top_builddir)/lib/ingraphs/libingraphs_C.la \
	$(top_builddir)/lib/cgraph/libcgraph.la \
	$(top_builddir)/lib/cdt/libcdt.la

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
dijkstra.1.pdf: dijkstra.1.ps
	@PS2PDF@ $<

dijkstra.1.ps: $(srcdir)/dijkstra.1
	@GROFF@ -Tps -man $< >$@
endif
endif

gvgen_SOURCES = gvgen.c graph_generator.c

gvgen_LDADD = \
	$(top_builddir)/lib/cgraph/libcgraph.la @MATH_LIBS@

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
gvgen.1.pdf: gvgen.1.ps
	@PS2PDF@ $<

gvgen.1.ps: $(srcdir)/gvgen.1
	@GROFF@ -Tps -man $< >$@
endif
endif

EXTRA_DIST = $(man_MANS) $(pdf) bcomps.vcxproj* \
	acyclic.vcxproj* bcomps.vcxproj* ccomps.vcxproj* dijkstra.vcxproj* gc.vcxproj* \
    gvcolor.vcxproj* gvgen.vcxproj* gvpack.vcxproj* gxl2gv.vcxproj* \
	mm2gv.vcxproj* nop.vcxproj* sccmap.vcxproj* tred.vcxproj* unflatten.vcxproj* \
	gmlscan.c gmlparse.c gmlparse.h gml2gv.vcxproj* \
	graphml2gv.vcxproj* gv2gml.vcxproj*

# FIXME - these are missing
#	gv2gxl.vcxproj*
		
CLEANFILES = stamp.h 

DISTCLEANFILES = $(pdf) gmlparse.[ch] gmlscan.c \
	gxl2gv.1.ps sccmap.1.ps ccomps.1.ps tred.1.ps unflatten.1.ps acyclic.1.ps \
	nop.1.ps gvcolor.1.ps bcomps.1.ps gc.1.ps gvpack.1.ps mm2gv.1.ps gml2gv.1.ps \
	graphml2gv.1.ps dijkstra.1.ps gvgen.1.ps
