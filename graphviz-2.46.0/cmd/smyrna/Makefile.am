# $Id$Revision$
## Process this file with automake to produce Makefile.in

pdfdir = $(pkgdatadir)/doc/pdf

AM_CPPFLAGS = \
	-DSMYRNA_PATH=\""$(pkgdatadir)/smyrna"\" \
        -I$(top_srcdir)/lib \
        -I$(top_srcdir)/lib/cgraph \
        -I$(top_srcdir)/lib/cdt \
        -I$(top_srcdir)/lib/utilities \
        -I$(top_srcdir)/lib/xdot \
        -I$(top_srcdir)/lib/gvpr \
        -I$(top_srcdir)/lib/common \
        -I$(top_srcdir)/cmd/smyrna/gui \
	$(GTK_CFLAGS) $(GLUT_CFLAGS) $(GTKGLEXT_CFLAGS) $(GLADE_CFLAGS) $(FREETYPE2_CFLAGS) $(FONTCONFIG_CFLAGS) $(GTS_CFLAGS) $(XRENDER_CFLAGS)

bin_PROGRAMS =
man_MANS = 
pdf_DATA = 
if WITH_SMYRNA
if ENABLE_SHARED
bin_PROGRAMS += smyrna 
endif
if ENABLE_STATIC
bin_PROGRAMS += smyrna_static
endif
man_MANS += smyrna.1
if ENABLE_MAN_PDFS
pdf_DATA += smyrna.1.pdf
endif
endif

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
pdf = smyrna.1.pdf
else
pdf = ERROR_NO_PS2PDF
endif
else
pdf =
endif

noinst_HEADERS = arcball.h draw.h glexpose.h \
	glmotion.h gltemplate.h gui/appmouse.h gui/callbacks.h \
	hotkeymap.h materials.h md5.h polytess.h selectionfuncs.h \
	smyrna_utils.h smyrnadefs.h topfisheyeview.h \
	topviewdefs.h topviewfuncs.h trackball.h tvnodes.h \
	viewport.h viewportcamera.h support.h \
	gui/datalistcallbacks.h gui/frmobjectui.h \
	gui/glcompui.h gui/gui.h gui/menucallbacks.h \
	gui/toolboxcallbacks.h gui/topviewsettings.h gvprpipe.h hier.h glutrender.h

smyrna_SOURCES = arcball.c draw.c glexpose.c \
	glmotion.c gltemplate.c gui/appmouse.c gui/callbacks.c \
	gvprpipe.c hier.c hotkeymap.c main.c md5.c polytess.c \
	selectionfuncs.c smyrna_utils.c topfisheyeview.c \
	topviewfuncs.c trackball.c  tvnodes.c \
	viewport.c viewportcamera.c \
	gui/datalistcallbacks.c gui/frmobjectui.c \
	gui/glcompui.c gui/gui.c gui/menucallbacks.c \
	gui/toolboxcallbacks.c gui/topviewsettings.c glutrender.c

smyrna_LDADD = $(top_builddir)/lib/cgraph/libcgraph_C.la \
		$(top_builddir)/lib/cdt/libcdt_C.la \
		$(top_builddir)/lib/xdot/libxdot_C.la \
		$(top_builddir)/lib/glcomp/libglcomp_C.la \
		$(top_builddir)/lib/topfish/libtopfish_C.la \
		$(top_builddir)/lib/common/libcommon_C.la \
		$(top_builddir)/lib/gvpr/libgvpr_C.la \
		$(top_builddir)/lib/expr/libexpr_C.la \
		$(top_builddir)/lib/ingraphs/libingraphs_C.la \
		$(top_builddir)/lib/neatogen/libneatogen_C.la \
		$(GTK_LIBS) $(GLUT_LIBS) $(GTKGLEXT_LIBS) $(GLADE_LIBS) $(X_LIBS) $(EXPAT_LIBS) $(GTS_LIBS) $(MATH_LIBS) $(EXTRA_SMYRNA_LDFLAGS)

smyrna_static_SOURCES = $(smyrna_SOURCES)
smyrna_static_LDADD = $(top_builddir)/lib/cgraph/libcgraph_C.la \
		$(top_builddir)/lib/cdt/libcdt_C.la \
		$(top_builddir)/lib/xdot/libxdot_C.la \
		$(top_builddir)/lib/glcomp/libglcomp_C.la \
		$(top_builddir)/lib/topfish/libtopfish_C.la \
		$(top_builddir)/lib/common/libcommon_C.la \
		$(top_builddir)/lib/gvpr/libgvpr_C.la \
		$(top_builddir)/lib/expr/libexpr_C.la \
		$(top_builddir)/lib/ingraphs/libingraphs_C.la \
		$(top_builddir)/lib/neatogen/libneatogen_C.la \
		$(GTK_LIBS) $(GLUT_LIBS) $(GTKGLEXT_LIBS) $(GLADE_LIBS) $(X_LIBS) $(EXPAT_LIBS) $(GTS_LIBS) $(MATH_LIBS)

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
smyrna.1.pdf: smyrna.1.ps
	@PS2PDF@ $<

smyrna.1.ps: $(srcdir)/smyrna.1
	@GROFF@ -Tps -man $< >$@
endif
endif

EXTRA_DIST = smyrna.vcxproj* $(man_MANS) $(pdf) smyrna.1

DISTCLEANFILES = $(pdf) smyrna.1.ps

