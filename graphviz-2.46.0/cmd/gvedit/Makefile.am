# $Id$ $Revision$
## Process this file with automake to produce Makefile.in

SUBDIRS = ui images

pdfdir = $(pkgdatadir)/doc/pdf
gveditdir = $(pkgdatadir)/gvedit

mans = gvedit.1
if ENABLE_MAN_PDFS
if HAVE_PS2PDF
pdfs = gvedit.1.pdf
else
pdfs = ERROR_NO_PS2PDF
endif
else
pdfs =
endif

if WITH_QT
bin_PROGRAMS = gvedit
gvedit_DATA = attrs.txt
man_MANS = $(mans)
if ENABLE_MAN_PDFS
pdf_DATA = gvedit.1.pdf
endif
endif

AM_CPPFLAGS = \
	-I$(top_srcdir)/lib \
	-I$(top_srcdir)/lib/common \
	-I$(top_srcdir)/lib/gvc \
	-I$(top_srcdir)/lib/pathplan \
	-I$(top_srcdir)/lib/cgraph \
	-I$(top_srcdir)/lib/cdt

gvedit_CPPFLAGS = \
	$(AM_CPPFLAGS) \
	$(QTGUI_CFLAGS) \
	$(QTCORE_CFLAGS) \
	-DDEMAND_LOADING=1 \
	-DGVEDIT_DATADIR=\""$(pkgdatadir)/gvedit"\"

gvedit_LDADD = \
	$(top_builddir)/lib/gvc/libgvc.la \
	$(top_builddir)/lib/cgraph/libcgraph.la \
	$(top_builddir)/lib/cdt/libcdt.la \
	$(QTGUI_LIBS) \
	$(QTCORE_LIBS) \
	$(EXPAT_LIBS) $(Z_LIBS) 

noinst_HEADERS = csettings.h imageviewer.h mainwindow.h mdichild.h ui_settings.h 

MOCABLES = moc_csettings.cpp moc_imageviewer.cpp moc_mainwindow.cpp moc_mdichild.cpp qrc_mdi.cpp

gvedit_SOURCES = main.cpp csettings.cpp imageviewer.cpp mainwindow.cpp mdichild.cpp $(MOCABLES)

$(MOCABLES): mocables

.PHONY: mocables
mocables: qMakefile
	$(MAKE) -f qMakefile prefix=${prefix} exec_prefix=${exec_prefix} mocables compiler_rcc_make_all

qMakefile: gvedit.pro
	$(QMAKE) -o qMakefile gvedit.pro

attrs.txt: $(top_srcdir)/doc/infosrc/attrs
	cp $(top_srcdir)/doc/infosrc/attrs attrs.txt

if ENABLE_MAN_PDFS
if HAVE_PS2PDF
gvedit.1.pdf: gvedit.1.ps
	@PS2PDF@ $<

gvedit.1.ps: $(srcdir)/gvedit.1
	@GROFF@ -Tps -man $< >$@
endif
endif

EXTRA_DIST = gvedit.vcxproj* gvedit.pro.in mdi.qrc $(mans) $(pdfs)

CLEANFILES = moc_csettings.cpp moc_imageviewer.cpp moc_mainwindow.cpp moc_mdichild.cpp qrc_mdi.cpp

DISTCLEANFILES = attrs.txt $(pdf) qMakefile gvedit.1.ps

clean-local:
	-$(MAKE) -f qMakefile clean

distclean-local:
	-$(MAKE) -f qMakefile distclean

