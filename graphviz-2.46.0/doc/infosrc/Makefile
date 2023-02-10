#
# Generator for on-line documentation.
#
# Required tools: cc, dot, awk, ksh, yacc, gs, python.
#  dot is used in mkarrows.sh, mkshapes.sh, mkstyles.sh, mktapers.sh, mksplines.sh
#  ps_to_png.sh and other make rules rely on netpbm tools.
#  
#  Install python modules with:
#  $ pip install --user -r requirements.txt
#
#  python is used to run:
#  - jconvert.py, which converts the json schema graphviz_json_schema.json to html.
#    This also relies on the python package json2html.
#  - mkoutput.py, which converts templates/output.html.j2 to output.html
#    This requires the jinja2 python package.
#
# The main products are web pages:
#    arrows.html  - arrow_grammar
#    colors.html  - color_names svgcolor_names brewer_colors
#    command.html - raw HTML
#    lang.html    - grammar
#    output.html  - outputs
#    shapes.html  - shapelist, html_grammar
#
# The files listed after each give the main data files used to
# generate the page.
#
# If shapelist is extended, add entry to SGIF.

INSTALL = cp
INSTALL_DIR = ../info

XGIF = forward.gif back.gif both.gif nohead.gif

MGIF = constraint.gif record.gif html1.gif html2.gif fill.gif round.gif \
       mrecord.gif html3.gif html4.gif colorlist.gif record2.gif fixed.gif tee.gif

GRADPNG = g_lin0.png g_wlin0.png g_rad0.png g_lin45.png g_wlin45.png g_rad45.png g_lin90.png g_wlin90.png g_rad90.png \
          g_lin180.png g_wlin180.png g_rad180.png g_lin270.png g_wlin270.png g_rad270.png g_lin360.png g_wlin360.png g_rad360.png

STYLEPNG = n_solid.png n_dashed.png n_dotted.png n_bold.png n_filled.png n_rounded.png n_striped.png \
           n_wedged.png n_diagonals.png \
           e_solid.png e_dashed.png e_dotted.png e_bold.png \
           c_solid.png c_dashed.png c_dotted.png c_bold.png c_filled.png c_rounded.png c_striped.png

TAPERPNG = normal_forward.png normal_back.png normal_both.png normal_none.png \
           none_forward.png none_back.png none_both.png none_none.png

SPLINEPNG = spline_none.png spline_line.png spline_spline.png spline_ortho.png spline_curved.png spline_polyline.png

MPNG = plugins.png sdlshapes.png $(GRADPNG) $(STYLEPNG) $(TAPERPNG) $(SPLINEPNG)

SGIF = Mcircle.gif Mdiamond.gif Msquare.gif box.gif \
      circle.gif diamond.gif doublecircle.gif \
      doubleoctagon.gif egg.gif ellipse.gif hexagon.gif house.gif \
      invhouse.gif invtrapezium.gif invtriangle.gif none.gif underline.gif \
      octagon.gif oval.gif parallelogram.gif plaintext.gif note.gif \
      point.gif polygon.gif trapezium.gif triangle.gif cylinder.gif \
      tripleoctagon.gif pentagon.gif septagon.gif rect.gif plain.gif \
      rectangle.gif square.gif star.gif tab.gif folder.gif box3d.gif \
      component.gif promoter.gif cds.gif terminator.gif utr.gif \
      primersite.gif restrictionsite.gif fivepoverhang.gif \
      threepoverhang.gif noverhang.gif assembly.gif signature.gif \
      insulator.gif ribosite.gif rnastab.gif proteasesite.gif \
      proteinstab.gif rpromoter.gif rarrow.gif larrow.gif lpromoter.gif

AGIF = a_normal.gif a_inv.gif a_dot.gif a_invdot.gif a_odot.gif \
       a_invodot.gif a_curve.gif a_icurve.gif a_none.gif \
       a_tee.gif a_empty.gif a_invempty.gif a_open.gif \
       a_diamond.gif a_odiamond.gif a_box.gif a_obox.gif a_crow.gif \
       a_halfopen.gif a_ediamond.gif a_lteeoldiamond.gif

A2GIF= aa_box.gif aa_lbox.gif aa_rbox.gif aa_obox.gif aa_olbox.gif aa_orbox.gif\
       aa_crow.gif aa_lcrow.gif aa_rcrow.gif \
       aa_diamond.gif aa_ldiamond.gif aa_rdiamond.gif aa_odiamond.gif aa_oldiamond.gif aa_ordiamond.gif\
       aa_dot.gif aa_odot.gif \
       aa_inv.gif aa_linv.gif aa_rinv.gif aa_oinv.gif aa_olinv.gif aa_orinv.gif\
       aa_none.gif \
       aa_normal.gif aa_lnormal.gif aa_rnormal.gif aa_onormal.gif aa_olnormal.gif aa_ornormal.gif\
       aa_tee.gif aa_ltee.gif aa_rtee.gif \
       aa_vee.gif aa_lvee.gif aa_rvee.gif \
       aa_curve.gif aa_lcurve.gif aa_rcurve.gif \
       aa_icurve.gif aa_licurve.gif aa_ricurve.gif

       

GIF = $(SGIF) $(AGIF) $(MGIF) $(A2GIF) $(XGIF)
DOTS = html2.gv html3.gv html4.gv tee.gv
HTML = colors.html command.html lang.html output.html shapes.html \
       arrows.html schema.html
MISC = graphviz_json_schema.json w3data.js
INSTALL_FILES = $(HTML) $(DOTS) $(SGIF) $(AGIF) $(A2GIF) $(MGIF) $(XGIF) $(MPNG) $(MISC)

.SUFFIXES: .gv .gif .dot .png

.dot.gif :
	dot -Tgif -o $@ $<

.gv.gif :
	dot -Tgif -o $@ $<

.gv.png :
	dot -Tpng -o $@ $<

.PHONY: all
all : $(HTML) $(GIF) $(MPNG) $(MJPG)

.PHONY: install
install : all
	$(INSTALL) $(INSTALL_FILES) $(INSTALL_DIR)

record2.gif : record.dot
	dot -Tgif -Grankdir=LR -o $@ $<

sdlshapes.ps : sdlshapes.dot
	dot -Tps -lsdl.ps sdlshapes.dot > sdlshapes.ps

sdlshapes.png : sdlshapes.ps
	./ps_to_png.sh sdlshapes.ps sdlshapes.png

colors.html : colors.1 colors.n ../../lib/common/color_names ../../lib/common/svgcolor_names ../../lib/common/brewer_colors mkcolors.awk brewer.awk svg.awk
	mkdir -p colortmp
	awk -f brewer.awk ../../lib/common/brewer_colors
	cat colors.1 > colors.html
	awk -f mkcolors.awk ../../lib/common/color_names >> colors.html
	echo '<H2><A NAME="svg">The SVG color scheme</A></H2>' >> colors.html
	awk -f svg.awk ../../lib/common/svgcolor_names > svg
	awk -f mkcolors.awk svg >> colors.html
	rm -rf svg
	echo '<H2><A NAME="brewer">Brewer color schemes</A></H2>' >> colors.html
	(for f in $$(ls colortmp); do awk -f mkcolors.awk -s colortmp/$$f >> colors.html; done)
	cat colors.n >> colors.html
	rm -rf colortmp


output.html : outputs mkoutput.py plugins.png schema.html templates/output.html.j2
	./mkoutput.py < outputs > output.html

schema.html : jconvert.py graphviz_json_schema.json
	./jconvert.py graphviz_json_schema.json schema.html

html.html : html.1 html_grammar html.2 html.3 html1.gif html2.gif html3.gif html4.gif mklang
	./mklang html_grammar gramtmp
	cat html.1 > html.html
	cat gramtmp >> html.html
	cat html.2 >> html.html
	cat html1.dot >> html.html
	cat html.3 >> html.html
	rm gramtmp

shapes : shapelist mkshapes.sh
	./mkshapes.sh
	touch shapes
 
shapes.html : shapes mkshhtml.py html.html record.gif record2.gif sdlshapes.png templates/shapes.html.j2 shapelist
	./mkshhtml.py html.html < shapelist > shapes.html

lang.html : grammar lang.1 lang.2 mklang
	./mklang grammar gramtmp
	cat lang.1 > lang.html
	cat gramtmp >> lang.html
	cat lang.2 >> lang.html
	rm gramtmp

arrows.html : arrow_grammar arrows.1 arrows.2 mkarrowtbl.sh mklang $(AGIF) $(A2GIF)
	./mklang arrow_grammar gramtmp
	cat arrows.1 > arrows.html
	cat gramtmp >> arrows.html
	cat arrows.2 >> arrows.html
	./mkarrowtbl.sh $(A2GIF) >> arrows.html
	rm gramtmp

mklang : mklang.y
	yacc mklang.y
	cc -o mklang y.tab.c
	rm -rf y.tab.c

$(SGIF) : mkshapes.sh shapelist
	./mkshapes.sh

$(AGIF) :
	./mkarrows.sh $@

$(A2GIF) :
	./mkarrows.sh -s $@

$(GRADPNG) : mkgrads.sh
	./mkgrads.sh $@

$(STYLEPNG) : mkstyles.sh
	./mkstyles.sh $@

$(TAPERPNG) : mktapers.sh
	./mktapers.sh $@

$(SPLINEPNG) : mksplines.sh
	./mksplines.sh $@

.PHONY: clean
clean :
	rm -f mklang sdlshapes.ps sdlshapes.ppm html.html

.PHONY: distclean
distclean : clean
	rm -f colors.html output.html shapes.html lang.html arrows.html schema.html
	rm -f $(A2GIF) $(AGIF) $(SGIF) $(MPJG) $(MGIF) $(MPNG) $(XGIF) shapes
	(for s in $$(cat shapelist); do rm -f $$s.gif; done)

EXTRA_DIST = $(XGIF) mklang.y mkarrows.sh mkshapes.sh mkstyles.sh mktapers.sh \
             mkarrowtbl.sh mkoutput.py mkshhtml.py \
	           ps_to_png.sh arrow_grammar grammar html_grammar \
             shapelist colors.1 colors.n \
             output.1 output.2 html.1 html.2 html1.dot html.3 \
             shapes.1 shapes.2 shapes.3 lang.1 lang.2 arrows.1 arrows.2 \
             brewer.awk mkcolors.awk svg.awk sz.awk \
             colorlist.dot html1.dot html4.dot round.dot constraint.dot \
             html2.dot mrecord.dot sdlshapes.dot fill.dot html3.dot record.dot \
             X11 outputs eqn.gif plugins.gv
