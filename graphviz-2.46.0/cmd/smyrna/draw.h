/* $Id$ $Revision$ */
/* vim:set shiftwidth=4 ts=8: */

/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: See CVS logs. Details at http://www.graphviz.org/
 *************************************************************************/

#ifndef DRAW_H
#define DRAW_H
#include "smyrnadefs.h"
#include <gtk/gtkgl.h>
#include <xdot/xdot.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <glcomp/glcompfont.h>

#ifdef __cplusplus
extern "C" {
#endif
/* DRAWING FUNCTIONS 
 * these are opengl based xdot drawing functions 
 * topview drawings are not here
 */
extern drawfunc_t OpFns[];
extern void drawCircle(float x, float y, float radius, float zdepth);
extern void drawBorders(ViewInfo * view);
extern void drawEllipse(float xradius, float yradius, int angle1, int angle2);
extern void draw_selpoly(glCompPoly* selPoly);

#ifdef __cplusplus
}				/* end extern "C" */
#endif
#endif
