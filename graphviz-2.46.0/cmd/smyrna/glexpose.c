/* $Id$Revision: */
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
#include "glexpose.h"
#include "draw.h"
#include "topviewfuncs.h"
#include <glcomp/glutils.h>
#include "topfisheyeview.h"
#include "gui/toolboxcallbacks.h"
#include "arcball.h"
#include "hotkeymap.h"
#include "polytess.h"

GLuint texture[3];
static int Status=0;									// Status Indicator

void LoadGLTextures()									// Load Bitmaps And Convert To Textures
{
	int imageWidth,imageHeight;

	unsigned char *data = glCompLoadPng ("c:/graphviz-ms/bin/Data/Crate.png", &imageWidth, &imageHeight);

	if (!data)
	{
	    printf ("Data/Crate.bmp could not be located\n");
	    exit(-1);
	}
	// Load The Bitmap, Check For Errors, If Bitmap's Not Found Quit
	glGenTextures(3, &texture[0]);					// Create Three Textures

		// Create Nearest Filtered Texture
		glBindTexture(GL_TEXTURE_2D, texture[0]);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, 3, imageWidth,imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

		// Create Linear Filtered Texture
		glBindTexture(GL_TEXTURE_2D, texture[1]);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, 3, imageWidth,imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

		// Create MipMapped Texture
		glBindTexture(GL_TEXTURE_2D, texture[2]);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_NEAREST);
		gluBuild2DMipmaps(GL_TEXTURE_2D, 3, imageWidth,imageHeight, GL_RGBA, GL_UNSIGNED_BYTE,data);
	Status=1;									// Set The Status To TRUE
}

static void drawRotatingAxis(void)
{
    static GLUquadricObj *quadratic;
    float AL = 45;

    if (get_mode(view) != MM_ROTATE)
	    return;

    if (!quadratic) {
	quadratic = gluNewQuadric();	// Create A Pointer To The Quadric Object
	gluQuadricNormals(quadratic, GLU_SMOOTH);	// Create Smooth Normals
	gluQuadricDrawStyle(quadratic, GLU_LINE);


    }

	glPushMatrix();
	glLoadIdentity();
	glMultMatrixf(view->arcball->Transform.M);	/*arcball transformations , experimental */
	glLineWidth(3);
	glBegin(GL_LINES);
	glColor3f(1, 1, 0);

	glVertex3f(0, 0, 0);
	glVertex3f(0, AL, 0);

	glVertex3f(0, 0, 0);
	glVertex3f(AL, 0, 0);

	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, AL);

	glEnd();
	glColor4f(0, 1, 0, 0.3);
	gluSphere(quadratic, AL, 20, 20);
	glLineWidth(1);
	glPopMatrix();


}



/*
	refreshes camera settings using view parameters such as pan zoom etc
	if a camera is selected viewport is switched to 3D
	params:ViewInfo	, global view variable defined in viewport.c
	return value:always 1
*/
static int glupdatecamera(ViewInfo * view)
{
    if (view->active_camera == -1)
	glTranslatef(-view->panx, -view->pany, view->panz);


    /*toggle to active camera */
    else {
	glMultMatrixf(view->arcball->Transform.M);	/*arcball transformations , experimental */
	glTranslatef(-view->cameras[view->active_camera]->targetx,
		     -view->cameras[view->active_camera]->targety, 0);
    }
    view->clipX1=0;
    view->clipX2=0;
    view->clipY1=0;
    view->clipY2=0;
    view->clipZ1=0;
    view->clipZ2=0;
    GetOGLPosRef(1, view->h - 5, &(view->clipX1), &(view->clipY1),
		 &(view->clipZ1));
    GetOGLPosRef(view->w - 1, 1, &(view->clipX2), &(view->clipY2),
		 &(view->clipZ2));

    if (view->active_camera == -1) {
	glScalef(1 / view->zoom * -1, 1 / view->zoom * -1,
		 1 / view->zoom * -1);
    } else {
	glScalef(1 / view->cameras[view->active_camera]->r,
		 1 / view->cameras[view->active_camera]->r,
		 1 / view->cameras[view->active_camera]->r);
    }


    return 1;
}

/*
	draws grid (little dots , with no use)
	params:ViewInfo	, global view variable defined in viewport.c
	return value:none
*/
static void glexpose_grid(ViewInfo * view)
{
    //drawing grids
    float x, y;
    if (view->gridVisible) {
	glPointSize(1);
	glBegin(GL_POINTS);
	glColor4f(view->gridColor.R, view->gridColor.G, view->gridColor.B,
		  view->gridColor.A);
	for (x = view->bdxLeft; x <= view->bdxRight;
	     x = x + view->gridSize) {
	    for (y = view->bdyBottom; y <= view->bdyTop;
		 y = y + view->gridSize) {
		glVertex3f(x, y, 0);
	    }
	}
	glEnd();
    }
}

/*
	draws active graph depending on graph type
	params:ViewInfo	, global view variable defined in viewport.c
	return value:1 if there is a graph to draw else 0 
*/
static int glexpose_drawgraph(ViewInfo * view)
{

    if (view->activeGraph > -1) {
	if (!view->Topview->fisheyeParams.active)
	    renderSmGraph(view->g[view->activeGraph],view->Topview);	    
	else {
	    drawtopologicalfisheye(view->Topview);
	}

	return 1;
    }
    return 0;
}

/*
	main gl expose ,any time sreen needs to be redrawn, this function is called by gltemplate
	,all drawings are initialized in this function
	params:ViewInfo	, global view variable defined in viewport.c
	return value:0 if something goes wrong with GL 1 , otherwise
*/
int glexpose_main(ViewInfo * view)
{
    static int doonce = 0;
    if (!glupdatecamera(view))
	return 0;

    if (view->activeGraph >= 0) {
	if (!doonce) {
	    doonce = 1;
	    btnToolZoomFit_clicked(NULL, NULL);
	    btnToolZoomFit_clicked(NULL, NULL);
	}
    }
    else
	return 0;



    glexpose_grid(view);
    drawBorders(view);
    glexpose_drawgraph(view);
    drawRotatingAxis();
    draw_selpoly(&view->Topview->sel.selPoly);
    glCompSetDraw(view->widgets);

	 /*DEBUG*/ return 1;
}
