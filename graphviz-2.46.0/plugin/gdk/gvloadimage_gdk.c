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

#include "config.h"

#include <stdlib.h>

#include <gvc/gvplugin_loadimage.h>
#include <gvc/gvio.h>

#ifdef HAVE_PANGOCAIRO
#include <cairo.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <gdk/gdkcairo.h>

#ifdef _WIN32 //*dependencies
    #pragma comment( lib, "gvc.lib" )
    #pragma comment( lib, "glib-2.0.lib" )
    #pragma comment( lib, "cairo.lib" )
    #pragma comment( lib, "gobject-2.0.lib" )
    #pragma comment( lib, "graph.lib" )
    #pragma comment( lib, "gdk-pixbuf.lib" )
#endif

typedef enum {
    FORMAT_BMP_CAIRO,
    FORMAT_JPEG_CAIRO,
    FORMAT_PNG_CAIRO,
    FORMAT_ICO_CAIRO,
    FORMAT_TIFF_CAIRO,
} format_type;

#if 0
// FIXME - should be using a stream reader
static cairo_status_t
reader (void *closure, unsigned char *data, unsigned int length)
{
    if (length == fread(data, 1, length, (FILE *)closure)
     || feof((FILE *)closure))
        return CAIRO_STATUS_SUCCESS;
    return CAIRO_STATUS_READ_ERROR;
}
#endif

#ifdef HAVE_CAIRO_SURFACE_SET_MIME_DATA
static void gdk_set_mimedata_from_file (cairo_surface_t *image, const char *mime_type, const char *file)
{
    FILE *fp;
    unsigned char *data = NULL;
    long len;
    const char *id_prefix = "gvloadimage_gdk-";
    char *unique_id;
    size_t unique_id_len;

    fp = fopen (file, "rb");
    if (fp == NULL)
        return;
    fseek (fp, 0, SEEK_END);
    len = ftell(fp);
    fseek (fp, 0, SEEK_SET);
    if (len > 0)
        data = malloc ((size_t)len);
    if (data) {
        if (fread(data, (size_t)len, 1, fp) != 1) {
            free (data);
            data = NULL;
        }
    }
    fclose(fp);

    if (data) {
        cairo_surface_set_mime_data (image, mime_type, data, (unsigned long)len, free, data);
        unique_id_len = strlen(id_prefix) + strlen(file) + 1;
        unique_id = malloc (unique_id_len);
        snprintf (unique_id, unique_id_len, "%s%s", id_prefix, file);
        cairo_surface_set_mime_data (image, CAIRO_MIME_TYPE_UNIQUE_ID, (unsigned char *)unique_id, unique_id_len, free, unique_id);
    }
}

static void gdk_set_mimedata(cairo_surface_t *image, usershape_t *us)
{
    switch (us->type) {
        case FT_PNG:
            gdk_set_mimedata_from_file (image, CAIRO_MIME_TYPE_PNG, us->name);
            break;
        case FT_JPEG:
            gdk_set_mimedata_from_file (image, CAIRO_MIME_TYPE_JPEG, us->name);
            break;
        default:
            break;
    }
}

#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
static void gdk_freeimage(usershape_t *us)
{
#ifndef HAVE_CAIRO_SURFACE_SET_MIME_DATA
    g_object_unref((GdkPixbuf*)(us->data));
#else /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
    cairo_surface_destroy ((cairo_surface_t *)(us->data));
#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
}

#ifndef HAVE_CAIRO_SURFACE_SET_MIME_DATA
static GdkPixbuf* gdk_loadimage(GVJ_t * job, usershape_t *us)
#else /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
static cairo_surface_t* gdk_loadimage(GVJ_t * job, usershape_t *us)
#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
{
#ifdef HAVE_CAIRO_SURFACE_SET_MIME_DATA
    cairo_t *cr = (cairo_t *) job->context; /* target context */
#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
    GdkPixbuf *image = NULL;
#ifdef HAVE_CAIRO_SURFACE_SET_MIME_DATA
    cairo_surface_t *cairo_image = NULL;
    cairo_pattern_t *pattern;
#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */

    assert(job);
    assert(us);
    assert(us->name);

    if (us->data) {
#ifndef HAVE_CAIRO_SURFACE_SET_MIME_DATA
        if (us->datafree == gdk_freeimage)
             image = (GdkPixbuf*)(us->data); /* use cached data */
        else {
             us->datafree(us);        /* free incompatible cache data */
             us->datafree = NULL;
             us->data = NULL;
#else /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
        if (us->datafree == gdk_freeimage) {
	    cairo_image = cairo_surface_reference ((cairo_surface_t *)(us->data)); /* use cached data */
	} else {
	    us->datafree(us);        /* free incompatible cache data */
	    us->datafree = NULL;
	    us->data = NULL;
#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
        }
    }
#ifndef HAVE_CAIRO_SURFACE_SET_MIME_DATA
    if (!image) { /* read file into cache */
	if (!gvusershape_file_access(us))
	    return NULL;
#else /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
    if (!cairo_image) { /* read file into cache */
        if (!gvusershape_file_access(us))
            return NULL;
#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
        switch (us->type) {
            case FT_PNG:
            case FT_JPEG:
            case FT_BMP:
            case FT_ICO:
            case FT_TIFF:
                // FIXME - should be using a stream reader
                image = gdk_pixbuf_new_from_file(us->name, NULL);
                break;
            default:
                image = NULL;
        }
#ifdef HAVE_CAIRO_SURFACE_SET_MIME_DATA

#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
        if (image) {
#ifndef HAVE_CAIRO_SURFACE_SET_MIME_DATA
            us->data = (void*)image;
#else /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
            cairo_save (cr);
            gdk_cairo_set_source_pixbuf (cr, image, 0, 0);
            pattern = cairo_get_source (cr);
            assert(cairo_pattern_get_type (pattern) == CAIRO_PATTERN_TYPE_SURFACE);
            cairo_pattern_get_surface (pattern, &cairo_image);
            cairo_image = cairo_surface_reference (cairo_image);
            cairo_restore (cr);
            gdk_set_mimedata (cairo_image, us);
            us->data = (void*)cairo_surface_reference (cairo_image);
#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
            us->datafree = gdk_freeimage;
        }
#ifndef HAVE_CAIRO_SURFACE_SET_MIME_DATA
	gvusershape_file_release(us);
#else /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
        gvusershape_file_release(us);
#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
    }
#ifndef HAVE_CAIRO_SURFACE_SET_MIME_DATA
    return image;
#else /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
    return cairo_image;
#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
}

static void gdk_loadimage_cairo(GVJ_t * job, usershape_t *us, boxf b, boolean filled)
{
    cairo_t *cr = (cairo_t *) job->context; /* target context */
#ifndef HAVE_CAIRO_SURFACE_SET_MIME_DATA
    GdkPixbuf *image;
#else /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
    cairo_surface_t *image;
#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */

    image = gdk_loadimage(job, us);
    if (image) {
        cairo_save(cr);
	cairo_translate(cr, b.LL.x, -b.UR.y);
	cairo_scale(cr, (b.UR.x - b.LL.x)/(us->w), (b.UR.y - b.LL.y)/(us->h)); 
#ifndef HAVE_CAIRO_SURFACE_SET_MIME_DATA
        gdk_cairo_set_source_pixbuf (cr, image, 0, 0);
#else /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
        cairo_set_source_surface (cr, image, 0, 0);
#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
        cairo_paint (cr);
        cairo_restore(cr);
#ifdef HAVE_CAIRO_SURFACE_SET_MIME_DATA
        cairo_surface_destroy (image);
#endif /* HAVE_CAIRO_SURFACE_SET_MIME_DATA */
    }
}

static gvloadimage_engine_t engine_gdk = {
    gdk_loadimage_cairo
};

#endif

gvplugin_installed_t gvloadimage_gdk_types[] = {
#ifdef HAVE_PANGOCAIRO
    {FORMAT_BMP_CAIRO,  "bmp:cairo", 1, &engine_gdk, NULL},
    {FORMAT_JPEG_CAIRO, "jpe:cairo", 2, &engine_gdk, NULL},
    {FORMAT_JPEG_CAIRO, "jpg:cairo", 2, &engine_gdk, NULL},
    {FORMAT_JPEG_CAIRO, "jpeg:cairo", 2, &engine_gdk, NULL},
    {FORMAT_PNG_CAIRO,  "png:cairo", -1, &engine_gdk, NULL},
    {FORMAT_ICO_CAIRO,  "ico:cairo", 1, &engine_gdk, NULL},
//    {FORMAT_TIFF_CAIRO, "tif:cairo", 1, &engine_gdk, NULL},
//    {FORMAT_TIFF_CAIRO, "tiff`:cairo", 1, &engine_gdk, NULL},
#endif
    {0, NULL, 0, NULL, NULL}
};
