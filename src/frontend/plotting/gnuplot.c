/**********
 * From xgraph.c:
 * Copyright 1992 Regents of the University of California.  All rights reserved.
 * Author: 1992 David A. Gates, U. C. Berkeley CAD Group
 *
 * Author: 2008 Stefano Pedretti
**********/

/*
 * gnuplot plots.
 */

#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "dvec.h"
#include "fteparse.h"
#include "gnuplot.h"

#include <variable.h>

#define GP_MAXVECTORS 64

void
ft_gnuplot(double *xlims, double *ylims, char *filename, char *title, char *xlabel, char *ylabel, GRIDTYPE gridtype, PLOTTYPE plottype, struct dvec *vecs)
{

    FILE *file, *file_data;
    struct dvec *v, *scale;
    double xval, yval;
    int i, numVecs, linewidth;
    bool xlog, ylog, nogrid, markers;
    char buf[BSIZE_SP], pointstyle[BSIZE_SP], *text;

    char filename_data[15];
    sprintf(filename_data, "%s.data", filename);


    /* Sanity checking. */
    for ( v = vecs, numVecs = 0; v; v = v->v_link2 ) {
	numVecs++;
    }
    if (numVecs == 0) {
	return;
    } else if (numVecs > GP_MAXVECTORS) {
	fprintf( cp_err, "Error: too many vectors for gnuplot.\n" );
	return;
    }
    if (!cp_getvar("xbrushwidth", VT_NUM, &linewidth))
        linewidth = 1;
    if (linewidth < 1) linewidth = 1;

    if (!cp_getvar("pointstyle", VT_STRING, pointstyle)) {
        markers = FALSE;
    } else {
	if (cieq(pointstyle,"markers")) {
	    markers = TRUE;
	} else {
	    markers = FALSE;
	}
    }


    /* Make sure the gridtype is supported. */
    switch (gridtype) {
    case GRID_LIN:
	nogrid = xlog = ylog = FALSE;
	break;
    case GRID_XLOG:
	xlog = TRUE;
	nogrid = ylog = FALSE;
	break;
    case GRID_YLOG:
	ylog = TRUE;
	nogrid = xlog = FALSE;
	break;
    case GRID_LOGLOG:
	xlog = ylog = TRUE;
	nogrid = FALSE;
	break;
    case GRID_NONE:
	nogrid = TRUE;
	xlog = ylog = FALSE;
	break;
    default:
	fprintf( cp_err, "Error: grid type unsupported by gnuplot.\n" );
	return;
    }

    /* Open the output gnuplot file. */
    if (!(file = fopen(filename, "w"))) {
	perror(filename);
	return;
    }

    /* Set up the file header. */
    if (title) {
	text = cp_unquote(title);
	fprintf( file, "set title \"%s\"\n", text );
	tfree(text);
    }
    if (xlabel) {
	text = cp_unquote(xlabel);
	fprintf( file, "set xlabel \"%s\"\n", text );
	tfree(text);
    }
    if (ylabel) {
	text = cp_unquote(ylabel);
	fprintf( file, "set ylabel \"%s\"\n", text );
	tfree(text);
    }
    if (nogrid) {
	fprintf( file, "set grid\n" );
    }
    if (xlog) {
    	fprintf( file, "set logscale x\n" );
	if (xlims) {
	    fprintf( file, "set xrange [% e: % e]\n", log10(xlims[0]),log10(xlims[1]) );
	}
    } else {
    	fprintf( file, "unset logscale x \n" );
	if (xlims) {
	    fprintf( file, "set xrange [% e: % e]\n", xlims[0],xlims[1] );
	}
    }
    if (ylog) {
    	fprintf( file, "set logscale y \n" );
	if (ylims) {
	    fprintf( file, "set yrange [% e: % e]\n", log10(ylims[0]),log10(ylims[1]) );
	}
    } else {
    	fprintf( file, "unset logscale y \n" );
	if (ylims) {
	    fprintf( file, "set yrange [% e: % e]\n", ylims[0],ylims[1] );
	}
    }

    fprintf( file, "#set xtics 1\n" );
    fprintf( file, "#set x2tics 1\n" );
    fprintf( file, "#set ytics 1\n" );
    fprintf( file, "#set y2tics 1\n" );

/* TODO
    fprintf( file, "LineWidth: %d\n", linewidth );
    if (plottype == PLOT_COMB) {
	fprintf( file, "BarGraph: True\n" );
	fprintf( file, "NoLines: True\n" );
    } else if (plottype == PLOT_POINT) {
	if (markers) {
	    fprintf( file, "Markers: True\n" );
	} else {
	    fprintf( file, "LargePixels: True\n" );
	}
	fprintf( file, "NoLines: True\n" );
    }
*/

    /* Open the output gnuplot data file. */

    if (!(file_data = fopen(filename_data, "w"))) {
	perror(filename);
	return;
    }

    fprintf( file, "plot ", v->v_name ); 
    i = 0;

    /* Write out the data and setup arrays */
    for ( v = vecs; v; v = v->v_link2 ) {
	scale = v->v_scale;
	if (v->v_name) {
	    i= i+2;
	    fprintf( file, "\'%s\' using %d:%d with lines t \"%s\" ,", filename_data , i , i+1 ,  v->v_name );
	}
    }
    fprintf( file, "\n");
    fprintf (file, "set terminal postscript eps\nset out %s.eps\nreplot\nset term pop", filename);

   for ( i = 0; i < scale->v_length; i++ ) {
	for ( v = vecs; v; v = v->v_link2 ) {
        	scale = v->v_scale;

	        xval = isreal(scale) ?
		scale->v_realdata[i] : realpart(&scale->v_compdata[i]);

        	yval = isreal(v) ?
		v->v_realdata[i] : realpart(&v->v_compdata[i]);

        	fprintf( file_data, "% e % e ", xval, yval );
      	}
    fprintf( file_data, "\n");
    }
    

    (void) fclose( file );
    (void) fclose( file_data );

    (void) sprintf( buf, "gnuplot %s &", filename );
    (void) system( buf );


    return;
}
