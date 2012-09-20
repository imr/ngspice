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

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/fteparse.h"
#include "gnuplot.h"


#define GP_MAXVECTORS 64


void
ft_gnuplot(double *xlims, double *ylims, char *filename, char *title, char *xlabel, char *ylabel, GRIDTYPE gridtype, PLOTTYPE plottype, struct dvec *vecs)
{
    FILE *file, *file_data;
    struct dvec *v, *scale = NULL;
    double xval, yval;
    int i, numVecs, linewidth, err;
    bool xlog, ylog, nogrid, markers;
    char buf[BSIZE_SP], pointstyle[BSIZE_SP], *text, plotstyle[BSIZE_SP];

    char filename_data[128];
    char filename_plt[128];

    sprintf(filename_data, "%s.data", filename);
    sprintf(filename_plt, "%s.plt", filename);

    /* Sanity checking. */
    for (v = vecs, numVecs = 0; v; v = v->v_link2)
        numVecs++;

    if (numVecs == 0) {
        return;
    } else if (numVecs > GP_MAXVECTORS) {
        fprintf(cp_err, "Error: too many vectors for gnuplot.\n");
        return;
    }
    if (!cp_getvar("xbrushwidth", CP_NUM, &linewidth))
        linewidth = 1;
    if (linewidth < 1) linewidth = 1;

    if (!cp_getvar("pointstyle", CP_STRING, pointstyle)) {
        markers = FALSE;
    } else {
        if (cieq(pointstyle,"markers"))
            markers = TRUE;
        else
            markers = FALSE;
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
        fprintf(cp_err, "Error: grid type unsupported by gnuplot.\n");
        return;
    }

    /* Open the output gnuplot file. */
    if ((file = fopen(filename_plt, "w")) == NULL) {
        perror(filename);
        return;
    }

    /* Set up the file header. */
    if (title) {
        text = cp_unquote(title);
        fprintf(file, "set title \"%s\"\n", text);
        tfree(text);
    }
    if (xlabel) {
        text = cp_unquote(xlabel);
        fprintf(file, "set xlabel \"%s\"\n", text);
        tfree(text);
    }
    if (ylabel) {
        text = cp_unquote(ylabel);
        fprintf(file, "set ylabel \"%s\"\n", text);
        tfree(text);
    }
    if (!nogrid) {
        if (linewidth > 1)
            fprintf(file, "set grid lw %d \n" , linewidth);
        else
            fprintf(file, "set grid\n");
    }
    if (xlog) {
        fprintf(file, "set logscale x\n");
        if (xlims)
            fprintf(file, "set xrange [%e:%e]\n", xlims[0], xlims[1]);
    } else {
        fprintf(file, "unset logscale x \n");
        if (xlims)
            fprintf(file, "set xrange [%e:%e]\n", xlims[0], xlims[1]);
    }
    if (ylog) {
        fprintf(file, "set logscale y \n");
        if (ylims)
            fprintf(file, "set yrange [%e:%e]\n", ylims[0], ylims[1]);
    } else {
        fprintf(file, "unset logscale y \n");
        if (ylims)
            fprintf(file, "set yrange [%e:%e]\n", ylims[0], ylims[1]);
    }

    fprintf(file, "#set xtics 1\n");
    fprintf(file, "#set x2tics 1\n");
    fprintf(file, "#set ytics 1\n");
    fprintf(file, "#set y2tics 1\n");

    if (linewidth > 1)
        fprintf(file, "set border lw %d\n", linewidth);

    if (plottype == PLOT_COMB) {
        strcpy(plotstyle, "boxes");
    } else if (plottype == PLOT_POINT) {
        if (markers) {
            // fprintf(file, "Markers: True\n");
        } else {
            // fprintf(file, "LargePixels: True\n");
        }
        strcpy(plotstyle, "points");
    } else {
        strcpy(plotstyle, "lines");
    }

    /* Open the output gnuplot data file. */
    if ((file_data = fopen(filename_data, "w")) == NULL) {
        perror(filename);
        return;
    }

    fprintf(file, "plot ");
    i = 0;

    /* Write out the gnuplot command */
    for (v = vecs; v; v = v->v_link2) {
        scale = v->v_scale;
        if (v->v_name) {
            i = i + 2;
            if (i > 2) fprintf(file, ",\\\n");
            fprintf(file, "\'%s\' using %d:%d with %s lw %d title \"%s\" ",
                    filename_data, i-1, i, plotstyle, linewidth, v->v_name);
        }
    }
    fprintf(file, "\n");
    fprintf(file, "set terminal push\n");
    fprintf(file, "set terminal postscript eps color\n");
    fprintf(file, "set out \'%s.eps\'\n", filename);
    fprintf(file, "replot\n");
    fprintf(file, "set term pop\n");
    fprintf(file, "replot\n");

    (void) fclose(file);

    /* Write out the data and setup arrays */
    for (i = 0; i < scale->v_length; i++) {
        for (v = vecs; v; v = v->v_link2) {
            scale = v->v_scale;

            xval = isreal(scale) ?
                   scale->v_realdata[i] : realpart(scale->v_compdata[i]);

            yval = isreal(v) ?
                   v->v_realdata[i] : realpart(v->v_compdata[i]);

            fprintf(file_data, "% e % e ", xval, yval);
        }
        fprintf(file_data, "\n");
    }

    (void) fclose(file_data);

#if defined(__MINGW32__) || defined(_MSC_VER)
    /* for external fcn system() */
    // (void) sprintf(buf, "start /B wgnuplot %s -" ,  filename_plt);
    (void) sprintf(buf, "start /B wgnuplot -persist %s " ,  filename_plt);
    _flushall();
#else
    /* for external fcn system() from LINUX environment */
    (void) sprintf(buf, "xterm -e gnuplot %s - &", filename_plt);
#endif
    err = system(buf);

    return;
}


/* simple printout of data into a file, similar to data table in ft_gnuplot
   command: wrsimple file vecs
 */
void
ft_writesimple(double *xlims, double *ylims, char *filename, char *title, char *xlabel, char *ylabel, GRIDTYPE gridtype, PLOTTYPE plottype, struct dvec *vecs)
{
    FILE *file_data;
    struct dvec *v, *scale = NULL;
    double xval;
    int i, numVecs;
    bool appendwrite;

    char filename_data[128];

    NG_IGNORE(xlims);
    NG_IGNORE(ylims);
    NG_IGNORE(title);
    NG_IGNORE(xlabel);
    NG_IGNORE(ylabel);
    NG_IGNORE(gridtype);
    NG_IGNORE(plottype);

    sprintf(filename_data, "%s.data", filename);
    appendwrite = cp_getvar("appendwrite", CP_BOOL, NULL);

    /* Sanity checking. */
    for (v = vecs, numVecs = 0; v; v = v->v_link2)
        numVecs++;

    if (numVecs == 0)
        return;

    /* Open the output data file. */
    if ((file_data = fopen(filename_data, appendwrite ? "a" : "w")) == NULL) {
        perror(filename);
        return;
    }

    i = 0;
    for (v = vecs; v; v = v->v_link2)
        scale = v->v_scale;

    /* Write out the data as simple arrays */
    for (i = 0; i < scale->v_length; i++) {
        for (v = vecs; v; v = v->v_link2) {
            scale = v->v_scale;

            xval = isreal(scale) ?
                   scale->v_realdata[i] : realpart(scale->v_compdata[i]);

            if (isreal(v))
                fprintf(file_data, "% e % e ", xval, v->v_realdata[i]);
            else
                fprintf(file_data, "% e % e % e ", xval, realpart(v->v_compdata[i]), imagpart(v->v_compdata[i]));
        }
        fprintf(file_data, "\n");
    }

    (void) fclose(file_data);

    return;
}
