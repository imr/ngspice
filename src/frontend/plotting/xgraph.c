/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1992 David A. Gates, U. C. Berkeley CAD Group
**********/

/*
 * Xgraph-11 plots.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/fteparse.h"
#include "xgraph.h"
#include "ngspice/fteext.h"


#define XG_MAXVECTORS 64

void
ft_xgraph(double *xlims, double *ylims, char *filename, char *title, char *xlabel, char *ylabel, GRIDTYPE gridtype, PLOTTYPE plottype, struct dvec *vecs)
{
    FILE *file;
    struct dvec *v, *scale;
    double xval, yval;
    int i, numVecs, linewidth;
    bool xlog, ylog, nogrid, markers;
    char buf[BSIZE_SP], pointstyle[BSIZE_SP], *text;

    /* Sanity checking. */
    for (v = vecs, numVecs = 0; v; v = v->v_link2)
        numVecs++;

    if (numVecs == 0) {
        return;
    } else if (numVecs > XG_MAXVECTORS) {
        fprintf(cp_err, "Error: too many vectors for Xgraph.\n");
        return;
    }

    if (!cp_getvar("xbrushwidth", CP_NUM, &linewidth))
        linewidth = 1;

    if (linewidth < 1)
        linewidth = 1;

    if (!cp_getvar("pointstyle", CP_STRING, pointstyle)) {
        markers = FALSE;
    } else {
        if (cieq(pointstyle, "markers"))
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
        fprintf(cp_err, "Error: grid type unsupported by Xgraph.\n");
        return;
    }

    /* Open the output file. */
    /* add user defined path (nname has to be freed after usage) */
    char *nname = set_output_path(filename);
    if ((file = fopen(nname, "w")) == NULL) {
        perror(filename);
        tfree(nname);
        return;
    }
    tfree(nname);

    /* Set up the file header. */
    if (title) {
        text = cp_unquote(title);
        fprintf(file, "TitleText: %s\n", text);
        tfree(text);
    }
    if (xlabel) {
        text = cp_unquote(xlabel);
        fprintf(file, "XUnitText: %s\n", text);
        tfree(text);
    }
    if (ylabel) {
        text = cp_unquote(ylabel);
        fprintf(file, "YUnitText: %s\n", text);
        tfree(text);
    }
    if (nogrid) {
        fprintf(file, "Ticks: True\n");
    }

    if (xlog) {
        fprintf(file, "LogX: True\n");
        if (xlims) {
            fprintf(file, "XLowLimit:  % e\n", log10(xlims[0]));
            fprintf(file, "XHighLimit: % e\n", log10(xlims[1]));
        }
    } else {
        if (xlims) {
            fprintf(file, "XLowLimit:  % e\n", xlims[0]);
            fprintf(file, "XHighLimit: % e\n", xlims[1]);
        }
    }

    if (ylog) {
        fprintf(file, "LogY: True\n");
        if (ylims) {
            fprintf(file, "YLowLimit:  % e\n", log10(ylims[0]));
            fprintf(file, "YHighLimit: % e\n", log10(ylims[1]));
        }
    } else {
        if (ylims) {
            fprintf(file, "YLowLimit:  % e\n", ylims[0]);
            fprintf(file, "YHighLimit: % e\n", ylims[1]);
        }
    }

    fprintf(file, "LineWidth: %d\n", linewidth);
    fprintf(file, "BoundBox: True\n");

    if (plottype == PLOT_COMB) {
        fprintf(file, "BarGraph: True\n");
        fprintf(file, "NoLines: True\n");
    } else if (plottype == PLOT_POINT) {
        if (markers)
            fprintf(file, "Markers: True\n");
        else
            fprintf(file, "LargePixels: True\n");
        fprintf(file, "NoLines: True\n");
    }

    /* Write out the data. */
    for (v = vecs; v; v = v->v_link2) {
        scale = v->v_scale;
        if (v->v_name)
            fprintf(file, "\"%s\"\n", v->v_name);

        for (i = 0; i < scale->v_length; i++) {
            xval = isreal(scale) ?
                scale->v_realdata[i] : realpart(scale->v_compdata[i]);
            yval = isreal(v) ?
                v->v_realdata[i] : realpart(v->v_compdata[i]);
            fprintf(file, "% e % e\n", xval, yval);
        }
        fprintf(file, "\n");
    }

    (void) fclose(file);
    (void) sprintf(buf, "xgraph %s &", filename);
    (void) system(buf);
}
