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


static void
quote_gnuplot_string(FILE *stream, char *s)
{
    fputc('"', stream);

    for (; *s; s++)
        switch (*s) {
        case '\n':
            fputs("\\n", stream);
            break;
        case '"':
        case '\\':
            fputc('\\', stream);
        default:
            fputc(*s, stream);
        }

    fputc('"', stream);
}


void
ft_gnuplot(double *xlims, double *ylims, char *filename, char *title, char *xlabel, char *ylabel, GRIDTYPE gridtype, PLOTTYPE plottype, struct dvec *vecs)
{
    FILE *file, *file_data;
    struct dvec *v, *scale = NULL;
    double xval, yval, prev_xval, extrange;
    int i, dir, numVecs, linewidth, err, terminal_type;
    bool xlog, ylog, nogrid, markers;
    char buf[BSIZE_SP], pointstyle[BSIZE_SP], *text, plotstyle[BSIZE_SP], terminal[BSIZE_SP];

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

    if (ylims && (fabs((ylims[1]-ylims[0])/ylims[0]) < 1.0e-6)) {
        fprintf(cp_err, "Error: range min ... max too small for using gnuplot.\n");
        fprintf(cp_err, "  Consider plotting with offset %g.\n", ylims[0]);
        return;
    }

    extrange = 0.05 * (ylims[1] - ylims[0]);

    if (!cp_getvar("gnuplot_terminal", CP_STRING, terminal)) {
        terminal_type = 1;
    } else {
        terminal_type = 1;
        if (cieq(terminal,"png"))
            terminal_type = 2;
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
#if !defined(__MINGW32__) && !defined(_MSC_VER)
    fprintf(file, "set terminal X11 noenhanced\n");
#else
    fprintf(file, "set termoption noenhanced\n");
#endif
    if (title) {
        text = cp_unquote(title);
        fprintf(file, "set title ");
        quote_gnuplot_string(file, text);
        fprintf(file, "\n");
        tfree(text);
    }
    if (xlabel) {
        text = cp_unquote(xlabel);
        fprintf(file, "set xlabel ");
        quote_gnuplot_string(file, text);
        fprintf(file, "\n");
        tfree(text);
    }
    if (ylabel) {
        text = cp_unquote(ylabel);
        fprintf(file, "set ylabel ");
        quote_gnuplot_string(file, text);
        fprintf(file, "\n");
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
            fprintf(file, "set xrange [%1.0e:%1.0e]\n", 
                pow(10, floor(log10(xlims[0]))), pow(10, ceil(log10(xlims[1]))));
        fprintf(file, "set mxtics 10\n");
        fprintf(file, "set grid mxtics\n");
    } else {
        fprintf(file, "unset logscale x \n");
        if (xlims)
            fprintf(file, "set xrange [%e:%e]\n", xlims[0], xlims[1]);
    }
    if (ylog) {
        fprintf(file, "set logscale y \n");
        if (ylims)
            fprintf(file, "set yrange [%1.0e:%1.0e]\n", 
                pow(10, floor(log10(ylims[0]))), pow(10, ceil(log10(ylims[1]))));
        fprintf(file, "set mytics 10\n");
        fprintf(file, "set grid mytics\n");
    } else {
        fprintf(file, "unset logscale y \n");
        if (ylims)
            fprintf(file, "set yrange [%e:%e]\n", ylims[0] - extrange, ylims[1] + extrange);
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
    fprintf(file, "set format y \"%%g\"\n");
    fprintf(file, "set format x \"%%g\"\n");
    fprintf(file, "plot ");
    i = 0;

    /* Write out the gnuplot command */
    for (v = vecs; v; v = v->v_link2) {
        scale = v->v_scale;
        if (v->v_name) {
            i = i + 2;
            if (i > 2) fprintf(file, ",\\\n");
            fprintf(file, "\'%s\' using %d:%d with %s lw %d title ",
                    filename_data, i-1, i, plotstyle, linewidth);
            quote_gnuplot_string(file, v->v_name);
        }
    }
    fprintf(file, "\n");

    /* do not print an eps or png file if filename start with 'np_' */
    if (!ciprefix("np_", filename)) {
        fprintf(file, "set terminal push\n");
        if (terminal_type == 1) {
            fprintf(file, "set terminal postscript eps color noenhanced\n");
            fprintf(file, "set out \'%s.eps\'\n", filename);
        }
        else {
            fprintf(file, "set terminal png noenhanced\n");
            fprintf(file, "set out \'%s.png\'\n", filename);
        }
        fprintf(file, "replot\n");
        fprintf(file, "set term pop\n");
    }

    fprintf(file, "replot\n");

    (void) fclose(file);

    /* Write out the data and setup arrays */
    bool mono = (plottype == PLOT_MONOLIN);
    dir = 0;
    prev_xval = NAN;
    for (i = 0; i < scale->v_length; i++) {
        for (v = vecs; v; v = v->v_link2) {
            scale = v->v_scale;

            xval = isreal(scale) ?
                   scale->v_realdata[i] : realpart(scale->v_compdata[i]);

            yval = isreal(v) ?
                   v->v_realdata[i] : realpart(v->v_compdata[i]);

            if (i > 0 && (mono || (scale->v_plot && scale->v_plot->pl_scale == scale))) {
                if (dir * (xval - prev_xval) < 0) {
                    /* direction reversal, start a new graph */
                    fprintf(file_data, "\n");
                    dir = 0;
                } else if (!dir && xval > prev_xval) {
                    dir = 1;
                } else if (!dir && xval < prev_xval) {
                    dir = -1;
                }
            }

            fprintf(file_data, "%e %e ", xval, yval);

            prev_xval = xval;
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

}


/* simple printout of data into a file, similar to data table in ft_gnuplot
   command: wrdata file vecs, vectors of different length (from different plots)
   may be printed. Data are written in pairs: scale vector, value vector. If
   data are complex, a triple is printed (scale, real, imag).
   Setting 'singlescale' as variable, the scale vector will be printed once only,
   if scale vectors are of same length (there is little risk here!).
   Width of numbers printed is set by option 'numdgt'.
 */
void
ft_writesimple(double *xlims, double *ylims, char *filename, char *title, char *xlabel, char *ylabel, GRIDTYPE gridtype, PLOTTYPE plottype, struct dvec *vecs)
{
    FILE *file_data;
    struct dvec *v;
    int i, numVecs, maxlen, preci;
    bool appendwrite, singlescale, vecnames;

    NG_IGNORE(xlims);
    NG_IGNORE(ylims);
    NG_IGNORE(title);
    NG_IGNORE(xlabel);
    NG_IGNORE(ylabel);
    NG_IGNORE(gridtype);
    NG_IGNORE(plottype);

    appendwrite = cp_getvar("appendwrite", CP_BOOL, NULL);
    singlescale = cp_getvar("wr_singlescale", CP_BOOL, NULL);
    vecnames = cp_getvar("wr_vecnames", CP_BOOL, NULL);

    /* Sanity checking. */
    for (v = vecs, numVecs = 0; v; v = v->v_link2)
        numVecs++;

    if (numVecs == 0)
        return;

    /* print scale vector only once */
    if (singlescale) {
        /* check if all vectors have equal scale length */
        maxlen = vecs->v_length; /* first length of vector read */
        for (v = vecs; v; v = v->v_link2)
            if (v->v_scale->v_length != maxlen) {
                fprintf(stderr,
                        "Error: Option 'singlescale' not possible.\n"
                        "       Vectors %s and %s have different lengths!\n"
                        "       No data written to %s!\n\n",
                        vecs->v_name, v->v_name, filename);
                return;
            }
    }
    else {
        /* find maximum scale length from all vectors */
        maxlen = 0;
        for (v = vecs; v; v = v->v_link2)
            maxlen = MAX(v->v_scale->v_length, maxlen);
    }

    /* Open the output data file. */
    if ((file_data = fopen(filename, appendwrite ? "a" : "w")) == NULL) {
        perror(filename);
        return;
    }

    /* If option numdgt is set, use it for printout precision. */
    if (cp_numdgt > 0)
        preci = cp_numdgt;
    else
        preci = 8;

    /* Print names of vectors to first line */
    if (vecnames) {
        bool prscale = TRUE;
        for (v = vecs; v; v = v->v_link2) {
            struct dvec *scale = v->v_scale;
            /* If wr_singlescale is set, print scale name only in first column */
            if (prscale)
                fprintf(file_data, " %-*s", preci + 7, scale->v_name);

            if (isreal(v))
                fprintf(file_data, " %-*s", preci + 7, v->v_name);
            else
                fprintf(file_data, " %-*s %-*s", preci + 7, v->v_name, preci + 7, v->v_name);
            if (singlescale)
                /* the following names are printed without scale vector names */
                prscale = FALSE;
        }
        fprintf(file_data, "\n");
    }

    /* Write out the data as simple arrays */
    for (i = 0; i < maxlen; i++) {
        bool prscale = TRUE;
        /* print scale from the first vector, then only if wr_singlescale is not set */
        for (v = vecs; v; v = v->v_link2) {
            struct dvec *scale = v->v_scale;
            /* if no more scale and value data, just print spaces */
            if (i >= scale->v_length) {
                if (prscale)
                    fprintf(file_data, "%*s", preci + 8, "");

                if (isreal(v))
                    fprintf(file_data, "%*s", preci + 8, "");
                else
                    fprintf(file_data, "%*s", 2 * (preci + 8), "");
            }
            else {
                if (prscale) {
                    double xval = isreal(scale)
                        ? scale->v_realdata[i]
                        : realpart(scale->v_compdata[i]);
                    fprintf(file_data, "% .*e ", preci, xval);
                }

                if (isreal(v))
                    fprintf(file_data, "% .*e ", preci, v->v_realdata[i]);
                else
                    fprintf(file_data, "% .*e % .*e ", preci, realpart(v->v_compdata[i]), preci, imagpart(v->v_compdata[i]));
            }
            if (singlescale)
                /* the following vectors are printed without scale vector */
                prscale = FALSE;
        }
        fprintf(file_data, "\n");
    }

    (void) fclose(file_data);
}
