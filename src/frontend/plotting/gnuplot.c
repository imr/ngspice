/**********
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
#if defined(__MINGW32__) || defined(_MSC_VER)
#undef BOOLEAN
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <locale.h>

#define GP_MAXVECTORS 64


static void
quote_gnuplot_string(FILE *stream, char *s)
{
    fputc('"', stream);

    for (; *s; s++)
        /* NOTE: The FALLTHROUGH comment is used to suppress a GCC warning
         * when flag -Wimplicit-fallthrough is present */
        switch (*s) {
        case '\n':
            fputs("\\n", stream);
            break;
        case '"':
        case '\\':
            fputc('\\', stream);
            /* FALLTHROUGH */
        default:
            fputc(*s, stream);
        }

    fputc('"', stream);
}


static double **dmatrix(int nrow, int ncol)
{
    double **d;
    int i;
    if (nrow < 2 && ncol < 2) {
        /* Who could want a 1x1 matrix? */
        return NULL;
    }
    d = TMALLOC(double *, nrow);
    for (i = 0; i < nrow; i++) {
        d[i] = TMALLOC(double, ncol);
    }
    return d;
}


static void dmatrix_free(double **d, int nrow, int ncol)
{
    int i;
    (void) ncol;
    if (d && nrow > 1) {
        for (i = 0; i < nrow; i++) {
            tfree(d[i]);
        }
        tfree(d);
    }
}


static bool has_contour_data(struct dvec *vecs)
{
    struct plot *curpl = NULL;
    struct dvec *v = NULL, *xvec = NULL, *yvec = NULL;
    int xdim, ydim, i, npoints;
    bool len_mismatch = FALSE, wrong_type = FALSE;

    if (!vecs) {
        return FALSE;
    }
    curpl = vecs->v_plot;
    if (!curpl) {
        return FALSE;
    }
    xdim = curpl->pl_xdim2d;
    ydim = curpl->pl_ydim2d;
    if (xdim < 2 || ydim < 2) {
        return FALSE;
    }

    for (v = vecs, i = 0; v; v = v->v_link2) {
        i++;
    }
    if (i > 1) {
        printf("Specify only one expr for an xycontour plot:");
        for (v = vecs; v; v = v->v_link2) {
            printf(" '%s'", v->v_name);
        }
        printf("\n");
        return FALSE;
    } else if (i < 1) {
        return FALSE;
    }
    if (!(vecs->v_flags & VF_REAL)) {
        wrong_type = TRUE;
    }
    npoints = xdim * ydim;
    if (vecs->v_length != npoints) {
        len_mismatch = TRUE;
    }

#ifdef VERBOSE_GNUPLOT
    printf("curpl vecs:");
    for (v = curpl->pl_dvecs; v; v = v->v_next) {
        printf(" '%s'", v->v_name);
    }
    printf("\n");
    printf("vecs:      ");
    for (v = vecs; v; v = v->v_next) {
        printf(" '%s'", v->v_name);
    }
    printf("\n");
#endif

    for (v = vecs; v; v = v->v_next) {
        /* Find the x and y vectors from the last part of the list.
           Passing by the the plotarg expr parsing elements at the front.
        */
        if (!(v->v_flags & VF_REAL)) {
            /* Only real types allowed */
            wrong_type = TRUE;
        }
        if (v->v_length != npoints) {
            /* Assume length 1 is a constant number */
            if (v->v_length != 1) {
                len_mismatch = TRUE;
            }
        }
        if (eq(v->v_name, "y")) {
            yvec = v;
            continue;
        }
        if (eq(v->v_name, "x")) {
            xvec = v;
        }
    }
    if (len_mismatch) {
        printf("Vector lengths mismatch, ignoring xycontour\n");
    }
    if (wrong_type) {
        printf("Non-real expr or constant, ignoring xycontour\n");
    }
    if (!xvec || !yvec || len_mismatch || wrong_type) {
        return FALSE;
    }
    return TRUE;
}


/* Precondition: has_contour_data was called and returned TRUE */
static int write_contour_data(FILE *filed, struct dvec *vecs)
{
    struct plot *curpl = NULL;
    struct dvec *v = NULL, *xvec = NULL, *yvec = NULL;
    int xdim, ydim, i, j, npoints, idx;
    double *ycol;
    double **zmat;

    if (!filed || !vecs) {
        return 1;
    }
    curpl = vecs->v_plot;
    if (!curpl) {
        return 1;
    }
    xdim = curpl->pl_xdim2d;
    ydim = curpl->pl_ydim2d;
    if (xdim < 2 || ydim < 2) {
        return 1;
    }

    npoints = xdim * ydim;
    for (v = vecs; v; v = v->v_next) {
        /* Use the x and y vectors from the last part of the list. */
        if (eq(v->v_name, "y")) {
            yvec = v;
            continue;
        }
        if (eq(v->v_name, "x")) {
            xvec = v;
        }
    }
    if (!xvec || !yvec) {
        return 1;
    }

    /* First output row has the x vector values */
    fprintf(filed, "%d", xdim);
    for (i = 0, j = 0; i < xdim; j += ydim) {
        if (j >= xvec->v_length) {
            return 1;
        }
        fprintf(filed, " %e", 1.0e6 * xvec->v_realdata[j]);
        i++;
    }
    fprintf(filed, "\n");

    ycol = TMALLOC(double, ydim);
    for (i = 0; i < ydim; i++) {
        ycol[i] = 1.0e6 * yvec->v_realdata[i];
    }
    zmat = dmatrix(ydim, xdim);
    idx = 0;
    for (i = 0; i < xdim; i++) {
        for (j = 0; j < ydim; j++) {
            zmat[j][i] = vecs->v_realdata[idx];
            idx++;
        }
    }
    if (idx != npoints) {
        tfree(ycol);
        dmatrix_free(zmat, ydim, xdim);
        return 1;
    }

    /* Subsequent output rows have a y vector value and the z matrix
       values corresponding to that y vector. There is a z matrix value
       for each x vector column.
    */
    for (i = 0; i < ydim; i++) {
        fprintf(filed, "%e", ycol[i]);
        for (j = 0; j < xdim; j++) {
            fprintf(filed, " %e", zmat[i][j]);
        }
        fprintf(filed, "\n");
    }

    tfree(ycol);
    dmatrix_free(zmat, ydim, xdim);
    return 0;
}


void ft_gnuplot(double *xlims, double *ylims,
        double xdel, double ydel,
        const char *filename, const char *title,
        const char *xlabel, const char *ylabel,
        GRIDTYPE gridtype, PLOTTYPE plottype,
        struct dvec *vecs, bool xycontour)
{
    FILE *file, *file_data;
    struct dvec *v, *scale = NULL;
    double xval, yval, prev_xval, extrange;
    int i, dir, numVecs, linewidth, gridlinewidth, err, terminal_type;
    bool xlog, ylog, nogrid, markers, nolegend, contours = FALSE;
    char buf[BSIZE_SP], pointstyle[BSIZE_SP], *text, plotstyle[BSIZE_SP], terminal[BSIZE_SP];

    char filename_data[128];
    char filename_plt[128];
    char *vtypename = NULL;

#ifdef SHARED_MODULE
    char* llocale = setlocale(LC_NUMERIC, NULL);
    setlocale(LC_NUMERIC, "C");
#endif

    snprintf(filename_data, 128, "%s.data", filename);
    snprintf(filename_plt, 128, "%s.plt", filename);

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

    if (xycontour) {
        contours = has_contour_data(vecs);
    }

    extrange = 0.05 * (ylims[1] - ylims[0]);

    if (!cp_getvar("gnuplot_terminal", CP_STRING,
            terminal, sizeof(terminal))) {
        terminal_type = 1;
    }
    else {
        terminal_type = 1;
        if (cieq(terminal,"png")) {
            terminal_type = 2;
        }
        else if (cieq(terminal,"png/quit")) {
            terminal_type = 3;
        }
        else if (cieq(terminal, "eps")) {
            terminal_type = 4;
        }
        else if (cieq(terminal, "eps/quit")) {
            terminal_type = 5;
        }
        else if (cieq(terminal, "xterm")) {
            terminal_type = 6;
        }
    }

    /* get linewidth for plotting the graph from .spiceinit */
    if (!cp_getvar("xbrushwidth", CP_NUM, &linewidth, 0))
        linewidth = 1;
    if (linewidth < 1)
        linewidth = 1;
    /* get linewidth for grid from .spiceinit */
    if (!cp_getvar("xgridwidth", CP_NUM, &gridlinewidth, 0))
        gridlinewidth = 1;
    if (gridlinewidth < 1)
        gridlinewidth = 1;


    if (!cp_getvar("pointstyle", CP_STRING, pointstyle, sizeof(pointstyle))) {
        markers = FALSE;
    } else {
        if (cieq(pointstyle,"markers"))
            markers = TRUE;
        else
            markers = FALSE;
    }

    if (!cp_getvar("nolegend", CP_BOOL, NULL, 0)) {
        nolegend = FALSE;
    }
    else {
        nolegend = TRUE;
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
#ifndef EXT_ASC
    fprintf(file, "set encoding utf8\n");
#endif
    fprintf(file, "set termoption noenhanced\n");

    if (contours) {
        fprintf(file, "set view map\n");
        fprintf(file, "set contour\n");
        fprintf(file, "unset surface\n");
        fprintf(file, "set cntrparam levels 20\n");
        fprintf(file, "set yrange reverse\n");
        fprintf(file, "set xlabel 'X microns'\n");
        fprintf(file, "set ylabel 'Y microns'\n");
        fprintf(file, "set key outside right\n");
        fprintf(file, "set title '%s - %s", vecs->v_plot->pl_title,
            vecs->v_name);
        vtypename = ft_typabbrev(vecs->v_type);
        if (vtypename) {
            fprintf(file, " %s'\n", vtypename);
        } else {
            fprintf(file, "'\n");
        }
    } else {
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
            if (gridlinewidth > 1)
                fprintf(file, "set grid lw %d \n" , gridlinewidth);
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

        if (xdel > 0.)
            fprintf(file, "set xtics %e\n", xdel);
        else
            fprintf(file, "#set xtics 1\n");
        fprintf(file, "#set x2tics 1\n");
        if (ydel > 0.)
            fprintf(file, "set ytics %e\n", ydel);
        else
            fprintf(file, "#set ytics 1\n");
        fprintf(file, "#set y2tics 1\n");

        if (gridlinewidth > 1)
            fprintf(file, "set border lw %d\n", gridlinewidth);

        if(nolegend)
            fprintf(file, "set key off\n");

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
    }

    /* Open the output gnuplot data file. */
    if ((file_data = fopen(filename_data, "w")) == NULL) {
        perror(filename);
        return;
    }
    if (contours) {
        if ((terminal_type != 3) && (terminal_type != 5)) {
            fprintf(file,
            "splot '%s' nonuniform matrix using 1:2:3 with lines lw 2 "
            "title ' '\n", filename_data);
        }
    } else {
        fprintf(file, "set format y \"%%g\"\n");
        fprintf(file, "set format x \"%%g\"\n");

        if ((terminal_type != 3) && (terminal_type != 5)) {
            fprintf(file, "plot ");
            i = 0;

            /* Write out the gnuplot command */
            for (v = vecs; v; v = v->v_link2) {
                scale = v->v_scale;
                if (v->v_name) {
                    i = i + 2;
                    if (i > 2) fprintf(file, ",\\\n");
                    fprintf(file, "\'%s\' using %d:%d with %s lw %d title ",
                        filename_data, i - 1, i, plotstyle, linewidth);
                    quote_gnuplot_string(file, v->v_name);
                }
            }
            fprintf(file, "\n");
        }
    }

    /* terminal_type
    1: do not print an eps or png file
    2: print png file, keep command window open
    3: print png file, quit command window
    4: print eps file, keep command window open
    5: print eps file, quit command window
    */
    if ((terminal_type == 2) || (terminal_type == 4))
        fprintf(file, "set terminal push\n");
    if ((terminal_type == 4) || (terminal_type == 5)) {
        fprintf(file, "set terminal postscript eps color noenhanced\n");
        fprintf(file, "set out \'%s.eps\'\n", filename);
    }
    if ((terminal_type == 2) || (terminal_type == 3)) {
        fprintf(file, "set terminal png noenhanced\n");
        fprintf(file, "set out \'%s.png\'\n", filename);
    }
    if ((terminal_type == 2) || (terminal_type == 4)) {
        fprintf(file, "replot\n");
        fprintf(file, "set term pop\n");
        fprintf(file, "replot\n");
    }

    if (contours) {
        if ((terminal_type == 3) || (terminal_type == 5)) {
            fprintf(file,
            "splot '%s' nonuniform matrix using 1:2:3 with lines lw 2 "
            "title ' '\n", filename_data);
            fprintf(file, "exit\n");
        }
    } else {
        if ((terminal_type == 3) || (terminal_type == 5)) {
            fprintf(file, "plot ");
            i = 0;

            /* Write out the gnuplot command */
            for (v = vecs; v; v = v->v_link2) {
                scale = v->v_scale;
                if (v->v_name) {
                    i = i + 2;
                    if (i > 2) fprintf(file, ",\\\n");
                    fprintf(file, "\'%s\' using %d:%d with %s lw %d title ",
                        filename_data, i - 1, i, plotstyle, linewidth);
                    quote_gnuplot_string(file, v->v_name);
                }
            }
            fprintf(file, "\n");
            fprintf(file, "exit\n");
        }
    }



    (void) fclose(file);

    if (contours) {
        if (write_contour_data(file_data, vecs) != 0) {
            fprintf(stderr, "Error when writing contour data file\n");
            (void) fclose(file_data);
            return;
        }
    } else {
        /* Write out the data and setup arrays */
        bool mono = (plottype != PLOT_RETLIN);
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
    }

    (void) fclose(file_data);

#if defined(__MINGW32__) || defined(_MSC_VER)
    /* for external fcn system() */
    // (void) sprintf(buf, "start /B wgnuplot %s -" ,  filename_plt);
    (void) sprintf(buf, "start /B wgnuplot -persist %s " ,  filename_plt);
    _flushall();
#else
    /* for external fcn system() from LINUX environment */
    if (terminal_type == 3) {
        fprintf(cp_out, "writing plot to file %s.png\n", filename);
        (void) sprintf(buf, "gnuplot %s", filename_plt);
    }
    else if (terminal_type == 5) {
        fprintf(cp_out, "writing plot to file %s.eps\n", filename);
        (void) sprintf(buf, "gnuplot %s", filename_plt);
    }
    else if (terminal_type == 6) {
        (void) sprintf(buf, "xterm -e gnuplot %s - &", filename_plt);
    }
    else {
        (void) sprintf(buf, "gnuplot -persist %s &", filename_plt);
    }
#endif
    err = system(buf);

    /* delete the plt and data files */
    if ((terminal_type == 3) || (terminal_type == 5)) {
        /* wait for gnuplot generating eps or png file */
#if defined(__MINGW32__) || defined(_MSC_VER)
        Sleep(200);
#else
        usleep(200000);
#endif
        if (remove(filename_data)) {
            fprintf(stderr, "Could not remove file %s\n", filename_data);
            perror(NULL);
        }
        if (remove(filename_plt)) {
            fprintf(stderr, "Could not remove file %s\n", filename_plt);
            perror(NULL);
        }
    }
#ifdef SHARED_MODULE
    /* go back to what it was before */
    setlocale(LC_NUMERIC, llocale);
#endif
}


/* simple printout of data into a file, similar to data table in ft_gnuplot
   command: wrdata file vecs, vectors of different length (from different plots)
   may be printed. Data are written in pairs: scale vector, value vector. If
   data are complex, a triple is printed (scale, real, imag).
   Setting 'singlescale' as variable, the scale vector will be printed once only,
   if scale vectors are of same length (there is little risk here!).
   Width of numbers printed is set by option 'numdgt'.
 */
void ft_writesimple(double *xlims, double *ylims,
        const char *filename, const char *title,
        const char *xlabel, const char *ylabel,
        GRIDTYPE gridtype, PLOTTYPE plottype,
        struct dvec *vecs)
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

    appendwrite = cp_getvar("appendwrite", CP_BOOL, NULL, 0);
    singlescale = cp_getvar("wr_singlescale", CP_BOOL, NULL, 0);
    vecnames = cp_getvar("wr_vecnames", CP_BOOL, NULL, 0);

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
