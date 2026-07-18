/**********
 * Enhancement-94: matplotlib ("pyplot") plots.
 *
 * A backend for `plotit()` that mirrors the gnuplot backend (`gnuplot.c`):
 * it writes the selected vectors to a `<file>.data` table and a `<file>.py`
 * matplotlib script, then shells out to Python. Modelled on ft_gnuplot().
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/fteparse.h"
#include "pyplot.h"
#if defined(__MINGW32__) || defined(_MSC_VER)
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <locale.h>

#define PY_MAXVECTORS 64


/* Write `s` as a single-quoted Python string literal, escaping backslashes
   and single quotes. */
static void
quote_python_string(FILE *stream, const char *s)
{
    fputc('\'', stream);
    for (; s && *s; s++) {
        if (*s == '\\' || *s == '\'')
            fputc('\\', stream);
        fputc(*s, stream);
    }
    fputc('\'', stream);
}


void ft_pyplot(double *xlims, double *ylims,
        double xdel, double ydel,
        const char *filename, const char *title,
        const char *xlabel, const char *ylabel,
        GRIDTYPE gridtype, PLOTTYPE plottype,
        struct dvec *vecs)
{
    FILE *file, *file_data;
    struct dvec *v, *scale = NULL;
    int i, col, numVecs, err, nper, nrows, row;
    bool xlog, ylog, nogrid, markers, boxes, impulses, have_style, have_figsize;
    char pointstyle[BSIZE_SP], terminal[BSIZE_SP], python[BSIZE_SP], style[BSIZE_SP];
    char figsize[BSIZE_SP], fmt[16];
    char lwarg[32];         /* Enhancement-183: "linewidth=%g, " or "" */
    double linewidth = 0.0;
    char backend[BSIZE_SP]; /* Enhancement-183: matplotlib backend override */
    bool have_backend;
    /* Enhancement-183: hold a full directory path (the deck's folder) + base
       name, not just a bare "pyplot.data" -- 128 was too small for a path. */
    char filename_data[1024], filename_py[1024];
    char buf[2 * 1024 + BSIZE_SP];
    char *text;
    double figw = 0.0, figh = 0.0;
    bool hardcopy = FALSE;

    NG_IGNORE(xdel);
    NG_IGNORE(ydel);

#ifdef SHARED_MODULE
    char *llocale = setlocale(LC_NUMERIC, NULL);
    setlocale(LC_NUMERIC, "C");
#endif

    snprintf(filename_data, sizeof(filename_data), "%s.data", filename);
    snprintf(filename_py, sizeof(filename_py), "%s.py", filename);

    for (v = vecs, numVecs = 0; v; v = v->v_link2)
        numVecs++;

    if (numVecs == 0) {
        return;
    } else if (numVecs > PY_MAXVECTORS) {
        fprintf(cp_err, "Error: too many vectors for pyplot.\n");
        return;
    }

    /* `set pyplot_terminal=png|svg|pdf` -> render headless (Agg) to
       <file>.<fmt> rather than opening an interactive window. Enhancement-99
       adds the svg and pdf vector formats alongside png. */
    fmt[0] = '\0';
    if (cp_getvar("pyplot_terminal", CP_STRING, terminal, sizeof(terminal))) {
        if (cieq(terminal, "png") || cieq(terminal, "png/quit")) {
            strcpy(fmt, "png");
            hardcopy = TRUE;
        } else if (cieq(terminal, "svg") || cieq(terminal, "svg/quit")) {
            strcpy(fmt, "svg");
            hardcopy = TRUE;
        } else if (cieq(terminal, "pdf") || cieq(terminal, "pdf/quit")) {
            strcpy(fmt, "pdf");
            hardcopy = TRUE;
        }
    }

    /* Enhancement-99: `set pyplot_figsize=W,H` -> figure size in inches. */
    have_figsize = FALSE;
    if (cp_getvar("pyplot_figsize", CP_STRING, figsize, sizeof(figsize))) {
        if (sscanf(figsize, "%lf%*[ ,xX]%lf", &figw, &figh) == 2
                && figw > 0.0 && figh > 0.0)
            have_figsize = TRUE;
    }

    /* the Python interpreter, overridable with `set pyplot_python=...`. */
    if (!cp_getvar("pyplot_python", CP_STRING, python, sizeof(python)))
        strcpy(python, "python3");

    /* Enhancement-183: `set pyplot_backend=<name>` -> select the matplotlib
       backend explicitly (e.g. TkAgg, QtAgg, MacOSX, WebAgg, Agg). Overrides
       the automatic backend, including the 'Agg' otherwise forced for the
       png/svg/pdf terminals -- so it is the user's responsibility to pick a
       file-capable/headless backend when combining it with those. */
    have_backend = cp_getvar("pyplot_backend", CP_STRING, backend, sizeof(backend))
                   ? TRUE : FALSE;

    /* Enhancement-98: `set pyplot_subplots=N` -> stacked subplots sharing the
       x-axis, N traces per panel (0/unset = a single axis, as before). */
    if (!cp_getvar("pyplot_subplots", CP_NUM, &nper, 0))
        nper = 0;
    if (nper < 0)
        nper = 0;
    nrows = (nper > 0) ? ((numVecs + nper - 1) / nper) : 1;

    /* Enhancement-98: `set pyplot_style=<name>` -> a matplotlib style sheet
       (e.g. dark, ggplot, bmh). "dark" aliases matplotlib's dark_background. */
    have_style = cp_getvar("pyplot_style", CP_STRING, style, sizeof(style)) ? TRUE : FALSE;
    if (have_style && cieq(style, "dark"))
        strcpy(style, "dark_background");

    /* Enhancement-183: `set pyplot_linewidth=<w>` -> matplotlib line width (in
       points) applied to every trace; unset/<=0 leaves matplotlib's default. */
    lwarg[0] = '\0';
    if (cp_getvar("pyplot_linewidth", CP_REAL, &linewidth, 0) && linewidth > 0.0)
        (void) snprintf(lwarg, sizeof lwarg, "linewidth=%g, ", linewidth);

    markers = FALSE;
    if (cp_getvar("pointstyle", CP_STRING, pointstyle, sizeof(pointstyle)))
        if (cieq(pointstyle, "markers"))
            markers = TRUE;

    impulses = (plottype == PLOT_COMB);
    boxes = (plottype == PLOT_BOXES);
    if (plottype == PLOT_POINT)
        markers = TRUE;

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
        fprintf(cp_err, "Error: grid type unsupported by pyplot.\n");
        return;
    }

    /* Write the data table: for each row, an (x, y) pair per vector, taken
       from each vector's own scale (real part for complex data). */
    if ((file_data = fopen(filename_data, "w")) == NULL) {
        perror(filename);
        return;
    }
    scale = vecs->v_scale;
    for (i = 0; i < scale->v_length; i++) {
        for (v = vecs; v; v = v->v_link2) {
            struct dvec *sc = v->v_scale;
            double xval = (i < sc->v_length)
                ? (isreal(sc) ? sc->v_realdata[i] : realpart(sc->v_compdata[i]))
                : NAN;
            double yval = (i < v->v_length)
                ? (isreal(v) ? v->v_realdata[i] : realpart(v->v_compdata[i]))
                : NAN;
            fprintf(file_data, "%e %e ", xval, yval);
        }
        fprintf(file_data, "\n");
    }
    (void) fclose(file_data);

    /* Write the matplotlib script. */
    if ((file = fopen(filename_py, "w")) == NULL) {
        perror(filename);
        return;
    }
    fprintf(file, "#!/usr/bin/env python3\n");
    fprintf(file, "# generated by ngspice 'pyplot' (Enhancement-94)\n");
    fprintf(file, "import numpy as np\n");
    /* Enhancement-183: an explicit `pyplot_backend` wins; otherwise the file
       terminals render headless with Agg (unchanged). matplotlib.use() must
       precede `import matplotlib.pyplot`. */
    if (have_backend) {
        fprintf(file, "import matplotlib\n");
        fprintf(file, "matplotlib.use(");
        quote_python_string(file, backend);
        fprintf(file, ")\n");
    } else if (hardcopy) {
        fprintf(file, "import matplotlib\n");
        fprintf(file, "matplotlib.use('Agg')\n");
    }
    fprintf(file, "import matplotlib.pyplot as plt\n");
    /* Enhancement-98: apply a matplotlib style sheet if requested (ignore an
       unknown name rather than aborting the plot). */
    if (have_style) {
        fprintf(file, "try:\n    plt.style.use(");
        quote_python_string(file, style);
        fprintf(file, ")\nexcept Exception:\n    pass\n");
    }
    fprintf(file, "d = np.loadtxt(");
    quote_python_string(file, filename_data);
    fprintf(file, ")\n");
    fprintf(file, "if d.ndim == 1:\n    d = d.reshape(-1, %d)\n", 2 * numVecs);
    /* Enhancement-98: one axis, or `nrows` stacked subplots sharing the x-axis.
       `axes` is always a 2-D array (squeeze=False) so it is indexed uniformly. */
    if (have_figsize)
        fprintf(file,
                "fig, axes = plt.subplots(%d, 1, sharex=True, squeeze=False, "
                "figsize=(%g, %g))\n", nrows, figw, figh);
    else
        fprintf(file, "fig, axes = plt.subplots(%d, 1, sharex=True, squeeze=False)\n", nrows);

    col = 0;
    row = 0;
    i = 0;
    for (v = vecs; v; v = v->v_link2) {
        row = (nper > 0) ? (i / nper) : 0;
        fprintf(file, "axes[%d, 0].", row);
        if (boxes)
            fprintf(file, "step(d[:, %d], d[:, %d], where='mid', %s", col, col + 1, lwarg);
        else if (impulses)
            fprintf(file, "stem(d[:, %d], d[:, %d], markerfmt=' ', %s", col, col + 1, lwarg);
        else if (markers)
            fprintf(file, "plot(d[:, %d], d[:, %d], marker='.', linestyle='None', ",
                    col, col + 1);
        else
            fprintf(file, "plot(d[:, %d], d[:, %d], %s", col, col + 1, lwarg);
        fprintf(file, "label=");
        quote_python_string(file, v->v_name ? v->v_name : "");
        fprintf(file, ")\n");
        col += 2;
        i++;
    }

    /* Per-axis cosmetics applied to every panel; the x-label goes on the
       bottom panel only, the title becomes the figure suptitle. */
    fprintf(file, "for _ax in axes[:, 0]:\n");
    if (ylabel) {
        text = cp_unquote(ylabel);
        fprintf(file, "    _ax.set_ylabel(");
        quote_python_string(file, text);
        fprintf(file, ")\n");
        tfree(text);
    }
    if (xlog)
        fprintf(file, "    _ax.set_xscale('log')\n");
    if (ylog)
        fprintf(file, "    _ax.set_yscale('log')\n");
    if (!nogrid)
        fprintf(file, "    _ax.grid(True, which='both')\n");
    if (xlims)
        fprintf(file, "    _ax.set_xlim(%e, %e)\n", xlims[0], xlims[1]);
    if (ylims && !ylog)
        fprintf(file, "    _ax.set_ylim(%e, %e)\n", ylims[0], ylims[1]);
    fprintf(file, "    _ax.legend()\n");
    if (xlabel) {
        text = cp_unquote(xlabel);
        fprintf(file, "axes[-1, 0].set_xlabel(");
        quote_python_string(file, text);
        fprintf(file, ")\n");
        tfree(text);
    }
    if (title) {
        text = cp_unquote(title);
        fprintf(file, "fig.suptitle(");
        quote_python_string(file, text);
        fprintf(file, ")\n");
        tfree(text);
    }
    fprintf(file, "fig.tight_layout()\n");
    if (hardcopy) {
        fprintf(file, "fig.savefig(");
        quote_python_string(file, filename);
        fprintf(file, " + '.%s', dpi=100)\n", fmt);
        fprintf(file, "print('pyplot: wrote %s.%s')\n", filename, fmt);
    } else {
        fprintf(file, "plt.show()\n");
    }
    (void) fclose(file);

    /* Run it: synchronously for a PNG, in the background for a window. */
#if defined(__MINGW32__) || defined(_MSC_VER)
    if (hardcopy)
        (void) snprintf(buf, sizeof(buf), "%s %s", python, filename_py);
    else
        (void) snprintf(buf, sizeof(buf), "start /B %s %s", python, filename_py);
    _flushall();
#else
    if (hardcopy)
        (void) snprintf(buf, sizeof(buf), "%s %s", python, filename_py);
    else
        (void) snprintf(buf, sizeof(buf), "%s %s &", python, filename_py);
#endif
    err = system(buf);
    if (err == -1)
        fprintf(cp_err, "Error: could not run '%s'.\n", buf);

#ifdef SHARED_MODULE
    setlocale(LC_NUMERIC, llocale);
#endif
}
