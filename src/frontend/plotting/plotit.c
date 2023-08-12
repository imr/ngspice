#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/dstring.h"
#include "ngspice/wordlist.h"
#include "ngspice/graph.h"
#include "ngspice/cpdefs.h"
#include "ngspice/pnode.h"
#include "ngspice/sim.h"
#include "ngspice/fteext.h"
#include "ngspice/compatmode.h"

#include <circuits.h>

#include "plotit.h"
#include "points.h"
#include "agraf.h"
#include "gnuplot.h"
#include "graf.h"

static bool sameflag;
/* All these things are static so that "samep" will work.
  They are outside of plotit() to allow deleting */
static double *xcompress = NULL, *xindices = NULL;
static double *xlim = NULL, *ylim = NULL;
static double *xdelta = NULL, *ydelta = NULL;
static char *xlabel = NULL, *ylabel = NULL, *title = NULL;
#ifdef TCL_MODULE
#include "ngspice/tclspice.h"
#endif


static struct dvec *vec_self(struct dvec *v);
static struct dvec *vec_scale(struct dvec *v);
static void find_axis_limits(double *lim, bool oneval, bool f_real,
        struct dvec *vecs,
        struct dvec *(*p_get_axis_dvec)(struct dvec *dvec),
        double *lims);


/* Remove the malloced parameters upon ngspice quit. These are set to NULL
 * to allow the function to be used at any time and safely called more than
 * one time. */
void pl_rempar(void)
{
    tfree(xcompress);
    tfree(xindices);
    tfree(xlim);
    tfree(ylim);
    tfree(xdelta);
    tfree(ydelta);
    tfree(xlabel);
    tfree(ylabel);
}

/* This routine gets parameters from the command line, which are of
 * the form "name number ..." It returns a pointer to the parameter
 * values.
 *
 * Parameters
 * wl: Wordlist prefixed with dummy node from which the parameter value or
 *      values is to be extracted. On return, the nodes corresponding to the
 *      name of the parameter and the following value nodes are removed.
 * name: Name of parameter
 * number: number of values for the parameter
 *
 * Return values
 * Allocated list of values extracted from the wordlist
 *
 * Remarks
 * The dummy node at the front of wl guarantees that removing the nodes
 * for this parameter will not change the first node of the wordlist. 
 */
static double *getlims(wordlist *wl, const char *name, int number)
{
    wordlist *wk;
    int n;

    if (number < 1) { /* Parameter takes no argument */
        return (double *) NULL;
    }

    /* Locate parameter name in the wordlist */
    wordlist * const beg = wl_find(name, wl->wl_next);
    if (!beg) { /* not foumd */
        return (double *) NULL;
    }

    wk = beg->wl_next; /* Start of values for parameter */

    double * const d = TMALLOC(double, number); /* alloc for returned vals */

    for (n = 0; n < number; n++) { /* loop over values */
        char *ss;

        if (!wk) {
            fprintf(cp_err,
                    "Syntax error: not enough parameters for \"%s\".\n", name);
            txfree(d);
            return (double *) NULL;
        }

        ss = wk->wl_word;
        if (ft_numparse(&ss, FALSE, d + n) < 0) { /* put val in d[n] */
            fprintf(cp_err,
                    "Syntax error: bad parameters for \"%s\".\n", name);
            txfree(d);
            return (double *) NULL;
        }

        wk = wk->wl_next;
    } /* end of loop over numbers */

    wl_delete_slice(beg, wk); /* remove param name and its value nodes */

    return d;
} /* end of function getlims */



/* Extend a data vector to length by replicating the last element, or
 * truncate it if it is too long. If the vector is empty, it is
 * extended with NAN */
static void xtend(struct dvec *v, int length)
{
    int i;

    if (v->v_length == length) { /* no change required */
        return;
    }

    if (v->v_length > length) { /* too long */
        dvec_trunc(v, length);
        return;
    }

    /* Else must be extended */
    i = v->v_length;

    dvec_realloc(v, length, NULL);

    if (isreal(v)) {
        double d = NAN;
        if (i > 0) { /* At least one value */
            d = v->v_realdata[i - 1];
        }
        while (i < length) { /* Fill new elements at end */
            v->v_realdata[i++] = d;
        }
    }
    else {
        ngcomplex_t c = {NAN, NAN};
        if (i > 0) { /* At least one value */
            c = v->v_compdata[i - 1];
        }
        while (i < length) { /* Fill new elements at end */
            v->v_compdata[i++] = c;
        }
    }
} /* end of function xtend */



/* Collapse every *xcomp elements into one, and use only the elements
 * between xind[0] and xind[1]. Decimate would be a better description
 * than compress */
static void compress(struct dvec *d, double *xcomp, double *xind)
{
    if (xind) {
        int newlen;
        const int ilo = (int) xind[0];
        const int ihi = (int) xind[1];
        if ((ihi >= ilo) && (ilo > 0) && (ilo < d->v_length) &&
            (ihi > 1) && (ihi <= d->v_length)) {
            newlen = ihi - ilo;
            if (isreal(d)) {
                double *dd = TMALLOC(double, newlen);
                memcpy(dd, d->v_realdata + ilo, (size_t) newlen * sizeof(double));
                dvec_realloc(d, newlen, dd);
            } else {
                ngcomplex_t *cc = TMALLOC(ngcomplex_t, newlen);
                memcpy(cc, d->v_compdata + ilo, (size_t) newlen * sizeof(ngcomplex_t));
                dvec_realloc(d, newlen, cc);
            }
        }
    }

    if (xcomp) {
        const int cfac = (int) *xcomp;
        if ((cfac > 1) && (cfac < d->v_length)) {
            int i, j;
            const int n = d->v_length;
            for (i = 0, j = 0; j < n; i++, j += cfac) {
                if (isreal(d)) {
                    d->v_realdata[i] = d->v_realdata[j];
                }
                else {
                    d->v_compdata[i] = d->v_compdata[j];
                }
            }
            dvec_trunc(d, i);
        }
    }
} /* end of function compress */



/* Check for and remove a one-word keyword (without an argument). */
static bool getflag(wordlist *wl, const char *name)
{
    wl = wl_find(name, wl->wl_next);

    if (!wl) {
        return FALSE;
    }

    wl_delete_slice(wl, wl->wl_next);

    return TRUE;
} /* end of function getflag */



/* Return a copy of the value of parameter and deletes the keyword and
 * value nodes in the wordlist. The search for the keyword begins after
 * the node wl. (This behavior is due to a dummy node being added
 * to the front of the wordlist.)
 *
 * Parameters
 * wl: wordlist to process
 * sz_keyword: keyword to locate
 *
 * Return values
 * NULL: The keyword was not found or its value was missing
 * allocation consisting of the value node as a string.
 *
 * Example
 * wl= "a" <-> "xlabel" <-> "voltage" <-> "b"
 * sz_keyword = "xlabel"
 * On return,
 * wl= "a" <-> "b"
 * return value = "voltage"
*/
static char *getword(wordlist *wl, const char *sz_keyword)
{
    wordlist *kw = wl_find(sz_keyword, wl->wl_next);

    if (kw == (wordlist *) NULL) { /* not found */
        return (char *) NULL;
    }

    wordlist *value = kw->wl_next; /* value follows keyword */
    if (value == (wordlist *) NULL) { /* no value for keyword */
        fprintf(cp_err,
                "Syntax error: missing value for plot keyword \"%s\".\n",
                sz_keyword);
        return (char *) NULL;
    }

    char *sz_ret = copy(value->wl_word); /* save value */
    wl_delete_slice(kw, value->wl_next); /* remove kw and val nodes */

    return sz_ret;
} /* end of funtion getword */



/* The common routine for all plotting commands. This does hardcopy
 * and graphics plotting.
 *
 * Parameters
 * wl: plotting command
 * hcopy: File used for plotting
 * devname: "Device" for plotting, e.g. Gnuplot
 */
bool plotit(wordlist *wl, const char *hcopy, const char *devname)
{
    if (!wl) { /* no wordlist -> cannot plot */
        return FALSE;
    }

    static double *xprevgraph = NULL;
    int prevgraph = 0;

    static bool nointerp = FALSE;
    static bool kicad = FALSE;
    static bool plain = FALSE;
    static GRIDTYPE gtype = GRID_LIN;
    static PLOTTYPE ptype = PLOT_LIN;

    bool gfound = FALSE, pfound = FALSE, oneval = FALSE, contour2d = FALSE, digitop = FALSE;
    double ylims[2], xlims[2];
    struct pnode *pn, *names = NULL;
    struct dvec *d = NULL, *vecs = NULL, *lv = NULL, *lastvs = NULL;
    char *xn;
    int i, xt;
    wordlist *wwl;
    char *nxlabel = NULL, *nylabel = NULL, *ntitle = NULL;
    double tstep, tstart, tstop, ttime;

    /* Save start of vectors on entry for cleaning up junk left behind
     * by ft_getpnames_quotes() */
    struct dvec *dv_head_orig =
            plot_cur ? plot_cur->pl_dvecs : (struct dvec *) NULL;

    /* Dstring for building plot command */
    DS_CREATE(ds_cline, 200);
    int rc_ds = 0; /* return code from dstring operations */

    /* return value, error by default */
    bool rtn = FALSE;

    /* Create a copy of the input wordlist with a dummy node at the
     * beginning of the list. The dummy node is used to ensure that
     * the keyword and value nodes and the labels and title nodes
     * that are removed are not at the beginning of the list.
     * As a result, the head of the list remains unchanged while
     * the undesired nodes are being removed. */
    wl = wl_cons(NULL, wl_copy(wl));

    /* First get the command line, without the limits.
       Wii be used for zoomed windows. Besides returning the values,
       which are not wanted here, the getlims calls remove the
       nodes for the keyword and its value from wwl. */
    wwl = wl_copy(wl);
    txfree(getlims(wwl, "xl", 2));
    txfree(getlims(wwl, "xlimit", 2));
    txfree(getlims(wwl, "yl", 2));
    txfree(getlims(wwl, "ylimit", 2));

    /* Save title, xlabel and ylabel for use later and remove the
     * corresponding nodes from the wordlist */
    nxlabel = getword(wwl, "xlabel");
    nylabel = getword(wwl, "ylabel");
    ntitle = getword(wwl, "title");
    /* remove sgraphid */
    txfree(getlims(wwl, "sgraphid", 1));

    /* Build the plot command. This construction had been done with wordlists
     * and reversing, and flattening, but it is clearer as well as much more
     * efficient to use a dstring. */
    char *flatstr = wl_flatten(wwl->wl_next);
    rc_ds |= ds_cat_printf(&ds_cline, "plot %s", flatstr);
    wl_free(wwl);
    tfree(flatstr);

    /* Add title, xlabel or ylabel, if available, with quotes ''. */
    if (nxlabel) {
        rc_ds |= ds_cat_printf(&ds_cline, " xlabel '%s'", nxlabel);
        tfree(nxlabel);
    }
    if (nylabel) {
        rc_ds |= ds_cat_printf(&ds_cline, " ylabel '%s'", nylabel);
        tfree(nylabel);
    }
    if (ntitle) {
        rc_ds |= ds_cat_printf(&ds_cline, " title '%s'", ntitle);
        tfree(ntitle);
    }
    if (rc_ds != 0) {
        fprintf(cp_err, "Unable to build plot command line.\n");
        goto quit1;
    }

    /* See if contours for 2D Cider data can be plotted with gnuplot */
    contour2d = getflag(wl, "xycontour");

    /* Now extract all the parameters. */
    digitop = getflag(wl, "digitop");

    sameflag = getflag(wl, "samep");

    if (!sameflag || !xlim) {
        txfree(xlim);
        xlim = getlims(wl, "xl", 2);
        if (!xlim) {
            xlim = getlims(wl, "xlimit", 2);
        }
    }
    else {
        txfree(getlims(wl, "xl", 2));
        txfree(getlims(wl, "xlimit", 2));
    }

    if (!sameflag || !ylim) {
        txfree(ylim);
        ylim = getlims(wl, "yl", 2);
        if (!ylim) {
            ylim = getlims(wl, "ylimit", 2);
        }
    }
    else {
        txfree(getlims(wl, "yl", 2));
        txfree(getlims(wl, "ylimit", 2));
    }

    if (!sameflag || !xcompress) {
        txfree(xcompress);
        xcompress = getlims(wl, "xcompress", 1);
        if (!xcompress) {
            xcompress = getlims(wl, "xcomp", 1);
        }
    }
    else {
        txfree(getlims(wl, "xcompress", 1));
        txfree(getlims(wl, "xcomp", 1));
    }

    if (!sameflag || !xindices) {
        txfree(xindices);
        xindices = getlims(wl, "xindices", 2);
        if (!xindices) {
            xindices = getlims(wl, "xind", 2);
        }
    }
    else {
        txfree(getlims(wl, "xindices", 2));
        txfree(getlims(wl, "xind", 2));
    }

    if (!sameflag || !xdelta) {
        txfree(xdelta);
        xdelta = getlims(wl, "xdelta", 1);
        if (!xdelta) {
            xdelta = getlims(wl, "xdel", 1);
        }
    } else {
        txfree(getlims(wl, "xdelta", 1));
        txfree(getlims(wl, "xdel", 1));
    }

    if (!sameflag || !ydelta) {
        txfree(ydelta);
        ydelta = getlims(wl, "ydelta", 1);
        if (!ydelta) {
            ydelta = getlims(wl, "ydel", 1);
        }
    }
    else {
        txfree(getlims(wl, "ydelta", 1));
        txfree(getlims(wl, "ydel", 1));
    }

    if (!sameflag || !xprevgraph) {
        xprevgraph = getlims(wl, "sgraphid", 1);
        if(xprevgraph)
            prevgraph = (int)(*xprevgraph);
    } else {
        txfree(getlims(wl, "sgraphid", 1));
    }
    /* Get the grid type and the point type.  Note we can't do if-else
     * here because we want to catch all the grid types.
     */
    if (getflag(wl, "lingrid")) {
        if (gfound) {
            fprintf(cp_err,
                    "Warning: too many grid types given. "
                    "\"lingrid\" is ignored.\n");
        } else {
            gtype = GRID_LIN;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "loglog")) {
        if (gfound) {
            fprintf(cp_err,
                    "Warning: too many grid types given. "
                    "\"loglog\" is ignored.\n");
        }
        else {
            gtype = GRID_LOGLOG;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "nogrid")) {
        if (gfound) {
            fprintf(cp_err,
                    "Warning: too many grid types given. "
                    "\"nogrid\" is ignored.\n");
        }
        else {
            gtype = GRID_NONE;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "linear")) {
        if (gfound) {
            fprintf(cp_err,
                    "Warning: too many grid types given. "
                    "\"linear\" is ignored.\n");
        }
        else {
            gtype = GRID_LIN;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "xlog")) {
        if (gfound) {
            if (gtype == GRID_YLOG)
                gtype = GRID_LOGLOG;
            else {
                fprintf(cp_err,
                        "Warning: too many grid types given. "
                        "\"xlog\" is ignored.\n");
            }
        }
        else {
            gtype = GRID_XLOG;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "ylog")) {
        if (gfound) {
            if (gtype == GRID_XLOG) {
                gtype = GRID_LOGLOG;
            }
            else {
                fprintf(cp_err,
                        "Warning: too many grid types given. "
                        "\"xlog\" is ignored.\n");
            }
        }
        else {
            gtype = GRID_YLOG;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "polar")) {
        if (gfound) {
            fprintf(cp_err,
                    "Warning: too many grid types given. "
                    "\"polar\" is ignored.\n");
        }
        else {
            gtype = GRID_POLAR;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "smith")) {
        if (gfound) {
            fprintf(cp_err,
                    "Warning: too many grid types given. "
                    "\"smith\" is ignored.\n");
        }
        else {
            gtype = GRID_SMITH;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "smithgrid")) {
        if (gfound) {
            fprintf(cp_err,
                    "Warning: too many grid types given. "
                    "\"smithgrid\" is ignored.\n");
        }
        else {
            gtype = GRID_SMITHGRID;
            gfound = TRUE;
        }
    }

    if (!sameflag && !gfound) {
        char buf[BSIZE_SP];
        if (cp_getvar("gridstyle", CP_STRING, buf, sizeof(buf))) {
            if (eq(buf, "lingrid")) {
                gtype = GRID_LIN;
            }
            else if (eq(buf, "loglog")) {
                gtype = GRID_LOGLOG;
            }
            else if (eq(buf, "xlog")) {
                gtype = GRID_XLOG;
            }
            else if (eq(buf, "ylog")) {
                gtype = GRID_YLOG;
            }
            else if (eq(buf, "smith")) {
                gtype = GRID_SMITH;
            }
            else if (eq(buf, "smithgrid")) {
                gtype = GRID_SMITHGRID;
            }
            else if (eq(buf, "polar")) {
                gtype = GRID_POLAR;
            }
            else if (eq(buf, "nogrid")) {
                gtype = GRID_NONE;
            }
            else {
                fprintf(cp_err,
                        "Warning: unknown grid type \"%s\" is ignored. "
                        "The grid type will default to linear.\n", buf);
                gtype = GRID_LIN;
            }
            gfound = TRUE;
        }
        else {
            gtype = GRID_LIN;
        }
    }

    /* Now get the point type.  */
    if (getflag(wl, "linplot")) {
        if (pfound) {
            fprintf(cp_err,
                    "Warning: too many plot types given. "
                    "\"linplot\" is ignored.\n");
        }
        else {
            ptype = PLOT_LIN;
            pfound = TRUE;
        }
    }
    if (getflag(wl, "retraceplot")) {
        if (pfound) {
            fprintf(cp_err,
                    "Warning: too many plot types given. "
                    "\"retraceplot\" is ignored.\n");
        }
        else {
            ptype = PLOT_RETLIN;
            pfound = TRUE;
        }
    }
    if (getflag(wl, "combplot")) {
        if (pfound) {
            fprintf(cp_err,
                    "Warning: too many plot types given. "
                    "\"combplot\" is ignored.\n");
        }
        else {
            ptype = PLOT_COMB;
            pfound = TRUE;
        }
    }
    if (getflag(wl, "pointplot")) {
        if (pfound) {
            fprintf(cp_err,
                    "Warning: too many plot types given. "
                    "\"pointplot\" is ignored.\n");
        }
        else {
            ptype = PLOT_POINT;
            pfound = TRUE;
        }
    }

    if (!sameflag && !pfound) {
        char buf[BSIZE_SP];
        if (cp_getvar("plotstyle", CP_STRING, buf, sizeof(buf))) {
            if (eq(buf, "linplot")) {
                ptype = PLOT_LIN;
            }
            else if (eq(buf, "retraceplot")) {
                ptype = PLOT_RETLIN;
            }
            else if (eq(buf, "combplot")) {
                ptype = PLOT_COMB;
            }
            else if (eq(buf, "pointplot")) {
                ptype = PLOT_POINT;
            }
            else {
                fprintf(cp_err,
                        "Warning: strange plot type \"%s\" is ignored. "
                        "The plot type will default to linear.\n", buf);
                ptype = PLOT_LIN;
            }
            pfound = TRUE;
        }
        else {
            ptype = PLOT_LIN;
        }
    }

    if (!sameflag || !xlabel) {
        xlabel = getword(wl, "xlabel");
    }
    else {
        txfree(getword(wl, "xlabel"));
    }

    if (!sameflag || !ylabel) {
        ylabel = getword(wl, "ylabel");
    }
    else {
        txfree(getword(wl, "ylabel"));
    }

    if (!sameflag || !title) {
        title = getword(wl, "title");
    }
    else {
        txfree(getword(wl, "title"));
    }

    if (!sameflag) {
        nointerp = getflag(wl, "nointerp");
    }
    else if (getflag(wl, "nointerp")) {
        nointerp = TRUE;
    }

    if (!sameflag) {
        kicad = getflag(wl, "kicad");
    }
    else if (getflag(wl, "kicad")) {
        kicad = TRUE;
    }

    if (!sameflag) {
        plain = getflag(wl, "plainplot");
    }
    else if (getflag(wl, "plainplot")) {
        plain = TRUE;
    }

    plain = plain | cp_getvar("plainplot", CP_BOOL, NULL, 0);

    if (!wl->wl_next) {
        fprintf(cp_err, "Error: no vectors given\n");
        goto quit1;
    }

    /* if plain is set, we skip all function parsing and just plot the
       vectors by name. vc1 vs vc2 is also not supported.
       Thus we may plot vecs with node names containing + - / etc.
       Note: Evaluating the wordlist starting at wl->wl_next since the first
       node is a dummy node.*/
    if(plain) {
        wordlist* wli;
        for (wli = wl->wl_next; wli; wli = wli->wl_next) {
            d = vec_get(wli->wl_word);
            if (!d) {
                fprintf(stderr, "Error during 'plot': vector %s not found\n", wli->wl_word);
                goto quit;
            }
            if (vecs)
                lv->v_link2 = d;
            else
                vecs = d;
            for (lv = d; lv->v_link2; lv = lv->v_link2)
                ;
        }
    }
    else {
        /* kicad will generate vector names containing '/'. If compatibilty flag
           'ki' is set in .spiceinit or plot line flag 'kicad' is set,
           we will place " around this vector name. Division in the plot command
           will then work only if spaces are around ' / '.*/
        if (kicad || newcompat.ki) {
            wordlist* wlk;
            for (wlk = wl->wl_next; wlk; wlk = wlk->wl_next) {
                char* wlkword = strchr(wlk->wl_word, '/');
                if (wlkword) {
                    /* already " around token */
                    if (*(wlk->wl_word) == '"')
                        continue;
                    /* just '/' */
                    if (*(wlkword + 1) == '\0')
                        continue;
                    else {
                        char* newword = tprintf("\"%s\"", wlk->wl_word);
                        tfree(wlk->wl_word);
                        wlk->wl_word = newword;
                    }
                }
            }
        }

        /* Now parse the vectors.  We have a list of the form
         * "a b vs c d e vs f g h".  Since it's a bit of a hassle for
         * us to parse the vector boundaries here, we do this -- call
         * ft_getpnames_quotes() without the check flag, and then look for 0-length
         * vectors with the name "vs"...  This is a sort of a gross hack,
         * since we have to check for 0-length vectors ourselves after
         * evaulating the pnodes...
         *
         * Note: Evaluating the wordlist starting at wl->wl_next since the first
         * node is a dummy node.
         */

        names = ft_getpnames_quotes(wl->wl_next, FALSE);
        if (names == (struct pnode*)NULL) {
            goto quit1;
        }

        /* Now evaluate the names. */
        for (pn = names, lv = NULL; pn; pn = pn->pn_next) {
            struct dvec* pn_value = pn->pn_value;

            /* Test for a vs b construct */
            if (pn_value && (pn_value->v_length == 0) &&
                eq(pn_value->v_name, "vs")) {
                struct dvec* dv;

                if (!lv) { /* e.g. "plot vs b" */
                    fprintf(cp_err, "Error: misplaced vs arg\n");
                    goto quit;
                }
                if ((pn = pn->pn_next) == NULL) { /* "plot a vs" */
                    fprintf(cp_err, "Error: missing vs arg\n");
                    goto quit;
                }

                dv = ft_evaluate(pn);
                if (!dv) {
                    goto quit;
                }

                if (lastvs) {
                    lv = lastvs->v_link2;
                }
                else {
                    lv = vecs;
                }

                while (lv) {
                    lv->v_scale = dv;
                    lastvs = lv;
                    lv = lv->v_link2;
                }
            }
            else { /* An explicit scale vector is not given ("plot a") */
                struct dvec* const dv = ft_evaluate(pn);
                if (!dv) {
                    goto quit;
                }

                if (!d) {
                    vecs = dv;
                }
                else {
                    d->v_link2 = dv;
                }

                for (d = dv; d->v_link2; d = d->v_link2) {
                    ;
                }

                lv = dv;
            }
        } /* end of loop evaluating the names */
        d->v_link2 = NULL; /* terminate list */
    } /* if not plain */

    /* Now check for 0-length vectors. */
    for (d = vecs; d; d = d->v_link2) {
        if (!d->v_length) {
            fprintf(cp_err, "Error(plotit.c--plotit): %s: zero length vector\n",
                    d->v_name);
            goto quit;
        }
    }

    /* Add n * spacing (e.g. 1.5) to digital event node based vectors */
    if (digitop) {
        double spacing = 1.5;
        double nn = 0.;
        int ii = 0, jj = 0;

        for (d = vecs; d; d = d->v_link2) {
            if ((d->v_flags & VF_EVENT_NODE) &&
                !(d->v_flags & VF_PERMANENT) &&
                d->v_scale && (d->v_scale->v_flags & VF_EVENT_NODE) &&
                (d->v_scale->v_type == SV_TIME) && (d->v_type == SV_VOLTAGE) &&
                (d->v_length > 1)) {
                for (ii = 0; ii < d->v_length; ii++) {
                    d->v_realdata[ii] += nn;
                }
                nn += spacing;
                jj++ ;
            }
        }
        if (!ydelta)
            ydelta = TMALLOC(double, 1);
        *ydelta = spacing;
        /* new plot */
        if (!ylim) {
            ylim = TMALLOC(double, 2);
            ylim[0] = 0;
            /* make ylim[1] a multiple of 2*1.5 */
            if (jj % 2 == 0)
                ylim[1] = nn;
            else
                ylim[1] = nn + 1.5;
        }
        /* re-scaled plot */
        else {
            /* just figure out that ylim[0] < ylim[1] */
            if (ylim[0] > ylim[1]) {
                double interm = ylim[1];
                ylim[1] = ylim[0];
                ylim[0] = interm;
            }
            if (ylim[0] < 1.1)
            /* catch the bottom line */
                ylim[0] = 0;
            else
            /* If we redraw, set again multiples of 'spacing' */
                ylim[0] = ((int)(ylim[0] / spacing) + 1) * spacing;
            ylim[1] = ((int)(ylim[1] / spacing) + 1) * spacing;
        }
        /* suppress y labeling */
        if (gtype == GRID_NONE)
            gtype = GRID_DIGITAL_NONE;
        else
            gtype = GRID_DIGITAL;
    }

    /* If there are higher dimensional vectors, transform them into a
     * family of vectors.
     */
    for (d = vecs, lv = NULL; d; d = d->v_link2) {
        /* Link the family of vectors that is created through the v_link2 link.
         * Note that vec_mkfamily links all of the vector that are created
         * through v_link2 also, so the family of vectors can be added to
         * the plot list by stepping to the end and linking to the next
         * vector */
        if (d->v_numdims > 1) { /* multi-dim vector */
            if (lv) {
                lv->v_link2 = vec_mkfamily(d);
            }
            else {
                vecs = lv = vec_mkfamily(d);
            }

            /* Step to end of the family of vectors */
            while (lv->v_link2) {
                lv = lv->v_link2;
            }

            /* And link last vector in family to next vector to plot */
            lv->v_link2 = d->v_link2;
            d = lv;
        }
        else {
            /* Ordinary 1-dim vector, so set prev vector to this one in
             * preparation for next increment in loop */
            lv = d;
        }
    } /* end of loop over vectors being plotted */

    /* Now fill in the scales for vectors who aren't already fixed up. */
    for (d = vecs; d; d = d->v_link2) {
        if (!d->v_scale) {
            if (d->v_plot->pl_scale) {
                d->v_scale = d->v_plot->pl_scale;
            }
            else {
                d->v_scale = d;
            }
        }
    }

    /* The following line displays the unit at the time of
       temp-sweep, res-sweep, and i-sweep. This may not be a so good solution. by H.T */
    if (strcmp(vecs->v_scale->v_name, "temp-sweep") == 0) {
        vecs->v_scale->v_type = SV_TEMP;
    }
    if (strcmp(vecs->v_scale->v_name, "res-sweep") == 0) {
        vecs->v_scale->v_type = SV_RES;
    }
    if (strcmp(vecs->v_scale->v_name, "i-sweep") == 0) {
        vecs->v_scale->v_type = SV_CURRENT;
    }

    /* See if the log flag is set anywhere... */
    if (!gfound) {
        for (d = vecs; d; d = d->v_link2) {
            if (d->v_scale && (d->v_scale->v_gridtype == GRID_XLOG)) {
                gtype = GRID_XLOG;
            }
        }
        for (d = vecs; d; d = d->v_link2) {
            if (d->v_gridtype == GRID_YLOG) {
                if ((gtype == GRID_XLOG) || (gtype == GRID_LOGLOG)) {
                    gtype = GRID_LOGLOG;
                }
                else {
                    gtype = GRID_YLOG;
                }
            }
        }
        for (d = vecs; d; d = d->v_link2) {
            if (d->v_gridtype == GRID_SMITH ||
                    d->v_gridtype == GRID_SMITHGRID ||
                    d->v_gridtype == GRID_POLAR) {
                gtype = d->v_gridtype;
                break;
            }
        }
    }

    /* See if there are any default plot types...  Here, like above, we
     * don't do entirely the best thing when there is a mixed set of
     * default plot types...
     */
    if (!sameflag && !pfound) {
        ptype = PLOT_LIN;
        for (d = vecs; d; d = d->v_link2) {
            if (d->v_plottype != PLOT_LIN) {
                ptype = d->v_plottype;
                break;
            }
        }
    }

    /* Check and see if this is pole zero stuff. */
    if ((vecs->v_type == SV_POLE) || (vecs->v_type == SV_ZERO)) {
        oneval = TRUE;
    }

    for (d = vecs; d; d = d->v_link2) {
        if (((d->v_type == SV_POLE) || (d->v_type == SV_ZERO)) !=
               oneval ? 1 : 0) {
            fprintf(cp_err,
                    "Error: plot must be either all pole-zero "
                    "or contain no poles or zeros\n");
            goto quit;
        }
    }

    if (gtype == GRID_POLAR || gtype == GRID_SMITH ||
            gtype == GRID_SMITHGRID) {
        oneval = TRUE;
    }

    /* If a vector contains a single point, copy the point so that there are
     * as many copies as the scale vector has elements. */
    for (d = vecs; d; d = d->v_link2) {
        if (d->v_length == 1) { /* single value */
            xtend(d, d->v_scale->v_length);
        }
    }

    /* Now patch up each vector with the compression (decimation) and
     * the strchr selection. */
    if (xcompress || xindices) {
        for (d = vecs; d; d = d->v_link2) {
            compress(d, xcompress, xindices);
            d->v_scale = vec_copy(d->v_scale);
            compress(d->v_scale, xcompress, xindices);
        }
    }

    /* Transform for smith plots */
    if (gtype == GRID_SMITH) {
        struct dvec **prevvp = &vecs;

        /* Loop over vectors being plotted */
        for (d = vecs; d; d = d->v_link2) {
            if (d->v_flags & VF_PERMANENT) {
                struct dvec * const n = vec_copy(d);
                n->v_flags &= ~VF_PERMANENT;
                n->v_link2 = d->v_link2;
                d = n;
                *prevvp = d;
            }
            prevvp = &d->v_link2;

            if (isreal(d)) {
                fprintf(cp_err,
                        "Warning: plotting real data \"%s\" on a Smith grid\n",
                        d->v_name);

                const int n_elem = d->v_length;
                int j;
                for (j = 0; j < n_elem; j++) {
                    const double r = d->v_realdata[j];
                    d->v_realdata[j] = (r - 1) / (r + 1);
                }
            }
            else {
                ngcomplex_t * const v0 = d->v_compdata;
                const int n_elem = d->v_length;
                int j;
                for (j = 0; j < n_elem; j++) {
                    ngcomplex_t * const p_cur = v0 + j;
                    (void) SMITH_tfm(realpart(*p_cur), imagpart(*p_cur),
                            &realpart(*p_cur), &imagpart(*p_cur));
                } /* end of loop over elements in vector */
            } /* complex data */
        } /* end of loop over vectors being plotted */
    } /* end of case of Smith grid */

    /* Figure out the proper x-axis and y-axis limits. */
    find_axis_limits(ylim, oneval, FALSE, vecs, &vec_self, ylims);
    find_axis_limits(xlim, oneval, TRUE, vecs, &vec_scale, xlims);

    if ((xlims[0] <= 0.0) &&
            ((gtype == GRID_XLOG) || (gtype == GRID_LOGLOG))) {
        fprintf(cp_err, "Error: X values must be > 0 for log scale\n");
        goto quit;
    }
    if ((ylims[0] <= 0.0) &&
            ((gtype == GRID_YLOG) || (gtype == GRID_LOGLOG))) {
        fprintf(cp_err, "Error: Y values must be > 0 for log scale\n");
        goto quit;
    }

    /* Fix the plot limits for smith and polar grids. */
    if ((!xlim || !ylim) && (gtype == GRID_POLAR)) {
        double mx, my, rad;
        /* (0,0) must be in the center of the screen. */
        mx = (fabs(xlims[0]) > fabs(xlims[1])) ? fabs(xlims[0]) : fabs(xlims[1]);
        my = (fabs(ylims[0]) > fabs(ylims[1])) ? fabs(ylims[0]) : fabs(ylims[1]);
        /* rad = (mx > my) ? mx : my; */
        /* AM.Roldán
         * Change this reason that this was discussed, as in the case of 1 + i want to plot point
         * is outside the drawing area so I'll stay as the maximum size of the hypotenuse of
         * the complex value
         */
        rad = hypot(mx, my);
        xlims[0] = -rad;
        xlims[1] = rad;
        ylims[0] = -rad;
        ylims[1] = rad;
    }
    else if ((!xlim || !ylim) &&
            (gtype == GRID_SMITH || gtype == GRID_SMITHGRID)) {
        xlims[0] = -1.0;
        xlims[1] = 1.0;
        ylims[0] = -1.0;
        ylims[1] = 1.0;
    }

    if (xlim) {
        tfree(xlim);
    }
    if (ylim) {
        tfree(ylim);
    }

    /* We don't want to try to deal with Smith plots for asciiplot. */
    if (devname && eq(devname, "lpr")) {
        /* check if we should (can) linearize */
        if (ft_curckt && ft_curckt->ci_ckt &&
                (strcmp(ft_curckt->ci_name, plot_cur->pl_title) == 0) &&
                if_tranparams(ft_curckt, &tstart, &tstop, &tstep) &&
                ((tstop - tstart) * tstep > 0.0) &&
                ((tstop - tstart) >= tstep) &&
                plot_cur && plot_cur->pl_dvecs &&
                plot_cur->pl_scale &&
                isreal(plot_cur->pl_scale) &&
                ciprefix("tran", plot_cur->pl_typename)) {
            int newlen = (int)((tstop - tstart) / tstep + 1.5);

            double *newscale;

            struct dvec *v, *newv_scale =
                dvec_alloc(copy(vecs->v_scale->v_name),
                           vecs->v_scale->v_type,
                           vecs->v_scale->v_flags,
                           newlen, NULL);

            newv_scale->v_gridtype = vecs->v_scale->v_gridtype;

            newscale = newv_scale->v_realdata;
            for (i = 0, ttime = tstart; i < newlen; i++, ttime += tstep) {
                newscale[i] = ttime;
            }

            for (v = vecs; v; v = v->v_link2) {
                double *newdata = TMALLOC(double, newlen);

                if (!ft_interpolate(v->v_realdata, newdata,
                        v->v_scale->v_realdata, v->v_scale->v_length,
                        newscale, newlen, 1)) {
                    fprintf(cp_err, "Error: can't interpolate %s\n", v->v_name);
                    goto quit;
                }

                dvec_realloc(v, newlen, newdata);

                /* Why go to all this trouble if agraf ignores it? */
                nointerp = TRUE;
            }

            vecs->v_scale = newv_scale;
        }

        ft_agraf(xlims, ylims,
                 vecs->v_scale, vecs->v_plot, vecs,
                 xdelta ? *xdelta : 0.0,
                 ydelta ? *ydelta : 0.0,
                 ((gtype == GRID_XLOG) || (gtype == GRID_LOGLOG)),
                 ((gtype == GRID_YLOG) || (gtype == GRID_LOGLOG)),
                 nointerp);
        rtn = TRUE;
        goto quit;
    }

    /* See if there is one common v_type we can give for the y scale... */
    for (d = vecs->v_link2; d; d = d->v_link2) {
        if (d->v_type != vecs->v_type) {
            break;
        }
    }

    const int y_type = (int) (d ? SV_NOTYPE : vecs->v_type);

    if (devname && eq(devname, "gnuplot")) {
        /* Interface to Gnuplot Plot Program */
        ft_gnuplot(xlims, ylims,
                   xdelta ? *xdelta : 0.0,
                   ydelta ? *ydelta : 0.0,
                   hcopy,
                   title ? title : vecs->v_plot->pl_title,
                   xlabel ? xlabel : ft_typabbrev(vecs->v_scale->v_type),
                   ylabel ? ylabel : ft_typabbrev(y_type),
                   gtype, ptype, vecs, contour2d);
        rtn = TRUE;
        goto quit;
    }

    if (devname && eq(devname, "writesimple")) {
        /* Interface to simple write output */
        ft_writesimple(xlims, ylims, hcopy,
                       title ? title : vecs->v_plot->pl_title,
                       xlabel ? xlabel : ft_typabbrev(vecs->v_scale->v_type),
                       ylabel ? ylabel : ft_typabbrev(y_type),
                       gtype, ptype, vecs);
        rtn = TRUE;
        goto quit;
    }

#ifdef TCL_MODULE
    if (devname && eq(devname, "blt")) {
        /* Just send the pairs to Tcl/Tk */
        for (d = vecs; d; d = d->v_link2)
            blt_plot(d, oneval ? NULL : d->v_scale, (d == vecs) ? 1 : 0);
        rtn = TRUE;
        goto quit;
    }
#endif

    /* Find the number of vectors being plotted */
    for (d = vecs, i = 0; d; d = d->v_link2) {
        i++;
    }

    /* Figure out the X name and the X type.  This is sort of bad... */
    xn = vecs->v_scale->v_name;
    xt = vecs->v_scale->v_type;


    if (!gr_init(xlims, ylims, (oneval ? NULL : xn),
            title ? title : vecs->v_plot->pl_title,
            hcopy, i,
            xdelta ? *xdelta : 0.0,
            ydelta ? *ydelta : 0.0,
            gtype, ptype, xlabel, ylabel, xt, y_type,
            plot_cur->pl_typename, ds_get_buf(&ds_cline), prevgraph)) {
        goto quit;
    }

    /* Now plot all the graphs. */
    for (d = vecs; d; d = d->v_link2) {
        ft_graf(d, oneval ? NULL : d->v_scale, FALSE);
    }

    gr_clean();
    rtn = TRUE; /* Indicate success */

quit:
    ds_free(&ds_cline); /* free dstring resources, if any */
    free_pnode(names);
    FREE(title);
    FREE(xlabel);
    FREE(ylabel);

quit1:
    /* Free any vectors left behing while parsing the plot arguments. These
     * are vectors created by ft_evaluate() */
    if (plot_cur != (struct plot *) NULL) {
        struct dvec *dv = plot_cur->pl_dvecs;
        while(dv != dv_head_orig) {
            struct dvec *dv_next = dv->v_next;
            vec_free(dv);
            dv = dv_next;
        }
    }

    wl_free(wl);
    return rtn;
} /* end of function plotit */



/* Return itself */
static struct dvec *vec_self(struct dvec *v)
{
    return v;
}



/* Return scale vector */
static struct dvec *vec_scale(struct dvec *v)
{
    return v->v_scale;
}



/* This function finds the range limits for an  x-axis or y-axis.
 *
 * Parameters
 * lim: Existing limits
 * oneval: Flag that there is no scale vector
 * f_real: Flag that the real component of a complex value should be used
 *      when finding the range if true and the imaginary part if false
 * vecs: Vectors being used to determine the range. It is related to athe
 *      oneval flag in that it determines
 * p_get_axis_dvec: Address of function used to get range information
 *      from a vector. It should be either the address of vec_self to use
 *      the vector itself (for y range with scale value) or the address of
 *      vec_scale for its scale vector (for x range of scale).
 * lims: Address of an array of 2 double values to receive the limits.
 **/
static void find_axis_limits(double *lim, bool oneval, bool f_real,
        struct dvec *vecs,
        struct dvec *(*p_get_axis_dvec)(struct dvec *dvec),
        double *lims)
{
    if (lim != (double *) NULL) {
        lims[0] = lim[0];
        lims[1] = lim[1];
    }
    else if (oneval) {
        struct dvec *d;
        lims[0] = HUGE;
        lims[1] = -lims[0];
        for (d = vecs; d; d = d->v_link2) {
            /* dd = ft_minmax(d, FALSE); */
            /* With this we seek the maximum and minimum of imaginary part
             * that will go to Y axis
             */
            const double * const dd = ft_minmax(d, f_real);

            if (lims[0] > dd[0]) {
                lims[0] = dd[0];
            }
            if (lims[1] < dd[1]) {
                lims[1] = dd[1];
            }
        }
    }
    else { /* have scale vector */
        struct dvec *d;
        lims[0] = HUGE;
        lims[1] = -lims[0];
        for (d = vecs; d; d = d->v_link2) {
            const double * const dd = ft_minmax((*p_get_axis_dvec)(d), TRUE);
            if (lims[0] > dd[0]) {
                lims[0] = dd[0];
            }
            if (lims[1] < dd[1]) {
                lims[1] = dd[1];
            }
        }
        for (d = vecs; d; d = d->v_link2) {
            struct dvec *d2 = (*p_get_axis_dvec)(d);
            short v_flags = d2->v_flags;
            if (v_flags & VF_MINGIVEN) {
                double v_minsignal = d2->v_minsignal;
                if (lims[0] < v_minsignal) {
                    lims[0] = v_minsignal;
                }
            }
            if (v_flags & VF_MAXGIVEN) {
                double v_maxsignal = d2->v_maxsignal;
                if (lims[1] > v_maxsignal) {
                    lims[1] = v_maxsignal;
                }
            }
        } /* end of loop over vectors being plotted */
    } /* end of case of vector with scale vector */

    /* Do some coercion of the limits to make them reasonable. */
    if ((lims[0] == 0.0) && (lims[1] == 0.0)) {
        lims[0] = -1.0;
        lims[1] = 1.0;
    }
    if (lims[0] > lims[1]) {
        SWAP(double, lims[0], lims[1]);
    }
    if (AlmostEqualUlps(lims[0], lims[1], 10)) {
        lims[0] *= (lims[0] > 0) ? 0.9 : 1.1;
        lims[1] *= (lims[1] > 0) ? 1.1 : 0.9;
    }

} /* end of function find_axis_limits */



