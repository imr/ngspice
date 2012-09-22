/* The 'compose' command.  This is a more powerful and convenient form
 * of the 'let' command.  */
#include "ngspice/ngspice.h"
#include "ngspice/complex.h"
#include "ngspice/dvec.h"
#include "ngspice/bool.h"
#include "ngspice/sim.h"
#include "ngspice/pnode.h"
#include "ngspice/fteext.h"
#include "ngspice/cpextern.h"

#include "quote.h"
#include "com_compose.h"
#include "completion.h"


/* Copy the data from a vector into a buffer with larger dimensions. */
static void
dimxpand(struct dvec *v, int *newdims, double *data)
{
    ngcomplex_t *cdata = (ngcomplex_t *) data;
    bool realflag = isreal(v);
    int i, j, o, n, t, u;
    int ncount[MAXDIMS], ocount[MAXDIMS];

    for (i = 0; i < MAXDIMS; i++)
        ncount[i] = ocount[i] = 0;

    for (;;) {
        for (o = n = i = 0; i < v->v_numdims; i++) {
            for (j = i, t = u = 1; j < v->v_numdims; j++) {
                t *= v->v_dims[j];
                u *= newdims[j];
            }
            o += ocount[i] * t;
            n += ncount[i] * u;
        }

        if (realflag) {
            data[n] = v->v_realdata[o];
        } else {
            realpart(cdata[n]) = realpart(v->v_compdata[o]);
            imagpart(cdata[n]) = imagpart(v->v_compdata[o]);
        }

        /* Now find the nextstrchr element... */
        for (i = v->v_numdims - 1; i >= 0; i--)
            if ((ocount[i] < v->v_dims[i] - 1) && (ncount[i] < newdims[i] - 1)) {
                ocount[i]++;
                ncount[i]++;
                break;
            } else {
                ocount[i] = ncount[i] = 0;
            }

        if (i < 0)
            break;
    }
}


/* The general syntax is 'compose name parm = val ...'
 * The possible parms are:
 *  start       The value at which the vector should start.
 *  stop        The value at which the vector should end.
 *  step        The difference between sucessive elements.
 *  lin         The number of points, linearly spaced.
 *  log         The number of points, logarithmically spaced.
 *  dec         The number of points per decade, logarithmically spaced.
 *  center      Where to center the range of points.
 *  span        The size of the range of points.
 *  unif        ??
 *  gauss       The number of points in the gaussian distribution.
 *  mean        The mean value for the gaussian dist.
 *  sd          The standard deviation for the gauss. dist.
 *  random      The number of randomly selected points.
 *  pool        The name of a vector (must be already defined) to get
 *              random values -- default is 'unitvec(npoints)'
 *
 * The case 'compose name values val val ...' takes the values and creates a
 * new vector -- the vals may be arbitrary expressions.
 *
 * NOTE: most of this doesn't work -- there will be plenty of unused variable
 * lint messages...
 */

void
com_compose(wordlist *wl)
{
    double start = 0.0;
    double stop = 0.0;
    double step = 0.0;
    double lin = 0.0;
    double center;
    double span;
    double mean, sd;
    bool startgiven = FALSE, stopgiven = FALSE, stepgiven = FALSE;
    bool lingiven = FALSE;
    bool loggiven = FALSE, decgiven = FALSE, gaussgiven = FALSE;
    bool randmgiven = FALSE;
    bool spangiven = FALSE;
    bool centergiven = FALSE;
    bool meangiven = FALSE;
    bool poolgiven = FALSE;
    bool sdgiven = FALSE;
    int  log, dec, gauss, randm;
    char *pool;
    int i;

    char *resname, *s, *var, *val;
    double *td, tt;
    double *data = NULL;
    ngcomplex_t *cdata = NULL;
    int length = 0;
    int dim, type = SV_NOTYPE, blocksize;
    bool realflag = TRUE;
    int dims[MAXDIMS];
    struct dvec *result, *vecs = NULL, *v, *lv = NULL;
    struct pnode *pn, *first_pn = NULL;
    bool reverse = FALSE;

    resname = cp_unquote(wl->wl_word);
    vec_remove(resname);
    wl = wl->wl_next;

    if (eq(wl->wl_word, "values")) {
        /* Build up the vector from the rest of the line... */
        wl = wl->wl_next;
        if ((pn = ft_getpnames(wl, TRUE)) == NULL)
            return;
        first_pn = pn;
        while (pn) {
            if ((v = ft_evaluate(pn)) == NULL)
                return;

            if (!vecs)
                vecs = lv = v;
            else
                lv->v_link2 = v;

            for (lv = v; lv->v_link2; lv = lv->v_link2)
                ;
            pn = pn->pn_next;
        }

        /* Now make sure these are all of the same dimensionality.  We
         * can coerce the sizes...
         */
        dim = vecs->v_numdims;
        if (dim < 2)
            dim = (vecs->v_length > 1) ? 1 : 0;

        if (dim == MAXDIMS) {
            fprintf(cp_err, "Error: max dimensionality is %d\n",
                    MAXDIMS);
            return;
        }

        for (v = vecs; v; v = v->v_link2)
            if (v->v_numdims < 2)
                v->v_dims[0] = v->v_length;

        for (v = vecs->v_link2, length = 1; v; v = v->v_link2) {
            i = v->v_numdims;
            if (i < 2)
                i = (v->v_length > 1) ? 1 : 0;
            if (i != dim) {
                fprintf(cp_err,
                        "Error: all vectors must be of the same dimensionality\n");
                return;
            }
            length++;
            if (iscomplex(v))
                realflag = FALSE;
        }

        for (i = 0; i < dim; i++) {
            dims[i] = vecs->v_dims[i];
            for (v = vecs->v_link2; v; v = v->v_link2)
                if (v->v_dims[i] > dims[i])
                    dims[i] = v->v_dims[i];
        }
        dim++;
        dims[dim - 1] = length;
        for (i = 0, blocksize = 1; i < dim - 1; i++)
            blocksize *= dims[i];

        if (realflag)
            data = TMALLOC(double, length * blocksize);
        else
            cdata = TMALLOC(ngcomplex_t, length * blocksize);

        /* Now copy all the data over... If the sizes are too small
         * then the extra elements are left as 0.
         */
        for (v = vecs, i = 0; v; v = v->v_link2) {
            if (dim == 1) {
                if (realflag && isreal(v)) {
                    data[i] = v->v_realdata[0];
                } else if (isreal(v)) {
                    realpart(cdata[i]) = realpart(v->v_compdata[0]);
                    imagpart(cdata[i]) = 0.0;
                } else {
                    realpart(cdata[i]) = realpart(v->v_compdata[0]);
                    imagpart(cdata[i]) = imagpart(v->v_compdata[0]);
                }
                i++;
                continue;
            }
            dimxpand(v, dims, (realflag ? (data + i * blocksize) :
                               (double *) (cdata + i * blocksize)));
        }

        length *= blocksize;
    } else {
        /* Parse the line... */
        while (wl) {
            if ((s = strchr(wl->wl_word, '=')) != NULL && s[1]) {
                /* This is var=val. */
                *s = '\0';
                var = wl->wl_word;
                val = s + 1;
                wl = wl->wl_next;
            } else if (strchr(wl->wl_word, '=')) {
                /* This is var= val. */
                *s = '\0';
                var = wl->wl_word;
                wl = wl->wl_next;
                if (wl) {
                    val = wl->wl_word;
                    wl = wl->wl_next;
                } else {
                    fprintf(cp_err, "Error: bad syntax\n");
                    return;
                }
            } else {
                /* This is var =val or var = val. */
                var = wl->wl_word;
                wl = wl->wl_next;
                if (wl) {
                    val = wl->wl_word;
                    if (*val != '=') {
                        fprintf(cp_err,
                                "Error: bad syntax\n");
                        return;
                    }
                    val++;
                    if (!*val) {
                        wl = wl->wl_next;
                        if (wl) {
                            val = wl->wl_word;
                        } else {
                            fprintf(cp_err,
                                    "Error: bad syntax\n");
                            return;
                        }
                    }
                    wl = wl->wl_next;
                } else {
                    fprintf(cp_err, "Error: bad syntax\n");
                    return;
                }
            }
            if (cieq(var, "start")) {
                startgiven = TRUE;
                if ((td = ft_numparse(&val, FALSE)) == NULL) {
                    fprintf(cp_err,
                            "Error: bad parm %s = %s\n", var, val);
                    return;
                }
                start = *td;
            } else if (cieq(var, "stop")) {
                stopgiven = TRUE;
                if ((td = ft_numparse(&val, FALSE)) == NULL) {
                    fprintf(cp_err,
                            "Error: bad parm %s = %s\n", var, val);
                    return;
                }
                stop = *td;
            } else if (cieq(var, "step")) {
                stepgiven = TRUE;
                if ((td = ft_numparse(&val, FALSE)) == NULL) {
                    fprintf(cp_err,
                            "Error: bad parm %s = %s\n", var, val);
                    return;
                }
                step = *td;
            } else if (cieq(var, "center")) {
                centergiven = TRUE;
                if ((td = ft_numparse(&val, FALSE)) == NULL) {
                    fprintf(cp_err,
                            "Error: bad parm %s = %s\n", var, val);
                    return;
                }
                center = *td;
            } else if (cieq(var, "span")) {
                spangiven = TRUE;
                if ((td = ft_numparse(&val, FALSE)) == NULL) {
                    fprintf(cp_err,
                            "Error: bad parm %s = %s\n", var, val);
                    return;
                }
                span = *td;
            } else if (cieq(var, "mean")) {
                meangiven = TRUE;
                if ((td = ft_numparse(&val, FALSE)) == NULL) {
                    fprintf(cp_err,
                            "Error: bad parm %s = %s\n", var, val);
                    return;
                }
                mean = *td;
            } else if (cieq(var, "sd")) {
                sdgiven = TRUE;
                if ((td = ft_numparse(&val, FALSE)) == NULL) {
                    fprintf(cp_err,
                            "Error: bad parm %s = %s\n", var, val);
                    return;
                }
                sd = *td;
            } else if (cieq(var, "lin")) {
                lingiven = TRUE;
                if ((td = ft_numparse(&val, FALSE)) == NULL) {
                    fprintf(cp_err,
                            "Error: bad parm %s = %s\n", var, val);
                    return;
                }
                lin = *td;
            } else if (cieq(var, "log")) {
                loggiven = TRUE;
                if ((td = ft_numparse(&val, FALSE)) == NULL) {
                    fprintf(cp_err,
                            "Error: bad parm %s = %s\n", var, val);
                    return;
                }
                log = (int)(*td);
            } else if (cieq(var, "dec")) {
                decgiven = TRUE;
                if ((td = ft_numparse(&val, FALSE)) == NULL) {
                    fprintf(cp_err,
                            "Error: bad parm %s = %s\n", var, val);
                    return;
                }
                dec = (int)(*td);
            } else if (cieq(var, "gauss")) {
                gaussgiven = TRUE;
                if ((td = ft_numparse(&val, FALSE)) == NULL) {
                    fprintf(cp_err,
                            "Error: bad parm %s = %s\n", var, val);
                    return;
                }
                gauss = (int)(*td);
            } else if (cieq(var, "random")) {
                randmgiven = TRUE;
                if ((td = ft_numparse(&val, FALSE)) == NULL) {
                    fprintf(cp_err,
                            "Error: bad parm %s = %s\n", var, val);
                    return;
                }
                randm = (int)(*td);
            } else if (cieq(var, "pool")) {
                poolgiven = TRUE;
                pool = val;
            }
        }

#ifdef LINT
        /* XXX Now, doesn't this look just a little suspicious */
        if (centergiven || spangiven || meangiven || sdgiven || poolgiven)
            j = k = l = m = q = inds = center + span + mean + sd +
                log + dec + gauss + randm + pool;
#endif
        /* Now see what we have... start and stop are pretty much
         * compatible with everything...
         */
        if (stepgiven && (step == 0.0)) {
            fprintf(cp_err, "Error: step cannot = 0.0\n");
            return;
        }

        if (startgiven && stopgiven && (start > stop)) {
            tt = start;
            start = stop;
            stop = tt;
            reverse = TRUE;
        }

        if (lingiven + loggiven + decgiven + randmgiven + gaussgiven > 1) {
            fprintf(cp_err,
                    "Error: can have at most one of (lin, log, dec, random, gauss)\n");
            return;
        } else if (lingiven + loggiven + decgiven + randmgiven + gaussgiven == 0) {
            /* Hmm, if we have a start, stop, and step we're ok. */
            if (startgiven && stopgiven && stepgiven) {
                lingiven = TRUE;
                lin = (stop - start) / step + 1;
                stepgiven = FALSE;  /* Problems below... */
            } else {
                fprintf(cp_err,
                        "Error: either one of (lin, log, dec, random, gauss) must be given, or all\n");
                fprintf(cp_err,
                        "\tof (start, stop, and step) must be given.\n");
                return;
            }
        }

        if (lingiven) {
            /* Create a linear sweep... */
            data = TMALLOC(double, (int) lin);
            if (stepgiven && startgiven && stopgiven) {
                if (step != (stop - start) / lin * (reverse ? -1 : 1)) {
                    fprintf(cp_err,
                            "Warning: bad step -- should be %g\n",
                            (stop - start) / lin * (reverse ? -1 : 1));
                    stepgiven = FALSE;
                }
            }
            if (!startgiven) {
                if (stopgiven && stepgiven)
                    start = stop - step * lin;
                else if (stopgiven)
                    start = stop - lin;
                else
                    start = 0;
                startgiven = TRUE;
            }
            if (!stopgiven) {
                if (stepgiven)
                    stop = start + lin * step;
                else
                    stop = start + lin;
                stopgiven = TRUE;
            }
            if (!stepgiven) {
                step = (stop - start) / lin;
            }
            if (reverse)
                for (i = 0, tt = stop; i < lin; i++, tt -= step)
                    data[i] = tt;
            else
                for (i = 0, tt = start; i < lin; i++, tt += step)
                    data[i] = tt;
            length = (int)lin;
        } else if (loggiven || decgiven) {
            /* Create a log sweep... */
        } else if (randmgiven) {
            /* Create a set of random values... */
        } else if (gaussgiven) {
            /* Create a gaussian distribution... */
        }
    }

    result = alloc(struct dvec);
    ZERO(result, struct dvec);
    result->v_name = copy(resname);
    result->v_type = type;

    if (realflag) {
        result->v_flags = VF_REAL | VF_PERMANENT;
        result->v_realdata = data;
    } else {
        result->v_flags =  VF_COMPLEX | VF_PERMANENT;
        result->v_compdata = cdata;
    }

    result->v_length = length;
    result->v_numdims = 1;
    result->v_dims[0] = length;

    vec_new(result);
    cp_addkword(CT_VECTOR, result->v_name);
    free_pnode(first_pn);
    tfree(resname); /*DG: resname has been copied so its remains allocated: memory leak One can remove this and not copy resname*/
}
