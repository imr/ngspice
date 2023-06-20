/* The 'compose' command.  This is a more powerful and convenient form
 * of the 'let' command.  */
#include <math.h>       /* log10 */

#include "ngspice/ngspice.h"
#include "ngspice/complex.h"
#include "ngspice/dvec.h"
#include "ngspice/bool.h"
#include "ngspice/sim.h"
#include "ngspice/pnode.h"
#include "ngspice/fteext.h"
#include "ngspice/cpextern.h"
#include "ngspice/randnumb.h"
#include "ngspice/evtproto.h"
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
            cdata[n] = v->v_compdata[o];
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
 *  step        The difference between successive elements.
 *  lin         The number of points, linearly spaced.
 *  log         The number of points, logarithmically spaced.
 *  dec         The number of points per decade, logarithmically spaced.
 *  oct         The number of points per octave, logarithmically spaced.
 *  center      Where to center the range of points.
 *  span        The size of the range of points.
 *  gauss       The number of points in the gaussian distribution.
 *  mean        The mean value for the gaussian or uniform distributions.
 *  sd          The standard deviation for the gaussian distribution.
 *  unif        The number of points in the uniform distribution.
 *
 * The case 'compose name values val val ...' takes the values and creates a
 * new vector -- the vals may be arbitrary expressions. Negative vals have to
 * be put into brackets, like (-1.6).
 */

void
com_compose(wordlist *wl)
{
    double start = 0.0;
    double stop = 0.0;
    double step = 0.0;
    double lin = 0.0;
    double center = 0.0;
    double span = 0.0;
    double mean = 0.0;
    double sd = 0.0;
    bool startgiven = FALSE, stopgiven = FALSE, stepgiven = FALSE;
    bool lingiven = FALSE;
    bool loggiven = FALSE, decgiven = FALSE, octgiven = FALSE, gaussgiven = FALSE;
    bool unifgiven = FALSE;
    bool spangiven = FALSE;
    bool centergiven = FALSE;
    bool meangiven = FALSE;
    bool sdgiven = FALSE;
    int  log = 0, dec = 0, oct = 0, gauss = 0, unif = 0;
    int i;

    double tt;
    double *data = NULL;
    ngcomplex_t *cdata = NULL;
    int length = 0;
    int dim, type = SV_NOTYPE, blocksize;
    bool realflag = TRUE;
    int dims[MAXDIMS];
    struct dvec *result, *vecs = NULL, *v, *lv = NULL;
    struct pnode *pn, *names = NULL;

    char *resname = cp_unquote(wl->wl_word);

    vec_remove(resname);
    wl = wl->wl_next;

    if (eq(wl->wl_word, "values")) {
        /* Build up the vector from the rest of the line... */
        wl = wl->wl_next;

        names = ft_getpnames(wl, TRUE);
        if (!names)
            goto done;

        for (pn = names; pn; pn = pn->pn_next) {
            if ((v = ft_evaluate(pn)) == NULL)
                goto done;

            if (!vecs)
                vecs = lv = v;
            else
                lv->v_link2 = v;

            for (lv = v; lv->v_link2; lv = lv->v_link2)
                ;
        }

        /* Now make sure these are all of the same dimensionality.  We
         * can coerce the sizes...
         */
        dim = vecs->v_numdims;
        if (dim < 2)
            dim = (vecs->v_length > 1) ? 1 : 0;

        if (dim == MAXDIMS) {
            fprintf(cp_err, "Error: compose -> max dimensionality is %d\n",
                    MAXDIMS);
            goto done;
        }

        for (v = vecs; v; v = v->v_link2)
            if (v->v_numdims < 2)
                v->v_dims[0] = v->v_length;

        /* Init real flag according to type of first element */
        realflag = !iscomplex(vecs);

        for (v = vecs->v_link2, length = 1; v; v = v->v_link2) {
            i = v->v_numdims;
            if (i < 2)
                i = (v->v_length > 1) ? 1 : 0;
            if (i != dim) {
                fprintf(cp_err,
                        "Error: compose -> all vectors must be of the same dimensionality\n");
                goto done;
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
                /* 3 possibilities
                 * 1) Composed vector is real (and current value is real)
                 * 2) Composed vector is complex
                 *      a) and current value is real
                 *      b) and current value is complex
                 * It is not possible for the composed vector to be real and
                 * the current value to be complex because it would have
                 * caused the composed vector to be complex. */
                if (realflag) { /* composed vector is real */
                    data[i] = v->v_realdata[0];
                }
                else { /* complex composed vector */
                    ngcomplex_t *cdata_cur = cdata + i;
                    if (isreal(v)) {
                        /* Current value is real, so build complex value from it
                         * and no imaginary part */
                        realpart(*cdata_cur) = *v->v_realdata;
                        imagpart(*cdata_cur) = 0.0;
                    }
                    else {
                        *cdata_cur = *v->v_compdata;
                    }
                }

                i++;
                continue;
            }
            dimxpand(v, dims, (realflag ? (data + i * blocksize) :
                               (double *) (cdata + i * blocksize)));
        }

        length *= blocksize;
#ifdef XSPICE
    } else if (eq(wl->wl_word, "xspice")) {
        /* Make vectors from an event node. */

        result = EVTfindvec(resname);
        if (result == NULL) {
            fprintf(cp_err, "There is no event node %s or it has no data\n",
                    resname);
            goto done;
        }
        result->v_flags |= VF_PERMANENT;
        result->v_scale->v_flags |= VF_PERMANENT;
        vec_new(result->v_scale);
        cp_addkword(CT_VECTOR, result->v_scale->v_name);
        txfree(resname); // It was copied
        goto finished;
#endif
    } else {
        /* Parse the line... */

        while (wl) {
            char *s, *var, *val;
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
                    fprintf(cp_err, "Error: compose -> bad syntax\n");
                    goto done;
                }
            } else {
                /* This is var =val or var = val. */
                var = wl->wl_word;
                wl = wl->wl_next;
                if (wl) {
                    val = wl->wl_word;
                    if (*val != '=') {
                        fprintf(cp_err,
                                "Error: compose -> bad syntax\n");
                        goto done;
                    }
                    val++;
                    if (!*val) {
                        wl = wl->wl_next;
                        if (wl) {
                            val = wl->wl_word;
                        } else {
                            fprintf(cp_err,
                                    "Error: compose -> bad syntax\n");
                            goto done;
                        }
                    }
                    wl = wl->wl_next;
                } else {
                    fprintf(cp_err, "Error: compose -> bad syntax\n");
                    goto done;
                }
            }
            if (cieq(var, "start")) {
                startgiven = TRUE;
                if (ft_numparse(&val, FALSE, &start) < 0) {
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
            }
            else if (cieq(var, "stop")) {
                stopgiven = TRUE;
                if (ft_numparse(&val, FALSE, &stop) < 0) {
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
            }
            else if (cieq(var, "step")) {
                stepgiven = TRUE;
                if (ft_numparse(&val, FALSE, &step) < 0) {
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
            }
            else if (cieq(var, "center")) {
                centergiven = TRUE;
                if (ft_numparse(&val, FALSE, &center) < 0) {
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
            }
            else if (cieq(var, "span")) {
                spangiven = TRUE;
                if (ft_numparse(&val, FALSE, &span) < 0) {
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
            }
            else if (cieq(var, "mean")) {
                meangiven = TRUE;
                if (ft_numparse(&val, FALSE, &mean) < 0) {
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
            }
            else if (cieq(var, "sd")) {
                sdgiven = TRUE;
                if (ft_numparse(&val, FALSE, &sd) < 0) {
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
            }
            else if (cieq(var, "lin")) {
                lingiven = TRUE;
                if (ft_numparse(&val, FALSE, &lin) < 0) {
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
            }
            else if (cieq(var, "log")) {
                double dbl_val;
                loggiven = TRUE;
                if (ft_numparse(&val, FALSE, &dbl_val) <= 0) {
                    /* Cannot convert value to int */
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
                log = (int) dbl_val;
            }
            else if (cieq(var, "dec")) {
                double dbl_val;
                decgiven = TRUE;
                if (ft_numparse(&val, FALSE, &dbl_val) <= 0) {
                    /* Cannot convert value to int */
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
                dec = (int) dbl_val;
            }
            else if (cieq(var, "oct")) {
                double dbl_val;
                octgiven = TRUE;
                if (ft_numparse(&val, FALSE, &dbl_val) <= 0) {
                    /* Cannot convert value to integer */
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
                oct = (int) dbl_val;
            }
            else if (cieq(var, "gauss")) {
                double dbl_val;
                gaussgiven = TRUE;
                if (ft_numparse(&val, FALSE, &dbl_val) <= 0) {
                    /* Cannot convert value to int */
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
                gauss = (int) dbl_val;
            }
            else if (cieq(var, "unif")) {
                double dbl_val;
                unifgiven = TRUE;
                if (ft_numparse(&val, FALSE, &dbl_val)<= 0) {
                    /* cannot convert to int */
                    fprintf(cp_err,
                            "Error: compose -> bad parm %s = %s\n", var, val);
                    goto done;
                }
                unif = (int) dbl_val;
            }
            else {
                fprintf(cp_err, "Error: compose -> bad parm %s\n", var);
                goto done;
            }
        }

        /* Now see what we have... start and stop are pretty much
         * compatible with everything (except gauss)...
         */
        if (centergiven && spangiven && !startgiven && !stopgiven) {
            start = center - span/2.0;
            stop  = center + span/2.0;
            startgiven = TRUE;
            stopgiven = TRUE;
        }

        if (stepgiven && (step == 0.0)) {
            fprintf(cp_err, "Error: compose -> step cannot = 0.0\n");
            goto done;
        }

        if (lingiven + loggiven + decgiven + octgiven + unifgiven + gaussgiven > 1) {
            fprintf(cp_err,
                    "Error: compose -> can have at most one of (lin, log, dec, oct, unif, gauss)\n");
            goto done;
        }
        else if (lingiven + loggiven + decgiven + octgiven + unifgiven + gaussgiven == 0) {
            /* Hmm, if we have a start, stop, and step we're ok. */
            if (startgiven && stopgiven && stepgiven) {
                lingiven = TRUE;
                /* Ensure that step has the right sign */
                if ((stop - start > 0) != (step > 0)) {
                  step = -step;
                }
                lin = (stop - start) / step + 1.;
                stepgiven = FALSE;  /* Problems below... */
            }
            else {
                fprintf(cp_err,
                        "Error: compose -> either one of (lin, log, dec, oct, unif, gauss) must be given, or all\n");
                fprintf(cp_err,
                        "\tof (start, stop, and step) must be given.\n");
                goto done;
            }
        }

        if (lingiven) {
            /* Create a linear sweep... */
            if (lin <= 0) {
                fprintf(cp_err,
                        "Error: compose -> The number of linearly spaced points, lin, must be positive.\n");
                goto done;
            }
            length = (int)lin;
            data = TMALLOC(double, length);
            if (stepgiven && startgiven && stopgiven) {
                if (step != (stop - start) / (lin - 1.0)) {
                    fprintf(cp_err,
                            "Warning: compose -> bad step -- should be %g. ",
                            (stop - start) / (lin - 1.0));
                    fprintf(cp_err,
                            "Specify only three out of start, stop, step, lin.\n");
                    stepgiven = FALSE;
                }
            }
            if (!startgiven) {
                if (stopgiven && stepgiven)
                    start = stop - step * (lin - 1.0);
                else if (stopgiven)
                    start = stop - lin + 1.0;
                else
                    start = 0;
                startgiven = TRUE;
            }
            if (!stopgiven) {
                if (stepgiven)
                    stop = start + step * (lin - 1.0);
                else
                    stop = start + lin - 1.;
                stopgiven = TRUE;
            }
            if (!stepgiven) {
                step = (stop - start) / (lin - 1.0);
            }

            for (i = 0, tt = start; i < length; i++, tt += step) {
                data[i] = tt;
        }

        }
        else if (loggiven || decgiven || octgiven) {
            /* Create a log sweep... */
            if (centergiven && spangiven) {
                if (center <= span/2.0) {
                    fprintf(cp_err,
                            "Error: compose -> center must be greater than span/2\n");
                    goto done;
                }
                if ((center <= 0) || (span <= 0)) {
                    fprintf(cp_err,
                            "Error: compose -> center and span must be greater than 0\n");
                    goto done;
                }
            }
            else if (startgiven && stopgiven) {
                if ((start <= 0) || (stop <= 0)) {
                    fprintf(cp_err,
                            "Error: compose -> start and stop must be greater than 0\n");
                    goto done;
                }
            }
            else {
                fprintf(cp_err,
                        "Error: compose -> start and stop or center and span needed in case of log, dec or oct\n");
                goto done;
            }
            if (decgiven) {
                log = (int)round(dec * log10(stop / start)) + 1;
            } else if (octgiven) {
                log = (int)round(oct * log10(stop / start) / log10(2)) + 1;
            }

            length = log;
            data = TMALLOC(double, length);

            data[0] = start;
            for (i = 0; i < length; i++)
                data[i] = start * pow(stop/start, (double)i/(log-1.0));

        }
        else if (unifgiven) {
            /* Create a set of uniform distributed values... */
            if (startgiven || stopgiven) {
                if (!startgiven || !stopgiven) {
                    fprintf(cp_err,
                            "Error: compose -> For uniform distribution (start, stop) can be only given as bundle.\n");
                    goto done;
                }
                if (meangiven || spangiven) {
                    fprintf(cp_err,
                            "Error: compose -> For uniform distribution (start, stop) can't be mixed with mean or span.\n");
                    goto done;
                }
                mean = (start + stop) / 2.0;
                span = fabs(stop - start);
                meangiven = TRUE;
                spangiven = TRUE;
            }
            if (unif <= 0) {
                fprintf(cp_err,
                        "Error: compose -> The number of uniformly distributed points, unif, must be positive.\n");
                goto done;
            }
            if (!meangiven) {
                /* Use mean default value 0.5 */
                mean = 0.5;
            }
            if (!spangiven) {
                /* Use span default value 1.0 */
                span = 1.0;
            }
            length = unif;
            data = TMALLOC(double, length);
            for (i = 0; i < length; i++)
                data[i] = mean + span * 0.5 * drand();

        }
        else if (gaussgiven) {
            /* Create a gaussian distribution... */
            if (gauss <= 0) {
                fprintf(cp_err,
                        "Error: compose -> The number of Gaussian distributed points, gauss, must be positive.\n");
                goto done;
            }
            if (!meangiven) {
                /* Use mean default value 0 */
                mean = 0;
            }
            if (!sdgiven) {
                /* Use sd default value 1.0 */
                sd = 1.0;
            }
            length = gauss;
            data = TMALLOC(double, length);
            for (i = 0; i < length; i++) {
                data[i] = mean + sd * gauss1();
            }
        }
    }

    /* Create a vector with the data that was processed */
    if (realflag) {
        result = dvec_alloc(resname,
                            type,
                            VF_REAL | VF_PERMANENT,
                            length, data);
    } else {
        result = dvec_alloc(resname,
                            type,
                            VF_COMPLEX | VF_PERMANENT,
                            length, cdata);
    }

    /* The allocation for resname has been assigned to the result vector, so
     * set to NULL so that it is not freed */
 finished:
    resname = NULL;

    /* Set dimension info */
    result->v_numdims = 1;
    result->v_dims[0] = length;

    vec_new(result);
    cp_addkword(CT_VECTOR, result->v_name);

done:
    free_pnode(names);
    txfree(resname);
} /* end of function com_compose */
