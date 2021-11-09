/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Code to do fourier transforms on data.  Note that we do interpolation
 * to get a uniform grid.  Note that if polydegree is 0 then no interpolation
 * is done.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/fteparse.h"
#include "ngspice/sperror.h"
#include "ngspice/const.h"
#include "ngspice/sim.h"

#include "fourier.h"
#include "variable.h"


static char *pnum(double num);
static int CKTfour(int ndata, int numFreq, double *thd, double *Time, double *Value,
                   double FundFreq, double *Freq, double *Mag, double *Phase, double *nMag,
                   double *nPhase);



#define DEF_FOURGRIDSIZE 200


/* CKTfour(ndata, numFreq, thd, Time, Value, FundFreq, Freq, Mag, Phase, nMag, nPhase)
 *         len    10       ?    inp   inp    inp       out   out  out    out   out
 */

int
fourier(wordlist *wl, struct plot *current_plot)
{
    struct dvec *time, *vec;
    struct pnode *pn, *names;
    double fundfreq, *data = NULL;
    int nfreqs, fourgridsize, polydegree;
    double *freq, *mag, *phase, *nmag, *nphase;  /* Outputs from CKTfour */
    double thd, *timescale = NULL;
    char *s;
    int i, err, fw;
    char xbuf[20];
    int shift;
    int rv = 1;

    struct dvec *n;
    int newveccount = 1;
    static int callstof = 1;

    if (!current_plot)
        return 1;

    sprintf(xbuf, "%1.1e", 0.0);
    shift = (int) strlen(xbuf) - 7;
    if (!current_plot || !current_plot->pl_scale) {
        fprintf(cp_err, "Error: no vectors loaded.\n");
        return 1;
    }

    if (!cp_getvar("nfreqs", CP_NUM, &nfreqs, 0) || nfreqs < 1)
        nfreqs = 10;
    if (!cp_getvar("polydegree", CP_NUM, &polydegree, 0) || polydegree < 0)
        polydegree = 1;
    if (!cp_getvar("fourgridsize", CP_NUM, &fourgridsize, 0) || fourgridsize < 1)
        fourgridsize = DEF_FOURGRIDSIZE;

    time = current_plot->pl_scale;
    if (!isreal(time)) {
        fprintf(cp_err, "Error: fourier needs real time scale\n");
        return 1;
    }
    s = wl->wl_word;
    if (ft_numparse(&s, FALSE, &fundfreq) < 0 || fundfreq <= 0.0) {
        fprintf(cp_err, "Error: bad fundamental freq %s\n", wl->wl_word);
        return 1;
    }

    freq = TMALLOC(double, nfreqs);
    mag = TMALLOC(double, nfreqs);
    phase = TMALLOC(double, nfreqs);
    nmag = TMALLOC(double, nfreqs);
    nphase = TMALLOC(double, nfreqs);

    wl = wl->wl_next;
    names = ft_getpnames_quotes(wl, TRUE);
    for (pn = names; pn; pn = pn->pn_next) {
        vec = ft_evaluate(pn);
        for (; vec; vec = vec->v_link2) {

            if (vec->v_length != time->v_length) {
                fprintf(cp_err,
                        "Error: lengths don't match: %d, %d\n",
                        vec->v_length, time->v_length);
                continue;
            }

            if (!isreal(vec)) {
                fprintf(cp_err, "Error: %s isn't real!\n", vec->v_name);
                continue;
            }

            if (polydegree) {
                double *dp, d;
                /* Build the grid... */
                timescale = TMALLOC(double, fourgridsize);
                data = TMALLOC(double, fourgridsize);
                dp = ft_minmax(time, TRUE);
                /* Now get the last fund freq... */
                d = 1 / fundfreq;   /* The wavelength... */
                if (dp[1] - dp[0] < d) {
                    fprintf(cp_err, "Error: wavelength longer than time span\n");
                    goto done;
                } else if (dp[1] - dp[0] > d) {
                    dp[0] = dp[1] - d;
                }

                d = (dp[1] - dp[0]) / fourgridsize;
                for (i = 0; i < fourgridsize; i++)
                    timescale[i] = dp[0] + i * d;

                /* Now interpolate the data... */
                if (!ft_interpolate(vec->v_realdata, data,
                                    time->v_realdata, vec->v_length,
                                    timescale, fourgridsize,
                                    polydegree)) {
                    fprintf(cp_err, "Error: can't interpolate\n");
                    goto done;
                }
            } else {
                fourgridsize = vec->v_length;
                data = vec->v_realdata;
                timescale = time->v_realdata;
            }

            err = CKTfour(fourgridsize, nfreqs, &thd, timescale,
                          data, fundfreq, freq, mag, phase, nmag,
                          nphase);
            if (err != OK) {
                ft_sperror(err, "fourier");
                goto done;
            }

            fprintf(cp_out, "Fourier analysis for %s:\n", vec->v_name);
            fprintf(cp_out,
                    "  No. Harmonics: %d, THD: %g %%, Gridsize: %d, Interpolation Degree: %d\n\n",
                    nfreqs, thd, fourgridsize,
                    polydegree);
            /* Each field will have width cp_numdgt + 6 (or 7
             * with HP-UX) + 1 if there is a - sign.
             */
            fw = ((cp_numdgt > 0) ? cp_numdgt : 6) + 5 + shift;
            fprintf(cp_out, "Harmonic %-*s %-*s %-*s %-*s %-*s\n",
                    fw, "Frequency", fw, "Magnitude",
                    fw, "Phase", fw, "Norm. Mag",
                    fw, "Norm. Phase");
            fprintf(cp_out, "-------- %-*s %-*s %-*s %-*s %-*s\n",
                    fw, "---------", fw, "---------",
                    fw, "-----", fw, "---------",
                    fw, "-----------");
            for (i = 0; i < nfreqs; i++) {
                char *pnumfr, *pnumma, *pnumph,  *pnumnm,   *pnumnp;
                pnumfr = pnum(freq[i]);
                pnumma = pnum(mag[i]);
                pnumph = pnum(phase[i]);
                pnumnm = pnum(nmag[i]);
                pnumnp = pnum(nphase[i]);
                fprintf(cp_out,
                        " %-4d    %-*s %-*s %-*s %-*s %-*s\n",
                        i,
                        fw, pnumfr,
                        fw, pnumma,
                        fw, pnumph,
                        fw, pnumnm,
                        fw, pnumnp);
                tfree(pnumfr);
                tfree(pnumma);
                tfree(pnumph);
                tfree(pnumnm);
                tfree(pnumnp);
            }
            fputs("\n", cp_out);

            /* create and assign a new vector n */
            /* with size 3 * nfreqs in current plot */
            /* generate name for new vector, using vec->name */
            n = dvec_alloc(tprintf("fourier%d%d", callstof, newveccount),
                           SV_NOTYPE,
                           VF_REAL | VF_PERMANENT,
                           3 * nfreqs, NULL);

            n->v_numdims = 2;
            n->v_dims[0] = 3;
            n->v_dims[1] = nfreqs;

            vec_new(n);

            /* store data in vector: freq, mag, phase */
            for (i = 0; i < nfreqs; i++) {
                n->v_realdata[i] = freq[i];
                n->v_realdata[i + nfreqs] = mag[i];
                n->v_realdata[i + 2 * nfreqs] = phase[i];
            }
            newveccount++;

            if (polydegree) {
                tfree(timescale);
                tfree(data);
            }
            timescale = NULL;
            data = NULL;
        }
    }

    callstof++;
    rv = 0;

done:
    free_pnode(names);
    tfree(freq);
    tfree(mag);
    tfree(phase);
    tfree(nmag);
    tfree(nphase);
    if (polydegree) {
        tfree(timescale);
        tfree(data);
    }

    return rv;
}


void
com_fourier(wordlist *wl)
{
    fourier(wl, plot_cur);
}


static char *
pnum(double num)
{
    int i = cp_numdgt;

    if (i < 1)
        i = 6;

    if (num < 0.0)
        return tprintf("%.*g", i - 1, num);
    else
        return tprintf("%.*g", i, num);
}


/* CKTfour() - perform fourier analysis of an output vector.
 *
 * Due to the construction of the program which places all the output
 * data in the post-processor, the fourier analysis can not be done
 * directly.  This function allows the post processor to hand back
 * vectors of time and data values to have the fourier analysis
 * performed on them.  */
static int
CKTfour(int ndata,              /* number of entries in the Time and
                                   Value arrays */
        int numFreq,            /* number of harmonics to calculate */
        double *thd,            /* total harmonic distortion (percent)
                                   to be returned */
        double *Time,           /* times at which the voltage/current
                                   values were measured*/
        double *Value,          /* voltage or current vector whose
                                   transform is desired */
        double FundFreq,        /* the fundamental frequency of the
                                   analysis */
        double *Freq,           /* the frequency value of the various
                                   harmonics */
        double *Mag,            /* the Magnitude of the fourier
                                   transform */
        double *Phase,          /* the Phase of the fourier transform */
        double *nMag,           /* the normalized magnitude of the
                                   transform: nMag(fund)=1*/
        double *nPhase)         /* the normalized phase of the
                                   transform: Nphase(fund)=0 */
{
    /* Note: we can consider these as a set of arrays.  The sizes are:
     * Time[ndata], Value[ndata], Freq[numFreq], Mag[numfreq],
     * Phase[numfreq], nMag[numfreq], nPhase[numfreq]
     *
     * The arrays must all be allocated by the caller.
     * The Time and Value array must be reasonably distributed over at
     * least one full period of the fundamental Frequency for the
     * fourier transform to be useful.  The function will take the
     * last period of the frequency as data for the transform.
     *
     * We are assuming that the caller has provided exactly one period
     * of the fundamental frequency.  */
    int i;
    int j;
    double tmp;

    NG_IGNORE(Time);

    /* clear output/computation arrays */

    for (i = 0; i < numFreq; i++) {
        Mag[i] = 0;
        Phase[i] = 0;
    }

    for (i = 0; i < ndata; i++)
        for (j = 0; j < numFreq; j++) {
            Mag[j]   += Value[i] * sin(j*2.0*M_PI*i/((double)ndata));
            Phase[j] += Value[i] * cos(j*2.0*M_PI*i/((double)ndata));
        }

    Mag[0] = Phase[0]/ndata;
    Phase[0] = nMag[0] = nPhase[0] = Freq[0] = 0;
    *thd = 0;
    for (i = 1; i < numFreq; i++) {
        tmp = Mag[i] * 2.0 / ndata;
        Phase[i] *= 2.0 / ndata;
        Freq[i] = i * FundFreq;
        Mag[i] = hypot(tmp, Phase[i]);
        Phase[i] = atan2(Phase[i], tmp) * 180.0/M_PI;
        nMag[i] = Mag[i] / Mag[1];
        nPhase[i] = Phase[i] - Phase[1];
        if (i > 1)
            *thd += nMag[i] * nMag[i];
    }
    *thd = 100*sqrt(*thd);
    return (OK);
}
