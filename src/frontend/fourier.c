/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Code to do fourier transforms on data.  Note that we do interpolation
 * to get a uniform grid.  Note that if polydegree is 0 then no interpolation
 * is done.
 */

#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "ftedata.h"
#include "fteparse.h"
#include "sperror.h"
#include "const.h"
#include "fourier.h"


/* static declarations */
static char * pn(double num);
static int CKTfour(int ndata, int numFreq, double *thd, double *Time, double *Value, 
		   double FundFreq, double *Freq, double *Mag, double *Phase, double *nMag, 
		   double *nPhase);



#define DEF_FOURGRIDSIZE 200

/* CKTfour(ndata,numFreq,thd,Time,Value,FundFreq,Freq,Mag,Phase,nMag,nPhase)
 *         len   10      ?   inp  inp   inp      out  out out   out  out
 */

void
com_fourier(wordlist *wl)
{
    struct dvec *time, *vec;
    struct pnode *names, *first_name;
    double *ff, fundfreq, *dp, *stuff;
    int nfreqs, fourgridsize, polydegree;
    double *freq, *mag, *phase, *nmag, *nphase;  /* Outputs from CKTfour */
    double thd, *timescale, *grid, d;
    char *s;
    int i, err, fw;
    char xbuf[20];
    int shift;

    sprintf(xbuf, "%1.1e", 0.0);
    shift = strlen(xbuf) - 7;
    if (!plot_cur || !plot_cur->pl_scale) {
        fprintf(cp_err, "Error: no vectors loaded.\n");
        return;
    }

    if ((!cp_getvar("nfreqs", VT_NUM, (char *) &nfreqs)) || (nfreqs < 1))
        nfreqs = 10;
    if ((!cp_getvar("polydegree", VT_NUM, (char *) &polydegree)) ||
            (polydegree < 0))
        polydegree = 1;
    if ((!cp_getvar("fourgridsize", VT_NUM, (char *) &fourgridsize)) ||
            (fourgridsize < 1))
        fourgridsize = DEF_FOURGRIDSIZE;

    time = plot_cur->pl_scale;
    if (!isreal(time)) {
        fprintf(cp_err, "Error: fourier needs real time scale\n");
        return;
    }
    s = wl->wl_word;
    if (!(ff = ft_numparse(&s, FALSE)) || (*ff <= 0.0)) {
        fprintf(cp_err, "Error: bad fund freq %s\n", wl->wl_word);
        return;
    }
    fundfreq = *ff;

    freq = (double *) tmalloc(nfreqs * sizeof (double));
    mag = (double *) tmalloc(nfreqs * sizeof (double));
    phase = (double *) tmalloc(nfreqs * sizeof (double));
    nmag = (double *) tmalloc(nfreqs * sizeof (double));
    nphase = (double *) tmalloc(nfreqs * sizeof (double));

    wl = wl->wl_next;
    names = ft_getpnames(wl, TRUE);
    first_name = names;
    while (names) {
        vec = ft_evaluate(names);
        names = names->pn_next;
        while (vec) {
            if (vec->v_length != time->v_length) {
                fprintf(cp_err, 
                    "Error: lengths don't match: %d, %d\n",
                        vec->v_length, time->v_length);
                continue;
            }
            if (!isreal(vec)) {
                fprintf(cp_err, "Error: %s isn't real!\n", 
                        vec->v_name);
                continue;
            }

            if (polydegree) {
                /* Build the grid... */
                grid = (double *) tmalloc(fourgridsize *
                        sizeof (double));
                stuff = (double *) tmalloc(fourgridsize *
                        sizeof (double));
                dp = ft_minmax(time, TRUE);

                /* Now get the last fund freq... */
                d = 1 / fundfreq;   /* The wavelength... */
                if (dp[1] - dp[0] < d) {
                    fprintf(cp_err, 
                "Error: wavelength longer than time span\n");
                    return;
                } else if (dp[1] - dp[0] > d) {
                    dp[0] = dp[1] - d;
                }

                d = (dp[1] - dp[0]) / fourgridsize;
                for (i = 0; i < fourgridsize; i++)
                    grid[i] = dp[0] + i * d;
                
                /* Now interpolate the stuff... */
                if (!ft_interpolate(vec->v_realdata, stuff,
                        time->v_realdata, vec->v_length,
                        grid, fourgridsize, 
                        polydegree)) {
                    fprintf(cp_err, 
                        "Error: can't interpolate\n");
                    return;
                }
                timescale = grid;
            } else {
                fourgridsize = vec->v_length;
                stuff = vec->v_realdata;
                timescale = time->v_realdata;
            }

            err = CKTfour(fourgridsize, nfreqs, &thd, timescale,
                    stuff, fundfreq, freq, mag, phase, nmag,
                    nphase);
            if (err != OK) {
                ft_sperror(err, "fourier");
                return;
            }

            fprintf(cp_out, "Fourier analysis for %s:\n", 
                    vec->v_name);
            fprintf(cp_out, 
"  No. Harmonics: %d, THD: %g %%, Gridsize: %d, Interpolation Degree: %d\n\n",
                nfreqs,  thd, fourgridsize, 
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
            for (i = 0; i < nfreqs; i++)
                fprintf(cp_out,
                    " %-4d    %-*s %-*s %-*s %-*s %-*s\n",
                    i,
                    fw, pn(freq[i]),
                    fw, pn(mag[i]),
                    fw, pn(phase[i]),
                    fw, pn(nmag[i]),
                    fw, pn(nphase[i]));
            fputs("\n", cp_out);
            vec = vec->v_link2;
        }
    }
    free_pnode(first_name);
    tfree(freq);
    tfree(mag);
    tfree(phase);
    tfree(nmag);
    tfree(nphase);
    return;
}

static char *
pn(double num)
{
    char buf[BSIZE_SP];
    int i = cp_numdgt;

    if (i < 1)
        i = 6;

    if (num < 0.0)
        (void) sprintf(buf, "%.*g", i - 1, num);
    else
        (void) sprintf(buf, "%.*g", i, num);
    return (copy(buf));
}

/*
 * CKTfour() - perform fourier analysis of an output vector.
 *  Due to the construction of the program which places all the
 *  output data in the post-processor, the fourier analysis can not
 *  be done directly.  This function allows the post processor to
 *  hand back vectors of time and data values to have the fourier analysis
 *  performed on them.
 *
 */


static int
CKTfour(int ndata, int numFreq, double *thd, double *Time, double *Value, double FundFreq, double *Freq, double *Mag, double *Phase, double *nMag, double *nPhase)
                /* number of entries in the Time and Value arrays */
                    /* number of harmonics to calculate */
                    /* total harmonic distortion (percent) to be returned */
                    /* times at which the voltage/current values were measured*/
                    /* voltage or current vector whose transform is desired */
                        /* the fundamental frequency of the analysis */
                    /* the frequency value of the various harmonics */
                    /* the Magnitude of the fourier transform */
                    /* the Phase of the fourier transform */
                    /* the normalized magnitude of the transform: nMag(fund)=1*/
                    /* the normalized phase of the transform: Nphase(fund)=0 */
    /* note we can consider these as a set of arrays:  The sizes are:
     *  Time[ndata], Value[ndata]
     *  Freq[numFreq],Mag[numfreq],Phase[numfreq],nMag[numfreq],nPhase[numfreq]
     * The arrays must all be allocated by the caller.
     * The Time and Value array must be reasonably distributed over at
     * least one full period of the fundamental Frequency for the
     * fourier transform to be useful.  The function will take the
     * last period of the frequency as data for the transform.
     */

{
/* we are assuming that the caller has provided exactly one period
 * of the fundamental frequency.
 */
    int i;
    int j;
    double tmp;
    /* clear output/computation arrays */

    for(i=0;i<numFreq;i++) {
        Mag[i]=0;
        Phase[i]=0;
    }
    for(i=0;i<ndata;i++) {
        for(j=0;j<numFreq;j++) {
            Mag[j]   += (Value[i]*sin(j*2.0*M_PI*i/((double) ndata)));
            Phase[j] += (Value[i]*cos(j*2.0*M_PI*i/((double) ndata)));
        }
    }

    Mag[0] = Phase[0]/ndata;
    Phase[0]=nMag[0]=nPhase[0]=Freq[0]=0;
    *thd = 0;
    for(i=1;i<numFreq;i++) {
        tmp = Mag[i]*2.0 /ndata;
        Phase[i] *= 2.0/ndata;
        Freq[i] = i * FundFreq;
        Mag[i] = sqrt(tmp*tmp+Phase[i]*Phase[i]);
        Phase[i] = atan2(Phase[i],tmp)*180.0/M_PI;
        nMag[i] = Mag[i]/Mag[1];
        nPhase[i] = Phase[i]-Phase[1];
        if(i>1) *thd += nMag[i]*nMag[i];
    }
    *thd = 100*sqrt(*thd);
    return(OK);
}

#ifdef notdef
    /* What is this code?  An old DFT? */
    double initial; /*  starting time */
    double final;   /* final time */
    double elapsed; /* elapsed time */
    double tmp;
    int start=0;
    int n;
    int m;
    int edge;

    *thd = 0;
    final = Time[ndata-1];
    initial = Time[0];
    elapsed = final - initial;
    if( (elapsed-1/FundFreq)< -.01/FundFreq ){
        /* not enough data for a full period */
        return(E_BADPARM);
    }
    elapsed = 1/FundFreq;   /* set to desired elapsed time */
    initial = final - elapsed;  /* set to desired starting time */
    while(Time[start]<initial) { start++; } /* to find first time in interval*/
    start++; /* throw away one more point - come back to it later */
    for(m=0;m<numFreq;m++) {
        Mag[m]=0;
        Phase[m]=0;
    }
    /* ok - here's the hard part - compute the dft of Data[start::ndata]
     * temporarily, put the real/imag. parts of the DFT in Mag[] and Phase[]
     * later we will convert each term to phase-magnitude 
     */

    for(n=start;n<ndata-1;n++) {
        for(m=0;m<numFreq;m++) {
            Mag[m]   += .5 * (Time[n+1]-Time[n-1]) * Value[n] * 
                    sin(2.0*M_PI*m*((Time[n]-initial)/elapsed));
            Phase[m] += .5 * (Time[n+1]-Time[n-1]) * Value[n] * 
                    cos(2.0*M_PI*m*((Time[n]-initial)/elapsed));
            /* know Time[n+-1] exists because stop at = ndata-2, */
            /* and did a start++ earlier - come back and clean up ends later */
        }
    }
    /* now to deal with the endpoints.  The (ndata-1)th point has a smaller 
     * interval 
     */
    for(m=0;m<numFreq;m++) {
        Mag[m]   += 0.5 * (Time[n]-Time[n-1]) * Value[n] * 
                sin(2.0*M_PI*m*((Time[n]-initial)/elapsed));
        Phase[m] += 0.5 * (Time[n]-Time[n-1]) * Value[n] * 
                cos(2.0*M_PI*m*((Time[n]-initial)/elapsed));
    }
    /* now to deal with the start of the interval.  first, deal with
     * the start-1'th point - exactly the same regardless of case 
     * because of the extra start++ earlier.
     */
    for(m=0;m<numFreq;m++) {
        Mag[m]   += 0.5 * (Time[start]-initial) * Value[start-1] * 
                sin(2.0*M_PI*m*((Time[start-1]-initial)/elapsed));
        Phase[m] += 0.5 * (Time[start]-initial) * Value[start-1] * 
                cos(2.0*M_PI*m*((Time[start-1]-initial)/elapsed));
    }
    /* now deal with the possibility that the above point, which was
     * the first one contained within the interval may have been
     * inside the interval, or ON the boundry - in the latter case,
     * we don't want to deal with the previous point at all.
     */
    if(Time[start-1]> initial) {
        /* interesting case - need to handle previous point */
        /* first, make sure that there is a point on the other side of
         * the beginning of time.
         */
        if(start-2 < 0) { 
            /* point doesn't exist, so we have to fudge
             * things slightly - by bumping edge up, we re-use the first
             * point in the interval for the last point before the 
             * interval - should be only for very small error in
             * interval boundaries, so shouldn't be significant, and is
             * better than ignoring the interval
             */
            edge = start-1;
        } else {
            edge = start-2;
        }
        for(m=0;m<numFreq;m++) {
            Mag[m]   += .5 * (Time[start-1]-initial) * Value[edge] * 
                    sin(2.0*M_PI*m*((Time[edge]-initial)/elapsed));
            Phase[m] += .5 * (Time[start-1]-initial) * Value[edge] * 
                    cos(2.0*M_PI*m*((Time[edge]-initial)/elapsed));
        }
    }
    
    Mag[0]=Phase[0]/elapsed;
    Phase[0]=nMag[0]=nPhase[0]=Freq[0]=0;

    for(m=1;m<numFreq;m++) {
        tmp = Mag[m] * 2.0 / (elapsed);
        Phase[m] *= 2.0 / (elapsed);
        Freq[m] = m * FundFreq;
        Mag[m] = sqrt(tmp * tmp + Phase[m] * Phase[m]);
        Phase[m] = atan2(Phase[m],tmp) * 180.0/M_PI;
        nMag[m] = Mag[m] / Mag[1];
        nPhase[m] = Phase[m] - Phase[1];
        if(m>1) *thd += nMag[m] * nMag[m];
    }
    *thd = 100 * sqrt(*thd);
    return(OK);

#endif
