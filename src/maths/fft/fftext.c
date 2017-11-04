/*******************************************************************
	This file extends the fftlib with calls to maintain the cosine and bit reversed tables
	for you (including mallocs and free's).  Call the init routine for each fft size you will
	be using.  Then you can call the fft routines below which will make the fftlib library
	call with the appropriate tables passed.  When you are done with all fft's you can call
	fftfree to release the storage for the tables.  Note that you can call fftinit repeatedly
	with the same size, the extra calls will be ignored. So, you could make a macro to call
	fftInit every time you call ffts. For example you could have someting like:
	#define FFT(a,n) if(!fftInit(roundtol(LOG2(n)))) ffts(a,roundtol(LOG2(n)),1);else printf("fft error\n");
*******************************************************************/
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "fftlib.h"
#include "matlib.h"
#include "ngspice/fftext.h"
#include "ngspice/config.h"
#include "ngspice/memory.h"

#ifndef M_PI
#define M_PI		3.141592653589793238462643383279502884197	// pi
#endif

#define eq(a,b)  (!strcmp((a), (b)))

// pointers to storage of Utbl's and  BRLow's
static double *UtblArray[8*sizeof(int)] =
{0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};

static short *BRLowArray[8*sizeof(int)/2]  = {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};

int fftInit(int M)
{
// malloc and init cosine and bit reversed tables for a given size fft, ifft, rfft, rifft
    /* INPUTS */
    /* M = log2 of fft size	(ex M=10 for 1024 point fft) */
    /* OUTPUTS */
    /* private cosine and bit reversed tables	*/

    int theError = 1;
    /*** I did NOT test cases with M>27 ***/
    if ((M >= 0) && ((size_t) M < 8*sizeof(int))) {
        theError = 0;
        if (UtblArray[M] == NULL) {	// have we not inited this size fft yet?
            // init cos table
            UtblArray[M] = TMALLOC(double, POW2(M)/4+1);
            if (UtblArray[M] == NULL)
                theError = 2;
            else {
                fftCosInit(M, UtblArray[M]);
            }
            if (M > 1) {
                if (BRLowArray[M/2] == NULL) {	// init bit reversed table for cmplx fft
                    BRLowArray[M/2] = TMALLOC(short, POW2(M/2-1));
                    if (BRLowArray[M/2] == NULL)
                        theError = 2;
                    else {
                        fftBRInit(M, BRLowArray[M/2]);
                    }
                }
            }
            if (M > 2) {
                if (BRLowArray[(M-1)/2] == NULL) {	// init bit reversed table for real fft
                    BRLowArray[(M-1)/2] = TMALLOC(short, POW2((M-1)/2-1));
                    if (BRLowArray[(M-1)/2] == NULL)
                        theError = 2;
                    else {
                        fftBRInit(M-1, BRLowArray[(M-1)/2]);
                    }
                }
            }
        }
    }
    return theError;
}

void fftFree(void)
{
// release storage for all private cosine and bit reversed tables
    int i1;
    for (i1=8*sizeof(int)/2-1; i1>=0; i1--) {
        if (BRLowArray[i1] != NULL) {
            tfree(BRLowArray[i1]);
            BRLowArray[i1] = NULL;
        }
    }
    for (i1=8*sizeof(int)-1; i1>=0; i1--) {
        if (UtblArray[i1] != NULL) {
            tfree(UtblArray[i1]);
            UtblArray[i1] = NULL;
        }
    }
}

int
fft_windows(char *window, double *win, double *time, int length, double maxt, double span, int order)
{
    int i;
    double sigma, scale;

    /* window functions - should have an average of one */
    if (eq(window, "none"))
        for (i = 0; i < length; i++)
            win[i] = 1.0;
    else if (eq(window, "rectangular"))
        for (i = 0; i < length; i++) {
            if (maxt-time[i] > span)
                win[i] = 0.0;
            else
                win[i] = 1.0;
        }
    else if (eq(window, "triangle") || eq(window, "bartlet") || eq(window, "bartlett"))
        for (i = 0; i < length; i++) {
            if (maxt-time[i] > span)
                win[i] = 0.0;
            else
                win[i] = 2.0 - fabs(2+4*(time[i]-maxt)/span);
        }
    else if (eq(window, "hann") || eq(window, "hanning") || eq(window, "cosine"))
        for (i = 0; i < length; i++) {
            if (maxt-time[i] > span)
                win[i] = 0.0;
            else
                win[i] = 1.0 - cos(2*M_PI*(time[i]-maxt)/span);
        }
    else if (eq(window, "hamming"))
        for (i = 0; i < length; i++) {
            if (maxt-time[i] > span)
                win[i] = 0.0;
            else
                win[i] = 1.0 - 0.46/0.54*cos(2*M_PI*(time[i]-maxt)/span);
        }
    else if (eq(window, "blackman"))
        for (i = 0; i < length; i++) {
            if (maxt-time[i] > span) {
                win[i] = 0;
            } else {
                win[i]  = 1.0;
                win[i] -= 0.50/0.42*cos(2*M_PI*(time[i]-maxt)/span);
                win[i] += 0.08/0.42*cos(4*M_PI*(time[i]-maxt)/span);
            }
        }
    else if (eq(window, "flattop"))
        for (i = 0; i < length; i++) {
            if (maxt-time[i] > span) {
                win[i] = 0;
            } else {
                win[i]  = 1.0;
                win[i] -= 1.93*cos(2*M_PI*(time[i]-maxt)/span);
                win[i] += 1.29*cos(4*M_PI*(time[i]-maxt)/span);
                win[i] -= 0.388*cos(6*M_PI*(time[i]-maxt)/span);
                win[i] += 0.032*cos(8*M_PI*(time[i]-maxt)/span);
            }
        }
    else if (eq(window, "gaussian")) {
        sigma = 1.0/order;
        scale = 0.83/sigma;
        for (i = 0; i < length; i++) {
            if (maxt-time[i] > span)
                win[i] = 0;
            else
                win[i] = scale*exp(-0.5 * pow((time[i]-maxt/2)/(sigma*maxt/2), 2));
        }
    } else {
        printf( "Warning: unknown window type %s\n", window);
        return 0;
    }

    return 1;
}

/*************************************************
 The following calls are easier than calling to fftlib directly.
 Just make sure fftlib has been called for each M first.
**************************************************/

void ffts(double *data, int M, int Rows)
{
    /* Compute in-place complex fft on the rows of the input array	*/
    /* INPUTS */
    /* *ioptr = input data array	*/
    /* M = log2 of fft size	(ex M=10 for 1024 point fft) */
    /* Rows = number of rows in ioptr array (use 1 for Rows for a single fft)	*/
    /* OUTPUTS */
    /* *ioptr = output data array	*/
    ffts1(data, M, Rows, UtblArray[M], BRLowArray[M/2]);
}

void iffts(double *data, int M, int Rows)
{
    /* Compute in-place inverse complex fft on the rows of the input array	*/
    /* INPUTS */
    /* *ioptr = input data array	*/
    /* M = log2 of fft size	(ex M=10 for 1024 point fft) */
    /* Rows = number of rows in ioptr array (use 1 for Rows for a single fft)	*/
    /* OUTPUTS */
    /* *ioptr = output data array	*/
    iffts1(data, M, Rows, UtblArray[M], BRLowArray[M/2]);
}

void rffts(double *data, int M, int Rows)
{
    /* Compute in-place real fft on the rows of the input array	*/
    /* The result is the complex spectra of the positive frequencies */
    /* except the location for the first complex number contains the real */
    /* values for DC and Nyquest */
    /* See rspectprod for multiplying two of these spectra together- ex. for fast convolution */
    /* INPUTS */
    /* *ioptr = real input data array	*/
    /* M = log2 of fft size	(ex M=10 for 1024 point fft) */
    /* Rows = number of rows in ioptr array (use 1 for Rows for a single fft)	*/
    /* OUTPUTS */
    /* *ioptr = output data array	in the following order */
    /* Re(x[0]), Re(x[N/2]), Re(x[1]), Im(x[1]), Re(x[2]), Im(x[2]), ... Re(x[N/2-1]), Im(x[N/2-1]). */
    rffts1(data, M, Rows, UtblArray[M], BRLowArray[(M-1)/2]);
}

void riffts(double *data, int M, int Rows)
{
    /* Compute in-place real ifft on the rows of the input array	*/
    /* data order as from rffts */
    /* INPUTS */
    /* *ioptr = input data array in the following order	*/
    /* M = log2 of fft size	(ex M=10 for 1024 point fft) */
    /* Re(x[0]), Re(x[N/2]), Re(x[1]), Im(x[1]), Re(x[2]), Im(x[2]), ... Re(x[N/2-1]), Im(x[N/2-1]). */
    /* Rows = number of rows in ioptr array (use 1 for Rows for a single fft)	*/
    /* OUTPUTS */
    /* *ioptr = real output data array	*/
    riffts1(data, M, Rows, UtblArray[M], BRLowArray[(M-1)/2]);
}

void rspectprod(double *data1, double *data2, double *outdata, int N)
{
// When multiplying a pair of spectra from rfft care must be taken to multiply the
// two real values seperately from the complex ones. This routine does it correctly.
// the result can be stored in-place over one of the inputs
    /* INPUTS */
    /* *data1 = input data array	first spectra */
    /* *data2 = input data array	second spectra */
    /* N = fft input size for both data1 and data2 */
    /* OUTPUTS */
    /* *outdata = output data array spectra */
    if(N>1) {
        outdata[0] = data1[0] * data2[0];	// multiply the zero freq values
        outdata[1] = data1[1] * data2[1];	// multiply the nyquest freq values
        cvprod(data1 + 2, data2 + 2, outdata + 2, N/2-1);	// multiply the other positive freq values
    } else {
        outdata[0] = data1[0] * data2[0];
    }
}


#ifdef BOURKE

static void
fftext(double *x, double *y, long int n, long int nn, int dir)
{
    /*
      http://local.wasp.uwa.edu.au/~pbourke/other/dft/
      download 22.05.08
      Used with permission from the author Paul Bourke
    */

    /*
      This computes an in-place complex-to-complex FFT
      x and y are the real and imaginary arrays
      n is the number of points, has to be to the power of 2
      nn is the number of points w/o zero padded values
      dir =  1 gives forward transform
      dir = -1 gives reverse transform
    */

    long i, i1, j, k, i2, l, l1, l2;
    double c1, c2, tx, ty, t1, t2, u1, u2, z;
    int m = 0, M = 1;

    /* get the exponent to the base of 2 from the number of points */
    while (M < n) {
        M *= 2;
        m++;
    }

    /* Do the bit reversal */
    i2 = n >> 1;
    j = 0;
    for (i = 0; i < n-1; i++) {
        if (i < j) {
            tx = x[i];
            ty = y[i];
            x[i] = x[j];
            y[i] = y[j];
            x[j] = tx;
            y[j] = ty;
        }
        k = i2;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    /* Compute the FFT */
    c1 = -1.0;
    c2 = 0.0;
    l2 = 1;
    for (l = 0; l < m; l++) {
        l1 = l2;
        l2 <<= 1;
        u1 = 1.0;
        u2 = 0.0;
        for (j = 0; j < l1; j++) {
            for (i = j; i < n; i += l2) {
                i1 = i + l1;
                t1 = u1 * x[i1] - u2 * y[i1];
                t2 = u1 * y[i1] + u2 * x[i1];
                x[i1] = x[i] - t1;
                y[i1] = y[i] - t2;
                x[i] += t1;
                y[i] += t2;
            }
            z =  u1 * c1 - u2 * c2;
            u2 = u1 * c2 + u2 * c1;
            u1 = z;
        }
        c2 = sqrt((1.0 - c1) / 2.0);
        if (dir == 1)
            c2 = -c2;
        c1 = sqrt((1.0 + c1) / 2.0);
    }

    /* Scaling for forward transform */
    if (dir == 1) {
        double scale = 1.0 / nn;
        for (i = 0; i < n; i++) {
            x[i] *= scale; /* don't consider zero padded values */
            y[i] *= scale;
        }
    }
}

#endif /* BOURKE */
