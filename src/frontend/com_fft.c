/**********
Copyright 2008 Holger Vogt.  All rights reserved.
Author:   2008 Holger Vogt
**********/

/*
 * Code to do fast fourier transform on data.
 */

#include "ngspice.h"
#include "ftedefs.h"
#include "dvec.h"
#include "sim.h"

#include "com_fft.h"
#include "variable.h"
#include "missing_math.h"

void
com_fft(wordlist *wl)
{
    complex **fdvec;
    double  **tdvec;
    double  *freq, *win, *time;
    double  delta_t, span;
    int     fpts, i, j, tlen, ngood;
    struct dvec  *f, *vlist, *lv, *vec;
    struct pnode *names, *first_name;

    float *reald, *imagd;
    int size, sign, isreal;
    float scaling;

    if (!plot_cur || !plot_cur->pl_scale) {
        fprintf(cp_err, "Error: no vectors loaded.\n");
        return;
    }
    if (!isreal(plot_cur->pl_scale) || 
        ((plot_cur->pl_scale)->v_type != SV_TIME)) {
        fprintf(cp_err, "Error: fft needs real time scale\n");
        return;
    }

		
    tlen = (plot_cur->pl_scale)->v_length;
    time = (plot_cur->pl_scale)->v_realdata;
    span = time[tlen-1] - time[0];
    delta_t = span/(tlen - 1);
    
    // size of input vector is power of two and larger than spice vector
    size = 1;
    while (size < tlen)
        size *= 2;

    // output vector has length of size/2
    fpts = size/2;    

    // window function   
    win = (double *) tmalloc(tlen * sizeof (double));
    {
       char   window[BSIZE_SP];
       double maxt = time[tlen-1];
       if (!cp_getvar("specwindow", VT_STRING, window)) 
           strcpy(window,"blackman");
       if (eq(window, "none"))
          for(i=0; i<tlen; i++) {
             win[i] = 1;
          }
       else if (eq(window, "rectangular"))
           for(i=0; i<tlen; i++) {
             if (maxt-time[i] > span) {
                win[i] = 0;
             } else {
                win[i] = 1;
             }
          }
       else if (eq(window, "hanning") || eq(window, "cosine"))
          for(i=0; i<tlen; i++) {
             if (maxt-time[i] > span) {
                win[i] = 0;
             } else {
                win[i] = 1 - cos(2*M_PI*(time[i]-maxt)/span);
             }
          }
       else if (eq(window, "hamming"))
          for(i=0; i<tlen; i++) {
             if (maxt-time[i] > span) {
                win[i] = 0;
             } else {
                win[i] = 1 - 0.92/1.08*cos(2*M_PI*(time[i]-maxt)/span);
             }
          }
       else if (eq(window, "triangle") || eq(window, "bartlet"))
          for(i=0; i<tlen; i++) {
             if (maxt-time[i] > span) {
                win[i] = 0;
             } else {
                win[i] = 2 - fabs(2+4*(time[i]-maxt)/span);
             }
          }
       else if (eq(window, "blackman")) {
          int order;
          if (!cp_getvar("specwindoworder", VT_NUM, &order)) order = 2;
          if (order < 2) order = 2;  /* only order 2 supported here */
          for(i=0; i<tlen; i++) {
             if (maxt-time[i] > span) {
                win[i] = 0;
             } else {
                win[i]  = 1;
                win[i] -= 0.50/0.42*cos(2*M_PI*(time[i]-maxt)/span);
                win[i] += 0.08/0.42*cos(4*M_PI*(time[i]-maxt)/span);
             }
          }
       } else if (eq(window, "gaussian")) {
          int order;
          double scale;
          extern double erfc();
          if (!cp_getvar("specwindoworder", VT_NUM, &order)) order = 2;
          if (order < 2) order = 2;
          scale = pow(2*M_PI/order,0.5)*(0.5-erfc(pow(order,0.5)));
          for(i=0; i<tlen; i++) {
             if (maxt-time[i] > span) {
                win[i] = 0;
             } else {
                win[i] = exp(-0.5*order*(1-2*(maxt-time[i])/span)
                                       *(1-2*(maxt-time[i])/span))/scale;
             }
          }
	   } else {
          fprintf(cp_err, "Warning: unknown window type %s\n", window);
          tfree(win);
          return;
       }
    }

    names = ft_getpnames(wl, TRUE);
    first_name = names;
    vlist = NULL;
    ngood = 0;
    while (names) {
        vec = ft_evaluate(names);
        names = names->pn_next;
        while (vec) {
            if (vec->v_length != tlen) {
                fprintf(cp_err, "Error: lengths of %s vectors don't match: %d, %d\n",
                        vec->v_name, vec->v_length, tlen);
                vec = vec->v_link2;
                continue;
            }
            if (!isreal(vec)) {
                fprintf(cp_err, "Error: %s isn't real!\n", 
                        vec->v_name);
                vec = vec->v_link2;
                continue;
            }
            if (vec->v_type == SV_TIME) {
                vec = vec->v_link2;
                continue;
            }
            if (!vlist)
                vlist = vec;
            else
                lv->v_link2 = vec;
            lv = vec;
            vec = vec->v_link2;
            ngood++;
        }
    }
    free_pnode_o(first_name); /* h_vogt 081206 */
    if (!ngood) {
       return;
    }
 
    plot_cur = plot_alloc("spectrum");
    plot_cur->pl_next = plot_list;
    plot_list = plot_cur;
    plot_cur->pl_title = copy((plot_cur->pl_next)->pl_title);
    plot_cur->pl_name = copy("Spectrum");
    plot_cur->pl_date = copy(datestring( ));

    freq = (double *) tmalloc(fpts * sizeof(double));
    f = alloc(struct dvec);
    ZERO(f, struct dvec);
    f->v_name = copy("frequency");
    f->v_type = SV_FREQUENCY;
    f->v_flags = (VF_REAL | VF_PERMANENT | VF_PRINT);
    f->v_length = fpts;
    f->v_realdata = freq;
    vec_new(f);

    for (i = 0; i<fpts; i++) freq[i] = i*1./span*tlen/size;


    tdvec = (double  **) tmalloc(ngood * sizeof(double  *));
    fdvec = (complex **) tmalloc(ngood * sizeof(complex *));
    for (i = 0, vec = vlist; i<ngood; i++) {
       tdvec[i] = vec->v_realdata; /* real input data */
       fdvec[i] = (complex *) tmalloc(fpts * sizeof(complex)); /* complex output data */
       f = alloc(struct dvec);
       ZERO(f, struct dvec);
       f->v_name = vec_basename(vec);
       f->v_type = SV_NOTYPE; //vec->v_type;
       f->v_flags = (VF_COMPLEX | VF_PERMANENT);
       f->v_length = fpts;
       f->v_compdata = fdvec[i];
       vec_new(f);
       vec = vec->v_link2;
    }


    sign = 1;
    isreal = 1;
    
    reald = (float*)tmalloc(size*sizeof(float));
    imagd = (float*)tmalloc(size*sizeof(float));
    
    printf("CPU: Delta Freq %f Hz, input length %d, output length %d\n", 1./span*tlen/size, size, fpts);
   
    for (i = 0; i<ngood; i++) {
        for (j = 0; j < tlen; j++){
    	    reald[j] = tdvec[i][j]*win[j];
            imagd[j] = 0;
        }            
        for (j = tlen; j < size; j++){
    	    reald[j] = 0;
            imagd[j] = 0;
        }  
                 
        fftext(reald, imagd, size, sign);
        scaling = 0.3;
            
        for (j=0;j<fpts;j++){
	    fdvec[i][j].cx_real = reald[j]/scaling;
	    fdvec[i][j].cx_imag = imagd[j]/scaling;
        }    		
    }
        
    tfree(reald);
    tfree(imagd);
    
    tfree(tdvec);
    tfree(fdvec);    
}


static void fftext(float* x, float* y, long int n, int dir)
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
   dir =  1 gives forward transform
   dir = -1 gives reverse transform 
*/

   long i,i1,j,k,i2,l,l1,l2;
   double c1,c2,tx,ty,t1,t2,u1,u2,z;
   int m=0, mm=1;
   
   /* get the exponent to the base of 2 from the number of points */
   while (mm < n) {
       mm *= 2;
       m++;
   }

   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;
   for (i=0;i<n-1;i++) {
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
   for (l=0;l<m;l++) {
      l1 = l2;
      l2 <<= 1;
      u1 = 1.0; 
      u2 = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<n;i+=l2) {
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
      for (i=0;i<n;i++) {
         x[i] /= n;
         y[i] /= n;
      }
   }
}
