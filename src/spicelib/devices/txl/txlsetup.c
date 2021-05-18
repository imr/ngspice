/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/


#include "ngspice/ngspice.h"
#include "ngspice/fteext.h"
#include "ngspice/smpdefs.h"
#include "txldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


static int 		ReadTxL(TXLinstance*, CKTcircuit*);
/*static int 		multC();*/
static int 		main_pade(double, double, double, double, double, TXLine*);
static int 		mac(double, double, double*, double*, double*, double*, double*);
/*static int 		divC();*/
static int 		div_C(double, double, double, double, double*, double*);
static int 		div3(double, double, double, double, double*, double*);
/*static double 	approx1();*/
/*static double 	approx2();*/
static int 		find_roots(double, double, double, double*, double*, double*);
/*static double 	f3();*/
/*static double 	f2();*/
/*static int 		expC();*/
/*static double 	exp_approx1();*/
/*static double 	exp_approx2();*/
static int 		exp_pade(double, double, double, double, double, TXLine*);
/*static int 		exp_div3();*/
static int 		exp_find_roots(double, double, double, double*, double*, double* );
static double		eval2(double, double, double, double);
static int 		get_c(double, double, double, double, double, double, double, double*, double*);
static int 		get_h3(TXLine*);
static int 		Gaussian_Elimination2(int);
static int 		Gaussian_Elimination1(int);
static int 		pade(double);
static int 		update_h1C_c(TXLine *);
static void		y_pade(double, double, double, double, TXLine*);
static double		root3(double, double, double, double);
static NDnamePt 	insert_ND(char*, NDnamePt*);
static NODE 		*insert_node(char*);
static NODE 		*NEW_node(void);
/*static VI_list_txl *new_vi_txl();*/

NODE     		*node_tab = NULL;
NDnamePt 		ndn_btree = NULL;
VI_list_txl 	*pool_vi_txl = NULL;

/* pade.c */
/**
static double xx1, xx2, xx3, xx4, xx5, xx6;
static double cc1, cc2, cc3, cc4, cc5, cc6;
**/

/* y.c */
static double sqtCdL;
static double b1, b2, b3, b4, b5;
static double p1, p2, p3, q1, q2, q3;
static double c1, c2, c3, x1, x2, x3;
static double A[3][4];

/* exp.c */
static double RG, tau, RC, GL;
static double a0, a1, a2, a3, a4, a5;
static double ep1, ep2, ep3, eq1, eq2, eq3;
static double ec1, ec2, ec3, ex1, ex2, ex3;
static int    ifImg;
static double AA[3][4];

#define epsi 1.0e-16
#define epsi2 1.0e-28

/* ARGSUSED */
int
TXLsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit*ckt, int *state)
{
  TXLmodel *model = (TXLmodel *)inModel;
  TXLinstance *here;     
  CKTnode *tmp;
  int error;

  NG_IGNORE(state);

    /*  loop through all the models */
    for( ; model != NULL; model = TXLnextModel(model)) {

        if (!model->Rgiven) {
           SPfrontEnd->IFerrorf (ERR_FATAL,
               "model %s: lossy line series resistance not given", model->TXLmodName);
          return(E_BADPARM);
        }
        if (!model->Ggiven) {
           SPfrontEnd->IFerrorf (ERR_FATAL,
               "model %s: lossy line parallel conductance not given", model->TXLmodName);
          return(E_BADPARM);
        }
        if (!model->Lgiven) {
          SPfrontEnd->IFerrorf (ERR_FATAL,
              "model %s: lossy line series inductance not given", model->TXLmodName);
          return (E_BADPARM);
        }
        if (!model->Cgiven) {
          SPfrontEnd->IFerrorf (ERR_FATAL,
              "model %s: lossy line parallel capacitance not given", model->TXLmodName);
          return (E_BADPARM);
        }
        if (!model->lengthgiven) {
          SPfrontEnd->IFerrorf (ERR_FATAL,
              "model %s: lossy line length must be given", model->TXLmodName);
          return (E_BADPARM);
        }

        /* loop through all the instances of the model */
        for (here = TXLinstances(model); here != NULL ;
                here=TXLnextInstance(here)) {
            
/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            if (! here->TXLibr1Given) {
                    error = CKTmkCur(ckt, &tmp, here->TXLname, "branch1");
                    if (error) return (error);
                    here->TXLibr1 = tmp->number;
            }
            if (! here->TXLibr2Given) {
                    error = CKTmkCur(ckt, &tmp, here->TXLname, "branch2");
                    if (error) return (error);
                    here->TXLibr2 = tmp->number;
            }

            TSTALLOC(TXLposPosPtr, TXLposNode, TXLposNode);
            TSTALLOC(TXLposNegPtr, TXLposNode, TXLnegNode);
            TSTALLOC(TXLnegPosPtr, TXLnegNode, TXLposNode);
            TSTALLOC(TXLnegNegPtr, TXLnegNode, TXLnegNode);
            TSTALLOC(TXLibr1PosPtr, TXLibr1, TXLposNode);
            TSTALLOC(TXLibr2NegPtr, TXLibr2, TXLnegNode);
            TSTALLOC(TXLnegIbr2Ptr, TXLnegNode, TXLibr2);
            TSTALLOC(TXLposIbr1Ptr, TXLposNode, TXLibr1);
            TSTALLOC(TXLibr1Ibr1Ptr, TXLibr1, TXLibr1);
            TSTALLOC(TXLibr2Ibr2Ptr, TXLibr2, TXLibr2);
            TSTALLOC(TXLibr1NegPtr, TXLibr1, TXLnegNode);
            TSTALLOC(TXLibr2PosPtr, TXLibr2, TXLposNode);
            TSTALLOC(TXLibr1Ibr2Ptr, TXLibr1, TXLibr2);
            TSTALLOC(TXLibr2Ibr1Ptr, TXLibr2, TXLibr1);

			here->in_node_name = CKTnodName(ckt,here->TXLposNode);
			here->out_node_name = CKTnodName(ckt,here->TXLnegNode);
			ReadTxL(here, ckt);

        }
    }

    return(OK);
}

int
TXLunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
  TXLmodel *model;
  TXLinstance *here;
  
  for (model = (TXLmodel *) inModel; model != NULL;
      model = TXLnextModel(model)) {
    for (here = TXLinstances(model); here != NULL;
         here = TXLnextInstance(here)) {
  
          if (here->TXLibr2) {
               CKTdltNNum(ckt, here->TXLibr2);
               here->TXLibr2 = 0;
          }

          if (here->TXLibr1) {
               CKTdltNNum(ckt, here->TXLibr1);
               here->TXLibr1 = 0;
          }

    here->TXLdcGiven=0;

    }
  }
  return OK;
}

/***
static VI_list_txl
*new_vi_txl(void)
{
	VI_list_txl *q;

	if (pool_vi_txl) {
		q = pool_vi_txl;
		pool_vi_txl = pool_vi_txl->pool;
		return(q);
	} else
		return(TMALLOC(VI_list_txl, 1));
}
***/

static int 
ReadTxL(TXLinstance *tx, CKTcircuit *ckt)
{
   double R, L, G, C, l;
   char *p, *n;
   NODE *nd;
   ETXLine *et;
   TXLine *t, *t2;
   RLINE *line;
   ERLINE *er;
   double LL = 1e-12;
	
   NG_IGNORE(ckt);

   p = tx->in_node_name;
   n = tx->out_node_name;

   line = TMALLOC(RLINE, 1);
   er = TMALLOC(ERLINE, 1);
   et = TMALLOC(ETXLine, 1);
   t = TMALLOC(TXLine, 1);
   t2 = TMALLOC(TXLine, 1);
   tx->txline = t;
   tx->txline2 = t2;
   t->newtp = 0;
   t2->newtp = 0;
   t->vi_head = t->vi_tail = NULL;
   nd = insert_node(p);
   et->link = nd->tptr;
   nd->tptr = et;
   et->line = t;
   t->in_node = nd;
   t2->in_node = nd;
   er->link = nd->rlptr;
   nd->rlptr = er;
   er->rl = line;
   line->in_node = nd;
   et = TMALLOC(ETXLine, 1);
   nd = insert_node(n);
   et->link = nd->tptr;
   nd->tptr = et;
   et->line = t;
   t->out_node = nd;
   t2->out_node = nd;
   er = TMALLOC(ERLINE, 1);
   er->link = nd->rlptr;
   nd->rlptr = er;
   er->rl = line;
   line->out_node = nd;
   t->dc1 = t->dc2 = 0.0;
   t2->dc1 = t2->dc2 = 0.0;
   t->lsl = 0;
   t2->lsl = 0;
   l = 0.0;

   R = TXLmodPtr(tx)->R;
   L = TXLmodPtr(tx)->L;
   L = MAX(L, LL);
   C = TXLmodPtr(tx)->C;
   G = TXLmodPtr(tx)->G;
   if (tx->TXLlengthgiven == TRUE)
   		l = tx->TXLlength;
   else l = TXLmodPtr(tx)->length;	


   if (l == 0.0) {
	   fprintf(stderr, "(Error) transmission line of zero length\n");
	   controlled_exit(EXIT_FAILURE);
   }
   else {
		if (R / L  < 5.0e+5) {
			line->g = 1.0e+2;
			if (G < 1.0e-2) {
				t->lsl = 1;  /* lossless line */
				t->taul = sqrt(C * L) * l * 1.0e12;
				t->h3_aten = t->sqtCdL = sqrt(C / L);
				t->h2_aten = 1.0;
				t->h1C = 0.0;
			}
		}
		else line->g = 1.0 / (R * l);
	}

   if (! t->lsl)
   		main_pade(R, L, G, C, l, t);

   return(1);
}


/****************************************************************
     pade.c  :  Calculate the Pade Approxximation of Y(s)
 ****************************************************************/


static int 
main_pade(double R, double L, double G, double C, double l, TXLine *h)
{
   y_pade(R, L, G, C, h);
   h->ifImg = exp_pade(R, L, G, C, l, h);
   get_h3(h);
   h->taul *= 1.0e12;
   update_h1C_c(h);

   return(1);
}

static int 
div_C(double ar, double ai, double br, double bi, double *cr, double *ci)
{
   *cr = ar * br + ai * bi;
   *ci = - ar * bi + ai * br;
   *cr = *cr / (br * br + bi * bi);
   *ci = *ci / (br * br + bi * bi);
   return (1);
}

/***
static int expC(ar, ai, h, cr, ci)
   double ar, ai, *cr, *ci;
   float h;
{
   double e, cs, si;

   e = exp((double) ar * h);
   cs = cos((double) ai * h);
   si = sin((double) ai * h);
   *cr = e * cs;
   *ci = e * si;

   return(1);
}
***/

/***
static int multC(ar, ai, br, bi, cr, ci)
   double ar, ai, br, bi;
   double *cr, *ci;
{
   *cr = ar*br - ai*bi;
   *ci = ar*bi + ai*br;

   return(1);
}
***/

/***
static int divC(ar, ai, br, bi, cr, ci)
   double ar, ai, br, bi;
   double *cr, *ci;
{
   double t;
   t = br*br + bi*bi;
   *cr = (ar*br + ai*bi) / t;
   *ci = (ai*br - ar*bi) / t;

   return(1);
}
***/

static int 
get_h3(TXLine *h)
{
   double cc1, cc2, cc3, cc4, cc5, cc6;
   double xx1, xx2, xx3, xx4, xx5, xx6;

   h->h3_aten = h->h2_aten * h->sqtCdL;
   h->h3_term[0].x = xx1 = h->h1_term[0].x;
   h->h3_term[1].x = xx2 = h->h1_term[1].x;
   h->h3_term[2].x = xx3 = h->h1_term[2].x;
   h->h3_term[3].x = xx4 = h->h2_term[0].x;
   h->h3_term[4].x = xx5 = h->h2_term[1].x;
   h->h3_term[5].x = xx6 = h->h2_term[2].x;
   cc1 = h->h1_term[0].c;
   cc2 = h->h1_term[1].c;
   cc3 = h->h1_term[2].c;
   cc4 = h->h2_term[0].c;
   cc5 = h->h2_term[1].c;
   cc6 = h->h2_term[2].c;

   if (h->ifImg) {
      double r, i;

      h->h3_term[0].c = cc1 + cc1 * (cc4/(xx1-xx4) + 
         2.0*(cc5*xx1-xx6*cc6-xx5*cc5)/(xx1*xx1-2.0*xx5*xx1+xx5*xx5+xx6*xx6));
      h->h3_term[1].c = cc2 + cc2 * (cc4/(xx2-xx4) + 
         2.0*(cc5*xx2-xx6*cc6-xx5*cc5)/(xx2*xx2-2.0*xx5*xx2+xx5*xx5+xx6*xx6));
      h->h3_term[2].c = cc3 + cc3 * (cc4/(xx3-xx4) + 
         2.0*(cc5*xx3-xx6*cc6-xx5*cc5)/(xx3*xx3-2.0*xx5*xx3+xx5*xx5+xx6*xx6));

      h->h3_term[3].c = cc4 + cc4 * (cc1/(xx4-xx1) + cc2/(xx4-xx2) + cc3/(xx4-xx3));

      h->h3_term[4].c = cc5;
      h->h3_term[5].c = cc6;
      div_C(cc5, cc6, xx5-xx1, xx6, &r, &i);
      h->h3_term[4].c += r * cc1;
      h->h3_term[5].c += i * cc1;
      div_C(cc5, cc6, xx5-xx2, xx6, &r, &i);
      h->h3_term[4].c += r * cc2;
      h->h3_term[5].c += i * cc2;
      div_C(cc5, cc6, xx5-xx3, xx6, &r, &i);
      h->h3_term[4].c += r * cc3;
      h->h3_term[5].c += i * cc3;
   } else {
      h->h3_term[0].c = cc1 + cc1 * (cc4/(xx1-xx4) + cc5/(xx1-xx5) + cc6/(xx1-xx6));
      h->h3_term[1].c = cc2 + cc2 * (cc4/(xx2-xx4) + cc5/(xx2-xx5) + cc6/(xx2-xx6));
      h->h3_term[2].c = cc3 + cc3 * (cc4/(xx3-xx4) + cc5/(xx3-xx5) + cc6/(xx3-xx6));

      h->h3_term[3].c = cc4 + cc4 * (cc1/(xx4-xx1) + cc2/(xx4-xx2) + cc3/(xx4-xx3));
      h->h3_term[4].c = cc5 + cc5 * (cc1/(xx5-xx1) + cc2/(xx5-xx2) + cc3/(xx5-xx3));
      h->h3_term[5].c = cc6 + cc6 * (cc1/(xx6-xx1) + cc2/(xx6-xx2) + cc3/(xx6-xx3));
   }
      
   return(1);
}

static int 
update_h1C_c(TXLine *h)
{
   int i;
   double d = 0;

   for (i = 0; i < 3; i++) {
      h->h1_term[i].c *= h->sqtCdL;
      d += h->h1_term[i].c;
   }
   h->h1C = d;

   for (i = 0; i < 3; i++) 
      h->h2_term[i].c *= h->h2_aten;
   
   for (i = 0; i < 6; i++) 
      h->h3_term[i].c *= h->h3_aten;

   return(1);
}
/****************************************************************
     y.c  :  Calculate the Pade Approximation of Y(s)
 ****************************************************************/


static double 
eval2(double a, double b, double c, double x)
{
   return(a*x*x + b*x + c);
}

/***
static double approx1(st)
   double st;
{
   double s3, s2, s1;

   s1 = st;
   s2 = s1 * s1;
   s3 = s2 * s1;

   return((s3 + q1*s2 + q2*s1 + q3) / (s3 + p1*s2 + p2*s1 + p3));
}
***/
/***
static double approx2(st)
   double st;
{
   return(1.0 + c1/(st - x1) + c2/(st - x2) + c3/(st - x3));
}
***/

static void 
y_pade(double R, double L, double G, double C, TXLine *h)
{

   /* float RdL, GdC; */
   double RdL, GdC; 

   sqtCdL = sqrt(C / L);
   RdL = R / L;
   GdC = G / C;

   mac(GdC, RdL, &b1, &b2, &b3, &b4, &b5);

   A[0][0] = 1.0 - sqrt(GdC / RdL);
   A[0][1] = b1;
   A[0][2] = b2;
   A[0][3] = -b3;

   A[1][0] = b1;
   A[1][1] = b2;
   A[1][2] = b3;
   A[1][3] = -b4;

   A[2][0] = b2;
   A[2][1] = b3;
   A[2][2] = b4;
   A[2][3] = -b5;

   Gaussian_Elimination1(3);

   p3 = A[0][3];
   p2 = A[1][3];
   p1 = A[2][3];

   q1 = p1 + b1;
   q2 = b1 * p1 + p2 + b2;
   q3 = p3 * sqrt(GdC / RdL);

   find_roots(p1, p2, p3, &x1, &x2, &x3);
   c1 = eval2(q1 - p1, q2 - p2, q3 - p3, x1) / 
		   eval2(3.0, 2.0 * p1, p2, x1);
   c2 = eval2(q1 - p1, q2 - p2, q3 - p3, x2) / 
		   eval2(3.0, 2.0 * p1, p2, x2);
   c3 = eval2(q1 - p1, q2 - p2, q3 - p3, x3) / 
		   eval2(3.0, 2.0 * p1, p2, x3);

   h->sqtCdL = sqtCdL;
   h->h1_term[0].c = c1;
   h->h1_term[1].c = c2;
   h->h1_term[2].c = c3;
   h->h1_term[0].x = x1;
   h->h1_term[1].x = x2;
   h->h1_term[2].x = x3;

}

static int 
Gaussian_Elimination1(int dims)
{
   int i, j, k, dim;
   double f;
   int imax;
   double max;

   dim = dims;

   for (i = 0; i < dim; i++) {
      imax = i;
      max = ABS(A[i][i]);
      for (j = i+1; j < dim; j++)
         if (ABS(A[j][i]) > max) {
	    imax = j;
	    max = ABS(A[j][i]);
	 } 
      if (max < epsi) {
         fprintf(stderr, " can not choose a pivot \n");
         controlled_exit(EXIT_FAILURE);
      }
      if (imax != i)
	 for (k = i; k <= dim; k++) {
	    SWAP(double, A[i][k], A[imax][k]);
	 }
      
      f = 1.0 / A[i][i];
      A[i][i] = 1.0;

      for (j = i+1; j <= dim; j++)
	 A[i][j] *= f;

      for (j = 0; j < dim ; j++) {
	 if (i == j)
	    continue;
	 f = A[j][i];
	 A[j][i] = 0.0;
	 for (k = i+1; k <= dim; k++)
	    A[j][k] -= f * A[i][k];
      }
   }
   return(1);
}

static double root3(double a1_in, double a2_in, double a3_in, double x)
{
   double t1, t2;

   t1 = x*x*x + a1_in*x*x + a2_in*x + a3_in;
   t2 = 3.0*x*x + 2.0*a1_in*x + a2_in;

   return(x - t1 / t2);
}

static int 
div3(double a1_in, double a2_in, double a3_in,
        double x, double *p1_in, double *p2_in)
{
   NG_IGNORE(a2_in);

   *p1_in = a1_in + x;
   *p2_in = -a3_in / x;

   return(1);
}


/****************************************************************
         Calculate the Maclaurin series of F(z)

   F(z) = sqrt((1+az) / (1+bz))
	= 1 + b1 z + b2 z^2 + b3 z^3 + b4 z^4 + b5 z^5 
 ****************************************************************/

/***
static double f3(double a, double b, double z)
{
   double t4, t3, t2, t1;
   double t14, t13, t12, t11;
   double sqt11;

   t1 = 1 / (1.0 + b * z);
   t2 = t1 * t1;
   t3 = t2 * t1;
   t4 = t3 * t1;

   t11 = (1.0 + a * z) * t1;
   t12 = (1.0 + a * z) * t2;
   t13 = (1.0 + a * z) * t3;
   t14 = (1.0 + a * z) * t4;

   sqt11 = sqrt(t11);


   return(
     -0.5 * (-2.0*a*b*t2 + 2.0*b*b*t13) * (a*t1 - b*t12) / (t11*sqt11)
     +3.0/8.0 * (a*t1-b*t12)*(a*t1-b*t12)*(a*t1-b*t12) / (t11*t11*sqt11)
     +0.5 * (4.0*a*b*b*t3 + 2.0*a*b*b*t3 - 6.0*b*b*b*t14) / sqt11
     -0.25 * (-2.0*a*b*t2 + 2.0*b*b*t13) * (a*t1-b*(1.0+a*z)) / 
     (t11*sqt11)
     );
}
***/

/***
static double f2(a, b, z)
   double a, b, z;
{
   double t3, t2, t1;
   double t13, t12, t11;
   double sqt11;

   t1 = 1 / (1.0 + b * z);
   t2 = t1 * t1;
   t3 = t2 * t1;

   t11 = (1.0 + a * z) * t1;
   t12 = (1.0 + a * z) * t2;
   t13 = (1.0 + a * z) * t3;

   sqt11 = sqrt(t11);

   return(
     -0.25 * (a*t1-b*t12) * (a*t1-b*t12) / (t11*sqt11)
     +0.5 * (-2.0*a*b*t2 + 2.0*b*b*t13) / sqt11
   );
}   
***/

static int mac(double at, double bt, double *b1_in, double *b2_in,
        double *b3_in, double *b4_in, double *b5_in)
   /* float at, bt; */
{
   double a, b;
   double y1, y2, y3, y4, y5;

   a = at;
   b = bt;

   y1 = *b1_in = 0.5 * (a - b);
   y2 = 0.5 * (3.0 * b * b - 2.0 * a * b - a * a) * y1 / (a - b);
   y3 = ((3.0 * b * b + a * a) * y1 * y1 + 0.5 * (3.0 * b * b 
	- 2.0 * a * b - a * a) * y2) / (a - b);
   y4 = ((3.0 * b * b - 3.0 * a * a) * y1 * y1 * y1 + (9.0 * b * b
	+ 3.0 * a * a) * y1 * y2 + 0.5 * (3.0 * b * b - 2.0 * a * b
	- a * a) * y3) / (a - b);
   y5 = (12.0 * a * a * y1 * y1 * y1 * y1 + y1 * y1 * y2 * (
	 18.0 * b * b - 18.0 * a * a) + (9.0 * b * b + 3.0 * a * a) *
	 (y2 * y2 + y1 * y3) + (3.0 * b * b + a * a) * y1 * y3 +
	 0.5 * (3.0 * b * b - 2.0 * a * b - a * a) * y4) / (a - b);

   *b2_in = y2 / 2.0;
   *b3_in = y3 / 6.0;
   *b4_in = y4 / 24.0;
   *b5_in = y5 / 120.0;

   return 1;
}


/****************************************************
 exp.c
 ****************************************************/

/***
static double exp_approx1(double st)
{
   double s3, s2, s1;

   s1 = st;
   s2 = s1 * s1;
   s3 = s2 * s1;

   return(exp((double) - st * tau - a0) * 
	(s3 + eq1*s2 + eq2*s1 + eq3) / (s3 + ep1*s2 + ep2*s1 + ep3));
}
***/

static int get_c(double eq1_in, double eq2_in, double eq3_in,
        double ep1_in, double ep2_in, double a, double b, double *cr, double *ci)
{
   double d, n;

   d = (3.0*(a*a-b*b)+2.0*ep1_in*a+ep2_in)*(3.0*(a*a-b*b)+2.0*ep1_in*a+ep2_in);
   d += (6.0*a*b+2.0*ep1_in*b)*(6.0*a*b+2.0*ep1_in*b);
   n = -(eq1_in*(a*a-b*b)+eq2_in*a+eq3_in)*(6.0*a*b+2.0*ep1_in*b);
   n += (2.0*eq1_in*a*b+eq2_in*b)*(3.0*(a*a-b*b)+2.0*ep1_in*a+ep2_in);
   *ci = n/d;
   n = (3.0*(a*a-b*b)+2.0*ep1_in*a+ep2_in)*(eq1_in*(a*a-b*b)+eq2_in*a+eq3_in);
   n += (6.0*a*b+2.0*ep1_in*b)*(2.0*eq1_in*a*b+eq2_in*b);
   *cr = n/d;

   return(1);
}

/***
static double exp_approx2(double st)
{
   if (ifImg) 
      return(1.0 + ec1/(st - ex1) + 2.0*(ec2*(st-ex2)-ec3*ex3) /
	 ((st-ex2)*(st-ex2) + ex3*ex3));
   else 
      return(1.0 + ec1/(st - ex1) + ec2/(st - ex2) + ec3/(st - ex3));
}
***/

static int 
exp_pade(double R, double L, double G, double C, double l, TXLine *h)
{
   double RdL, GdC;
  
   tau = sqrt(L*C);
   RdL = R / L;
   GdC = G / C;
   RG  = R * G;
   RC = R * C;
   GL = G * L;

   {
      double a, b, t;
      double y1, y2, y3, y4, y5, y6;

      a = RdL;
      b = GdC;
      t = tau;

      /*
      y1 = 0.5 * (a + b);
      y2 = a * b - y1 * y1;
      y3 = - a * b * y1 - 2.0 * y1 * y2 + y1 * y1 * y1;
      y4 = 2.0 * a * b * y1 * y1 - a * b * y2 - 2.0 * y2 * y2
          - 2.0 * y1 * y3 + 5.0 * y1 * y1 * y2
          - 2.0 * y1 * y1 * y1 * y1;
      y5 = 6.0 * a * b * (y1 * y2 - y1 * y1 * y1) - a * b * y3
           - 2.0 * y1 * y4
           - 6.0 * y2 * y3 + 12.0 * y2 * y2 * y1 + 7.0 * y1 * y1 * y3
           -10.0 * y1 * y1 * y1 * y2 - 8.0 * y1 * y1 * y1 * y2
           + 6.0 * y1 * y1 * y1 * y1 * y1;
   y6 = 24.0 * a * b * y1 * y1 * y1 * y1 - 36.0 * a * b * y1 * y1 * y2
        + 6.0 * a * b * y2 * y2 + 8.0 * a * b * y1 * y3 - 2.0 * y2 * y4
        - 2.0 * y1 * y5 + 2.0 * y1 * y1 * y4 - a * b * y4 -6.0 * y3 * y3
        + 44.0 * y1 * y2 * y3 + 60.0 * y1 * y1 * y1 * y1 * y2
	-24.0 * y1 * y1 * y1 * y1 * y1 * y1 + 12.0 * y2 * y2 * y2
	-54.0 * y1 * y1 * y2 * y2 + 7.0 * y1 * y1 * y4
	-24.0 * y1 * y1 * y1 * y3 - 24.0 * y1 * y1 * y2 * y2
	-8.0 * y1 * y1 * y1 * y3 + 24.0 * y1 * y1 * y1 * y1 * y2
	- 6.0 * y2 * y4;
	*/

       y1 = 0.5 * (a + b);
       y2 = a * b - y1 * y1;
       y3 = -3.0 * y1 * y2;
       y4 = -3.0 * y2 * y2 - 4.0 * y1 * y3;
       y5 = - 5.0 * y1 * y4 -10.0 * y2 * y3;
       y6 = -10.0 * y3 * y3 - 15.0 * y2 * y4 - 6.0 * y1 * y5;

       a0 = y1 * t;
       a1 = y2 * t * t / 2.0;
       a2 = y3 * t * t * t / 6.0;
       a3 = y4 * t * t * t * t / 24.0;
       a4 = y5 * t * t * t * t * t / 120.0;
       a5 = y6 * t * t * t * t * t * t / 720.0;

   }

   a0 *= l;
   a1 *= l;
   a2 *= l;
   a3 *= l;
   a4 *= l;
   a5 *= l;

   pade(l);

      h->taul = tau * l;
      h->h2_aten = exp(- a0);
      h->h2_term[0].c = ec1;
      h->h2_term[1].c = ec2;
      h->h2_term[2].c = ec3;
      h->h2_term[0].x = ex1;
      h->h2_term[1].x = ex2;
      h->h2_term[2].x = ex3;

   return(ifImg);
}

static int pade(double l)
{
   int i, j;
   double a[6];
   double b[6];

   a[1] = -a1;
   a[2] = -a2;
   a[3] = -a3;
   a[4] = -a4;
   a[5] = -a5;
   
   b[0] = 1.0;
   b[1] = a[1];
   for (i = 2; i <= 5; i++) {
      b[i] = 0.0;
      for (j = 1; j <= i; j++)
	 b[i] += j * a[j] * b[i-j];
      b[i] = b[i] / (double) i;
   }

   AA[0][0] = 1.0 - exp(a0 - l * sqrt(RG));
   AA[0][1] = b[1];
   AA[0][2] = b[2];
   AA[0][3] = -b[3];

   AA[1][0] = b[1];
   AA[1][1] = b[2];
   AA[1][2] = b[3];
   AA[1][3] = -b[4];

   AA[2][0] = b[2];
   AA[2][1] = b[3];
   AA[2][2] = b[4];
   AA[2][3] = -b[5];

   Gaussian_Elimination2(3);

   ep3 = AA[0][3];
   ep2 = AA[1][3];
   ep1 = AA[2][3];

   eq1 = ep1 + b[1];
   eq2 = b[1] * ep1 + ep2 + b[2];
   eq3 = ep3 * exp(a0 - l * sqrt(RG));

   ep3 = ep3 / (tau*tau*tau);
   ep2 = ep2 / (tau*tau);
   ep1 = ep1 / tau;
   eq3 = eq3 / (tau*tau*tau);
   eq2 = eq2 / (tau*tau);
   eq1 = eq1 / tau;
    /*
   printf("factor = %e\n", exp(-a0));
   printf("ep1 = %e ep2 = %e ep3 = %e\n", ep1, ep2, ep3);
     */
   exp_find_roots(ep1, ep2, ep3, &ex1, &ex2, &ex3);
    /*
   printf("roots are %e %e %e \n", ex1, ex2, ex3);
     */
   ec1 = eval2(eq1 - ep1, eq2 - ep2, eq3 - ep3, ex1) / 
		   eval2(3.0, 2.0 * ep1, ep2, ex1);
   if (ifImg) 
      get_c(eq1 - ep1, eq2 - ep2, eq3 - ep3, ep1, ep2, ex2, ex3, &ec2, &ec3);
   else {
      ec2 = eval2(eq1 - ep1, eq2 - ep2, eq3 - ep3, ex2) / 
		   eval2(3.0, 2.0 * ep1, ep2, ex2);
      ec3 = eval2(eq1 - ep1, eq2 - ep2, eq3 - ep3, ex3) / 
		   eval2(3.0, 2.0 * ep1, ep2, ex3);
   }
   return (1);
}

static int 
Gaussian_Elimination2(int dims)
{
   int i, j, k, dim;
   double f;
   double max;
   int imax;

   dim = dims;

   for (i = 0; i < dim; i++) {
      imax = i;
      max = ABS(AA[i][i]);
      for (j = i+1; j < dim; j++)
         if (ABS(AA[j][i]) > max) {
            imax = j;
            max = ABS(AA[j][i]);
         }
      if (max < epsi2) {
         fprintf(stderr, " can not choose a pivot \n");
         controlled_exit(EXIT_FAILURE);
      }
      if (imax != i)
	 for (k = i; k <= dim; k++) {
	    SWAP(double, AA[i][k], AA[imax][k]);
	 }
      
      f = 1.0 / AA[i][i];
      AA[i][i] = 1.0;

      for (j = i+1; j <= dim; j++)
	 AA[i][j] *= f;

      for (j = 0; j < dim ; j++) {
	 if (i == j)
	    continue;
	 f = AA[j][i];
	 AA[j][i] = 0.0;
	 for (k = i+1; k <= dim; k++)
	    AA[j][k] -= f * AA[i][k];
      }
   }
   return(1);
}

/***
static int 
exp_div3(double a1, double a2, double a3, double x, 
         double *p1, double *p2)
   {
   *p1 = a1 + x;
   *p2 = - a3 / x;

   return(1);
}
***/

/***
 ***/

static int exp_find_roots(double a1_in, double a2_in, double a3_in,
        double *ex1_in, double *ex2_in, double *ex3_in)
{
   double x, t;
   double p, q;

   q = (a1_in*a1_in-3.0*a2_in) / 9.0;
   p = (2.0*a1_in*a1_in*a1_in-9.0*a1_in*a2_in+27.0*a3_in) / 54.0;
   t = q*q*q - p*p;
   if (t >= 0.0) {
      t = acos(p /(q * sqrt(q)));
      x = -2.0*sqrt(q)*cos(t / 3.0) - a1_in/3.0;
   } else {
      if (p > 0.0) {
         t = pow(sqrt(-t)+p, 1.0 / 3.0);
         x = -(t + q / t) - a1_in/3.0;
      } else if (p == 0.0) {
         x = -a1_in/3.0;
      } else {
         t = pow(sqrt(-t)-p, 1.0 / 3.0);
         x = (t + q / t) - a1_in/3.0;
      }
   }
    {
        double ex1a;
        int i = 0;
        ex1a = x;
        for (t = root3(a1_in, a2_in, a3_in, x); ABS(t-x) > 5.0e-4;
                t = root3(a1_in, a2_in, a3_in, x)) {
            if (++i == 32) {
                x = ex1a;
                break;
            }
            else {
                x = t;
            }
        }
    }
   /***
   x = a1;
   for (t = root3(a1, a2, a3, x); ABS(t-x) > epsi2; 
			    t = root3(a1, a2, a3, x)) {
      x = t;
      i++;
      if (i > 1000) {
         x = 0.5 * (x + root3(a1, a2, a3, x));
         j++;
         if (j == 3)
            break;
         i = 0;
      }
   }
    ***/
   *ex1_in = x;
   div3(a1_in, a2_in, a3_in, x, &a1_in, &a2_in);

   t = a1_in * a1_in - 4.0 * a2_in;
   if (t < 0) {
      ifImg = 1;
      printf("***** Two Imaginary Roots.\n");
      *ex3_in = 0.5 * sqrt(-t);
      *ex2_in = -0.5 * a1_in;
   } else {
      ifImg = 0;
      t *= 1.0e-16;
      t = sqrt(t)*1.0e8;
      if (a1_in >= 0.0)
         *ex2_in = t = -0.5 * (a1_in + t);
      else
         *ex2_in = t = -0.5 * (a1_in - t);
      *ex3_in = a2_in / t;
      /*
      *ex2 = 0.5 * (-a1 + t);
      *ex3 = 0.5 * (-a1 - t);
       */
   }

   return(1);
}
static NDnamePt
insert_ND(char *name, NDnamePt *ndn)
{
   int       cmp;
   NDnamePt  p;

   if (*ndn == NULL) {
      p = *ndn = TMALLOC(NDname, 1);
      p->nd = NULL;
      p->right = p->left = NULL;
      strcpy(p->id, name);
      return(p);
   }
   cmp = strcmp((*ndn)->id, name);
   if (cmp == 0) 
      return(*ndn);
   else {
      if (cmp < 0)
         return(insert_ND(name, &((*ndn)->left)));
      else
         return(insert_ND(name, &((*ndn)->right)));
   }
}


static NODE 
*insert_node(char *name)
{
   NDnamePt n;
   NODE    *p;

   n = insert_ND(name, &ndn_btree);
   if (n->nd == NULL) {
      p = NEW_node();
      p->name = n;
      n->nd = p;
      p->next = node_tab;
      node_tab = p;
      return(p);
   } else
      return(n->nd);
}

static NODE
*NEW_node(void)
{
   NODE *n;

   n = TMALLOC(NODE, 1);
   n->mptr = NULL;
   n->gptr = NULL;
   n->cptr = NULL;
   n->rptr = NULL;
   n->tptr = NULL;
   n->cplptr = NULL;
   n->rlptr = NULL;
   n->ddptr = NULL;
   n->cvccsptr = NULL;
   n->vccsptr = NULL;
   n->CL = 0.0;
   n->V = n->dv = 0.0;
   n->gsum = n->cgsum = 0;
   n->is   = 0;
   n->tag  = 0;
   n->flag = 0;
   n->region = NULL;
   n->ofile = NULL;
   n->dvtag = 0;

   return(n);
}

static int find_roots(double a1_in, double a2_in, double a3_in,
        double *x1r, double *x2r, double *x3r)
{
   double x, t;
   double p, q;

   q = (a1_in*a1_in-3.0*a2_in) / 9.0;
   p = (2.0*a1_in*a1_in*a1_in-9.0*a1_in*a2_in+27.0*a3_in) / 54.0;
   t = q*q*q - p*p;
   if (t >= 0.0) {
      t = acos(p /(q * sqrt(q)));
      x = -2.0*sqrt(q)*cos(t / 3.0) - a1_in/3.0;
   } else {
      if (p > 0.0) {
	 t = pow(sqrt(-t)+p, 1.0 / 3.0);
         x = -(t + q / t) - a1_in/3.0;
      } else if (p == 0.0) {
	 x = -a1_in/3.0;
      } else {
 	 t = pow(sqrt(-t)-p, 1.0 / 3.0);
	 x = (t + q / t) - a1_in/3.0;
      }
   }
   {
      double x_backup = x;
      int i = 0;
      for (t = root3(a1_in, a2_in, a3_in, x); ABS(t-x) > 5.0e-4;
			t = root3(a1_in, a2_in, a3_in, x))
         if (++i == 32) {
	    x = x_backup;
	    break;
	 } else
	    x = t;
   }
   /*
   x = a1;
   i = 0;
   j = 0;
   for (t = root3(a1, a2, a3, x); ABS(t-x) > epsi; 
			    t = root3(a1, a2, a3, x)) {
      x = t;
      i++;
      if (i > 1000) {
	 x = 0.5 * (x + root3(a1, a2, a3, x));
	 j++;
	 if (j == 3)
	    break;
	 i = 0;
      }
   }
    */

   *x1r = x;
   div3(a1_in, a2_in, a3_in, x, &a1_in, &a2_in);

   t = a1_in * a1_in - 4.0 * a2_in;
   if (t < 0) {
      printf("***** Two Imaginary Roots in Characteristic Admittance.\n");
      controlled_exit(EXIT_FAILURE);
   }

   t *= 1.0e-18;
   t = sqrt(t) * 1.0e9;
   if (a1_in >= 0.0)
      *x2r = t = -0.5 * (a1_in + t);
   else
      *x2r = t = -0.5 * (a1_in - t);
   *x3r = a2_in / t;
   /*
   *x2 = 0.5 * (-a1 + t);
   *x3 = 0.5 * (-a1 - t);
    */
   return(1);
}

int
TXLdevDelete(GENinstance* inst)
{
    VI_list_txl *tmplist, *prevlist;
    TXLinstance* here = (TXLinstance*)inst;
    if (here->txline2)
        tfree(here->txline2);
    if (here->txline) {
        prevlist = tmplist = here->txline->vi_head;
        while(tmplist) {
           tmplist = tmplist->next;
           tfree(prevlist);
           prevlist = tmplist;
        }
        tfree(here->txline);
    }
    return OK;
}
