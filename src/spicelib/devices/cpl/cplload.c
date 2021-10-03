/**********
Copyright 1992 Regents of the University of California.	 All rights
reserved.
Author:	1992 Charles Hough
**********/


#include "ngspice/ngspice.h"
#include "cpldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "cplhash.h"

VI_list	*pool_vi;
static double ratio[MAX_CP_TX_LINES];
static VI_list *new_vi(void);
static void free_vi(VI_list*);
static int get_pvs_vi(int t1, int t2,
                      CPLine *cp,
                      double  v1_i[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
                      double  v2_i[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
                      double  i1_i[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
                      double  i2_i[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
                      double  v1_o[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
                      double  v2_o[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
                      double  i1_o[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
                      double  i2_o[MAX_CP_TX_LINES][MAX_CP_TX_LINES]);
static int update_cnv(CPLine*, double);
static int add_new_vi(CPLinstance*, CKTcircuit*, int);
static int right_consts(CPLinstance*, CPLine*, int, int, double, double, int*, int*, CKTcircuit*);
static int update_delayed_cnv(CPLine*, double);
static int multC(double, double, double, double, double*, double*);
static int expC(double,	double,	double,	double*, double*);
static int divC(double,	double,	double,	double,	double*, double*);
static void update_cnv_a(TMS*, double, double, double, double, double, double, double);
static void copy_cp(CPLine*, CPLine*);

/*ARGSUSED*/
int
CPLload(GENmodel *inModel, CKTcircuit *ckt)
{
	CPLmodel *model	= (CPLmodel *)inModel;
	CPLinstance *here;
	CPLine *cp, *cp2;
	int *k_p, *l_p;
	int time, time2;
	double h, h1, f;
/*	int hint; never used */
	double hf;
	NODE *nd;
	double v, v1, g;
	int i, j, k, l;
	int cond1;
	int noL, m, p, q;
	CKTnode	*node;
	VI_list	*vi, *vi_before;
	int before, delta;
	int resindex;

	double gmin;	 /* dc solution	*/

	h = ckt->CKTdelta;
	h1 = 0.5 * h;
	time2 =	(int) (ckt->CKTtime * 1e12);
/*	hint = (int)(h * 1e12); never used */
	hf = h * 1e12;
	time = (int) ((ckt->CKTtime - ckt->CKTdelta) * 1e12);

	cond1= ckt->CKTmode & MODEDC;

	gmin = 0.1 * ckt->CKTgmin;     /* dc solution */

	for( ; model !=	NULL; model = CPLnextModel(model)) {
		for (here = CPLinstances(model); here != NULL ;
			here=CPLnextInstance(here)) {

			cp = here->cplines;

			noL = cp->noL =	here->dimension;

			for(m =	0 ; m <	noL ; m++)     /* dc solution */
			{
			   *here->CPLposPosPtr[m] += gmin;
			   *here->CPLnegNegPtr[m] += gmin;
			   *here->CPLnegPosPtr[m] += gmin;
			   *here->CPLposNegPtr[m] += gmin;
			}

			if (cond1 || cp->vi_head == NULL) continue;

			if (cp->vi_tail->time >	time) {
				time = cp->vi_tail->time;
/*				hint = time2 - time; never used */
			}

			before = cp->vi_tail->time;
			vi_before = cp->vi_tail;

			if (time > cp->vi_tail->time) {

				copy_cp(cp, here->cplines2);
				add_new_vi(here, ckt, time);
				delta =	time - before;

				for (m = 0; m <	noL; m++) {
					nd = cp->in_node[m];
					v = vi_before->v_i[m];
					v1 = nd->V = cp->vi_tail->v_i[m];
					nd->dv = (v1 - v) / delta;
				}
				for (m = 0; m <	noL; m++) {
					nd = cp->out_node[m];
					v = vi_before->v_o[m];
					v1 = nd->V = cp->vi_tail->v_o[m];
					nd->dv = (v1 - v) / delta;
				}

				update_cnv(cp, delta);
				if (cp->ext) update_delayed_cnv(cp, delta);
			}
		}
	}

    model = (CPLmodel *)inModel;
    /*	loop through all the models */
    for( ; model != NULL; model	= CPLnextModel(model))	{

	/* loop	through	all the	instances of the model */
	for (here = CPLinstances(model); here != NULL ;
			here=CPLnextInstance(here)) {

			double mintaul = 123456789.0;

			cp = here->cplines;
			cp2 = here->cplines2;

			for (i = 0; i <	cp->noL; i++) {
				if (mintaul > cp->taul[i]) mintaul = cp->taul[i];
			}
			if (mintaul < hf) {

				fprintf(stderr,	"your time step	was too large for CPL tau.\n");
/*				fprintf(stderr,	"please	decrease max time step in .tran	card.\n");
				fprintf(stderr,	".tran tstep tstop tstart tmax.\n");
				fprintf(stderr,	"make tmax smaller than	%e and try again.\n",
				mintaul	* 1e-12);
				return (1111);
*/
				fprintf(stderr,	"tmax is now set to	%e.\n", 0.9 * mintaul * 1e-12);
				ckt->CKTmaxStep = 0.9 * mintaul * 1e-12;

			}

			noL = cp->noL =	here->dimension;
			if (cond1) {
				resindex = 0;
				for (m = 0; m <	noL; m++) {
					if (here->CPLlengthGiven)
						g = model->Rm[resindex]	* here->CPLlength;
					else g = model->Rm[resindex] * CPLmodPtr(here)->length;
					*(here->CPLposIbr1Ptr[m]) += 1.0;
					*(here->CPLnegIbr2Ptr[m]) += 1.0;
					*(here->CPLibr1Ibr1Ptr[m])	+= 1.0;
					*(here->CPLibr1Ibr2Ptr[m][m]) += 1.0;
					*(here->CPLibr2PosPtr[m][m]) += 1.0;
					*(here->CPLibr2NegPtr[m][m]) -= 1.0;
					*(here->CPLibr2Ibr1Ptr[m][m]) -= g;
					resindex = resindex + noL - m;
				}
				continue;
			}

			/* dc setup */
			if (here->CPLdcGiven ==	0 && !cond1) {
				for (i = 0; i <	cp->noL; i++) {
					nd = cp->in_node[i];
					for(node = ckt->CKTnodes;node; node = node->next) {
						if (strcmp(nd->name->id, node->name) ==	0) {
							cp->dc1[i] = ckt->CKTrhsOld[node->number];
							cp2->dc1[i] = nd->V = cp->dc1[i];
							break;
						}
					}
					nd = cp->out_node[i];
					for(node = ckt->CKTnodes;node; node = node->next) {
						if (strcmp(nd->name->id, node->name) ==	0) {
							cp->dc2[i] = ckt->CKTrhsOld[node->number];
							cp2->dc2[i] = nd->V = cp->dc2[i];
							break;
						}
					}
				}
				here->CPLdcGiven = 1;

				vi = new_vi();
				vi->time = 0;
				{
				for (i = 0; i <	cp->noL; i++) {
					for (j = 0; j <	cp->noL; j++) {
						TMS *tms;
						double a, b;

						tms = cp->h1t[i][j];
						if (!tms) {
							fprintf(stderr, "\nError in CPL %s: Forbidden combination of model parameters!\n", here->gen.GENname);
							controlled_exit(1);
						}
						if (tms->ifImg)	{
							tms->tm[0].cnv_i = - cp->dc1[j]	*
								tms->tm[0].c / tms->tm[0].x;
							tms->tm[0].cnv_o = - cp->dc2[j]	*
								tms->tm[0].c / tms->tm[0].x;
							divC(tms->tm[1].c, tms->tm[2].c,
								tms->tm[1].x, tms->tm[2].x, &a,	&b);
							tms->tm[1].cnv_i = - cp->dc1[j]	* a;
							tms->tm[1].cnv_o = - cp->dc2[j]	* a;
							tms->tm[2].cnv_i = - cp->dc1[j]	* b;
							tms->tm[2].cnv_o = - cp->dc2[j]	* b;
						} else
							for (k = 0; k <	3; k++)	{
								tms->tm[k].cnv_i = - cp->dc1[j]	*
									tms->tm[k].c / tms->tm[k].x;
								tms->tm[k].cnv_o = - cp->dc2[j]	*
									tms->tm[k].c / tms->tm[k].x;
							}

						for (l = 0; l <	cp->noL; l++) {
							tms = cp->h2t[i][j][l];
							if (!tms) {
								fprintf(stderr, "\nError in CPL %s: Forbidden combination of model parameters!\n", here->gen.GENname);
								controlled_exit(1);
							}
							for (k = 0; k <	3; k++)	{
								tms->tm[k].cnv_i = 0.0;
								tms->tm[k].cnv_o = 0.0;
							}
						}
						for (l = 0; l <	cp->noL; l++) {
							tms = cp->h3t[i][j][l];
							if (!tms) {
								fprintf(stderr, "\nError in CPL %s: Forbidden combination of model parameters!\n", here->gen.GENname);
								controlled_exit(1);
							}
							if (tms->ifImg)	{
								tms->tm[0].cnv_i = - cp->dc1[j]	*
									tms->tm[0].c / tms->tm[0].x;
								tms->tm[0].cnv_o = - cp->dc2[j]	*
									tms->tm[0].c / tms->tm[0].x;
								divC(tms->tm[1].c, tms->tm[2].c,
									tms->tm[1].x, tms->tm[2].x, &a,	&b);
								tms->tm[1].cnv_i = - cp->dc1[j]	* a;
								tms->tm[1].cnv_o = - cp->dc2[j]	* a;
								tms->tm[2].cnv_i = - cp->dc1[j]	* b;
								tms->tm[2].cnv_o = - cp->dc2[j]	* b;
							} else
								for (k = 0; k <	3; k++)	{
									tms->tm[k].cnv_i = - cp->dc1[j]	*
										tms->tm[k].c / tms->tm[k].x;
									tms->tm[k].cnv_o = - cp->dc2[j]	*
										tms->tm[k].c / tms->tm[k].x;
								}
						}
					}

					for (j = 0; j <	cp->noL; j++) {
						vi->i_i[j] = vi->i_o[j]	= 0.0;
						vi->v_i[j] = cp->dc1[j];
						vi->v_o[j] = cp->dc2[j];
					}
				}

				vi->next = NULL;
				cp->vi_tail = vi;
				cp->vi_head = vi;
				cp2->vi_tail = vi;
				cp2->vi_head = vi;

				}
			}

			for (m = 0; m <	noL; m++) {
				*(here->CPLibr1Ibr1Ptr[m])	= -1.0;
				*(here->CPLibr2Ibr2Ptr[m])	= -1.0;
			}

			for (m = 0; m <	noL; m++) {
				*(here->CPLposIbr1Ptr[m]) = 1.0;
				*(here->CPLnegIbr2Ptr[m]) = 1.0;
			}

			for (m = 0; m <	noL; m++) {
				for (p = 0; p <	noL; p++) {
					*(here->CPLibr1PosPtr[m][p]) =
						cp->h1t[m][p]->aten + h1 * cp->h1C[m][p];
					*(here->CPLibr2NegPtr[m][p]) =
						cp->h1t[m][p]->aten + h1 * cp->h1C[m][p];
				}
			}

			k_p = here->CPLibr1;
			l_p = here->CPLibr2;

			copy_cp(cp2, cp);

			if (right_consts(here,cp2, time,time2,h,h1,k_p,l_p,ckt)) {
				cp2->ext = 1;
				for (q = 0; q <	noL; q++) {
					cp->ratio[q] = ratio[q];
					if (ratio[q] > 0.0) {
						for (m = 0; m <	noL; m++) {
							for (p = 0; p <	noL; p++) {


				if (cp->h3t[m][p][q]) {
				f = ratio[q] * (h1 * cp->h3C[m][p][q] +
					cp->h3t[m][p][q]->aten);
						*(here->CPLibr1NegPtr[m][p]) = -f;
						*(here->CPLibr2PosPtr[m][p]) = -f;
				}
				if (cp->h2t[m][p][q]) {
				f = ratio[q] * (h1 * cp->h2C[m][p][q] +
					cp->h2t[m][p][q]->aten);
						*(here->CPLibr1Ibr2Ptr[m][p]) = -f;
						*(here->CPLibr2Ibr1Ptr[m][p]) = -f;
				}

						}
					}
					}
				}
			}
			else cp->ext = 0;
	}
	}

    return(OK);
}

static void
copy_cp(CPLine *new, CPLine *old)
{
	int i, j, k, l,	m;
	VI_list	*temp;

	new->noL = m = old->noL;
	new->ext = old->ext;
	for (i = 0; i <	m; i++)	{
		new->ratio[i] =	old->ratio[i];
		new->taul[i] = old->taul[i];

		for (j = 0; j <	m; j++)	{
			if (new->h1t[i][j] == NULL) {
				TMS *nptr = new->h1t[i][j] = TMALLOC(TMS, 1);
				memsaved(nptr);
			}
			new->h1t[i][j]->ifImg =	old->h1t[i][j]->ifImg;
			new->h1t[i][j]->aten = old->h1t[i][j]->aten;
			new->h1C[i][j] = old->h1C[i][j];

			for (k = 0; k <	3; k++)	{
				new->h1t[i][j]->tm[k].c	= old->h1t[i][j]->tm[k].c;
				new->h1t[i][j]->tm[k].x	= old->h1t[i][j]->tm[k].x;
				new->h1t[i][j]->tm[k].cnv_i = old->h1t[i][j]->tm[k].cnv_i;
				new->h1t[i][j]->tm[k].cnv_o = old->h1t[i][j]->tm[k].cnv_o;
				new->h1e[i][j][k] = old->h1e[i][j][k];
			}
			for (l = 0; l <	m; l++)	{
				if (new->h2t[i][j][l] == NULL) {
					TMS *nptr = new->h2t[i][j][l] = TMALLOC(TMS, 1);
					memsaved(nptr);
				}
				new->h2t[i][j][l]->ifImg = old->h2t[i][j][l]->ifImg;
				new->h2t[i][j][l]->aten	= old->h2t[i][j][l]->aten;
				new->h2C[i][j][l] = old->h2C[i][j][l];
				new->h3C[i][j][l] = old->h3C[i][j][l];
				for (k = 0; k <	3; k++)	{
					new->h2t[i][j][l]->tm[k].c = old->h2t[i][j][l]->tm[k].c;
					new->h2t[i][j][l]->tm[k].x = old->h2t[i][j][l]->tm[k].x;
					new->h2t[i][j][l]->tm[k].cnv_i
						= old->h2t[i][j][l]->tm[k].cnv_i;
					new->h2t[i][j][l]->tm[k].cnv_o
						= old->h2t[i][j][l]->tm[k].cnv_o;
				}

				if (new->h3t[i][j][l] == NULL) {
					TMS* nptr = new->h3t[i][j][l] = TMALLOC(TMS, 1);
                    memsaved(nptr);
				}
				new->h3t[i][j][l]->ifImg = old->h3t[i][j][l]->ifImg;
				new->h3t[i][j][l]->aten	= old->h3t[i][j][l]->aten;
				for (k = 0; k <	3; k++)	{
					new->h3t[i][j][l]->tm[k].c = old->h3t[i][j][l]->tm[k].c;
					new->h3t[i][j][l]->tm[k].x = old->h3t[i][j][l]->tm[k].x;
					new->h3t[i][j][l]->tm[k].cnv_i
						= old->h3t[i][j][l]->tm[k].cnv_i;
					new->h3t[i][j][l]->tm[k].cnv_o
						= old->h3t[i][j][l]->tm[k].cnv_o;
				}
			}
		}
	}


	while (new->vi_head->time < old->vi_head->time)	{
		temp = new->vi_head;
		new->vi_head = new->vi_head->next;
		free_vi(temp);
	}
}


static int
right_consts(CPLinstance *here,	CPLine *cp, int	t, int time, double h, double h1,
	     int *l1, int *l2, CKTcircuit *ckt)
{
   int i, j, k,	l;
   double e;
   double ff[MAX_CP_TX_LINES], gg[MAX_CP_TX_LINES];
   double v1_i[MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   double v2_i[MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   double i1_i[MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   double i2_i[MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   double v1_o[MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   double v2_o[MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   double i1_o[MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   double i2_o[MAX_CP_TX_LINES][MAX_CP_TX_LINES];
   int ext;
   int noL;

    NG_IGNORE(here);

   noL = cp->noL;

   for (j = 0; j < noL;	j++) {
       double ff1;

      ff[j] = 0.0;
      gg[j] = 0.0;
      for (k = 0; k < noL; k++)
     if	(cp->h1t[j][k])	{
	if (cp->h1t[j][k]->ifImg) {
	   double er, ei, a, b,	a1, b1;
	   TMS *tms;
	   tms = cp->h1t[j][k];
	   cp->h1e[j][k][0] = e	= exp(tms->tm[0].x * h);
	   expC(tms->tm[1].x, tms->tm[2].x, h, &er, &ei);
	   cp->h1e[j][k][1] = er;
	   cp->h1e[j][k][2] = ei;

	   ff1 = tms->tm[0].c *	e * h1;
	   ff[j]  -= tms->tm[0].cnv_i *	e;
	   gg[j]  -= tms->tm[0].cnv_o *	e;
	   ff[j]  -= ff1 * cp->in_node[k]->V;
	   gg[j]  -= ff1 * cp->out_node[k]->V;

	   multC(tms->tm[1].c, tms->tm[2].c, er, ei, &a1, &b1);
	   multC(tms->tm[1].cnv_i, tms->tm[2].cnv_i, er, ei, &a, &b);
	   ff[j] -= 2.0	* (a1 *	h1 * cp->in_node[k]->V + a);
	   multC(tms->tm[1].cnv_o, tms->tm[2].cnv_o, er, ei, &a, &b);
	   gg[j] -= 2.0	* (a1 *	h1 * cp->out_node[k]->V	+ a);
	} else {
	   ff1 = 0.0;
	   for (i = 0; i < 3; i++) {
	  cp->h1e[j][k][i] = e = exp(cp->h1t[j][k]->tm[i].x * h);
	  ff1 -= cp->h1t[j][k]->tm[i].c	* e;
	  ff[j]	 -= cp->h1t[j][k]->tm[i].cnv_i * e;
	  gg[j]	 -= cp->h1t[j][k]->tm[i].cnv_o * e;
	   }
	   ff[j] += ff1	* h1 * cp->in_node[k]->V;
	   gg[j] += ff1	* h1 * cp->out_node[k]->V;
	}
     }
   }

   ext = get_pvs_vi(t, time, cp, v1_i, v2_i, i1_i, i2_i,
	  v1_o,	v2_o, i1_o, i2_o);

   for (j  = 0;	j < noL; j++) {	      /**  current eqn	**/
       TERM *tm;

      for (k = 0; k < noL; k++)	      /**  node	voltage	 **/
     for (l = 0; l < noL; l++)	  /**  different mode  **/
	if (cp->h3t[j][k][l]) {
	   if (cp->h3t[j][k][l]->ifImg)	{
	  double er, ei, a, b, a1, b1, a2, b2;
	  TMS *tms;
	  tms =	cp->h3t[j][k][l];
	  expC(tms->tm[1].x, tms->tm[2].x, h, &er, &ei);
	  a2 = h1 * tms->tm[1].c;
	  b2 = h1 * tms->tm[2].c;
	  a = tms->tm[1].cnv_i;
	  b = tms->tm[2].cnv_i;
	   multC(a, b, er, ei, &a, &b);
	  multC(a2, b2,	v1_i[l][k] * er + v2_i[l][k], v1_i[l][k] * ei, &a1, &b1);
	  tms->tm[1].cnv_i = a + a1;
	  tms->tm[2].cnv_i = b + b1;
	  a = tms->tm[1].cnv_o;
	  b = tms->tm[2].cnv_o;
	  multC(a, b, er, ei, &a, &b);
	  multC(a2, b2,	v1_o[l][k] * er + v2_o[l][k], v1_o[l][k] * ei, &a1, &b1);
	  tms->tm[1].cnv_o = a + a1;
	  tms->tm[2].cnv_o = b + b1;
	  tm = &(tms->tm[0]);
	  e = exp(tm->x * h);
	  tm->cnv_i = tm->cnv_i	* e + h1 * tm->c *
		  (v1_i[l][k] *	e + v2_i[l][k]);
	  tm->cnv_o = tm->cnv_o	* e + h1 * tm->c *
		  (v1_o[l][k] *	e + v2_o[l][k]);
	  ff[j]	+= tms->aten * v2_o[l][k] + tm->cnv_o +
	     2.0 * tms->tm[1].cnv_o;
	     gg[j] += tms->aten	* v2_i[l][k] + tm->cnv_i +
	    2.0	* tms->tm[1].cnv_i;
	   } else {
	  for (i = 0; i	< 3; i++) {   /**     3	poles	  **/
	     tm	= &(cp->h3t[j][k][l]->tm[i]);
	     e =  exp(tm->x * h);
	     tm->cnv_i = tm->cnv_i * e + h1 * tm->c * (v1_i[l][k] * e +	v2_i[l][k]);
	     tm->cnv_o = tm->cnv_o * e + h1 * tm->c * (v1_o[l][k] * e +	v2_o[l][k]);
	     ff[j] += tm->cnv_o;
	     gg[j] += tm->cnv_i;
	  }

	  ff[j]	+= cp->h3t[j][k][l]->aten * v2_o[l][k];
	  gg[j]	+= cp->h3t[j][k][l]->aten * v2_i[l][k];
	   }
	}
      for (k = 0; k < noL; k++)	      /**  line	current	 **/
     for (l = 0; l < noL; l++)	  /**  different mode  **/
	if (cp->h2t[j][k][l]) {
	   if (cp->h2t[j][k][l]->ifImg)	{
	  double er, ei, a, b, a1, b1, a2, b2;
	  TMS *tms;
	  tms =	cp->h2t[j][k][l];
	  expC(tms->tm[1].x, tms->tm[2].x, h, &er, &ei);
	  a2 = h1 * tms->tm[1].c;
	  b2 = h1 * tms->tm[2].c;
	  a = tms->tm[1].cnv_i;
	  b = tms->tm[2].cnv_i;
	  multC(a, b, er, ei, &a, &b);
	  multC(a2, b2,	i1_i[l][k] * er + i2_i[l][k], i1_i[l][k] * ei, &a1, &b1);
	  tms->tm[1].cnv_i = a + a1;
	  tms->tm[2].cnv_i = b + b1;
	  a = tms->tm[1].cnv_o;
	  b = tms->tm[2].cnv_o;
	  multC(a, b, er, ei, &a, &b);
	  multC(a2, b2,	i1_o[l][k] * er + i2_o[l][k], i1_o[l][k] * ei, &a1, &b1);
	  tms->tm[1].cnv_o = a + a1;
	  tms->tm[2].cnv_o = b + b1;
	  tm = &(tms->tm[0]);
	  e = exp(tm->x * h);
	  tm->cnv_i = tm->cnv_i	* e + h1 * tm->c *
		  (i1_i[l][k] *	e + i2_i[l][k]);
	  tm->cnv_o = tm->cnv_o	* e + h1 * tm->c *
		  (i1_o[l][k] *	e + i2_o[l][k]);
	  ff[j]	+= tms->aten * i2_o[l][k] + tm->cnv_o +
	     2.0 * tms->tm[1].cnv_o;
	     gg[j] += tms->aten	* i2_i[l][k] + tm->cnv_i +
	    2.0	* tms->tm[1].cnv_i;
	   } else {
	  for (i = 0; i	< 3; i++) {   /**     3	poles	  **/
	     tm	= &(cp->h2t[j][k][l]->tm[i]);
	     e =  exp(tm->x * h);
	     tm->cnv_i = tm->cnv_i * e + h1 * tm->c * (i1_i[l][k] * e +	i2_i[l][k]);
	     tm->cnv_o = tm->cnv_o * e + h1 * tm->c * (i1_o[l][k] * e +	i2_o[l][k]);
	     ff[j] += tm->cnv_o;
	     gg[j] += tm->cnv_i;
	  }

	  ff[j]	+= cp->h2t[j][k][l]->aten * i2_o[l][k];
	  gg[j]	+= cp->h2t[j][k][l]->aten * i2_i[l][k];
	   }
	}
   }

   for (i = 0; i < noL;	i++) {
	  *(ckt->CKTrhs	+ l1[i]) = ff[i];
	  *(ckt->CKTrhs	+ l2[i]) = gg[i];
   }

   return(ext);
}

static int
update_cnv(CPLine *cp, double h)
{
   int i, j, k;
   int noL;
   double ai, bi, ao, bo;
   double e, t;
   TMS *tms;
   TERM	*tm;

   noL = cp->noL;
   for (j = 0; j < noL;	j++)
      for (k = 0; k < noL; k++)	{
     ai	= cp->in_node[k]->V;
     ao	= cp->out_node[k]->V;
     bi	= cp->in_node[k]->dv;
     bo	= cp->out_node[k]->dv;

     if	(cp->h1t[j][k])	{
	if (cp->h1t[j][k]->ifImg) {
	   tms = cp->h1t[j][k];
	   if (tms == NULL)
	  continue;
	   tm =	&(tms->tm[0]);
	   e = cp->h1e[j][k][0];
	   t = tm->c / tm->x;
	   update_cnv_a(tms, h,	ai, ao,	ai - bi	* h, ao	- bo * h,
	      cp->h1e[j][k][1],	cp->h1e[j][k][2]);
	   bi *= t;
	   bo *= t;
	   tm->cnv_i = (tm->cnv_i - bi*h) * e +	(e - 1.0)*(ai*t	+
	      1.0e+12*bi/tm->x);
	   tm->cnv_o = (tm->cnv_o - bo*h) * e +	(e - 1.0)*(ao*t	+
	      1.0e+12*bo/tm->x);
	} else
	   for (i = 0; i < 3; i++) {
	  tm = &(cp->h1t[j][k]->tm[i]);

	  e = cp->h1e[j][k][i];

	  t = tm->c / tm->x;
	  bi *=	t;
	  bo *=	t;

	  tm->cnv_i = (tm->cnv_i - bi*h) * e + (e - 1.0)*(ai*t + 1.0e+12*bi/tm->x);
	  tm->cnv_o = (tm->cnv_o - bo*h) * e + (e - 1.0)*(ao*t + 1.0e+12*bo/tm->x);
	   }
     }
      }
   return 0;
}

static VI_list
*new_vi(void)
{
   VI_list *q;

	if (pool_vi) {
		q = pool_vi;
	    pool_vi = pool_vi->pool;
		return(q);
	}
	else {
		VI_list* nptr = TMALLOC(VI_list, 1);
		memsaved(nptr);
		return(nptr);
	}
}

static void
free_vi(VI_list	*q)
{
	q->pool	= pool_vi;
	pool_vi	= q;
}


static int
add_new_vi(CPLinstance *here, CKTcircuit *ckt, int time)
{
   VI_list *vi;
   int i, noL;
   CPLine *cp, *cp2;

   cp =	here->cplines;
   cp2 = here->cplines2;

   vi =	new_vi();
   vi->time = time;
   noL = cp->noL;
   for (i = 0; i < noL;	i++) {
	  /*
      vi->v_i[i] = cp->in_node[i]->V;
      vi->v_o[i] = cp->out_node[i]->V;
	  */
      vi->v_i[i] = *(ckt->CKTrhsOld + here->CPLposNodes[i]);
      vi->v_o[i] = *(ckt->CKTrhsOld + here->CPLnegNodes[i]);
      vi->i_i[i] = *(ckt->CKTrhsOld + here->CPLibr1[i]);
      vi->i_o[i] = *(ckt->CKTrhsOld + here->CPLibr2[i]);
   }
   cp->vi_tail->next = vi;
   cp2->vi_tail->next =	vi;
   vi->next = NULL;
   cp->vi_tail = vi;
   cp2->vi_tail	= vi;

   return(1);
}


static int
get_pvs_vi(int t1, int t2,
   CPLine *cp,
   double v1_i[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
   double v2_i[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
   double i1_i[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
   double i2_i[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
   double v1_o[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
   double v2_o[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
   double i1_o[MAX_CP_TX_LINES][MAX_CP_TX_LINES],
   double i2_o[MAX_CP_TX_LINES][MAX_CP_TX_LINES])
{
   double ta[MAX_CP_TX_LINES], tb[MAX_CP_TX_LINES];
   VI_list *vi,	*vi1;
   double f;
   int i, j;
   int	mini = -1;
   double minta	= 123456789.0;
   int ext = 0;
   int noL;

   noL = cp->noL;

   for (i = 0; i < noL;	i++) {
      ta[i] = t1 - cp->taul[i];
      tb[i] = t2 - cp->taul[i];
      if (ta[i]	< minta) {
		 minta = ta[i];
		 mini =	i;
      }
   }

   for (i = 0; i < noL;	i++) {

      ratio[i] = 0.0;

      if (tb[i]	<= 0) {
		 for (j	= 0; j < noL; j++) {
			i1_i[i][j] = i2_i[i][j]	= i1_o[i][j] = i2_o[i][j] = 0.0;
			v1_i[i][j] = v2_i[i][j]	= cp->dc1[j];
			v1_o[i][j] = v2_o[i][j]	= cp->dc2[j];
		 }
      }	else {
		 if (ta[i] <= 0) {
			for (j = 0; j <	noL; j++) {
			   i1_i[i][j] =	i1_o[i][j] = 0.0;
			   v1_i[i][j] =	cp->dc1[j];
			   v1_o[i][j] =	cp->dc2[j];
			}
			vi1 = cp->vi_head;
			vi = vi1->next;
		 } else	{
			vi1 = cp->vi_head;
			for (vi	= vi1->next; vi->time <	ta[i]; ) {
			   /* if (i == mini)
					 free_vi(vi1); */
			   vi1 = vi;

			   /* new */
			   vi =	vi->next;
			   if (vi == NULL) goto	errordetect;
			}
			f = (ta[i] - vi1->time)	/ (vi->time - vi1->time);
			for (j = 0; j <	noL; j++) {
			   v1_i[i][j] =	vi1->v_i[j] + f	* (vi->v_i[j] -	vi1->v_i[j]);
			   v1_o[i][j] =	vi1->v_o[j] + f	* (vi->v_o[j] -	vi1->v_o[j]);
			   i1_i[i][j] =	vi1->i_i[j] + f	* (vi->i_i[j] -	vi1->i_i[j]);
			   i1_o[i][j] =	vi1->i_o[j] + f	* (vi->i_o[j] -	vi1->i_o[j]);
			}
			if (i == mini)
			   cp->vi_head = vi1;
		 }

		 if (tb[i] > t1) {
			/*
			fprintf(stderr,	"pvs: time = %d\n", t2);
			 */
			ext = 1;

			ratio[i] = f = (tb[i] -	t1) / (t2 - t1);

			if (vi)
			   for (; vi->next; vi = vi->next)
			       ;
			else
			   vi =	vi1;
			f = 1 -	f;
			for (j = 0; j <	noL; j++) {
			   v2_i[i][j] =	vi->v_i[j] * f;
			   v2_o[i][j] =	vi->v_o[j] * f;
			   i2_i[i][j] =	vi->i_i[j] * f;
		   i2_o[i][j] =	vi->i_o[j] * f;
	    }
	  } else {
			for (; vi->time	< tb[i];) {
			   vi1 = vi;

			   /* new */
			   vi =	vi->next;
			   if (vi == NULL) goto	errordetect;
			}

			f = (tb[i] - vi1->time)	/ (vi->time - vi1->time);
			for (j = 0; j <	noL; j++) {
			   v2_i[i][j] =	vi1->v_i[j] + f	* (vi->v_i[j] -	vi1->v_i[j]);
			   v2_o[i][j] =	vi1->v_o[j] + f	* (vi->v_o[j] -	vi1->v_o[j]);
			   i2_i[i][j] =	vi1->i_i[j] + f	* (vi->i_i[j] -	vi1->i_i[j]);
			   i2_o[i][j] =	vi1->i_o[j] + f	* (vi->i_o[j] -	vi1->i_o[j]);
			}
	}
      }
   }

   return(ext);

errordetect:
   fprintf(stderr,	"your maximum time step	is too large for tau.\n");
   fprintf(stderr,	"decrease max time step	in .tran card and try again\n");
   controlled_exit(0);
}


static int update_delayed_cnv(CPLine *cp, double h)
{
   int i, j, k;
   double *ratio1;
   double f;
   VI_list *vi;
   TMS *tms;
   int noL;

   h *=	0.5e-12;
   ratio1 = cp->ratio;
   vi =	cp->vi_tail;
   noL = cp->noL;

   for (k = 0; k < noL;	k++)	 /*  mode  */
      if (ratio1[k] > 0.0)
     for (i = 0; i < noL; i++)	/*  current eqn	 */
	for (j = 0; j <	noL; j++) {
	   tms = cp->h3t[i][j][k];
	   if (tms == NULL)
	  continue;
	   f = h * ratio1[k] * vi->v_i[j];
	   tms->tm[0].cnv_i += f *  tms->tm[0].c;
	   tms->tm[1].cnv_i += f *  tms->tm[1].c;
	   tms->tm[2].cnv_i += f *  tms->tm[2].c;

	   f = h * ratio1[k] * vi->v_o[j];
	   tms->tm[0].cnv_o += f *  tms->tm[0].c;
	   tms->tm[1].cnv_o += f *  tms->tm[1].c;
	   tms->tm[2].cnv_o += f *  tms->tm[2].c;

	   tms = cp->h2t[i][j][k];
	   f = h * ratio1[k] * vi->i_i[j];
	   tms->tm[0].cnv_i += f *  tms->tm[0].c;
	   tms->tm[1].cnv_i += f *  tms->tm[1].c;
	   tms->tm[2].cnv_i += f *  tms->tm[2].c;

	   f = h * ratio1[k] * vi->i_o[j];
	   tms->tm[0].cnv_o += f *  tms->tm[0].c;
	   tms->tm[1].cnv_o += f *  tms->tm[1].c;
	   tms->tm[2].cnv_o += f *  tms->tm[2].c;
	}
   return(1);
}


static int
expC(double ar,	double ai, double h, double *cr, double	*ci)
{
	double e, cs, si;

	e = exp(ar * h);
	cs = cos(ai * h);
	si = sin(ai * h);
	*cr = e	* cs;
	*ci = e	* si;

	return(1);
}

static int
multC(double ar, double	ai, double br, double bi,
      double *cr, double *ci)
{
	double tp;

	tp = ar*br - ai*bi;
	*ci = ar*bi+ai*br;
	*cr = tp;

	return (1);

}

static void
update_cnv_a(TMS *tms, double h, double ai, double ao, double bi, double	bo,
	     double er,	double ei)
{
   double a, b,	a1, b1;

   h *=	0.5e-12;
   multC(tms->tm[1].c, tms->tm[2].c, er, ei, &a1, &b1);
   multC(tms->tm[1].cnv_i, tms->tm[2].cnv_i, er, ei, &a, &b);
   tms->tm[1].cnv_i = a	+ h * (a1 * bi + ai * tms->tm[1].c);
   tms->tm[2].cnv_i = b	+ h * (b1 * bi + ai * tms->tm[2].c);

   multC(tms->tm[1].cnv_o, tms->tm[2].cnv_o, er, ei, &a, &b);
   tms->tm[1].cnv_o = a	+ h * (a1 * bo + ao * tms->tm[1].c);
   tms->tm[2].cnv_o = b	+ h * (b1 * bo + ao * tms->tm[2].c);
}

static int
divC(double ar,	double ai, double br, double bi, double	*cr, double *ci)
{
	double t;

	t = br*br + bi*bi;
	*cr = (ar*br + ai*bi) /	t;
	*ci = (ai*br - ar*bi) /	t;

	return(1);
}

