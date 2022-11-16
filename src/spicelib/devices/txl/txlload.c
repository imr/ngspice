/**********
Copyright 1992 Regents of the University of California.	 All rights
reserved.
Author:	1992 Charles Hough
**********/


#include "ngspice/ngspice.h"
#include "txldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


static double ratio[MAX_CP_TX_LINES];
static int update_cnv_txl(TXLine*, double);
static VI_list_txl *new_vi_txl(void);
static void free_vi_txl(VI_list_txl*);
static int add_new_vi_txl(TXLinstance*,	CKTcircuit*, int);
static int get_pvs_vi_txl(int, int, TXLine*, double*, double*, double*,	double*, double*, 
			  double*, double*, double*);
static int right_consts_txl(TXLine*, int, int, double, double, int, int, CKTcircuit*);
static int update_delayed_cnv_txl(TXLine*, double);
static int multC(double, double, double, double, double*, double*);
static int expC(double,	double,	double, double*,	double*);
static void copy_tx(TXLine *, TXLine *);

/*ARGSUSED*/
int
TXLload(GENmodel *inModel, CKTcircuit *ckt)
{
	TXLmodel *model = (TXLmodel *)inModel;
	TXLinstance *here;
	TXLine *tx, *tx2;
	int k, l;
	int time, time2;
	double h, h1, f;
	int hint;
	double hf;
	NODE *nd;
	double v, v1, g;
	int cond1;
	CKTnode	*node;
	VI_list_txl *vi, *vi_before;
	int i, before, delta;

	double gmin;	 /* dc solution	*/

	/* debug 
	printf("before txlload\n");
	SMPprint(ckt->CKTmatrix, NULL);
	*/

	h = ckt->CKTdelta;
	h1 = 0.5 * h;
	time2 =	(int) (ckt->CKTtime * 1e12);
	hint = (int)(h * 1e12);
	hf = h * 1e12;
	time = (int) ((ckt->CKTtime - ckt->CKTdelta) * 1e12);

	cond1= ckt->CKTmode & MODEDC;

	gmin = 0.1 * ckt->CKTgmin;     /* dc solution */

	for( ; model !=	NULL; model = TXLnextModel(model)) {
		for (here = TXLinstances(model); here != NULL ; 
			here=TXLnextInstance(here)) { 

			tx = here->txline;

			*here->TXLposPosPtr += gmin; /* dc solution */
			*here->TXLnegNegPtr += gmin;
			*here->TXLnegPosPtr += gmin;
			*here->TXLposNegPtr += gmin;

			if (cond1 || tx->vi_head == NULL) continue;

			if (time < tx->vi_tail->time) {
				time = tx->vi_tail->time;
				hint = time2 - time;
			}

			vi_before = tx->vi_tail;
			before = tx->vi_tail->time;

			if (time > tx->vi_tail->time) {

				copy_tx(tx, here->txline2);
				add_new_vi_txl(here, ckt, time);

				delta =	time - before;

				nd = tx->in_node;
				v = vi_before->v_i;
				nd->V =	tx->vi_tail->v_i;
				v1 = nd->V;
				nd->dv = (v1 - v) / delta;

				nd = tx->out_node;
				v = vi_before->v_o;
				v1 = nd->V = tx->vi_tail->v_o;
				nd->dv = (v1 - v) / delta;

				if (tx->lsl) continue;
				update_cnv_txl(tx, delta);
				if (tx->ext) update_delayed_cnv_txl(tx,	delta);
			}
		}
	}

    model = (TXLmodel *)inModel;
    for( ; model != NULL; model	= TXLnextModel(model))	{
	for (here = TXLinstances(model); here != NULL ;	
			here=TXLnextInstance(here)) { 

			tx = here->txline;
			tx2 = here->txline2;

			if (!tx->lsl &&	hf > tx->taul) {

				fprintf(stderr, "your time step is too large for TXL tau.\n");
/*				fprintf(stderr, "please decrease max time	step in	.tran card.\n");
				fprintf(stderr, ".tran tstep tstop tstart	tmax.\n");
				fprintf(stderr, "make tmax smaller than %e and try again.\n",
				tx->taul * 1e-12);
				return (1111);
*/
				fprintf(stderr,	"tmax is now set to	%e.\n", 0.9 * tx->taul * 1e-12);
				ckt->CKTmaxStep = 0.9 * tx->taul * 1e-12;

			}

			if (cond1) {
				if (here->TXLlengthgiven) 
					g = model->R * here->TXLlength;
				else g = model->R * TXLmodPtr(here)->length;
				*(here->TXLposIbr1Ptr) += 1.0;
				*(here->TXLnegIbr2Ptr) += 1.0;
				*(here->TXLibr1Ibr1Ptr)	+= 1.0;
				*(here->TXLibr1Ibr2Ptr)	+= 1.0;
				*(here->TXLibr2PosPtr) += 1.0;
				*(here->TXLibr2NegPtr) -= 1.0;
				*(here->TXLibr2Ibr1Ptr)	-= g;

				continue;

			}

			/* dc setup */
			if (here->TXLdcGiven ==	0 && !cond1) {
				nd = tx->in_node;
				for (node = ckt->CKTnodes;node;	node = node->next) {
					if (strcmp(nd->name->id, node->name) ==	0) {
						tx->dc1	= tx2->dc1 = ckt->CKTrhsOld[node->number]; 
						nd->V =	tx->dc1;
						break;
					}
				}
				nd = tx->out_node;
				for (node = ckt->CKTnodes;node;	node = node->next) {
					if (strcmp(nd->name->id, node->name) ==	0) {
						tx->dc2	= tx2->dc2 = ckt->CKTrhsOld[node->number]; 
						nd->V =	tx->dc2;
						break;
					}
				}
				here->TXLdcGiven = 1;

				vi = new_vi_txl();
				vi->time = 0;

				vi->i_i	= *(ckt->CKTrhsOld + here->TXLibr1);
				vi->i_o	= *(ckt->CKTrhsOld + here->TXLibr2);

				vi->v_i	= tx->dc1;
				vi->v_o	= tx->dc2;

				for (i = 0; i <	3; i++)	{
					tx->h1_term[i].cnv_i = 
					   - tx->dc1 * tx->h1_term[i].c	/ tx->h1_term[i].x;
					tx->h1_term[i].cnv_o = 
					   - tx->dc2 * tx->h1_term[i].c	/ tx->h1_term[i].x;
				}
				for (i = 0; i <	3; i++)	{
					tx->h2_term[i].cnv_i = 0.0;
					tx->h2_term[i].cnv_o = 0.0;
				}
				for (i = 0; i <	6; i++)	{
					tx->h3_term[i].cnv_i = 
				    - tx->dc1 *	tx->h3_term[i].c / tx->h3_term[i].x;
					tx->h3_term[i].cnv_o = 
				    - tx->dc2 *	tx->h3_term[i].c / tx->h3_term[i].x;
				}
				vi->next = NULL;
				tx->vi_tail = vi;
				tx->vi_head = vi;
				here->txline2->vi_tail = vi;
				here->txline2->vi_head = vi;

			}

			/* change 6,6	 1/18/93 
			*(here->TXLibr1Ibr1Ptr)	-= 1.0;	
			*(here->TXLibr2Ibr2Ptr)	-= 1.0;
	    *(here->TXLposIbr1Ptr) += 1.0;
	    *(here->TXLnegIbr2Ptr) += 1.0;
	    *(here->TXLibr1PosPtr) += tx->sqtCdL + h1 *	tx->h1C;
			*(here->TXLibr2NegPtr) += tx->sqtCdL + h1 * tx->h1C;
			*/
			*(here->TXLibr1Ibr1Ptr)	= -1.0;	
			*(here->TXLibr2Ibr2Ptr)	= -1.0;
	    *(here->TXLposIbr1Ptr) = 1.0;
	    *(here->TXLnegIbr2Ptr) = 1.0;
	    *(here->TXLibr1PosPtr) = tx->sqtCdL	+ h1 * tx->h1C;
			*(here->TXLibr2NegPtr) = tx->sqtCdL + h1 * tx->h1C;

			k = here->TXLibr1;
			l = here->TXLibr2;

			copy_tx(tx2, tx);

			if (right_consts_txl(tx2, time,	time2, h, h1, k, l, ckt)) {
				if (tx->lsl) {
					f = ratio[0] * tx->h3_aten;
					*(here->TXLibr1NegPtr) = -f;
					*(here->TXLibr2PosPtr) = -f;
					f = ratio[0] * tx->h2_aten;
					*(here->TXLibr1Ibr2Ptr)	= -f;
					*(here->TXLibr2Ibr1Ptr)	= -f;
				}
				else {
					tx->ext	= 1;
					tx->ratio = ratio[0];
					if (ratio[0] > 0.0) {
						f = ratio[0] * (h1 * (tx->h3_term[0].c 
							+ tx->h3_term[1].c + tx->h3_term[2].c 
							+ tx->h3_term[3].c + tx->h3_term[4].c
							+ tx->h3_term[5].c ) + tx->h3_aten);
						*(here->TXLibr1NegPtr) = -f;
						*(here->TXLibr2PosPtr) = -f;
						f = ratio[0] * (h1 * ( tx->h2_term[0].c	
							+ tx->h2_term[1].c + tx->h2_term[2].c )	
							+ tx->h2_aten);
						*(here->TXLibr1Ibr2Ptr)	= -f;
						*(here->TXLibr2Ibr1Ptr)	= -f;
					}
				}
			} 
			else tx->ext = 0;
	}
    }

	if (cond1) return (OK);

	/* debug 
	printf("after txlload\n");
	SMPprint(ckt->CKTmatrix, NULL);
	*/

    return(OK);
}

static void 
copy_tx(TXLine *new, TXLine *old)
{
	int i;
	VI_list_txl *temp;

	new->lsl = old->lsl;
	new->ext = old->ext;
	new->ratio = old->ratio;
	new->taul = old->taul;
	new->sqtCdL = old->sqtCdL;
	new->h2_aten = old->h2_aten;
	new->h3_aten = old->h3_aten;
	new->h1C = old->h1C;
	for (i=	0; i < 3; i++) {
		new->h1e[i] = old->h1e[i];

		new->h1_term[i].c = old->h1_term[i].c;
		new->h1_term[i].x = old->h1_term[i].x;
		new->h1_term[i].cnv_i =	old->h1_term[i].cnv_i;
		new->h1_term[i].cnv_o =	old->h1_term[i].cnv_o;

		new->h2_term[i].c = old->h2_term[i].c;
		new->h2_term[i].x = old->h2_term[i].x;
		new->h2_term[i].cnv_i =	old->h2_term[i].cnv_i;
		new->h2_term[i].cnv_o =	old->h2_term[i].cnv_o;
	}
	for (i=	0; i < 6; i++) {
		new->h3_term[i].c = old->h3_term[i].c;
		new->h3_term[i].x = old->h3_term[i].x;
		new->h3_term[i].cnv_i =	old->h3_term[i].cnv_i;
		new->h3_term[i].cnv_o =	old->h3_term[i].cnv_o;
	}

	new->ifImg = old->ifImg;
	if (new->vi_tail != old->vi_tail) {
		/* someting wrong */
		fprintf(stderr, "Error during evaluating TXL line\n");
		controlled_exit(0);
	}

	while (new->vi_head->time < old->vi_head->time)	{
		temp = new->vi_head;
		new->vi_head = new->vi_head->next;
		free_vi_txl(temp);
	}
		
}


static int 
update_cnv_txl(TXLine *tx, double h)
{
   int i;

   double ai, bi, ao, bo;
   double e, t;

   ai =	tx->in_node->V;
   ao =	tx->out_node->V;
   bi =	tx->in_node->dv;
   bo =	tx->out_node->dv;

   for (i = 0; i < 3; i++) {
      TERM *tm;
      tm = &(tx->h1_term[i]);

      e	= tx->h1e[i];
    
      t	= tm->c	/ tm->x;
      bi *= t;
      bo *= t;

      tm->cnv_i	= (tm->cnv_i - bi*h) * e + (e -	1.0)*(ai*t + 1.0e+12*bi/tm->x);
      tm->cnv_o	= (tm->cnv_o - bo*h) * e + (e -	1.0)*(ao*t + 1.0e+12*bo/tm->x);
   }
   return (1);
}


static VI_list_txl
*new_vi_txl(void)
{
   VI_list_txl *q;

   if (pool_vi_txl) {
      q	= pool_vi_txl;
      pool_vi_txl = pool_vi_txl->pool;
      return(q);
   } else 
      return(TMALLOC(VI_list_txl, 1));
}

static void 
free_vi_txl(VI_list_txl	*q)
{
   q->pool = pool_vi_txl;
   pool_vi_txl = q;
}


static int 
add_new_vi_txl(TXLinstance *here, CKTcircuit *ckt, int time)
{
   VI_list_txl *vi;
   TXLine *tx, *tx2;

   tx =	here->txline;
   tx2 = here->txline2;

   vi =	new_vi_txl();
   vi->time = time;
   tx->vi_tail->next = vi;
   tx2->vi_tail->next =	vi;
   vi->next = NULL;
   tx->vi_tail = vi;
   tx2->vi_tail	= vi;
   
   vi->v_i = *(ckt->CKTrhsOld +	here->TXLposNode);
   vi->v_o = *(ckt->CKTrhsOld +	here->TXLnegNode);
   vi->i_i = *(ckt->CKTrhsOld +	here->TXLibr1);
   vi->i_o = *(ckt->CKTrhsOld +	here->TXLibr2);
   return(1);
}


static int 
get_pvs_vi_txl(int t1, int t2, TXLine *tx, double *v1_i, double	*v2_i, double *i1_i, double *i2_i, 
	       double *v1_o, double *v2_o, double *i1_o, double	*i2_o)
{
   double ta, tb; 
   VI_list_txl *vi, *vi1;
   double f;
   int ext = 0;

   ta =	t1 - tx->taul;
   tb =	t2 - tx->taul;
   if (tb <= 0)	{
      *v1_i = *v2_i = tx->dc1; 
      *v2_o = *v1_o = tx->dc2;
      *i1_i = *i2_i = *i1_o = *i2_o = 0;
      return(ext);
   }

   if (ta <= 0)	{
      *i1_i = *i1_o = 0.0; 
      *v1_i = tx->dc1;
      *v1_o = tx->dc2; 
      vi1 = tx->vi_head;
      vi = vi1->next;
   } else {
      vi1 = tx->vi_head;
      for (vi =	vi1->next; vi->time < ta; vi = vi->next) {
	 /* free_vi_txl(vi1); */
	 vi1 = vi;
      }
      f	= (ta -	vi1->time) / (vi->time - vi1->time);
      *v1_i = vi1->v_i + f * (vi->v_i -	vi1->v_i);
      *v1_o = vi1->v_o + f * (vi->v_o -	vi1->v_o);
      *i1_i = vi1->i_i + f * (vi->i_i -	vi1->i_i);
      *i1_o = vi1->i_o + f * (vi->i_o -	vi1->i_o);
      tx->vi_head = vi1;
   }

   if (tb > t1)	{
     
      /* fprintf(stderr, "pvs: time = %d\n", t2); */
      ext = 1;
      /*     
      f	= tb - t1;
      *v2_i = tx->in_node->V + tx->in_node->dv * f;
      *v2_o = tx->out_node->V +	tx->out_node->dv * f;
      
      if (vi) {
	 for (;	vi->time != t1;	vi = vi->next) 
	    vi1	= vi;

	 f /= (double) (t1 - vi1->time);
	 *i2_i = vi->i_i + f * (vi->i_i	- vi1->i_i);
	 *i2_o = vi->i_o + f * (vi->i_o	- vi1->i_o);
      }	else {
	 *i2_i = vi1->i_i;
	 *i2_o = vi1->i_o;
      }
       */
      ratio[0] = f = (tb - t1) / (t2 - t1);
      if (vi)
	 for (;	vi->time != t1;	vi = vi->next)
	    ;
      else 
	 vi = vi1;
      f	= 1 - f;
      *v2_i = vi->v_i *	f;
      *v2_o = vi->v_o *	f;
      *i2_i = vi->i_i *	f;
      *i2_o = vi->i_o *	f;
   } else {
      for (; vi->time <	tb; vi = vi->next) 
	 vi1 = vi;
      
      f	= (tb -	vi1->time) / (vi->time - vi1->time);
      *v2_i = vi1->v_i + f * (vi->v_i -	vi1->v_i);
      *v2_o = vi1->v_o + f * (vi->v_o -	vi1->v_o);
      *i2_i = vi1->i_i + f * (vi->i_i -	vi1->i_i);
      *i2_o = vi1->i_o + f * (vi->i_o -	vi1->i_o);
   }

   return(ext);
}


static int
right_consts_txl(TXLine	*tx, int t, int	time, double h,	double h1, int l1, int l2, CKTcircuit *ckt)
/***  h1 = 0.5 * h  ***/
{
   int i;
   double ff=0.0, gg=0.0, e;
   double v1_i,	v2_i, i1_i, i2_i;
   double v1_o,	v2_o, i1_o, i2_o;
   int ext;

   if (! tx->lsl) {
   double ff1=0.0;
   for (i = 0; i < 3; i++) {
      tx->h1e[i] = e = exp(tx->h1_term[i].x * h);
      ff1 -= tx->h1_term[i].c *	e;
      ff  -= tx->h1_term[i].cnv_i * e;
      gg  -= tx->h1_term[i].cnv_o * e;
   }
   ff += ff1 * h1 * tx->in_node->V;
   gg += ff1 * h1 * tx->out_node->V;
   }

   ext = get_pvs_vi_txl(t, time, tx, &v1_i, &v2_i, &i1_i, &i2_i, &v1_o,	&v2_o, &i1_o, &i2_o);

   if (tx->lsl)	{
	   ff =	tx->h3_aten * v2_o + tx->h2_aten * i2_o;
	   gg =	tx->h3_aten * v2_i + tx->h2_aten * i2_i;
  } else {
   if (tx->ifImg) {
      double a,	b, er, ei, a1, b1, a2, b2;

      for (i = 0; i < 4; i++) {
     TERM *tm;
     tm	= &(tx->h3_term[i]);
     e =  exp(tm->x * h);
     tm->cnv_i = tm->cnv_i * e + h1 * tm->c * (v1_i * e	+ v2_i);
     tm->cnv_o = tm->cnv_o * e + h1 * tm->c * (v1_o * e	+ v2_o);
      }
      expC(tx->h3_term[4].x, tx->h3_term[5].x, h, &er, &ei);
      a2 = h1 *	tx->h3_term[4].c;
      b2 = h1 *	tx->h3_term[5].c;

      a	= tx->h3_term[4].cnv_i;
      b	= tx->h3_term[5].cnv_i;
      multC(a, b, er, ei, &a, &b);
      multC(a2,	b2, v1_i * er +	v2_i, v1_i * ei, &a1, &b1);
      tx->h3_term[4].cnv_i = a + a1;
      tx->h3_term[5].cnv_i = b + b1;

      a	= tx->h3_term[4].cnv_o;
      b	= tx->h3_term[5].cnv_o;
      multC(a, b, er, ei, &a, &b);
      multC(a2,	b2, v1_o * er +	v2_o, v1_o * ei, &a1, &b1);
      tx->h3_term[4].cnv_o = a + a1;
      tx->h3_term[5].cnv_o = b + b1;

      ff += tx->h3_aten	* v2_o;
      gg += tx->h3_aten	* v2_i;

      for (i = 0; i < 5; i++) {
     ff	+= tx->h3_term[i].cnv_o;
     gg	+= tx->h3_term[i].cnv_i;
      }
      ff += tx->h3_term[4].cnv_o;
      gg += tx->h3_term[4].cnv_i;

      {
     TERM *tm;
     tm	= &(tx->h2_term[0]);

     e =  exp(tm->x * h);
     tm->cnv_i = tm->cnv_i * e + h1 * tm->c * (i1_i * e	+ i2_i);
     tm->cnv_o = tm->cnv_o * e + h1 * tm->c * (i1_o * e	+ i2_o);
      }
      expC(tx->h2_term[1].x, tx->h2_term[2].x, h, &er, &ei);
      a2 = h1 *	tx->h2_term[1].c;
      b2 = h1 *	tx->h2_term[2].c;

      a	= tx->h2_term[1].cnv_i;
      b	= tx->h2_term[2].cnv_i;
      multC(a, b, er, ei, &a, &b);
      multC(a2,	b2, i1_i * er +	i2_i, i1_i * ei, &a1, &b1);
      tx->h2_term[1].cnv_i = a + a1;
      tx->h2_term[2].cnv_i = b + b1;

      a	= tx->h2_term[1].cnv_o;
      b	= tx->h2_term[2].cnv_o;
      multC(a, b, er, ei, &a, &b);
      multC(a2,	b2, i1_o * er +	i2_o, i1_o * ei, &a1, &b1);
      tx->h2_term[1].cnv_o = a + a1;
      tx->h2_term[2].cnv_o = b + b1;

      ff += tx->h2_aten	* i2_o + tx->h2_term[0].cnv_o +
	  2.0 *	tx->h2_term[1].cnv_o;
      gg += tx->h2_aten	* i2_i + tx->h2_term[0].cnv_i +
	  2.0 *	tx->h2_term[1].cnv_i;
   } else {
      for (i = 0; i < 6; i++) {
     TERM *tm;
     tm	= &(tx->h3_term[i]);

     e =  exp(tm->x * h);
     tm->cnv_i = tm->cnv_i * e + h1 * tm->c * (v1_i * e	+ v2_i);
     tm->cnv_o = tm->cnv_o * e + h1 * tm->c * (v1_o * e	+ v2_o);
      }

      ff += tx->h3_aten	* v2_o;
      gg += tx->h3_aten	* v2_i;

      for (i = 0; i < 6; i++) {
     ff	+= tx->h3_term[i].cnv_o;
     gg	+= tx->h3_term[i].cnv_i;
      }

      for (i = 0; i < 3; i++) {
     TERM *tm;
     tm	= &(tx->h2_term[i]);

     e =  exp(tm->x * h);
     tm->cnv_i = tm->cnv_i * e + h1 * tm->c * (i1_i * e	+ i2_i);
     tm->cnv_o = tm->cnv_o * e + h1 * tm->c * (i1_o * e	+ i2_o);
      }

      ff += tx->h2_aten	* i2_o;
      gg += tx->h2_aten	* i2_i;

      for (i = 0; i < 3; i++) {
     ff	+= tx->h2_term[i].cnv_o;
     gg	+= tx->h2_term[i].cnv_i;
      }
   }
   }

	*(ckt->CKTrhs +	l1) = ff;
	*(ckt->CKTrhs +	l2) = gg;

   return(ext);
}


static int update_delayed_cnv_txl(TXLine *tx, double h)
{
   double ratio1;
   double f;
   VI_list_txl *vi;
   TERM	*tms;

   h *=	0.5e-12;
   ratio1 = tx->ratio;
   vi =	tx->vi_tail;

   if (ratio1 > 0.0) {
      tms = tx->h3_term;
      f	= h * ratio1 * vi->v_i;
      tms[0].cnv_i += f	*  tms[0].c;
      tms[1].cnv_i += f	*  tms[1].c;
      tms[2].cnv_i += f	*  tms[2].c;
      tms[3].cnv_i += f	*  tms[3].c;
      tms[4].cnv_i += f	*  tms[4].c;
      tms[5].cnv_i += f	*  tms[5].c;

      f	= h * ratio1 * vi->v_o;
      tms[0].cnv_o += f	*  tms[0].c;
      tms[1].cnv_o += f	*  tms[1].c;
      tms[2].cnv_o += f	*  tms[2].c;
      tms[3].cnv_o += f	*  tms[3].c;
      tms[4].cnv_o += f	*  tms[4].c;
      tms[5].cnv_o += f	*  tms[5].c;

      tms = tx->h2_term;
      f	= h * ratio1 * vi->i_i;
      tms[0].cnv_i += f	*  tms[0].c;
      tms[1].cnv_i += f	*  tms[1].c;
      tms[2].cnv_i += f	*  tms[2].c;

      f	= h * ratio1 * vi->i_o;
      tms[0].cnv_o += f	*  tms[0].c;
      tms[1].cnv_o += f	*  tms[1].c;
      tms[2].cnv_o += f	*  tms[2].c;
   }

   return(1);
}

static int 
expC(double ar,	double ai, double h, double *cr, double *ci)
{
   double e, cs, si;

   e = exp(ar * h);
   cs =	cos(ai	* h);
   si =	sin(ai	* h);
   *cr = e * cs;
   *ci = e * si;

   return(1);
}

static int 
multC(double ar, double	ai, double br, double bi, double *cr, double *ci)
{
	double	tp;

	tp = ar*br - ai*bi;
	*ci = ar*bi + ai*br;
	*cr = tp;

	return (1);

}

