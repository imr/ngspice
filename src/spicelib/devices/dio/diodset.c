/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author:  1988 Jaijeet S Roychowdhury
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "diodefs.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

int
DIOdSetup(model,ckt)

DIOmodel *model;
CKTcircuit *ckt;
/* actually load the current resistance value into the 
         * sparse matrix previously provided 
         */
{
	DIOinstance *here;
	double arg;
	double csat;    /* area-scaled saturation current */
	double czero;
	double czof2;
	double evd;
	double evrev;
	double gd;
	double sarg;
	double vd;      /* current diode voltage */
	double vt;      /* K t / Q */
	double vte;
	double g2,g3;
	double cdiff2,cdiff3;
	double cjunc1,cjunc2,cjunc3;

	/*  loop through all the diode models */
	for( ; model != NULL; model = model->DIOnextModel ) {

		/* loop through all the instances of the model */
		for (here = model->DIOinstances; here != NULL ;
                here=here->DIOnextInstance) {
		    if (here->DIOowner != ARCHme) continue;

			    /*
			     *  this routine loads diodes for dc and transient analyses.
			     */




			csat=here->DIOtSatCur*here->DIOarea;
			vt = CONSTKoverQ * here->DIOtemp;
			vte=model->DIOemissionCoeff * vt;
	    		vd = *(ckt->CKTrhsOld + (here->DIOposPrimeNode)) -
			*(ckt->CKTrhsOld + (here->DIOnegNode));

			/*
			 *   compute derivatives
			 */
			if (vd >= -5*vte) {
				evd = exp(vd/vte);
				gd = csat*evd/vte+ckt->CKTgmin;
				g2 = 0.5*(gd-ckt->CKTgmin)/vte;
				cdiff2 = g2*model->DIOtransitTime;
				g3 = g2/3/vte;
				cdiff3 = g3*model->DIOtransitTime;
			} 
			else if((!(here->DIOtBrkdwnV))|| 
			    (vd >= -here->DIOtBrkdwnV)) {
				gd = -csat/vd+ckt->CKTgmin;
				g2=g3=cdiff2=cdiff3=0.0;
				/* off */
			} 
			else {
				/* reverse breakdown */
				/* why using csat instead of breakdowncurrent? */
				evrev=exp(-(here->DIOtBrkdwnV+vd)/vt);
				/*
				                cd = -csat*(evrev-1+here->DIOtBrkdwnV/vt);
						*/
				/* should there be a minus here above? */
				gd=csat*evrev/vt;
				g2 = -gd/2/vt;
				g3 = -g2/3/vt;
				cdiff3 = cdiff2 = 0;
			}
			/*
			                 *   junction charge storage elements
			                 */
			czero=here->DIOtJctCap*here->DIOarea;
			if (czero != 0.0) {
			if (vd < here->DIOtDepCap){
				arg=1-vd/model->DIOjunctionPot;
				sarg=exp(-model->DIOgradingCoeff*log(arg));
				/* the expression for depletion charge 
				                        model->DIOjunctionPot*czero*
                            (1-arg*sarg)/(1-model->DIOgradingCoeff);
						    */
				cjunc1 = czero*sarg;
				cjunc2 = cjunc1/2/model->DIOjunctionPot*model->DIOgradingCoeff/arg;
				cjunc3 = cjunc2/3/model->DIOjunctionPot/arg*(model->DIOgradingCoeff + 1);
			} 
			else {
				czof2=czero/model->DIOf2;
				/* depletion charge equation 
				                        czero*here->DIOtF1+czof2*
                        (model->DIOf3*(vd-here->DIOtDepCap)+
				                        (model->DIOgradingCoeff/(model->DIOjunctionPot+
				                        model->DIOjunctionPot))*(vd*vd-here->DIOtDepCap*
                        here->DIOtDepCap));
							*/
				cjunc2 = czof2/2/model->DIOjunctionPot*model->DIOgradingCoeff;
				cjunc3 =0.0;
			}
			} 
			else 
			{
			cjunc1 = cjunc2 = cjunc3 = 0.0;
			}

			/*
			                 *   store small-signal parameters
			                 */

			here->id_x2 = g2;
			here->id_x3 = g3;
			here->cdif_x2 = cdiff2;
			here->cdif_x3 = cdiff3;
			here->cjnc_x2 = cjunc2;
			here->cjnc_x3 = cjunc3;
		}
	}
		return(OK);
}
