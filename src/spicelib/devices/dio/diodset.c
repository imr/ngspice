/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author:  1988 Jaijeet S Roychowdhury
Modified by Dietmar Warning 2003
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "diodefs.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"


/* actually load the current resistance value into the sparse matrix
 * previously provided - for distortion analysis */
int
DIOdSetup(DIOmodel *model, CKTcircuit *ckt)
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
        double cd;
        double sqrt_ikf;
        double ikf_area_m;
	double sqrt_ikr;
        double ikr_area_m;
	double czeroSW;
	double cjunc1SW,cjunc2SW,cjunc3SW;

	/*  loop through all the diode models */
	for( ; model != NULL; model = model->DIOnextModel ) {

		/* loop through all the instances of the model */
		for (here = model->DIOinstances; here != NULL ;
                here=here->DIOnextInstance) {
		    if (here->DIOowner != ARCHme) continue;

			    /*
			     *  this routine loads diodes for dc and transient analyses.
			     */

                        csat=(here->DIOtSatCur*here->DIOarea+here->DIOtSatSWCur*here->DIOpj)*here->DIOm;
			vt = CONSTKoverQ * here->DIOtemp;
			vte=model->DIOemissionCoeff * vt;
	    		vd = *(ckt->CKTrhsOld + (here->DIOposPrimeNode)) -
			*(ckt->CKTrhsOld + (here->DIOnegNode));

			/*
			 *   compute derivatives
			 */
			if (vd >= -5*vte) {
				evd = exp(vd/vte);
                                cd = csat*(evd-1)+ckt->CKTgmin*vd;
				gd = csat*evd/vte+ckt->CKTgmin;
                                if((model->DIOforwardKneeCurrentGiven) && (model->DIOforwardKneeCurrent > 0.0) && (cd > 1.0e-18) ) {
                                  ikf_area_m = model->DIOforwardKneeCurrent*here->DIOarea*here->DIOm;
                                  sqrt_ikf = sqrt(cd/ikf_area_m);
                                  gd = ((1+sqrt_ikf)*gd - cd*gd/(2*sqrt_ikf*ikf_area_m))/(1+2*sqrt_ikf+cd/ikf_area_m)+ckt->CKTgmin;
                                }
				g2 = 0.5*(gd-ckt->CKTgmin)/vte;
				cdiff2 = g2 * here->DIOtTransitTime;
				g3 = g2/3/vte;
				cdiff3 = g3 * here->DIOtTransitTime;
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
                        czero=here->DIOtJctCap*here->DIOarea*here->DIOm;
			if (czero != 0.0) {
			  if (vd < here->DIOtDepCap){
			  	arg=1-vd/model->DIOjunctionPot;
			  	sarg=exp(-here->DIOtGradingCoeff*log(arg));
			  	/* the expression for depletion charge 
			  	                        model->DIOjunctionPot*czero*
                              (1-arg*sarg)/(1-here->DIOtGradingCoeff);
			  			    */
			  	cjunc1 = czero*sarg;
			  	cjunc2 = cjunc1/2/model->DIOjunctionPot*here->DIOtGradingCoeff/arg;
			  	cjunc3 = cjunc2/3/model->DIOjunctionPot/arg*(here->DIOtGradingCoeff + 1);
			  } 
			  else {
			  	czof2=czero/here->DIOtF2;
			  	/* depletion charge equation 
			  	                        czero*here->DIOtF1+czof2*
                          (model->DIOtF3*(vd-here->DIOtDepCap)+
			  	                        (here->DIOtGradingCoeff/(model->DIOjunctionPot+
			  	                        model->DIOjunctionPot))*(vd*vd-here->DIOtDepCap*
                          here->DIOtDepCap));
			  				*/
			  	cjunc2 = czof2/2/model->DIOjunctionPot*here->DIOtGradingCoeff;
			  	cjunc3 =0.0;
			  }
			} 
			else 
			{
			  cjunc1 = cjunc2 = cjunc3 = 0.0;
			}
                        czeroSW=+here->DIOtJctSWCap*here->DIOpj*here->DIOm;
			if (czeroSW != 0.0) {
			  if (vd < here->DIOtDepCap){
			  	arg=1-vd/model->DIOjunctionSWPot;
			  	sarg=exp(-model->DIOgradingSWCoeff*log(arg));
			  	cjunc1SW = czeroSW*sarg;
			  	cjunc2SW = cjunc1SW/2/model->DIOjunctionSWPot*model->DIOgradingSWCoeff/arg;
			  	cjunc3SW = cjunc2SW/3/model->DIOjunctionSWPot/arg*(model->DIOgradingSWCoeff + 1);
			  } 
			  else {
			  	czof2=czeroSW/here->DIOtF2SW;
			  	cjunc2SW = czof2/2/model->DIOjunctionSWPot*model->DIOgradingSWCoeff;
			  	cjunc3SW = 0.0;
			  }
			} 
			else 
			{
			  cjunc1SW = cjunc2SW = cjunc3SW = 0.0;
			}

			/*
			                 *   store small-signal parameters
			                 */

			here->id_x2 = g2;
			here->id_x3 = g3;
			here->cdif_x2 = cdiff2;
			here->cdif_x3 = cdiff3;
			here->cjnc_x2 = cjunc2+cjunc2SW;
			here->cjnc_x3 = cjunc3+cjunc3SW;
		}
	}
		return(OK);
}
