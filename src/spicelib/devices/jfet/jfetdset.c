/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/cktdefs.h"
#include "jfetdefs.h"
#include "ngspice/const.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

int
JFETdSetup(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current resistance value into the 
         * sparse matrix previously provided 
         */
{
    JFETmodel *model = (JFETmodel*)inModel;
    JFETinstance *here;
    double beta;
    double betap;
    double lcapgd1;
    double lcapgd2;
    double lcapgd3;
    double lcapgs2;
    double lcapgs3;
    double lcapgs1;
    double cd;
    double cdrain;
    double cg;
    double cgd;
    double csat;
    double czgd;
    double czgdf2;
    double czgs;
    double czgsf2;
    double evgd;
    double evgs;
    double fcpb2;
    double gdpr;
    double gds1;
    double gds2;
    double gds3;
    double lggd1;
    double lggd2;
    double lggd3;
    double lggs1;
    double lggs2;
    double lggs3;
    double gm1;
    double gm2;
    double gm3;
    double gmds;
    double gm2ds;
    double gmds2;
    double gspr;
    double sarg;
    double twob;
    double twop;
    double vds;
    double vgd;
    double vgs;
    double vgst;

    /*  loop through all the models */
    for( ; model != NULL; model = JFETnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = JFETinstances(model); here != NULL ;
                here=JFETnextInstance(here)) {

            /*
             *  dc model parameters 
             */
            beta = here->JFETtBeta * here->JFETarea * here->JFETm;
            gdpr=model->JFETdrainConduct*here->JFETarea * here->JFETm;
            gspr=model->JFETsourceConduct*here->JFETarea * here-> JFETm;
            csat=here->JFETtSatCur*here->JFETarea * here-> JFETm;
            /*
             *    initialization
             */
                vgs= model->JFETtype*(*(ckt->CKTrhsOld + here->JFETgateNode)
			- *(ckt->CKTrhsOld + here->JFETsourcePrimeNode));
                vgd= model->JFETtype*(*(ckt->CKTrhsOld + here->JFETgateNode)
			- *(ckt->CKTrhsOld + here->JFETdrainPrimeNode));
            /*
             *   determine dc current and derivatives 
             */
            vds=vgs-vgd;

	    if (vds < 0.0) {
		vds = -vds;
		SWAP(double, vgs, vgd);
		/* so now these have become the local variables */
		here->JFETmode = -1;
		} else {
		here->JFETmode = 1;
		}

            if (vgs <= -5*here->JFETtemp*CONSTKoverQ) {
                lggs1 = -csat/vgs+ckt->CKTgmin;
		lggs2=lggs3=0;
                cg = lggs1*vgs;
            } else {
                evgs = exp(vgs/(here->JFETtemp*CONSTKoverQ));
                lggs1 = csat*evgs/(here->JFETtemp*CONSTKoverQ)+ckt->CKTgmin;
		lggs2 = (lggs1-ckt->CKTgmin)/((here->JFETtemp*CONSTKoverQ)*2);
		lggs3 = lggs2/(3*(here->JFETtemp*CONSTKoverQ));
                cg = csat*(evgs-1)+ckt->CKTgmin*vgs;
            }
            if (vgd <= -5*(here->JFETtemp*CONSTKoverQ)) {
                lggd1 = -csat/vgd+ckt->CKTgmin;
		lggd2=lggd3=0;
                cgd = lggd1*vgd;
            } else {
                evgd = exp(vgd/(here->JFETtemp*CONSTKoverQ));
                lggd1 = csat*evgd/(here->JFETtemp*CONSTKoverQ)+ckt->CKTgmin;
		lggd2 = (lggd1-ckt->CKTgmin)/((here->JFETtemp*CONSTKoverQ)*2);
		lggd3 = lggd2/(3*(here->JFETtemp*CONSTKoverQ));
                cgd = csat*(evgd-1)+ckt->CKTgmin*vgd;
            }
            cg = cg+cgd;
            /*
             *   compute drain current and derivatives
             */
                vgst=vgs-here->JFETtThreshold;
                /*
                 *   cutoff region 
                 */
                if (vgst <= 0) {
                    cdrain=0;
                    gm1=gm2=gm3=0;
                    gds1=gds2=gds3=0;
		    gmds=gm2ds=gmds2=0;
                } else {
                    betap=beta*(1+model->JFETlModulation*vds);
                    twob=betap+betap;
                    if (vgst <= vds) {
                        /*
                         *   normal mode, saturation region 
                         */

			 /* note - for cdrain, all the g's refer to the
			  * derivatives which have not been divided to
			  * become Taylor coeffs. A notational 
			  * inconsistency but simplifies processing later.
			  */
                        cdrain=betap*vgst*vgst;
                        gm1=twob*vgst;
			gm2=twob;
			gm3=0;
                        gds1=model->JFETlModulation*beta*vgst*vgst;
			gds2=gds3=gmds2=0;
			gm2ds=2*model->JFETlModulation*beta;
			gmds=gm2ds*vgst;
                    } else {
                        /*
                         *   normal mode, linear region 
                         */
                        cdrain=betap*vds*(vgst+vgst-vds);
                        gm1=twob*vds;
			gm2=0;
			gm3=0;
			gmds=(beta+beta)*(1+2*model->JFETlModulation*vds);
			gm2ds=0;
			gds2=2*beta*(2*model->JFETlModulation*vgst - 1 -
				3*model->JFETlModulation*vds);
                        gds1=beta*(2*(vgst-vds) + 4*vgst*vds*
		model->JFETlModulation - 3*model->JFETlModulation*vds*vds);
			gmds2=4*beta*model->JFETlModulation;
			gds3= -6*beta*model->JFETlModulation;
                    }
                }
            /*
             *   compute equivalent drain current source 
             */
            cd=cdrain-cgd;
                /* 
                 *    charge storage elements 
                 */
                czgs=here->JFETtCGS*here->JFETarea * here->JFETm;
                czgd=here->JFETtCGD*here->JFETarea * here->JFETm;
                twop=here->JFETtGatePot+here->JFETtGatePot;
                fcpb2=here->JFETcorDepCap*here->JFETcorDepCap;
                czgsf2=czgs/model->JFETf2;
                czgdf2=czgd/model->JFETf2;
                if (vgs < here->JFETcorDepCap) {
                    sarg=sqrt(1-vgs/here->JFETtGatePot);
                    lcapgs1=czgs/sarg;
		    lcapgs2=lcapgs1/(here->JFETtGatePot*4*sarg*sarg);
		    lcapgs3=lcapgs2/(here->JFETtGatePot*2*sarg*sarg);
                } else {
                    lcapgs1=czgsf2*(model->JFETf3+vgs/twop);
		    lcapgs2=czgsf2/twop*0.5;
		    lcapgs3=0;
                }
                if (vgd < here->JFETcorDepCap) {
                    sarg=sqrt(1-vgd/here->JFETtGatePot);
                    lcapgd1=czgd/sarg;
		    lcapgd2=lcapgd1/(here->JFETtGatePot*4*sarg*sarg);
		    lcapgd3=lcapgd2/(here->JFETtGatePot*2*sarg*sarg);
                } else {
                    lcapgd1=czgdf2*(model->JFETf3+vgd/twop);
		    lcapgd2=czgdf2/twop*0.5;
		    lcapgd3=0;
                }
                /*
                 *   process to get Taylor coefficients, taking into
		 * account type and mode.
                 */

	if (here->JFETmode == 1)
		{
		/* normal mode - no source-drain interchange */
here->cdr_x = gm1;
here->cdr_y = gds1;
here->cdr_x2 = gm2;
here->cdr_y2 = gds2;
here->cdr_xy = gmds;
here->cdr_x3 = gm3;
here->cdr_y3 = gds3;
here->cdr_x2y = gm2ds;
here->cdr_xy2 = gmds2;

here->ggs1 = lggs1;
here->ggd1 = lggd1;
here->ggs2 = lggs2;
here->ggd2 = lggd2;
here->ggs3 = lggs3;
here->ggd3 = lggd3;
here->capgs1 = lcapgs1;
here->capgd1 = lcapgd1;
here->capgs2 = lcapgs2;
here->capgd2 = lcapgd2;
here->capgs3 = lcapgs3;
here->capgd3 = lcapgd3;
} else {
		/*
		 * inverse mode - source and drain interchanged
		 */


here->cdr_x = -gm1;
here->cdr_y = gm1 + gds1;
here->cdr_x2 = -gm2;
here->cdr_y2 = -(gm2 + gds2 + 2*gmds);
here->cdr_xy = gm2 + gmds;
here->cdr_x3 = -gm3;
here->cdr_y3 = gm3 + gds3 + 3*(gm2ds + gmds2 ) ;
here->cdr_x2y = gm3 + gm2ds;
here->cdr_xy2 = -(gm3 + 2*gm2ds + gmds2);

here->ggs1 = lggd1;
here->ggd1 = lggs1;

here->ggs2 = lggd2;
here->ggd2 = lggs2;

here->ggs3 = lggd3;
here->ggd3 = lggs3;

here->capgs1 = lcapgd1;
here->capgd1 = lcapgs1;

here->capgs2 = lcapgd2;
here->capgd2 = lcapgs2;

here->capgs3 = lcapgd3;
here->capgd3 = lcapgs3;
}

/* now to adjust for type and multiply by factors to convert to Taylor coeffs. */

here->cdr_x2 = 0.5*model->JFETtype*here->cdr_x2;
here->cdr_y2 = 0.5*model->JFETtype*here->cdr_y2;
here->cdr_xy = model->JFETtype*here->cdr_xy;
here->cdr_x3 = here->cdr_x3/6.;
here->cdr_y3 = here->cdr_y3/6.;
here->cdr_x2y = 0.5*here->cdr_x2y;
here->cdr_xy2 = 0.5*here->cdr_xy2;


here->ggs2 = model->JFETtype*lggs2;
here->ggd2 = model->JFETtype*lggd2;

here->capgs2 = model->JFETtype*lcapgs2;
here->capgd2 = model->JFETtype*lcapgd2;

		 }
		 }
		 return(OK);
		 }
