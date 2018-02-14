/**********
Based on jfetload.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to add PS model and new parameter definitions ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
   10 Feb 1994:  New code added to call psmodel.c routines
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfet2defs.h"
#include "ngspice/const.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "psmodel.h"
#include "ngspice/suffix.h"

int
JFET2load(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current resistance value into the 
         * sparse matrix previously provided 
         */
{
    JFET2model *model = (JFET2model*)inModel;
    JFET2instance *here;
    double capgd;
    double capgs;
    double cd;
    double cdhat = 0.0;
    double cdreq;
    double ceq;
    double ceqgd;
    double ceqgs;
    double cg;
    double cgd;
    double cghat = 0.0;
    double delvds;
    double delvgd;
    double delvgs;
    double gdpr;
    double gds;
    double geq;
    double ggd;
    double ggs;
    double gm;
    double gspr;
    double vds;
    double vgd;
    double vgs;
#ifndef PREDICTOR
    double xfact;
#endif
    int icheck;
    int ichk1;
    int error;

    double m;

    /*  loop through all the models */
    for( ; model != NULL; model = JFET2nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = JFET2instances(model); here != NULL ;
                here=JFET2nextInstance(here)) {
            /*
             *  dc model parameters 
             */
            gdpr=model->JFET2drainConduct*here->JFET2area;
            gspr=model->JFET2sourceConduct*here->JFET2area;
            /*
             *    initialization
             */
            icheck=1;
            if( ckt->CKTmode & MODEINITSMSIG) {
                vgs= *(ckt->CKTstate0 + here->JFET2vgs);
                vgd= *(ckt->CKTstate0 + here->JFET2vgd);
            } else if (ckt->CKTmode & MODEINITTRAN) {
                vgs= *(ckt->CKTstate1 + here->JFET2vgs);
                vgd= *(ckt->CKTstate1 + here->JFET2vgd);
            } else if ( (ckt->CKTmode & MODEINITJCT) &&
                    (ckt->CKTmode & MODETRANOP) &&
                    (ckt->CKTmode & MODEUIC) ) {
                vds=model->JFET2type*here->JFET2icVDS;
                vgs=model->JFET2type*here->JFET2icVGS;
                vgd=vgs-vds;
            } else if ( (ckt->CKTmode & MODEINITJCT) &&
                    (here->JFET2off == 0)  ) {
                vgs = -1;
                vgd = -1;
            } else if( (ckt->CKTmode & MODEINITJCT) ||
                    ((ckt->CKTmode & MODEINITFIX) && (here->JFET2off))) {
                vgs = 0;
                vgd = 0;
            } else {
#ifndef PREDICTOR
                if(ckt->CKTmode & MODEINITPRED) {
                    xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    *(ckt->CKTstate0 + here->JFET2vgs)= 
                            *(ckt->CKTstate1 + here->JFET2vgs);
                    vgs=(1+xfact)* *(ckt->CKTstate1 + here->JFET2vgs)-xfact*
                            *(ckt->CKTstate2 + here->JFET2vgs);
                    *(ckt->CKTstate0 + here->JFET2vgd)= 
                            *(ckt->CKTstate1 + here->JFET2vgd);
                    vgd=(1+xfact)* *(ckt->CKTstate1 + here->JFET2vgd)-xfact*
                            *(ckt->CKTstate2 + here->JFET2vgd);
                    *(ckt->CKTstate0 + here->JFET2cg)= 
                            *(ckt->CKTstate1 + here->JFET2cg);
                    *(ckt->CKTstate0 + here->JFET2cd)= 
                            *(ckt->CKTstate1 + here->JFET2cd);
                    *(ckt->CKTstate0 + here->JFET2cgd)=
                            *(ckt->CKTstate1 + here->JFET2cgd);
                    *(ckt->CKTstate0 + here->JFET2gm)=
                            *(ckt->CKTstate1 + here->JFET2gm);
                    *(ckt->CKTstate0 + here->JFET2gds)=
                            *(ckt->CKTstate1 + here->JFET2gds);
                    *(ckt->CKTstate0 + here->JFET2ggs)=
                            *(ckt->CKTstate1 + here->JFET2ggs);
                    *(ckt->CKTstate0 + here->JFET2ggd)=
                            *(ckt->CKTstate1 + here->JFET2ggd);
                } else {
#endif /*PREDICTOR*/
                    /*
                     *  compute new nonlinear branch voltages 
                     */
                    vgs=model->JFET2type*
                        (*(ckt->CKTrhsOld+ here->JFET2gateNode)-
                        *(ckt->CKTrhsOld+ 
                        here->JFET2sourcePrimeNode));
                    vgd=model->JFET2type*
                        (*(ckt->CKTrhsOld+here->JFET2gateNode)-
                        *(ckt->CKTrhsOld+
                        here->JFET2drainPrimeNode));
#ifndef PREDICTOR
                }
#endif /*PREDICTOR*/
                delvgs=vgs- *(ckt->CKTstate0 + here->JFET2vgs);
                delvgd=vgd- *(ckt->CKTstate0 + here->JFET2vgd);
                delvds=delvgs-delvgd;
                cghat= *(ckt->CKTstate0 + here->JFET2cg)+ 
                        *(ckt->CKTstate0 + here->JFET2ggd)*delvgd+
                        *(ckt->CKTstate0 + here->JFET2ggs)*delvgs;
                cdhat= *(ckt->CKTstate0 + here->JFET2cd)+
                        *(ckt->CKTstate0 + here->JFET2gm)*delvgs+
                        *(ckt->CKTstate0 + here->JFET2gds)*delvds-
                        *(ckt->CKTstate0 + here->JFET2ggd)*delvgd;
                /*
                 *   bypass if solution has not changed 
                 */
                if((ckt->CKTbypass) &&
                    (!(ckt->CKTmode & MODEINITPRED)) &&
                    (fabs(delvgs) < ckt->CKTreltol*MAX(fabs(vgs),
                        fabs(*(ckt->CKTstate0 + here->JFET2vgs)))+
                        ckt->CKTvoltTol) )
        if ( (fabs(delvgd) < ckt->CKTreltol*MAX(fabs(vgd),
                        fabs(*(ckt->CKTstate0 + here->JFET2vgd)))+
                        ckt->CKTvoltTol))
        if ( (fabs(cghat-*(ckt->CKTstate0 + here->JFET2cg)) 
                        < ckt->CKTreltol*MAX(fabs(cghat),
                        fabs(*(ckt->CKTstate0 + here->JFET2cg)))+
                        ckt->CKTabstol) ) if ( /* hack - expression too big */
                    (fabs(cdhat-*(ckt->CKTstate0 + here->JFET2cd))
                        < ckt->CKTreltol*MAX(fabs(cdhat),
                        fabs(*(ckt->CKTstate0 + here->JFET2cd)))+
                        ckt->CKTabstol) ) {

                    /* we can do a bypass */
                    vgs= *(ckt->CKTstate0 + here->JFET2vgs);
                    vgd= *(ckt->CKTstate0 + here->JFET2vgd);
                    vds= vgs-vgd;
                    cg= *(ckt->CKTstate0 + here->JFET2cg);
                    cd= *(ckt->CKTstate0 + here->JFET2cd);
                    cgd= *(ckt->CKTstate0 + here->JFET2cgd);
                    gm= *(ckt->CKTstate0 + here->JFET2gm);
                    gds= *(ckt->CKTstate0 + here->JFET2gds);
                    ggs= *(ckt->CKTstate0 + here->JFET2ggs);
                    ggd= *(ckt->CKTstate0 + here->JFET2ggd);
                    goto load;
                }
                /*
                 *  limit nonlinear branch voltages 
                 */
                ichk1=1;
                vgs = DEVpnjlim(vgs,*(ckt->CKTstate0 + here->JFET2vgs),
                        (here->JFET2temp*CONSTKoverQ), here->JFET2vcrit, &icheck);
                vgd = DEVpnjlim(vgd,*(ckt->CKTstate0 + here->JFET2vgd),
                        (here->JFET2temp*CONSTKoverQ), here->JFET2vcrit,&ichk1);
                if (ichk1 == 1) {
                    icheck=1;
                }
                vgs = DEVfetlim(vgs,*(ckt->CKTstate0 + here->JFET2vgs),
                        model->JFET2vto);
                vgd = DEVfetlim(vgd,*(ckt->CKTstate0 + here->JFET2vgd),
                        model->JFET2vto);
            }
            /*
             *   determine dc current and derivatives 
             */
            vds=vgs-vgd;
            if (vds < 0.0) {
               cd = -PSids(ckt, model, here, vgd, vgs,
                            &cgd, &cg, &ggd, &ggs, &gm, &gds);
               gds += gm;
               gm  = -gm;
            } else {
               cd =  PSids(ckt, model, here, vgs, vgd, 
                            &cg, &cgd, &ggs, &ggd, &gm, &gds);
            }
            cg = cg + cgd;
            cd = cd - cgd;

            if ( (ckt->CKTmode & (MODETRAN | MODEAC | MODEINITSMSIG) ) ||
                    ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) ){
                /* 
                 *    charge storage elements 
                 */
                double capds = model->JFET2capds*here->JFET2area;

                PScharge(ckt, model, here, vgs, vgd, &capgs, &capgd);

                *(ckt->CKTstate0 + here->JFET2qds) = capds * vds;
            
                /*
                 *   store small-signal parameters 
                 */
                if( (!(ckt->CKTmode & MODETRANOP)) || 
                        (!(ckt->CKTmode & MODEUIC)) ) {
                    if(ckt->CKTmode & MODEINITSMSIG) {
                        *(ckt->CKTstate0 + here->JFET2qgs) = capgs;
                        *(ckt->CKTstate0 + here->JFET2qgd) = capgd;
                        *(ckt->CKTstate0 + here->JFET2qds) = capds;
                        continue; /*go to 1000*/
                    }
                    /*
                     *   transient analysis 
                     */
                    if(ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->JFET2qgs) =
                                *(ckt->CKTstate0 + here->JFET2qgs);
                        *(ckt->CKTstate1 + here->JFET2qgd) =
                                *(ckt->CKTstate0 + here->JFET2qgd);
                        *(ckt->CKTstate1 + here->JFET2qds) =
                                *(ckt->CKTstate0 + here->JFET2qds);
                    }
                    error = NIintegrate(ckt,&geq,&ceq,capgs,here->JFET2qgs);
                    if(error) return(error);
                    ggs = ggs + geq;
                    cg = cg + *(ckt->CKTstate0 + here->JFET2cqgs);
                    error = NIintegrate(ckt,&geq,&ceq,capgd,here->JFET2qgd);
                    if(error) return(error);
                    ggd = ggd + geq;
                    cg = cg + *(ckt->CKTstate0 + here->JFET2cqgd);
                    cd = cd - *(ckt->CKTstate0 + here->JFET2cqgd);
                    cgd = cgd + *(ckt->CKTstate0 + here->JFET2cqgd);
                    error = NIintegrate(ckt,&geq,&ceq,capds,here->JFET2qds);
                    cd = cd + *(ckt->CKTstate0 + here->JFET2cqds);
                    if(error) return(error);
                    if (ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->JFET2cqgs) =
                                *(ckt->CKTstate0 + here->JFET2cqgs);
                        *(ckt->CKTstate1 + here->JFET2cqgd) =
                                *(ckt->CKTstate0 + here->JFET2cqgd);
                        *(ckt->CKTstate1 + here->JFET2cqds) =
                                *(ckt->CKTstate0 + here->JFET2cqds);
                    }
                }
            }
            /*
             *  check convergence 
             */
            if( (!(ckt->CKTmode & MODEINITFIX)) | (!(ckt->CKTmode & MODEUIC))) {
                if((icheck == 1) ||
		   (fabs(cghat-cg) >= ckt->CKTreltol*
		    MAX(fabs(cghat),fabs(cg))+ckt->CKTabstol) ||
		   (fabs(cdhat-cd) > ckt->CKTreltol*
		    MAX(fabs(cdhat),fabs(cd))+ckt->CKTabstol)) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }
            *(ckt->CKTstate0 + here->JFET2vgs) = vgs;
            *(ckt->CKTstate0 + here->JFET2vgd) = vgd;
            *(ckt->CKTstate0 + here->JFET2cg) = cg;
            *(ckt->CKTstate0 + here->JFET2cd) = cd;
            *(ckt->CKTstate0 + here->JFET2cgd) = cgd;
            *(ckt->CKTstate0 + here->JFET2gm) = gm;
            *(ckt->CKTstate0 + here->JFET2gds) = gds;
            *(ckt->CKTstate0 + here->JFET2ggs) = ggs;
            *(ckt->CKTstate0 + here->JFET2ggd) = ggd;
            /*
             *    load current vector
             */
load:

            m = here->JFET2m;

            ceqgd=model->JFET2type*(cgd-ggd*vgd);
            ceqgs=model->JFET2type*((cg-cgd)-ggs*vgs);
            cdreq=model->JFET2type*((cd+cgd)-gds*vds-gm*vgs);
            *(ckt->CKTrhs + here->JFET2gateNode) += m * (-ceqgs-ceqgd);
            *(ckt->CKTrhs + here->JFET2drainPrimeNode) +=
                    m * (-cdreq+ceqgd);
            *(ckt->CKTrhs + here->JFET2sourcePrimeNode) +=
                    m * (cdreq+ceqgs);
            /*
             *    load y matrix 
             */
            *(here->JFET2drainDrainPrimePtr)        += m * (-gdpr);
            *(here->JFET2gateDrainPrimePtr)         += m * (-ggd);
            *(here->JFET2gateSourcePrimePtr)        += m * (-ggs);
            *(here->JFET2sourceSourcePrimePtr)      += m * (-gspr);
            *(here->JFET2drainPrimeDrainPtr)        += m * (-gdpr);
            *(here->JFET2drainPrimeGatePtr)         += m * (gm-ggd);
            *(here->JFET2drainPrimeSourcePrimePtr)  += m * (-gds-gm);
            *(here->JFET2sourcePrimeGatePtr)        += m * (-ggs-gm);
            *(here->JFET2sourcePrimeSourcePtr)      += m * (-gspr);
            *(here->JFET2sourcePrimeDrainPrimePtr)  += m * (-gds);
            *(here->JFET2drainDrainPtr)             += m * (gdpr);
            *(here->JFET2gateGatePtr)               += m * (ggd+ggs);
            *(here->JFET2sourceSourcePtr)           += m * (gspr);
            *(here->JFET2drainPrimeDrainPrimePtr)   += m * (gdpr+gds+ggd);
            *(here->JFET2sourcePrimeSourcePrimePtr) += m * (gspr+gds+gm+ggs);
        }
    }
    return(OK);
}
