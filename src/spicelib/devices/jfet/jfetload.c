/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
Sydney University mods Copyright(c) 1989 Anthony E. Parker, David J. Skellern
        Laboratory for Communication Science Engineering
        Sydney University Department of Electrical Engineering, Australia
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfetdefs.h"
#include "ngspice/const.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

int
JFETload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current resistance value into the 
         * sparse matrix previously provided 
         */
{
    JFETmodel *model = (JFETmodel*)inModel;
    JFETinstance *here;
    double beta;
    double betap;
    double capgd;
    double capgs;
    double cd;
    double cdhat = 0.0;
    double cdrain;
    double cdreq;
    double ceq;
    double ceqgd;
    double ceqgs;
    double cg;
    double cgd;
    double cghat = 0.0;
    double csat;
    double czgd;
    double czgdf2;
    double czgs;
    double czgsf2;
    double delvds;
    double delvgd;
    double delvgs;
    double evgd;
    double evgs;
    double fcpb2;
    double gdpr;
    double gds;
    double geq;
    double ggd;
    double ggs;
    double gm;
    double gspr;
    double sarg;
    double twop;
    double vds;
    double vgd;
    double vgdt;
    double vgs;
    double vgst;
#ifndef PREDICTOR
    double xfact;
#endif
    /* Modification for Sydney University JFET model */
    double vto;
    double apart,cpart;
    double Bfac;
    /* end Sydney University mod. */
    int icheck;
    int ichk1;
    int error;
    
    double arg, vt_temp;

    double m;

    /*  loop through all the models */
    for( ; model != NULL; model = JFETnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = JFETinstances(model); here != NULL ;
                here=JFETnextInstance(here)) {

            /*
             *  dc model parameters 
             */
            beta = here->JFETtBeta * here->JFETarea;
            gdpr=model->JFETdrainConduct*here->JFETarea;
            gspr=model->JFETsourceConduct*here->JFETarea;
            csat=here->JFETtSatCur*here->JFETarea;
            /*
             *    initialization
             */
            icheck=1;
            if( ckt->CKTmode & MODEINITSMSIG) {
                vgs= *(ckt->CKTstate0 + here->JFETvgs);
                vgd= *(ckt->CKTstate0 + here->JFETvgd);
            } else if (ckt->CKTmode & MODEINITTRAN) {
                vgs= *(ckt->CKTstate1 + here->JFETvgs);
                vgd= *(ckt->CKTstate1 + here->JFETvgd);
            } else if ( (ckt->CKTmode & MODEINITJCT) &&
                    (ckt->CKTmode & MODETRANOP) &&
                    (ckt->CKTmode & MODEUIC) ) {
                vds=model->JFETtype*here->JFETicVDS;
                vgs=model->JFETtype*here->JFETicVGS;
                vgd=vgs-vds;
            } else if ( (ckt->CKTmode & MODEINITJCT) &&
                    (here->JFEToff == 0)  ) {
                vgs = -1;
                vgd = -1;
            } else if( (ckt->CKTmode & MODEINITJCT) ||
                    ((ckt->CKTmode & MODEINITFIX) && (here->JFEToff))) {
                vgs = 0;
                vgd = 0;
            } else {
#ifndef PREDICTOR
                if(ckt->CKTmode & MODEINITPRED) {
                    xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    *(ckt->CKTstate0 + here->JFETvgs)= 
                            *(ckt->CKTstate1 + here->JFETvgs);
                    vgs=(1+xfact)* *(ckt->CKTstate1 + here->JFETvgs)-xfact*
                            *(ckt->CKTstate2 + here->JFETvgs);
                    *(ckt->CKTstate0 + here->JFETvgd)= 
                            *(ckt->CKTstate1 + here->JFETvgd);
                    vgd=(1+xfact)* *(ckt->CKTstate1 + here->JFETvgd)-xfact*
                            *(ckt->CKTstate2 + here->JFETvgd);
                    *(ckt->CKTstate0 + here->JFETcg)= 
                            *(ckt->CKTstate1 + here->JFETcg);
                    *(ckt->CKTstate0 + here->JFETcd)= 
                            *(ckt->CKTstate1 + here->JFETcd);
                    *(ckt->CKTstate0 + here->JFETcgd)=
                            *(ckt->CKTstate1 + here->JFETcgd);
                    *(ckt->CKTstate0 + here->JFETgm)=
                            *(ckt->CKTstate1 + here->JFETgm);
                    *(ckt->CKTstate0 + here->JFETgds)=
                            *(ckt->CKTstate1 + here->JFETgds);
                    *(ckt->CKTstate0 + here->JFETggs)=
                            *(ckt->CKTstate1 + here->JFETggs);
                    *(ckt->CKTstate0 + here->JFETggd)=
                            *(ckt->CKTstate1 + here->JFETggd);
                } else {
#endif /*PREDICTOR*/
                    /*
                     *  compute new nonlinear branch voltages 
                     */
                    vgs=model->JFETtype*
                        (*(ckt->CKTrhsOld+ here->JFETgateNode)-
                        *(ckt->CKTrhsOld+ 
                        here->JFETsourcePrimeNode));
                    vgd=model->JFETtype*
                        (*(ckt->CKTrhsOld+here->JFETgateNode)-
                        *(ckt->CKTrhsOld+
                        here->JFETdrainPrimeNode));
#ifndef PREDICTOR
                }
#endif /*PREDICTOR*/
                delvgs=vgs- *(ckt->CKTstate0 + here->JFETvgs);
                delvgd=vgd- *(ckt->CKTstate0 + here->JFETvgd);
                delvds=delvgs-delvgd;
                cghat= *(ckt->CKTstate0 + here->JFETcg)+ 
                        *(ckt->CKTstate0 + here->JFETggd)*delvgd+
                        *(ckt->CKTstate0 + here->JFETggs)*delvgs;
                cdhat= *(ckt->CKTstate0 + here->JFETcd)+
                        *(ckt->CKTstate0 + here->JFETgm)*delvgs+
                        *(ckt->CKTstate0 + here->JFETgds)*delvds-
                        *(ckt->CKTstate0 + here->JFETggd)*delvgd;
                /*
                 *   bypass if solution has not changed 
                 */
                if((ckt->CKTbypass) &&
                    (!(ckt->CKTmode & MODEINITPRED)) &&
                    (fabs(delvgs) < ckt->CKTreltol*MAX(fabs(vgs),
                        fabs(*(ckt->CKTstate0 + here->JFETvgs)))+
                        ckt->CKTvoltTol) )
                if ( (fabs(delvgd) < ckt->CKTreltol*MAX(fabs(vgd),
                        fabs(*(ckt->CKTstate0 + here->JFETvgd)))+
                        ckt->CKTvoltTol))
                if ( (fabs(cghat-*(ckt->CKTstate0 + here->JFETcg)) 
                        < ckt->CKTreltol*MAX(fabs(cghat),
                        fabs(*(ckt->CKTstate0 + here->JFETcg)))+
                        ckt->CKTabstol) ) 
                if ( /* hack - expression too big */
                    (fabs(cdhat-*(ckt->CKTstate0 + here->JFETcd))
                        < ckt->CKTreltol*MAX(fabs(cdhat),
                        fabs(*(ckt->CKTstate0 + here->JFETcd)))+
                        ckt->CKTabstol) ) {

                    /* we can do a bypass */
                    vgs= *(ckt->CKTstate0 + here->JFETvgs);
                    vgd= *(ckt->CKTstate0 + here->JFETvgd);
                    vds= vgs-vgd;
                    cg= *(ckt->CKTstate0 + here->JFETcg);
                    cd= *(ckt->CKTstate0 + here->JFETcd);
                    cgd= *(ckt->CKTstate0 + here->JFETcgd);
                    gm= *(ckt->CKTstate0 + here->JFETgm);
                    gds= *(ckt->CKTstate0 + here->JFETgds);
                    ggs= *(ckt->CKTstate0 + here->JFETggs);
                    ggd= *(ckt->CKTstate0 + here->JFETggd);
                    goto load;
                }
                /*
                 *  limit nonlinear branch voltages 
                 */
                ichk1=1;
                vgs = DEVpnjlim(vgs,*(ckt->CKTstate0 + here->JFETvgs),
                        (here->JFETtemp*CONSTKoverQ), here->JFETvcrit, &icheck);
                vgd = DEVpnjlim(vgd,*(ckt->CKTstate0 + here->JFETvgd),
                        (here->JFETtemp*CONSTKoverQ), here->JFETvcrit,&ichk1);
                if (ichk1 == 1) {
                    icheck=1;
                }
                vgs = DEVfetlim(vgs,*(ckt->CKTstate0 + here->JFETvgs),
                        here->JFETtThreshold);
                vgd = DEVfetlim(vgd,*(ckt->CKTstate0 + here->JFETvgd),
                        here->JFETtThreshold);
            }
            /*
             *   determine dc current and derivatives 
             */
            vds=vgs-vgd;
            
            vt_temp=here->JFETtemp*CONSTKoverQ;
            if (vgs < -3*vt_temp) {
                arg=3*vt_temp/(vgs*CONSTe);
                arg = arg * arg * arg;
                cg = -csat*(1+arg)+ckt->CKTgmin*vgs;
                ggs = csat*3*arg/vgs+ckt->CKTgmin;
            } else {
                evgs = exp(vgs/vt_temp);
                ggs = csat*evgs/vt_temp+ckt->CKTgmin;
                cg = csat*(evgs-1)+ckt->CKTgmin*vgs;
            }
            
            if (vgd < -3*vt_temp) {
                arg=3*vt_temp/(vgd*CONSTe);
                arg = arg * arg * arg;
                cgd = -csat*(1+arg)+ckt->CKTgmin*vgd;
                ggd = csat*3*arg/vgd+ckt->CKTgmin;
            } else {
                evgd = exp(vgd/vt_temp);
                ggd = csat*evgd/vt_temp+ckt->CKTgmin;
                cgd = csat*(evgd-1)+ckt->CKTgmin*vgd;
            }
            
            cg = cg+cgd;

            /* Modification for Sydney University JFET model */
            vto = here->JFETtThreshold;
            if (vds >= 0) {
                vgst = vgs - vto;
                /*
                 * compute drain current and derivatives for normal mode
                 */
                if (vgst <= 0) {
                    /*
                     * normal mode, cutoff region
                     */
                    cdrain = 0;
                    gm = 0;
                    gds = 0;
                } else {
                    betap = beta*(1 + model->JFETlModulation*vds);
                    Bfac = model->JFETbFac;
                    if (vgst >= vds) {
                        /*
                         * normal mode, linear region
                         */
                        apart = 2*model->JFETb + 3*Bfac*(vgst - vds);
                        cpart = vds*(vds*(Bfac*vds - model->JFETb)+vgst*apart);
                        cdrain = betap*cpart;
                        gm = betap*vds*(apart + 3*Bfac*vgst);
                        gds = betap*(vgst - vds)*apart
                              + beta*model->JFETlModulation*cpart;
                    } else {
                        Bfac = vgst*Bfac;
                        gm = betap*vgst*(2*model->JFETb+3*Bfac);
                        /*
                         * normal mode, saturation region
                         */
                        cpart=vgst*vgst*(model->JFETb+Bfac);
                        cdrain = betap*cpart;
                        gds = model->JFETlModulation*beta*cpart;
                    }
                }
            } else {
                vgdt = vgd - vto;
                /*
                 * compute drain current and derivatives for inverse mode
                 */
                if (vgdt <= 0) {
                    /*
                     * inverse mode, cutoff region
                     */
                    cdrain = 0;
                    gm = 0;
                    gds = 0;
                } else {
                    betap = beta*(1 - model->JFETlModulation*vds);
                    Bfac = model->JFETbFac;
                    if (vgdt + vds >= 0) {
                        /*
                         * inverse mode, linear region
                         */
                        apart = 2*model->JFETb + 3*Bfac*(vgdt + vds);
                        cpart = vds*(-vds*(-Bfac*vds-model->JFETb)+vgdt*apart);
                        cdrain = betap*cpart;
                        gm = betap*vds*(apart + 3*Bfac*vgdt);
                        gds = betap*(vgdt + vds)*apart
                             - beta*model->JFETlModulation*cpart - gm;
                    } else {
                        Bfac = vgdt*Bfac;
                        gm = -betap*vgdt*(2*model->JFETb+3*Bfac);
                        /*
                         * inverse mode, saturation region
                         */
                        cpart=vgdt*vgdt*(model->JFETb+Bfac);
                        cdrain = - betap*cpart;
                        gds = model->JFETlModulation*beta*cpart-gm;
                    }
                }
            }

#ifdef notdef
            /* The original section is now commented out */
            /* end Sydney University mod */
            /*
             *   compute drain current and derivitives for normal mode 
             */
            if (vds >= 0) {
                vgst=vgs-here->JFETtThreshold;
                /*
                 *   normal mode, cutoff region 
                 */
                if (vgst <= 0) {
                    cdrain=0;
                    gm=0;
                    gds=0;
                } else {
                    betap=beta*(1+model->JFETlModulation*vds);
                    twob=betap+betap;
                    if (vgst <= vds) {
                        /*
                         *   normal mode, saturation region 
                         */
                        cdrain=betap*vgst*vgst;
                        gm=twob*vgst;
                        gds=model->JFETlModulation*beta*vgst*vgst;
                    } else {
                        /*
                         *   normal mode, linear region 
                         */
                        cdrain=betap*vds*(vgst+vgst-vds);
                        gm=twob*vds;
                        gds=twob*(vgst-vds)+model->JFETlModulation*beta*
                                vds*(vgst+vgst-vds);
                    }
                }
            } else {
                /*
                 *   compute drain current and derivitives for inverse mode 
                 */
                vgdt=vgd-here->JFETtThreshold;
                if (vgdt <= 0) {
                    /*
                     *   inverse mode, cutoff region 
                     */
                    cdrain=0;
                    gm=0;
                    gds=0;
                } else {
                    /*
                     *   inverse mode, saturation region 
                     */
                    betap=beta*(1-model->JFETlModulation*vds);
                    twob=betap+betap;
                    if (vgdt <= -vds) {
                        cdrain = -betap*vgdt*vgdt;
                        gm = -twob*vgdt;
                        gds = model->JFETlModulation*beta*vgdt*vgdt-gm;
                    } else {
                        /*
                         *  inverse mode, linear region 
                         */
                        cdrain=betap*vds*(vgdt+vgdt+vds);
                        gm=twob*vds;
                        gds=twob*vgdt-model->JFETlModulation*beta*vds*
                                (vgdt+vgdt+vds);
                    }
                }
            }
            /* end of original section, now deleted (replaced w/SU mod */
#endif

            /*
             *   compute equivalent drain current source 
             */
            cd=cdrain-cgd;
            if ( (ckt->CKTmode & (MODEDCTRANCURVE | MODETRAN | MODEAC | MODEINITSMSIG) ) ||
                    ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) ){
                /* 
                 *    charge storage elements 
                 */
                czgs=here->JFETtCGS*here->JFETarea;
                czgd=here->JFETtCGD*here->JFETarea;
                twop=here->JFETtGatePot+here->JFETtGatePot;
                fcpb2=here->JFETcorDepCap*here->JFETcorDepCap;
                czgsf2=czgs/model->JFETf2;
                czgdf2=czgd/model->JFETf2;
                if (vgs < here->JFETcorDepCap) {
                    sarg=sqrt(1-vgs/here->JFETtGatePot);
                    *(ckt->CKTstate0 + here->JFETqgs) = twop*czgs*(1-sarg);
                    capgs=czgs/sarg;
                } else {
                    *(ckt->CKTstate0 + here->JFETqgs) = czgs*here->JFETf1 +
                            czgsf2*(model->JFETf3 *(vgs-
                            here->JFETcorDepCap)+(vgs*vgs-fcpb2)/
                            (twop+twop));
                    capgs=czgsf2*(model->JFETf3+vgs/twop);
                }
                if (vgd < here->JFETcorDepCap) {
                    sarg=sqrt(1-vgd/here->JFETtGatePot);
                    *(ckt->CKTstate0 + here->JFETqgd) = twop*czgd*(1-sarg);
                    capgd=czgd/sarg;
                } else {
                    *(ckt->CKTstate0 + here->JFETqgd) = czgd*here->JFETf1+
                            czgdf2*(model->JFETf3* (vgd-
                            here->JFETcorDepCap)+(vgd*vgd-fcpb2)/
                            (twop+twop));
                    capgd=czgdf2*(model->JFETf3+vgd/twop);
                }
                /*
                 *   store small-signal parameters 
                 */
                if( (!(ckt->CKTmode & MODETRANOP)) || 
                        (!(ckt->CKTmode & MODEUIC)) ) {
                    if(ckt->CKTmode & MODEINITSMSIG) {
                        *(ckt->CKTstate0 + here->JFETqgs) = capgs;
                        *(ckt->CKTstate0 + here->JFETqgd) = capgd;
                        continue; /*go to 1000*/
                    }
                    /*
                     *   transient analysis 
                     */
                    if(ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->JFETqgs) =
                                *(ckt->CKTstate0 + here->JFETqgs);
                        *(ckt->CKTstate1 + here->JFETqgd) =
                                *(ckt->CKTstate0 + here->JFETqgd);
                    }
                    error = NIintegrate(ckt,&geq,&ceq,capgs,here->JFETqgs);
                    if(error) return(error);
                    ggs = ggs + geq;
                    cg = cg + *(ckt->CKTstate0 + here->JFETcqgs);
                    error = NIintegrate(ckt,&geq,&ceq,capgd,here->JFETqgd);
                    if(error) return(error);
                    ggd = ggd + geq;
                    cg = cg + *(ckt->CKTstate0 + here->JFETcqgd);
                    cd = cd - *(ckt->CKTstate0 + here->JFETcqgd);
                    cgd = cgd + *(ckt->CKTstate0 + here->JFETcqgd);
                    if (ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->JFETcqgs) =
                                *(ckt->CKTstate0 + here->JFETcqgs);
                        *(ckt->CKTstate1 + here->JFETcqgd) =
                                *(ckt->CKTstate0 + here->JFETcqgd);
                    }
                }
            }
            /*
             *  check convergence 
             */
            if( (!(ckt->CKTmode & MODEINITFIX)) |
                (!(ckt->CKTmode & MODEUIC))) {
                if((icheck == 1) ||
                   (fabs(cghat-cg) >= ckt->CKTreltol *
                    MAX(fabs(cghat), fabs(cg)) + ckt->CKTabstol) ||
                   (fabs(cdhat-cd) > ckt->CKTreltol *
                    MAX(fabs(cdhat), fabs(cd)) + ckt->CKTabstol)) {
                    ckt->CKTnoncon++;
                    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }
            *(ckt->CKTstate0 + here->JFETvgs) = vgs;
            *(ckt->CKTstate0 + here->JFETvgd) = vgd;
            *(ckt->CKTstate0 + here->JFETcg) = cg;
            *(ckt->CKTstate0 + here->JFETcd) = cd;
            *(ckt->CKTstate0 + here->JFETcgd) = cgd;
            *(ckt->CKTstate0 + here->JFETgm) = gm;
            *(ckt->CKTstate0 + here->JFETgds) = gds;
            *(ckt->CKTstate0 + here->JFETggs) = ggs;
            *(ckt->CKTstate0 + here->JFETggd) = ggd;
            /*
             *    load current vector
             */
load:

            m = here->JFETm;

            ceqgd=model->JFETtype*(cgd-ggd*vgd);
            ceqgs=model->JFETtype*((cg-cgd)-ggs*vgs);
            cdreq=model->JFETtype*((cd+cgd)-gds*vds-gm*vgs);
            *(ckt->CKTrhs + here->JFETgateNode) += m * (-ceqgs-ceqgd);
            *(ckt->CKTrhs + here->JFETdrainPrimeNode) +=
                    m * (-cdreq+ceqgd);
            *(ckt->CKTrhs + here->JFETsourcePrimeNode) +=
                    m * (cdreq+ceqgs);
            /*
             *    load y matrix 
             */
            *(here->JFETdrainDrainPrimePtr)        += m * (-gdpr);
            *(here->JFETgateDrainPrimePtr)         += m * (-ggd);
            *(here->JFETgateSourcePrimePtr)        += m * (-ggs);
            *(here->JFETsourceSourcePrimePtr)      += m * (-gspr);
            *(here->JFETdrainPrimeDrainPtr)        += m * (-gdpr);
            *(here->JFETdrainPrimeGatePtr)         += m * (gm-ggd);
            *(here->JFETdrainPrimeSourcePrimePtr)  += m * (-gds-gm);
            *(here->JFETsourcePrimeGatePtr)        += m * (-ggs-gm);
            *(here->JFETsourcePrimeSourcePtr)      += m * (-gspr);
            *(here->JFETsourcePrimeDrainPrimePtr)  += m * (-gds);
            *(here->JFETdrainDrainPtr)             += m * (gdpr);
            *(here->JFETgateGatePtr)               += m * (ggd+ggs);
            *(here->JFETsourceSourcePtr)           += m * (gspr);
            *(here->JFETdrainPrimeDrainPrimePtr)   += m * (gdpr+gds+ggd);
            *(here->JFETsourcePrimeSourcePrimePtr) += m * (gspr+gds+gm+ggs);
        }
    }
    return(OK);
}
