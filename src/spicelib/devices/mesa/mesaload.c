/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "mesadefs.h"
#include "ngspice/const.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*
#define true 1
#define false 0
*/

#define EPSILONGAAS (12.244*8.85418e-12)

#define W (here->MESAwidth)
#define L (here->MESAlength)

static void mesa1(MESAmodel*, MESAinstance*, double, double, double,
                  double*, double*, double*, double*,double*);

static void mesa2(MESAmodel*, MESAinstance*, double, double, double,
                  double*, double*, double*, double*,double*);
                  
static void mesa3(MESAmodel*, MESAinstance*, double, double, double,
                  double*, double*, double*, double*,double*);

void Pause(void);

int
MESAload(GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel*)inModel;
    MESAinstance *here;
    double capgd = 0.0;
    double capgs = 0.0;
    double cd;
    double cdhat = 0.0;
    double cdrain = 0.0;
    double cdreq;
    double ceq;
    double ceqgd;
    double ceqgs;
    double cg;
    double cgd;
    double cgs;
    double cghat = 0.0;
    double arg;
    double earg;
    double delvds;
    double delvgd;
    double delvgs;
    double delvgspp = 0.0;
    double delvgdpp = 0.0;
    double evgd;
    double evgs;
    double gds = 0.0;                            
    double geq;
    double ggd;
    double ggs;
    double gm = 0.0;
    double ggspp = 0.0;
    double cgspp = 0.0;
    double ggdpp = 0.0;
    double cgdpp = 0.0;
    double vcrits;
    double vcritd;
    double vds = 0.0;
    double vgd = 0.0;
    double vgs = 0.0;
    double vgspp = 0.0;
    double vgdpp = 0.0;    
    double vgs1 = 0.0;
    double vgd1 = 0.0;
#ifndef PREDICTOR
    double xfact;
#endif
    double vted;
    double vtes;
    double vtd;
    double vts;
    double von;
    double ccorr;
    int inverse = FALSE;
    int icheck;
    int ichk1;
    int error;

    double m;

    /*  loop through all the models */
    for( ; model != NULL; model = MESAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MESAinstances(model); here != NULL ;
                here=MESAnextInstance(here)) {

            /*
             *  dc model parameters 
             */
            vcrits = here->MESAvcrits;
            vcritd = here->MESAvcritd;
            vtd  = CONSTKoverQ * here->MESAtd;
            vts  = CONSTKoverQ * here->MESAts;
            vted = model->MESAn*vtd;
            vtes = model->MESAn*vts;
            /*
             *    initialization
             */
            icheck = 1;
            if( ckt->CKTmode & MODEINITSMSIG) {
                vgs = *(ckt->CKTstate0 + here->MESAvgs);
                vgd = *(ckt->CKTstate0 + here->MESAvgd);
                vgspp = *(ckt->CKTstate0 + here->MESAvgspp);
                vgdpp = *(ckt->CKTstate0 + here->MESAvgdpp);
            } else if (ckt->CKTmode & MODEINITTRAN) {
                vgs = *(ckt->CKTstate1 + here->MESAvgs);
                vgd = *(ckt->CKTstate1 + here->MESAvgd);
                vgspp = *(ckt->CKTstate1 + here->MESAvgspp);
                vgdpp = *(ckt->CKTstate1 + here->MESAvgdpp);
            } else if ( (ckt->CKTmode & MODEINITJCT) &&
                    (ckt->CKTmode & MODETRANOP) &&
                    (ckt->CKTmode & MODEUIC) ) {
                vds = here->MESAicVDS;
                vgs = here->MESAicVGS;
                vgd = vgs-vds;
                vgspp = vgs;
                vgdpp = vgd;
            } else if ( (ckt->CKTmode & MODEINITJCT) &&
                    (here->MESAoff == 0)  ) {
                vgs = -1;
                vgd = -1;
                vgspp = 0;
                vgdpp = 0;
            } else if( (ckt->CKTmode & MODEINITJCT) ||
                    ((ckt->CKTmode & MODEINITFIX) && (here->MESAoff))) {
                vgs = 0;
                vgd = 0;
                vgspp = 0;
                vgdpp = 0;
            } else {
#ifndef PREDICTOR
                if(ckt->CKTmode & MODEINITPRED) {
                    xfact = ckt->CKTdelta/ckt->CKTdeltaOld[2];
                    *(ckt->CKTstate0 + here->MESAvgs) = 
                            *(ckt->CKTstate1 + here->MESAvgs);
                    vgs = (1+xfact) * *(ckt->CKTstate1 + here->MESAvgs) -
                           xfact * *(ckt->CKTstate2 + here->MESAvgs);
                    *(ckt->CKTstate0 + here->MESAvgspp) =
                            *(ckt->CKTstate1 + here->MESAvgspp);
                    vgspp = (1+xfact) * *(ckt->CKTstate1 + here->MESAvgspp) -
                           xfact * *(ckt->CKTstate2 + here->MESAvgspp);
                    *(ckt->CKTstate0 + here->MESAvgd) = 
                            *(ckt->CKTstate1 + here->MESAvgd);
                    vgd = (1+xfact)* *(ckt->CKTstate1 + here->MESAvgd) -
                           xfact * *(ckt->CKTstate2 + here->MESAvgd);
                    *(ckt->CKTstate0 + here->MESAvgdpp) =
                            *(ckt->CKTstate1 + here->MESAvgdpp);
                    vgdpp = (1+xfact) * *(ckt->CKTstate1 + here->MESAvgdpp) -
                           xfact * *(ckt->CKTstate2 + here->MESAvgdpp);
                    *(ckt->CKTstate0 + here->MESAcg) = 
                            *(ckt->CKTstate1 + here->MESAcg);
                    *(ckt->CKTstate0 + here->MESAcd) = 
                            *(ckt->CKTstate1 + here->MESAcd);
                    *(ckt->CKTstate0 + here->MESAcgd) =
                            *(ckt->CKTstate1 + here->MESAcgd);
                    *(ckt->CKTstate0 + here->MESAcgs) =
                            *(ckt->CKTstate1 + here->MESAcgs);                            
                    *(ckt->CKTstate0 + here->MESAgm) =
                            *(ckt->CKTstate1 + here->MESAgm);
                    *(ckt->CKTstate0 + here->MESAgds) =
                            *(ckt->CKTstate1 + here->MESAgds);
                    *(ckt->CKTstate0 + here->MESAggs) =
                            *(ckt->CKTstate1 + here->MESAggs);
                    *(ckt->CKTstate0 + here->MESAggd) =
                            *(ckt->CKTstate1 + here->MESAggd);
                    *(ckt->CKTstate0 + here->MESAggspp) =
                            *(ckt->CKTstate1 + here->MESAggspp);
                    *(ckt->CKTstate0 + here->MESAcgspp) =
                            *(ckt->CKTstate1 + here->MESAcgspp);
                    *(ckt->CKTstate0 + here->MESAggdpp) =
                            *(ckt->CKTstate1 + here->MESAggdpp);
                    *(ckt->CKTstate0 + here->MESAcgdpp) =
                            *(ckt->CKTstate1 + here->MESAcgdpp);
                } else {
#endif /* PREDICTOR */
                    /*
                     *  compute new nonlinear branch voltages 
                     */
                    vgs = (*(ckt->CKTrhsOld+ here->MESAgatePrimeNode)-
                        *(ckt->CKTrhsOld+here->MESAsourcePrimeNode));
                    vgd = (*(ckt->CKTrhsOld+here->MESAgatePrimeNode)-
                        *(ckt->CKTrhsOld+here->MESAdrainPrimeNode));
                    vgspp = (*(ckt->CKTrhsOld+here->MESAgatePrimeNode)-
                        *(ckt->CKTrhsOld+here->MESAsourcePrmPrmNode));
                    vgdpp = (*(ckt->CKTrhsOld+here->MESAgatePrimeNode)-
                        *(ckt->CKTrhsOld+here->MESAdrainPrmPrmNode));
                        
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */
                delvgs=vgs - *(ckt->CKTstate0 + here->MESAvgs);
                delvgd=vgd - *(ckt->CKTstate0 + here->MESAvgd);
                delvds=delvgs - delvgd;
                delvgspp = vgspp - *(ckt->CKTstate0 + here->MESAvgspp);
                delvgdpp = vgdpp - *(ckt->CKTstate0 + here->MESAvgdpp);
                cghat= *(ckt->CKTstate0 + here->MESAcg) + 
                        *(ckt->CKTstate0 + here->MESAggd)*delvgd +
                        *(ckt->CKTstate0 + here->MESAggs)*delvgs +
                        *(ckt->CKTstate0 + here->MESAggspp)*delvgspp+
                        *(ckt->CKTstate0 + here->MESAggdpp)*delvgdpp;
                cdhat= *(ckt->CKTstate0 + here->MESAcd) +
                        *(ckt->CKTstate0 + here->MESAgm)*delvgs +
                        *(ckt->CKTstate0 + here->MESAgds)*delvds -
                        *(ckt->CKTstate0 + here->MESAggd)*delvgd;
                /*
                 *   bypass if solution has not changed 
                 */
                if((ckt->CKTbypass) &&
                    (!(ckt->CKTmode & MODEINITPRED)) &&
                    (fabs(delvgs) < ckt->CKTreltol*MAX(fabs(vgs),
                        fabs(*(ckt->CKTstate0 + here->MESAvgs)))+
                        ckt->CKTvoltTol) )
                if((fabs(delvgd) < ckt->CKTreltol*MAX(fabs(vgd),
                        fabs(*(ckt->CKTstate0 + here->MESAvgd)))+
                        ckt->CKTvoltTol))
                if((fabs(delvgspp) < ckt->CKTreltol*MAX(fabs(vgspp),
                        fabs(*(ckt->CKTstate0 + here->MESAvgspp)))+
                        ckt->CKTvoltTol))
                if((fabs(delvgdpp) < ckt->CKTreltol*MAX(fabs(vgdpp),
                        fabs(*(ckt->CKTstate0 + here->MESAvgdpp)))+
                        ckt->CKTvoltTol))
                if((fabs(cghat-*(ckt->CKTstate0 + here->MESAcg)) 
                        < ckt->CKTreltol*MAX(fabs(cghat),
                        fabs(*(ckt->CKTstate0 + here->MESAcg)))+
                        ckt->CKTabstol))
                if((fabs(cdhat-*(ckt->CKTstate0 + here->MESAcd))
                        < ckt->CKTreltol*MAX(fabs(cdhat),
                        fabs(*(ckt->CKTstate0 + here->MESAcd)))+
                        ckt->CKTabstol)) {

                    /* we can do a bypass */
                    vgs= *(ckt->CKTstate0 + here->MESAvgs);
                    vgd= *(ckt->CKTstate0 + here->MESAvgd);
                    vds= vgs-vgd;
                    vgspp = *(ckt->CKTstate0 + here->MESAvgspp);
                    vgdpp = *(ckt->CKTstate0 + here->MESAvgdpp);
                    cg= *(ckt->CKTstate0 + here->MESAcg);
                    cd= *(ckt->CKTstate0 + here->MESAcd);
                    cgs= *(ckt->CKTstate0 + here->MESAcgs);
                    cgd= *(ckt->CKTstate0 + here->MESAcgd);
                    gm= *(ckt->CKTstate0 + here->MESAgm);
                    gds= *(ckt->CKTstate0 + here->MESAgds);
                    ggs= *(ckt->CKTstate0 + here->MESAggs);
                    ggd= *(ckt->CKTstate0 + here->MESAggd);
                    ggspp = *(ckt->CKTstate0 + here->MESAggspp);
                    cgspp = *(ckt->CKTstate0 + here->MESAcgspp);
                    ggdpp = *(ckt->CKTstate0 + here->MESAggdpp);
                    cgdpp = *(ckt->CKTstate0 + here->MESAcgdpp);
                    goto load;
                }
                /*
                 *  limit nonlinear branch voltages 
                 */
                ichk1=1;
                vgs = DEVpnjlim(vgs,*(ckt->CKTstate0 + here->MESAvgs),vtes,
                        vcrits, &icheck);
                vgd = DEVpnjlim(vgd,*(ckt->CKTstate0 + here->MESAvgd),vted,
                        vcritd,&ichk1);
                if (ichk1 == 1) {
                    icheck=1;
                }
                vgs = DEVfetlim(vgs,*(ckt->CKTstate0 + here->MESAvgs),
                        here->MESAtVto);
                vgd = DEVfetlim(vgd,*(ckt->CKTstate0 + here->MESAvgd),
                        here->MESAtVto);
                if(here->MESAsourcePrmPrmNode == here->MESAsourcePrimeNode)
                  vgspp = vgs;
                if(here->MESAdrainPrmPrmNode == here->MESAdrainPrimeNode)
                  vgdpp = vgd;
            }
            /*
             *   determine dc current and derivatives 
             */
            vds = vgs-vgd;
                      
            arg = -vgs*model->MESAdel/vts;
            earg = exp(arg);
            evgs = exp(vgs/vtes);
            ggs  = here->MESAcsatfs*evgs/vtes+here->MESAggrwl*earg*(1-arg)+ckt->CKTgmin;
            cgs  = here->MESAcsatfs*(evgs-1)+here->MESAggrwl*vgs*earg+
                   ckt->CKTgmin*vgs;
            cg   = cgs;
             
            arg = -vgd*model->MESAdel/vtd;
            earg = exp(arg);
            evgd = exp(vgd/vted);
            ggd  = here->MESAcsatfd*evgd/vted+here->MESAggrwl*earg*(1-arg)+ckt->CKTgmin;
            cgd  = here->MESAcsatfd*(evgd-1)+here->MESAggrwl*vgd*earg+
                   ckt->CKTgmin*vgd;
            cg = cg+cgd;
            
            if(vds < 0) {
              vds = -vds;
              inverse = TRUE;
            }
            von = here->MESAtVto+model->MESAks*(*(ckt->CKTrhsOld+here->MESAsourcePrimeNode)-model->MESAvsg);
            if(model->MESAlevel == 2)
              mesa1(model,here,inverse?vgd:vgs,vds,von,&cdrain,&gm,&gds,&capgs,&capgd);
            else if(model->MESAlevel == 3)
              mesa2(model,here,inverse?vgd:vgs,vds,von,&cdrain,&gm,&gds,&capgs,&capgd);
            else if(model->MESAlevel == 4)
              mesa3(model,here,inverse?vgd:vgs,vds,von,&cdrain,&gm,&gds,&capgs,&capgd);
            if(inverse) {
              cdrain = -cdrain;
              vds = -vds;
              SWAP(double, capgs, capgd);
            }
            /*
             *   compute equivalent drain current source 
             */
            cd = cdrain - cgd;
            if ( (ckt->CKTmode & (MODETRAN|MODEINITSMSIG)) ||
                    ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) ){
                /* 
                 *    charge storage elements 
                 */
                vgs1 = *(ckt->CKTstate1 + here->MESAvgspp);
                vgd1 = *(ckt->CKTstate1 + here->MESAvgdpp);

                if(ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate1 + here->MESAqgs) = capgs*vgspp;
                    *(ckt->CKTstate1 + here->MESAqgd) = capgd*vgdpp;
                }
                *(ckt->CKTstate0+here->MESAqgs) = *(ckt->CKTstate1 + here->MESAqgs) +
                                                   capgs*(vgspp-vgs1);
                *(ckt->CKTstate0+here->MESAqgd) = *(ckt->CKTstate1 + here->MESAqgd) +
                                                   capgd*(vgdpp-vgd1);

                /*
                 *   store small-signal parameters 
                 */
                if( (!(ckt->CKTmode & MODETRANOP)) || 
                        (!(ckt->CKTmode & MODEUIC)) ) {
                    if(ckt->CKTmode & MODEINITSMSIG) {
                        *(ckt->CKTstate0 + here->MESAqgs) = capgs;
                        *(ckt->CKTstate0 + here->MESAqgd) = capgd;
                        continue; /*go to 1000*/
                    }
                    /*
                     *   transient analysis 
                     */
                    if(ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->MESAqgs) =
                                *(ckt->CKTstate0 + here->MESAqgs);
                        *(ckt->CKTstate1 + here->MESAqgd) =
                                *(ckt->CKTstate0 + here->MESAqgd);
                    }
                    error = NIintegrate(ckt,&geq,&ceq,capgs,here->MESAqgs);
                    if(error) return(error);
                    ggspp = geq;
                    cgspp = *(ckt->CKTstate0 + here->MESAcqgs);
                    cg = cg + cgspp;
                    error = NIintegrate(ckt,&geq,&ceq,capgd,here->MESAqgd);
                    if(error) return(error);
                    ggdpp = geq;
                    cgdpp = *(ckt->CKTstate0 + here->MESAcqgd);
                    cg = cg + cgdpp;
                    cd = cd - cgdpp;
                    if (ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->MESAcqgs) =
                                *(ckt->CKTstate0 + here->MESAcqgs);
                        *(ckt->CKTstate1 + here->MESAcqgd) =
                                *(ckt->CKTstate0 + here->MESAcqgd);
                    }
                }
            }
            /*
             *  check convergence 
             */
            if( (!(ckt->CKTmode & MODEINITFIX)) | (!(ckt->CKTmode & MODEUIC))) {
                if( (icheck == 1) 
                        || (fabs(cghat-cg) >= ckt->CKTreltol*
                            MAX(fabs(cghat),fabs(cg))+ckt->CKTabstol) ||
                        (fabs(cdhat-cd) > ckt->CKTreltol*
                            MAX(fabs(cdhat),fabs(cd))+ckt->CKTabstol) 
                        ) {
                    ckt->CKTnoncon++;
                }
            }
            *(ckt->CKTstate0 + here->MESAvgs) = vgs;
            *(ckt->CKTstate0 + here->MESAvgspp) = vgspp;
            *(ckt->CKTstate0 + here->MESAvgd) = vgd;
            *(ckt->CKTstate0 + here->MESAvgdpp) = vgdpp;
            *(ckt->CKTstate0 + here->MESAcg) = cg;
            *(ckt->CKTstate0 + here->MESAcd) = cd;
            *(ckt->CKTstate0 + here->MESAcgd) = cgd;
            *(ckt->CKTstate0 + here->MESAcgs) = cgs;
            *(ckt->CKTstate0 + here->MESAgm) = gm;
            *(ckt->CKTstate0 + here->MESAgds) = gds;
            *(ckt->CKTstate0 + here->MESAggs) = ggs;
            *(ckt->CKTstate0 + here->MESAggd) = ggd;
            *(ckt->CKTstate0 + here->MESAggspp) = ggspp;
            *(ckt->CKTstate0 + here->MESAcgspp) = cgspp;
            *(ckt->CKTstate0 + here->MESAggdpp) = ggdpp;
            *(ckt->CKTstate0 + here->MESAcgdpp) = cgdpp;
            
            /*
             *    load current vector
             */
load:

            m = here->MESAm;

            ccorr = model->MESAag*(cgs-cgd);
            ceqgd = cgd + cgdpp - ggd*vgd - ggdpp*vgdpp;
            ceqgs = cgs + cgspp - ggs*vgs - ggspp*vgspp;
            cdreq=((cd+cgd+cgdpp)-gds*vds-gm*vgs);
            *(ckt->CKTrhs + here->MESAgatePrimeNode) += m * (-ceqgs-ceqgd);
            ceqgd = (cgd-ggd*vgd);
            *(ckt->CKTrhs + here->MESAdrainPrimeNode) += m * (-cdreq+ceqgd+ccorr);
            ceqgd = (cgdpp-ggdpp*vgdpp);
            *(ckt->CKTrhs + here->MESAdrainPrmPrmNode) += ceqgd;
            ceqgs = (cgs-ggs*vgs);
            *(ckt->CKTrhs + here->MESAsourcePrimeNode) += m * (cdreq+ceqgs-ccorr);
            ceqgs = (cgspp-ggspp*vgspp);
            *(ckt->CKTrhs + here->MESAsourcePrmPrmNode) += ceqgs;

            /*
             *    load y matrix 
             */
            *(here->MESAdrainDrainPtr)               += m * (here->MESAdrainConduct);
            *(here->MESAsourceSourcePtr)             += m * (here->MESAsourceConduct);
            *(here->MESAgateGatePtr)                 += m * (here->MESAgateConduct);
            *(here->MESAsourcePrmPrmSourcePrmPrmPtr) += m * (here->MESAtGi+ggspp);
            *(here->MESAdrainPrmPrmDrainPrmPrmPtr)   += m * (here->MESAtGf+ggdpp);
            *(here->MESAgatePrimeGatePrimePtr)       += m * (ggd+ggs+here->MESAgateConduct+ggspp+ggdpp);
            *(here->MESAdrainPrimeDrainPrimePtr)     += m * (gds+ggd+here->MESAdrainConduct+here->MESAtGf);
            *(here->MESAsourcePrimeSourcePrimePtr)   += m * (gds+gm+ggs+here->MESAsourceConduct+here->MESAtGi);
            *(here->MESAdrainDrainPrimePtr)          -= m * (here->MESAdrainConduct);
            *(here->MESAdrainPrimeDrainPtr)          -= m * (here->MESAdrainConduct);
            *(here->MESAsourceSourcePrimePtr)        -= m * (here->MESAsourceConduct);
            *(here->MESAsourcePrimeSourcePtr)        -= m * (here->MESAsourceConduct);
            *(here->MESAgateGatePrimePtr)            -= m * (here->MESAgateConduct);
            *(here->MESAgatePrimeGatePtr)            -= m * (here->MESAgateConduct);
            *(here->MESAgatePrimeDrainPrimePtr)      -= m * (ggd);
            *(here->MESAgatePrimeSourcePrimePtr)     -= m * (ggs);
            *(here->MESAdrainPrimeGatePrimePtr)      += m * (gm-ggd);
            *(here->MESAdrainPrimeSourcePrimePtr)    += m * (-gds-gm);
            *(here->MESAsourcePrimeGatePrimePtr)     += m * (-ggs-gm);
            *(here->MESAsourcePrimeDrainPrimePtr)    -= m * (gds);
            *(here->MESAsourcePrimeSourcePrmPrmPtr)  -= m * (here->MESAtGi);
            *(here->MESAsourcePrmPrmSourcePrimePtr)  -= m * (here->MESAtGi);
            *(here->MESAgatePrimeSourcePrmPrmPtr)    -= m * (ggspp);
            *(here->MESAsourcePrmPrmGatePrimePtr)    -= m * (ggspp);
            *(here->MESAdrainPrimeDrainPrmPrmPtr)    -= m * (here->MESAtGf);
            *(here->MESAdrainPrmPrmDrainPrimePtr)    -= m * (here->MESAtGf);
            *(here->MESAgatePrimeDrainPrmPrmPtr)     -= m * (ggdpp);
            *(here->MESAdrainPrmPrmGatePrimePtr)     -= m * (ggdpp);
        }
    }
    return(OK);
}


#define ETA     (here->MESAtEta)
#define MU0     (here->MESAtMu)
#define RS      (here->MESAtRsi)
#define RD      (here->MESAtRdi)
#define SIGMA0  (model->MESAsigma0)
#define VSIGMAT (model->MESAvsigmat)
#define VSIGMA  (model->MESAvsigma)
#define THETA   (model->MESAtheta)
#define VS      (model->MESAvs)
#define ND      (model->MESAnd)
#define D       (model->MESAd)
#define TC      (model->MESAtc)
#define MC      (model->MESAmc)
#define M0      (model->MESAm)
#define ALPHA   (model->MESAalpha)
#define LAMBDA  (here->MESAtLambda)
#define NDELTA  (model->MESAndelta)
#define TH      (model->MESAth)
#define NDU     (model->MESAndu)
#define DU      (model->MESAdu)
#define NMAX    (model->MESAnmax)
#define GAMMA   (model->MESAgamma)


static void mesa1(MESAmodel *model, MESAinstance *here, double vgs,
                  double vds, double von, double *cdrain,
                  double *gm, double *gds, double *capgs, double *capgd)

{

  double vt;
  double etavth;  
  double vl;
  double rt;
  double mu;
  double beta;
  double vgt;
  double vgt0;
  double sigma;
  double vgte;
  double isat;
  double isata;
  double isatb;
  double ns;
  double a;
  double b;
  double c;
  double d;
  double e;
  double f;
  double g;
  double h;
  double m;
  double p;
  double q;
  double r;
  double s;
  double t;
  double u;
  double v;
  double w;
  double temp;
  double gch;
  double gchi;
  double vsate;
  double vdse;
  double cgc;
  double sqrt1;
  double delidgch;
  double delgchgchi;
  double delgchins;
  double delnsvgt;
  double delnsvgte;
  double delvgtevgt;
  double delidvsate;
  double delvsateisat;
  double delisatisata;
  double delisatavgte;
  double delisatabeta;
  double delisatisatb;
  double delvsategch;
  double delidvds;
  double ddevgte;
  double dvgtvgs;
  double dgchivgt;
  double dvgtevds;
  double dgchivds;
  double disatavgt;
  double disatavds;
  double disatbvgt;
  double dvsatevgt;
  double dvsatevds;
  double gmmadd;
  double gdsmadd;
  

  vt       = CONSTKoverQ * here->MESAts;
  etavth   = ETA*vt;
  rt       = RS+RD;
  vgt0     = vgs - von;
  s        = exp((vgt0-VSIGMAT)/VSIGMA);
  sigma    = SIGMA0/(1+s);
  vgt      = vgt0+sigma*vds;
  mu       = MU0+THETA*vgt;
  vl       = VS/mu*here->MESAlength;
  beta     = here->MESAbeta/(model->MESAvpo+3*vl);
  u        = vgt/vt-1;
  t        = sqrt(model->MESAdeltaSqr+u*u);
  vgte     = 0.5*vt*(2+u+t);
  a        = 2*beta*vgte;
  b        = exp(-vgt/etavth);
  if(vgte > model->MESAvpo)
    sqrt1 = 0;
  else
    sqrt1    = sqrt(1-vgte/model->MESAvpo);
  ns       = 1.0/(1.0/ND/D/(1-sqrt1) + 1.0/here->MESAn0*b);
  if(ns < 1.0e-38) {
    *cdrain = 0;
    *gm = 0.0;
    *gds = 0.0;
    *capgs = here->MESAcf;
    *capgd = here->MESAcf;
    return;
  }
  gchi     = here->MESAgchi0*mu*ns;
  gch      = gchi/(1+gchi*rt);
  f        = sqrt(1+2*a*RS);
  d        = 1+a*RS + f;
  e        = 1+TC*vgte;
  isata    = a*vgte/(d*e);
  isatb    = here->MESAisatb0*mu*exp(vgt/etavth);
  isat     = isata*isatb/(isata+isatb);
  vsate    = isat/gch;
  vdse     = vds*pow(1+pow(vds/vsate,MC),-1.0/MC);
  m        = M0+ALPHA*vgte;
  g        = pow(vds/vsate,m);
  h        = pow(1+g,1.0/m);
  here->MESAdelidgch0 = vds/h;
  delidgch = here->MESAdelidgch0*(1+LAMBDA*vds);
  *cdrain  = gch*delidgch;
  if(vgt > model->MESAvpo)
    temp = 0;
  else
    temp = sqrt(1-vgt/model->MESAvpo);
  cgc      = W*L*EPSILONGAAS/(temp+b)/D;
  c        = (vsate-vdse)/(2*vsate-vdse);
  c        = c*c;
  *capgs   = here->MESAcf+2.0/3.0*cgc*(1-c);
  c        = vsate/(2*vsate-vdse);
  c        = c*c;
  *capgd   = here->MESAcf+2.0/3.0*cgc*(1-c);
  temp     = 1+gchi*rt;
  delgchgchi   = 1.0/(temp*temp);
  delgchins    = here->MESAgchi0*mu;
  delnsvgt     = ns*ns*1.0/here->MESAn0/etavth*b;
  q            = 1 - sqrt1;
  if(sqrt1 == 0)
    delnsvgte = 0;
  else
    delnsvgte = 0.5*ns*ns/(model->MESAvpo*ND*D*sqrt1*q*q);
  delvgtevgt   = 0.5*(1+u/t);
  here->MESAdelidvds0 = gch/h;
  if(vds != 0.0)
    here->MESAdelidvds1 = (*cdrain)*pow(vds/vsate,m-1)/(vsate*(1+g));
  else
    here->MESAdelidvds1 = 0.0;
  delidvds     = here->MESAdelidvds0*(1+2*LAMBDA*vds) - here->MESAdelidvds1;
  delidvsate   = (*cdrain)*g/(vsate*(1+g));
  delvsateisat = 1.0/gch;
  r            = isata+isatb;
  r            = r*r;
  delisatisata = isatb*isatb/r;
  v            = 1.0+1.0/f;
  ddevgte      = 2*beta*RS*v*e+d*TC;
  temp         = d*d*e*e;
  delisatavgte = (2*a*d*e - a*vgte*ddevgte)/temp;
  delisatabeta = 2*vgte*vgte*(d*e-a*e*RS*v)/temp;
  delisatisatb = isata*isata/r;
  delvsategch  = -vsate/gch;
  dvgtvgs      = 1 - SIGMA0*vds*s/VSIGMA/((1+s)*(1+s));
  temp         = here->MESAgchi0*ns*THETA;
  dgchivgt     = delgchins*(delnsvgte*delvgtevgt+delnsvgt)+temp;
  dvgtevds     = delvgtevgt*sigma;
  dgchivds     = delgchins*(delnsvgte*dvgtevds+delnsvgt*sigma)+temp*sigma;
  temp         = delisatabeta*3*beta*vl*THETA/(mu*(model->MESAvpo+3*vl));
  disatavgt    = delisatavgte*delvgtevgt+temp;
  disatavds    = delisatavgte*dvgtevds+temp*sigma;
  disatbvgt    = isatb/etavth+isatb/mu*THETA;
  p            = delgchgchi*dgchivgt;
  w            = delgchgchi*dgchivds;
  dvsatevgt    = delvsateisat*(delisatisata*disatavgt+delisatisatb*disatbvgt)+delvsategch*p;
  dvsatevds    = delvsateisat*(delisatisata*disatavds+delisatisatb*disatbvgt*sigma)+delvsategch*w;
  if(ALPHA != 0) {
    if(vds == 0)
      gmmadd = 0;
    else
      gmmadd     = (*cdrain)*(log(1+g)/(m*m)-g*log(vds/vsate)/(m*(1+g)))*
                   ALPHA*delvgtevgt;
    gdsmadd    = gmmadd*sigma;
  } else {
    gmmadd     = 0;
    gdsmadd    = 0;
  }  
  here->MESAgm0 = p;
  here->MESAgm1 = delidvsate*dvsatevgt;
  here->MESAgm2 = dvgtvgs;
  g            = delidgch*p+here->MESAgm1;
  *gm          = (g+gmmadd)*dvgtvgs;
  here->MESAgds0 = delidvsate*dvsatevds+delidgch*w+gdsmadd;
  *gds         = delidvds+here->MESAgds0;
  
}



static void mesa2(MESAmodel *model, MESAinstance *here, double vgs,
                  double vds, double von, double *cdrain, double *gm,
                  double *gds, double *capgs, double *capgd)

{

  double vt;
  double rt;
  double vgt;
  double etavth;
  double vgt0;
  double sigma;
  double vgte;
  double isat;
  double isata;
  double isatb;
  double nsa;
  double nsb;
  double ns;
  double a;
  double b;
  double c;
  double d;
  double e;
  double f;
  double g;
  double h;
  double p;
  double q;
  double r;
  double s;
  double t;
  double gch;
  double gchi;
  double vsate;
  double vdse;
  double ca;
  double cb;
  double cgc;
  double delidgch;
  double delgchgchi;
  double delgchins;
  double delnsvgt;
  double delnsbvgt;
  double delnsavgte;
  double delvgtevgt;
  double delidvsate;
  double delvsateisat;
  double delisatisata;
  double delisatavgte;
  double delisatisatb;
  double delvsategch;
  double delisatbvgt;
  double delvsatevgt;
  double delidvds;
  double ddevgte;
  double delvgtvgs;


  vt     = CONSTKoverQ * here->MESAts;
  etavth = ETA*vt;
  rt     = RS+RD;
  vgt0   = vgs - von;
  s      = exp((vgt0-VSIGMAT)/VSIGMA);
  sigma  = SIGMA0/(1+s);
  vgt    = vgt0+sigma*vds;
  t      = vgt/vt-1;
  q      = sqrt(model->MESAdeltaSqr+t*t);
  vgte   = 0.5*vt*(2+t+q);
  a      = 2*model->MESAbeta*vgte;
  if(vgt > model->MESAvpod) {
    if(vgte > model->MESAvpo) {
      nsa = NDELTA*TH + NDU*DU;
      ca  = EPSILONGAAS/DU;
      delnsavgte = 0;
    } else {
      r = sqrt((model->MESAvpo-vgte)/model->MESAvpou);
      nsa = NDELTA*TH + NDU*DU*(1-r);
      ca  = EPSILONGAAS/DU/r;
      delnsavgte = NDU*DU/model->MESAvpou/2.0/r;
    }
  } else {
    if(model->MESAvpod - vgte < 0) {
      nsa = NDELTA*TH*(1-DU/TH);
      ca  = EPSILONGAAS/DU;
      delnsavgte = 0;
    } else {
      r   = sqrt(1+NDU/NDELTA*(model->MESAvpod - vgte)/model->MESAvpou);
      nsa = NDELTA*TH*(1-DU/TH*(r-1));
      ca  = EPSILONGAAS/DU/r;
      delnsavgte = DU*NDU/2.0/model->MESAvpou/r;
    }
  }
  b      = exp(vgt/etavth);
  cb     = EPSILONGAAS/(DU+TH)*b;
  nsb    = here->MESAnsb0*b;
  delnsbvgt = nsb/etavth;
  ns     = nsa*nsb/(nsa+nsb);
  if(ns < 1.0e-38) {
    *cdrain = 0;
    *gm = 0.0;
    *gds = 0.0;
    *capgs = here->MESAcf;
    *capgd = here->MESAcf;
    return;
  }
  gchi   = here->MESAgchi0*ns;
  gch    = gchi/(1+gchi*rt);
  f      = sqrt(1+2*a*RS);
  d      = 1+a*RS + f;
  e      = 1+TC*vgte;
  isata  = a*vgte/d/e;
  isatb  = here->MESAisatb0*b;
  isat   = isata*isatb/(isata+isatb);
  vsate  = isat/gch;
  vdse   = vds*pow(1+pow(vds/vsate,MC),-1.0/MC);
  g      = pow(vds/vsate,M0);
  h      = pow(1+g,1.0/M0);
  here->MESAdelidgch0 = vds/h;
  delidgch = here->MESAdelidgch0*(1+LAMBDA*vds);
  *cdrain  = gch*delidgch;
  cgc      = W*L*ca*cb/(ca+cb);
  c        = (vsate-vdse)/(2*vsate-vdse);
  c        = c*c;
  *capgs   = here->MESAcf+2.0/3.0*cgc*(1-c);
  c        = vsate/(2*vsate-vdse);
  c        = c*c;
  *capgd   = here->MESAcf+2.0/3.0*cgc*(1-c);
  c        = vgt/vt-1;
  delvgtevgt = 0.5*(1+t/q);
  here->MESAdelidvds0 = gch/h;
  if(vds != 0.0)
    here->MESAdelidvds1 = (*cdrain)*pow(vds/vsate,M0-1)/vsate/(1+g);
  else
    here->MESAdelidvds1 = 0.0;
  delidvds     = here->MESAdelidvds0*(1+2*LAMBDA*vds) -
                 here->MESAdelidvds1;
  delgchgchi = 1.0/(1+gchi*rt)/(1+gchi*rt);
  delgchins  = here->MESAgchi0;
  r          = nsa+nsb;
  r          = r*r;
  delnsvgt   = (nsb*nsb*delvgtevgt*delnsavgte + nsa*nsa*delnsbvgt)/r;
  delidvsate   = (*cdrain)*g/vsate/(1+g);
  delvsateisat = 1.0/gch;
  r            = isata+isatb;
  r            = r*r;
  delisatisata = isatb*isatb/r;
  ddevgte      = 2*model->MESAbeta*RS*(1+1.0/f)*e+d*TC;
  delisatavgte = (2*a*d*e - a*vgte*ddevgte)/d/d/e/e;
  delisatisatb = isata*isata/r;
  delisatbvgt  = isatb/etavth;
  delvsategch  = -vsate/gch;
  delvgtvgs    = 1-SIGMA0*vds*s/VSIGMA/(1+s)/(1+s);
  p            = delgchgchi*delgchins*delnsvgt;
  delvsatevgt  = delvsateisat*(delisatisata*delisatavgte*delvgtevgt +
                 delisatisatb*delisatbvgt) + delvsategch*p;
  here->MESAgm0 = p;
  here->MESAgm1 = delidvsate*delvsatevgt;
  here->MESAgm2 = delvgtvgs;
  g            = delidgch*p + here->MESAgm1;
  *gm          = g*delvgtvgs;
  here->MESAgds0 = g*sigma;
  *gds         = delidvds+here->MESAgds0;

}



static void mesa3(MESAmodel *model, MESAinstance *here, double vgs,
                  double vds, double von, double *cdrain, double *gm,
                  double *gds, double *capgs, double *capgd)

{

  double vt;
  double vgt;
  double vgt0;
  double sigma;
  double vgte;
  double isat;
  double isatm;
  double ns;
  double nsm;
  double a;
  double b;
  double c;
  double d;
  double e;
  double g;
  double h;
  double p;
  double q;
  double s;
  double t;
  double u;
  double temp;
  double etavth;
  double gch;
  double gchi;
  double gchim;
  double vsate;
  double vdse;
  double cgc;
  double cgcm;
  double rt;
  double vl;
  double delidgch;
  double delgchgchi;
  double delgchins;
  double delnsnsm;
  double delnsmvgt;
  double delvgtevgt;
  double delidvsate;
  double delvsateisat;
  double delisatisatm;
  double delisatmvgte;
  double delisatmgchim;
  double delvsategch;
  double delidvds;
  double delvgtvgs;
  double delvsatevgt;
       
  vt     = CONSTKoverQ * here->MESAts;     
  etavth = ETA*vt;
  vl     = VS/MU0*L;
  rt     = RS+RD;
  vgt0   = vgs - von;
  s      = exp((vgt0-VSIGMAT)/VSIGMA);
  sigma  = SIGMA0/(1+s);
  vgt    = vgt0+sigma*vds;
  u      = 0.5*vgt/vt-1;
  t      = sqrt(model->MESAdeltaSqr+u*u);
  vgte   = vt*(2+u+t);
  b      = exp(vgt/etavth);
  nsm    = 2*here->MESAn0*log(1+0.5*b);
  if(nsm < 1.0e-38) {
    *cdrain = 0;
    *gm = 0.0;
    *gds = 0.0;
    *capgs = here->MESAcf;
    *capgd = here->MESAcf;
    return;
  }
  c      = pow(nsm/NMAX,GAMMA);
  q      = pow(1+c,1.0/GAMMA);
  ns     = nsm/q;
  gchi   = here->MESAgchi0*ns;
  gch    = gchi/(1+gchi*rt);
  gchim  = here->MESAgchi0*nsm;
  h      = sqrt(1+2*gchim*model->MESArsi + vgte*vgte/(vl*vl));
  p      = 1+gchim*RS+h;
  isatm  = gchim*vgte/p;
  g      = pow(isatm/here->MESAimax,GAMMA);
  isat   = isatm/pow(1+g,1/GAMMA);
  vsate  = isat/gch;
  vdse   = vds*pow(1+pow(vds/vsate,MC),-1.0/MC);
  d      = pow(vds/vsate,M0);
  e      = pow(1+d,1.0/M0);
  delidgch      = vds*(1+LAMBDA*vds)/e;
  *cdrain       = gch*delidgch;
  cgcm          = 1.0/(1/model->MESAcas*D/model->MESAepsi +
                  1/model->MESAcbs*etavth/CHARGE/here->MESAn0*exp(-vgt/etavth));
  cgc           = W*L*cgcm/pow(1+c,1+1.0/GAMMA);
/*  
  {
  char buf[256];
  void far pascal OutputDebugString(char*);
  sprintf(buf,"\n%f\t%e",vgs,cgc);
  OutputDebugString(buf);  
  }
*/  
  a             = (vsate-vdse)/(2*vsate-vdse);
  a             = a*a;
  temp          = 2.0/3.0;
  *capgs        = here->MESAcf+temp*cgc*(1-a);
  a             = vsate/(2*vsate-vdse);
  a             = a*a;
  *capgd        = here->MESAcf+temp*cgc*(1-a);
  delidvsate    = (*cdrain)*d/vsate/(1+d);
  delidvds      = gch*(1+2*LAMBDA*vds)/e-(*cdrain)*
                  pow(vds/vsate,M0-1)/(vsate*(1+d));
  a             = 1+gchi*rt;
  delgchgchi    = 1.0/(a*a);
  delgchins     = here->MESAgchi0;
  delnsnsm      = ns/nsm*(1-c/(1+c));
  delnsmvgt     = here->MESAn0/etavth/(1.0/b + 0.5);
  delvgtevgt    = 0.5*(1+u/t);
  delvsateisat  = 1.0/gch;
  delisatisatm  = isat/isatm*(1-g/(1+g));
  delisatmvgte  = gchim*(p - vgte*vgte/(vl*vl*h))/(p*p);
  delvsategch   = -vsate/gch;
  delisatmgchim = vgte*(p - gchim*RS*(1+1.0/h))/(p*p);
  delvgtvgs     = 1-vds*SIGMA0/VSIGMA*s/((1+s)*(1+s));
  p             = delgchgchi*delgchins*delnsnsm*delnsmvgt;
  delvsatevgt   = (delvsateisat*delisatisatm*(delisatmvgte*delvgtevgt +
                  delisatmgchim*here->MESAgchi0*delnsmvgt)+delvsategch*p);
  g             = delidgch*p + delidvsate*delvsatevgt;
  *gm           = g*delvgtvgs;
  *gds          = delidvds + g*sigma;

}
