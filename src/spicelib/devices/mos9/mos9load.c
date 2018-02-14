/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "mos9defs.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS9load(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current value into the 
         * sparse matrix previously provided 
         */
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;
    double Beta;
    double DrainSatCur;
    double EffectiveLength;
    double EffectiveWidth;
    double GateBulkOverlapCap;
    double GateDrainOverlapCap;
    double GateSourceOverlapCap;
    double OxideCap;
    double SourceSatCur;
    double arg;
    double cbhat;
    double cdhat;
    double cdrain;
    double cdreq;
    double ceq;
    double ceqbd;
    double ceqbs;
    double ceqgb;
    double ceqgd;
    double ceqgs;
    double delvbd;
    double delvbs;
    double delvds;
    double delvgd;
    double delvgs;
    double evbd;
    double evbs;
    double gcgb;
    double gcgd;
    double gcgs;
    double geq;
    double sarg;
    double sargsw;
    double vbd;
    double vbs;
    double vds;
    double vdsat;
    double vgb1;
    double vgb;
    double vgd1;
    double vgd;
    double vgdo;
    double vgs1;
    double vgs;
    double von;
#ifndef PREDICTOR
    double xfact = 0.0;
#endif
    int xnrm;
    int xrev;
    double capgs = 0.0;   /* total gate-source capacitance */
    double capgd = 0.0;   /* total gate-drain capacitance */
    double capgb = 0.0;   /* total gate-bulk capacitance */
    int Check;
#ifndef NOBYPASS      
    double tempv;
#endif /*NOBYPASS*/    
    int error;
#ifdef CAPBYPASS
    int senflag;
#endif /* CAPBYPASS */
    int SenCond;
    double vt;  /* vt at instance temperature */


#ifdef CAPBYPASS
    senflag = 0;
#endif /* CAPBYPASS */
    if(ckt->CKTsenInfo){
        if(ckt->CKTsenInfo->SENstatus == PERTURBATION) {
            if((ckt->CKTsenInfo->SENmode == ACSEN)||
                (ckt->CKTsenInfo->SENmode == TRANSEN)){
#ifdef CAPBYPASS
                senflag = 1;
#endif /* CAPBYPASS */
            }
            goto next;
        }
    }

    /*  loop through all the MOS9 device models */
next: 
    for( ; model != NULL; model = MOS9nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS9instances(model); here != NULL ;
                here=MOS9nextInstance(here)) {

            vt = CONSTKoverQ * here->MOS9temp;
            Check=1;

            if(ckt->CKTsenInfo){
#ifdef SENSDEBUG
                printf("MOS9load \n");
#endif /* SENSDEBUG */

                if(ckt->CKTsenInfo->SENstatus == PERTURBATION)
                    if(here->MOS9senPertFlag == OFF)continue;

            }
            SenCond = ckt->CKTsenInfo && here->MOS9senPertFlag;

            /* first, we compute a few useful values - these could be
             * pre-computed, but for historical reasons are still done
             * here.  They may be moved at the expense of instance size
             */

            EffectiveWidth=here->MOS9w-2*model->MOS9widthNarrow+
                                                    model->MOS9widthAdjust;
            EffectiveLength=here->MOS9l - 2*model->MOS9latDiff+
                                                    model->MOS9lengthAdjust;

            if( (here->MOS9tSatCurDens == 0) || 
                    (here->MOS9drainArea == 0) ||
                    (here->MOS9sourceArea == 0)) {
                DrainSatCur = here->MOS9m * here->MOS9tSatCur;
                SourceSatCur = here->MOS9m * here->MOS9tSatCur;
            } else {
                DrainSatCur = here->MOS9m * here->MOS9tSatCurDens * 
                        here->MOS9drainArea;
                SourceSatCur = here->MOS9m * here->MOS9tSatCurDens * 
                        here->MOS9sourceArea;
            }
            GateSourceOverlapCap = model->MOS9gateSourceOverlapCapFactor * 
                    here->MOS9m * EffectiveWidth;
            GateDrainOverlapCap = model->MOS9gateDrainOverlapCapFactor * 
                    here->MOS9m * EffectiveWidth;
            GateBulkOverlapCap = model->MOS9gateBulkOverlapCapFactor * 
                    here->MOS9m * EffectiveLength;
            Beta = here->MOS9tTransconductance *
                    here->MOS9m * EffectiveWidth/EffectiveLength;
            OxideCap = model->MOS9oxideCapFactor * EffectiveLength * 
                    here->MOS9m * EffectiveWidth;


            if(SenCond){
#ifdef SENSDEBUG
                printf("MOS9senPertFlag = ON \n");
#endif /* SENSDEBUG */
                if((ckt->CKTsenInfo->SENmode == TRANSEN) &&
                (ckt->CKTmode & MODEINITTRAN)) {
                    vgs = *(ckt->CKTstate1 + here->MOS9vgs);
                    vds = *(ckt->CKTstate1 + here->MOS9vds);
                    vbs = *(ckt->CKTstate1 + here->MOS9vbs);
                    vbd = *(ckt->CKTstate1 + here->MOS9vbd);
                    vgb = vgs - vbs;
                    vgd = vgs - vds;
                }
                else if (ckt->CKTsenInfo->SENmode == ACSEN){
                    vgb = model->MOS9type * ( 
                        *(ckt->CKTrhsOp+here->MOS9gNode) -
                        *(ckt->CKTrhsOp+here->MOS9bNode));
                    vbs = *(ckt->CKTstate0 + here->MOS9vbs);
                    vbd = *(ckt->CKTstate0 + here->MOS9vbd);
                    vgd = vgb + vbd ;
                    vgs = vgb + vbs ;
                    vds = vbs - vbd ;
                }
                else{
                    vgs = *(ckt->CKTstate0 + here->MOS9vgs);
                    vds = *(ckt->CKTstate0 + here->MOS9vds);
                    vbs = *(ckt->CKTstate0 + here->MOS9vbs);
                    vbd = *(ckt->CKTstate0 + here->MOS9vbd);
                    vgb = vgs - vbs;
                    vgd = vgs - vds;
                }
#ifdef SENSDEBUG
                printf(" vbs = %.7e ,vbd = %.7e,vgb = %.7e\n",vbs,vbd,vgb);
                printf(" vgs = %.7e ,vds = %.7e,vgd = %.7e\n",vgs,vds,vgd);
#endif /* SENSDEBUG */
                goto next1;
            }


            /* 
             * ok - now to do the start-up operations
             *
             * we must get values for vbs, vds, and vgs from somewhere
             * so we either predict them or recover them from last iteration
             * These are the two most common cases - either a prediction
             * step or the general iteration step and they
             * share some code, so we put them first - others later on
             */

            if((ckt->CKTmode & (MODEINITFLOAT | MODEINITPRED | MODEINITSMSIG | 
                    MODEINITTRAN)) ||
                    ( (ckt->CKTmode & MODEINITFIX) && (!here->MOS9off) )  ) {
#ifndef PREDICTOR
                if(ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) {

                    /* predictor step */

                    xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    *(ckt->CKTstate0 + here->MOS9vbs) = 
                            *(ckt->CKTstate1 + here->MOS9vbs);
                    vbs = (1+xfact)* (*(ckt->CKTstate1 + here->MOS9vbs))
                            -(xfact * (*(ckt->CKTstate2 + here->MOS9vbs)));
                    *(ckt->CKTstate0 + here->MOS9vgs) = 
                            *(ckt->CKTstate1 + here->MOS9vgs);
                    vgs = (1+xfact)* (*(ckt->CKTstate1 + here->MOS9vgs))
                            -(xfact * (*(ckt->CKTstate2 + here->MOS9vgs)));
                    *(ckt->CKTstate0 + here->MOS9vds) = 
                            *(ckt->CKTstate1 + here->MOS9vds);
                    vds = (1+xfact)* (*(ckt->CKTstate1 + here->MOS9vds))
                            -(xfact * (*(ckt->CKTstate2 + here->MOS9vds)));
                    *(ckt->CKTstate0 + here->MOS9vbd) = 
                            *(ckt->CKTstate0 + here->MOS9vbs)-
                            *(ckt->CKTstate0 + here->MOS9vds);
                } else {
#endif /*PREDICTOR*/

                    /* general iteration */

                    vbs = model->MOS9type * ( 
                        *(ckt->CKTrhsOld+here->MOS9bNode) -
                        *(ckt->CKTrhsOld+here->MOS9sNodePrime));
                    vgs = model->MOS9type * ( 
                        *(ckt->CKTrhsOld+here->MOS9gNode) -
                        *(ckt->CKTrhsOld+here->MOS9sNodePrime));
                    vds = model->MOS9type * ( 
                        *(ckt->CKTrhsOld+here->MOS9dNodePrime) -
                        *(ckt->CKTrhsOld+here->MOS9sNodePrime));
#ifndef PREDICTOR
                }
#endif /*PREDICTOR*/

                /* now some common crunching for some more useful quantities */

                vbd=vbs-vds;
                vgd=vgs-vds;
                vgdo = *(ckt->CKTstate0 + here->MOS9vgs) - 
                        *(ckt->CKTstate0 + here->MOS9vds);
                delvbs = vbs - *(ckt->CKTstate0 + here->MOS9vbs);
                delvbd = vbd - *(ckt->CKTstate0 + here->MOS9vbd);
                delvgs = vgs - *(ckt->CKTstate0 + here->MOS9vgs);
                delvds = vds - *(ckt->CKTstate0 + here->MOS9vds);
                delvgd = vgd-vgdo;

                /* these are needed for convergence testing */

                if (here->MOS9mode >= 0) {
                    cdhat=
                        here->MOS9cd-
                        here->MOS9gbd * delvbd +
                        here->MOS9gmbs * delvbs +
                        here->MOS9gm * delvgs + 
                        here->MOS9gds * delvds ;
                } else {
                    cdhat=
                        here->MOS9cd -
                        ( here->MOS9gbd -
                        here->MOS9gmbs) * delvbd -
                        here->MOS9gm * delvgd + 
                        here->MOS9gds * delvds ;
                }
                cbhat=
                    here->MOS9cbs +
                    here->MOS9cbd +
                    here->MOS9gbd * delvbd +
                    here->MOS9gbs * delvbs ;
#ifndef NOBYPASS
                /* now lets see if we can bypass (ugh) */
                /* the following mess should be one if statement, but
                 * many compilers can't handle it all at once, so it
                 * is split into several successive if statements
                 */
                tempv = MAX(fabs(cbhat),fabs(here->MOS9cbs
                        + here->MOS9cbd))+ckt->CKTabstol;
                if((!(ckt->CKTmode & (MODEINITPRED|MODEINITTRAN|MODEINITSMSIG)
                        )) && (ckt->CKTbypass) )
                if ( (fabs(cbhat-(here->MOS9cbs + 
                        here->MOS9cbd)) < ckt->CKTreltol * 
                        tempv)) 
                if( (fabs(delvbs) < (ckt->CKTreltol * MAX(fabs(vbs),
                        fabs(*(ckt->CKTstate0+here->MOS9vbs)))+
                        ckt->CKTvoltTol)))
                if ( (fabs(delvbd) < (ckt->CKTreltol * MAX(fabs(vbd),
                        fabs(*(ckt->CKTstate0+here->MOS9vbd)))+
                        ckt->CKTvoltTol)) )
                if( (fabs(delvgs) < (ckt->CKTreltol * MAX(fabs(vgs),
                        fabs(*(ckt->CKTstate0+here->MOS9vgs)))+
                        ckt->CKTvoltTol)))
                if ( (fabs(delvds) < (ckt->CKTreltol * MAX(fabs(vds),
                        fabs(*(ckt->CKTstate0+here->MOS9vds)))+
                        ckt->CKTvoltTol)) )
                if( (fabs(cdhat- here->MOS9cd) <
                        ckt->CKTreltol * MAX(fabs(cdhat),fabs(
                        here->MOS9cd)) + ckt->CKTabstol) ) {
                    /* bypass code *
                     * nothing interesting has changed since last
                     * iteration on this device, so we just
                     * copy all the values computed last iteration out
                     * and keep going
                     */
                    vbs = *(ckt->CKTstate0 + here->MOS9vbs);
                    vbd = *(ckt->CKTstate0 + here->MOS9vbd);
                    vgs = *(ckt->CKTstate0 + here->MOS9vgs);
                    vds = *(ckt->CKTstate0 + here->MOS9vds);
                    vgd = vgs - vds;
                    vgb = vgs - vbs;
                    cdrain = here->MOS9mode * (here->MOS9cd + here->MOS9cbd);
                    if(ckt->CKTmode & (MODETRAN | MODETRANOP)) {
                        capgs = ( *(ckt->CKTstate0+here->MOS9capgs)+ 
                                  *(ckt->CKTstate1+here->MOS9capgs) +
                                  GateSourceOverlapCap );
                        capgd = ( *(ckt->CKTstate0+here->MOS9capgd)+ 
                                  *(ckt->CKTstate1+here->MOS9capgd) +
                                  GateDrainOverlapCap );
                        capgb = ( *(ckt->CKTstate0+here->MOS9capgb)+ 
                                  *(ckt->CKTstate1+here->MOS9capgb) +
                                  GateBulkOverlapCap );
                    }
                    goto bypass;
                }
#endif /*NOBYPASS*/
                /* ok - bypass is out, do it the hard way */

                von = model->MOS9type * here->MOS9von;

#ifndef NODELIMITING
                /* 
                 * limiting
                 *  we want to keep device voltages from changing
                 * so fast that the exponentials churn out overflows
                 * and similar rudeness
                 */

                if(*(ckt->CKTstate0 + here->MOS9vds) >=0) {
                    vgs = DEVfetlim(vgs,*(ckt->CKTstate0 + here->MOS9vgs)
                            ,von);
                    vds = vgs - vgd;
                    vds = DEVlimvds(vds,*(ckt->CKTstate0 + here->MOS9vds));
                    vgd = vgs - vds;
                } else {
                    vgd = DEVfetlim(vgd,vgdo,von);
                    vds = vgs - vgd;
                    if(!(ckt->CKTfixLimit)) {
                        vds = -DEVlimvds(-vds,-(*(ckt->CKTstate0 + 
                                here->MOS9vds)));
                    }
                    vgs = vgd + vds;
                }
                if(vds >= 0) {
                    vbs = DEVpnjlim(vbs,*(ckt->CKTstate0 + here->MOS9vbs),
                            vt,here->MOS9sourceVcrit,&Check);
                    vbd = vbs-vds;
                } else {
                    vbd = DEVpnjlim(vbd,*(ckt->CKTstate0 + here->MOS9vbd),
                            vt,here->MOS9drainVcrit,&Check);
                    vbs = vbd + vds;
                }
#endif /*NODELIMITING*/

            } else {

                /* ok - not one of the simple cases, so we have to
                 * look at all of the possibilities for why we were
                 * called.  We still just initialize the three voltages
                 */

                if((ckt->CKTmode & MODEINITJCT) && !here->MOS9off) {
                    vds= model->MOS9type * here->MOS9icVDS;
                    vgs= model->MOS9type * here->MOS9icVGS;
                    vbs= model->MOS9type * here->MOS9icVBS;
                    if((vds==0) && (vgs==0) && (vbs==0) && 
                            ((ckt->CKTmode & 
                                (MODETRAN|MODEDCOP|MODEDCTRANCURVE)) ||
                             (!(ckt->CKTmode & MODEUIC)))) {
                        vbs = -1;
                        vgs = model->MOS9type * here->MOS9tVto;
                        vds = 0;
                    }
                } else {
                    vbs=vgs=vds=0;
                } 
            }

            /*
             * now all the preliminaries are over - we can start doing the
             * real work
             */
            vbd = vbs - vds;
            vgd = vgs - vds;
            vgb = vgs - vbs;


            /*
             * bulk-source and bulk-drain diodes
             *   here we just evaluate the ideal diode current and the
             *   corresponding derivative (conductance).
             */

next1:      if(vbs <= -3*vt) {
                arg=3*vt/(vbs*CONSTe);
                arg = arg * arg * arg;
                here->MOS9cbs = -SourceSatCur*(1+arg)+ckt->CKTgmin*vbs;
                here->MOS9gbs = SourceSatCur*3*arg/vbs+ckt->CKTgmin;
            } else {
                evbs = exp(MIN(MAX_EXP_ARG,vbs/vt));
                here->MOS9gbs = SourceSatCur*evbs/vt + ckt->CKTgmin;
                here->MOS9cbs = SourceSatCur*(evbs-1) + ckt->CKTgmin*vbs;
            }
            if(vbd <= -3*vt) {
                arg=3*vt/(vbd*CONSTe);
                arg = arg * arg * arg;
                here->MOS9cbd = -DrainSatCur*(1+arg)+ckt->CKTgmin*vbd;
                here->MOS9gbd = DrainSatCur*3*arg/vbd+ckt->CKTgmin;
            } else {
                evbd = exp(MIN(MAX_EXP_ARG,vbd/vt));
                here->MOS9gbd = DrainSatCur*evbd/vt + ckt->CKTgmin;
                here->MOS9cbd = DrainSatCur*(evbd-1) + ckt->CKTgmin*vbd;
            }


            /* now to determine whether the user was able to correctly
             * identify the source and drain of his device
             */
            if(vds >= 0) {
                /* normal mode */
                here->MOS9mode = 1;
            } else {
                /* inverse mode */
                here->MOS9mode = -1;
            }

            {
            /*
             * subroutine moseq3(vds,vbs,vgs,gm,gds,gmbs,
             * qg,qc,qb,cggb,cgdb,cgsb,cbgb,cbdb,cbsb)
             */

            /*
             *     this routine evaluates the drain current, its derivatives and
             *     the charges associated with the gate, channel and bulk
             *     for mosfets based on semi-empirical equations
             */

            /*
            common /mosarg/ vto,beta,gamma,phi,phib,cox,xnsub,xnfs,xd,xj,xld,
            1   xlamda,uo,uexp,vbp,utra,vmax,xneff,xl,xw,vbi,von,vdsat,qspof,
            2   beta0,beta1,cdrain,xqco,xqc,fnarrw,fshort,lev
            common /status/ omega,time,delta,delold(7),ag(7),vt,xni,egfet,
            1   xmu,sfactr,mode,modedc,icalc,initf,method,iord,maxord,noncon,
            2   iterno,itemno,nosolv,modac,ipiv,ivmflg,ipostp,iscrch,iofile
            common /knstnt/ twopi,xlog2,xlog10,root2,rad,boltz,charge,ctok,
            1   gmin,reltol,abstol,vntol,trtol,chgtol,eps0,epssil,epsox,
            2   pivtol,pivrel
            */

            /* equivalence (xlamda,alpha),(vbp,theta),(uexp,eta),(utra,xkappa)*/

            double coeff0 = 0.0631353e0;
            double coeff1 = 0.8013292e0;
            double coeff2 = -0.01110777e0;
            double oneoverxl;   /* 1/effective length */
            double eta; /* eta from model after length factor */
            double phibs;   /* phi - vbs */
            double sqphbs;  /* square root of phibs */
            double dsqdvb;  /*  */
            double sqphis;  /* square root of phi */
            double sqphs3;  /* square root of phi cubed */
            double wps;
            double oneoverxj;   /* 1/junction depth */
            double xjonxl;  /* junction depth/effective length */
            double djonxj;
            double wponxj;
            double arga;
            double argb;
            double argc;
            double dwpdvb;
            double dadvb;
            double dbdvb;
            double gammas;
            double fbodys;
            double fbody;
            double onfbdy;
            double qbonco;
            double vbix;
            double wconxj;
            double dfsdvb;
            double dfbdvb;
            double dqbdvb;
            double vth;
            double dvtdvb;
            double csonco;
            double cdonco;
            double dxndvb = 0.0;
            double dvodvb = 0.0;
            double dvodvd = 0.0;
            double vgsx;
            double dvtdvd;
            double onfg;
            double fgate;
            double us;
            double dfgdvg;
            double dfgdvd;
            double dfgdvb;
            double dvsdvg;
            double dvsdvb;
            double dvsdvd;
            double xn = 0.0;
            double vdsc;
            double onvdsc = 0.0;
            double dvsdga;
            double vdsx;
            double dcodvb;
            double cdnorm;
            double cdo;
            double cd1;
            double fdrain = 0.0;
            double fd2;
            double dfddvg = 0.0;
            double dfddvb = 0.0;
            double dfddvd = 0.0;
            double gdsat;
            double cdsat;
            double gdoncd;
            double gdonfd;
            double gdonfg;
            double dgdvg;
            double dgdvd;
            double dgdvb;
            double emax;
            double emongd;
            double demdvg;
            double demdvd;
            double demdvb;
            double delxl;
            double dldvd;
            double dldem;
            double ddldvg;
            double ddldvd;
            double ddldvb;
            double dlonxl;
            double xlfact;
            double diddl;
            double gds0 = 0.0;
            double emoncd;
            double ondvt;
            double onxn;
            double wfact;
            double gms;
            double gmw;
            double fshort;

            /*
             *     bypasses the computation of charges
             */

            /*
             *     reference cdrain equations to source and
             *     charge equations to bulk
             */
            vdsat = 0.0;
            oneoverxl = 1.0/EffectiveLength;
            eta = model->MOS9eta * 8.15e-22/(model->MOS9oxideCapFactor*
                    EffectiveLength*EffectiveLength*EffectiveLength);
            /*
             *.....square root term
             */
            if ( (here->MOS9mode==1?vbs:vbd) <=  0.0 ) {
                phibs  =  here->MOS9tPhi-(here->MOS9mode==1?vbs:vbd);
                sqphbs  =  sqrt(phibs);
                dsqdvb  =  -0.5/sqphbs;
            } else {
                sqphis = sqrt(here->MOS9tPhi);
                sqphs3 = here->MOS9tPhi*sqphis;
                sqphbs = sqphis/(1.0+(here->MOS9mode==1?vbs:vbd)/
                    (here->MOS9tPhi+here->MOS9tPhi));
                phibs = sqphbs*sqphbs;
                dsqdvb = -phibs/(sqphs3+sqphs3);
            }
            /*
             *.....short channel effect factor
             */
            if ( (model->MOS9junctionDepth != 0.0) && 
                    (model->MOS9coeffDepLayWidth != 0.0) ) {
                wps = model->MOS9coeffDepLayWidth*sqphbs;
                oneoverxj = 1.0/model->MOS9junctionDepth;
                xjonxl = model->MOS9junctionDepth*oneoverxl;
                djonxj = model->MOS9latDiff*oneoverxj;
                wponxj = wps*oneoverxj;
                wconxj = coeff0+coeff1*wponxj+coeff2*wponxj*wponxj;
                arga = wconxj+djonxj;
                argc = wponxj/(1.0+wponxj);
                argb = sqrt(1.0-argc*argc);
                fshort = 1.0-xjonxl*(arga*argb-djonxj);
                dwpdvb = model->MOS9coeffDepLayWidth*dsqdvb;
                dadvb = (coeff1+coeff2*(wponxj+wponxj))*dwpdvb*oneoverxj;
                dbdvb = -argc*argc*(1.0-argc)*dwpdvb/(argb*wps);
                dfsdvb = -xjonxl*(dadvb*argb+arga*dbdvb);
            } else {
                fshort = 1.0;
                dfsdvb = 0.0;
            }
            /*
             *.....body effect
             */
            gammas = model->MOS9gamma*fshort;
            fbodys = 0.5*gammas/(sqphbs+sqphbs);
            fbody = fbodys+model->MOS9narrowFactor/EffectiveWidth;
            onfbdy = 1.0/(1.0+fbody);
            dfbdvb = -fbodys*dsqdvb/sqphbs+fbodys*dfsdvb/fshort;
            qbonco =gammas*sqphbs+model->MOS9narrowFactor*phibs/EffectiveWidth;
            dqbdvb = gammas*dsqdvb+model->MOS9gamma*dfsdvb*sqphbs-
                model->MOS9narrowFactor/EffectiveWidth;
                
            /*
             *.....static feedback effect
             */
            vbix = here->MOS9tVbi*model->MOS9type-eta*(here->MOS9mode*vds);
            /*
             *.....threshold voltage
             */
            vth = vbix+qbonco;
            dvtdvd = -eta;
            dvtdvb = dqbdvb;
            /*
             *.....joint weak inversion and strong inversion
             */
            von = vth;
            if ( model->MOS9fastSurfaceStateDensity != 0.0 ) {
                csonco = CHARGE*model->MOS9fastSurfaceStateDensity * 
                    1e4 /*(cm**2/m**2)*/ *EffectiveLength*EffectiveWidth *
                    here->MOS9m/OxideCap;
                cdonco = qbonco/(phibs+phibs);
                xn = 1.0+csonco+cdonco;
                von = vth+vt*xn;
                dxndvb = dqbdvb/(phibs+phibs)-qbonco*dsqdvb/(phibs*sqphbs);
                dvodvd = dvtdvd;
                dvodvb = dvtdvb+vt*dxndvb;
            } else {
                /*
                 *.....cutoff region
                 */
                if ( (here->MOS9mode==1?vgs:vgd) <= von ) {
                    cdrain = 0.0;
                    here->MOS9gm = 0.0;
                    here->MOS9gds = 0.0;
                    here->MOS9gmbs = 0.0;
                    goto innerline1000;
                }
            }
            /*
             *.....device is on
             */
            vgsx = MAX((here->MOS9mode==1?vgs:vgd),von);
            /*
             *.....mobility modulation by gate voltage
             */
            onfg = 1.0+model->MOS9theta*(vgsx-vth);
            fgate = 1.0/onfg;
            us = here->MOS9tSurfMob * 1e-4 /*(m**2/cm**2)*/ *fgate;
            dfgdvg = -model->MOS9theta*fgate*fgate;
            dfgdvd = -dfgdvg*dvtdvd;
            dfgdvb = -dfgdvg*dvtdvb;
            /*
             *.....saturation voltage
             */
            vdsat = (vgsx-vth)*onfbdy;
            if ( model->MOS9maxDriftVel <= 0.0 ) {
                dvsdvg = onfbdy;
                dvsdvd = -dvsdvg*dvtdvd;
                dvsdvb = -dvsdvg*dvtdvb-vdsat*dfbdvb*onfbdy;
            } else {
                vdsc = EffectiveLength*model->MOS9maxDriftVel/us;
                onvdsc = 1.0/vdsc;
                arga = (vgsx-vth)*onfbdy;
                argb = sqrt(arga*arga+vdsc*vdsc);
                vdsat = arga+vdsc-argb;
                dvsdga = (1.0-arga/argb)*onfbdy;
                dvsdvg = dvsdga-(1.0-vdsc/argb)*vdsc*dfgdvg*onfg;
                dvsdvd = -dvsdvg*dvtdvd;
                dvsdvb = -dvsdvg*dvtdvb-arga*dvsdga*dfbdvb;
            }
            /*
             *.....current factors in linear region
             */
            vdsx = MIN((here->MOS9mode*vds),vdsat);
            if ( vdsx == 0.0 ) goto line900;
            cdo = vgsx-vth-0.5*(1.0+fbody)*vdsx;
            dcodvb = -dvtdvb-0.5*dfbdvb*vdsx;
            /* 
             *.....normalized drain current
             */
            cdnorm = cdo*vdsx;
            here->MOS9gm = vdsx;
            if ((here->MOS9mode*vds) > vdsat) here->MOS9gds = -dvtdvd*vdsx;
            else here->MOS9gds = vgsx-vth-(1.0+fbody+dvtdvd)*vdsx;
            here->MOS9gmbs = dcodvb*vdsx;
            /* 
             *.....drain current without velocity saturation effect
             */
            cd1 = Beta*cdnorm;
            Beta = Beta*fgate;
            cdrain = Beta*cdnorm;
            here->MOS9gm = Beta*here->MOS9gm+dfgdvg*cd1;
            here->MOS9gds = Beta*here->MOS9gds+dfgdvd*cd1;
            here->MOS9gmbs = Beta*here->MOS9gmbs+dfgdvb*cd1;
            /*
             *.....velocity saturation factor
             */
            if ( model->MOS9maxDriftVel > 0.0 ) {
                fdrain = 1.0/(1.0+vdsx*onvdsc);
                fd2 = fdrain*fdrain;
                arga = fd2*vdsx*onvdsc*onfg;
                dfddvg = -dfgdvg*arga;
                if ((here->MOS9mode*vds) > vdsat) dfddvd = -dfgdvd*arga;
                else dfddvd = -dfgdvd*arga-fd2*onvdsc;
                dfddvb = -dfgdvb*arga;
                /*
                 *.....drain current
                 */
                here->MOS9gm = fdrain*here->MOS9gm+dfddvg*cdrain;
                here->MOS9gds = fdrain*here->MOS9gds+dfddvd*cdrain;
                here->MOS9gmbs = fdrain*here->MOS9gmbs+dfddvb*cdrain;
                cdrain = fdrain*cdrain;
                Beta = Beta*fdrain;
            }
            /*
             *.....channel length modulation
             */
            if ((here->MOS9mode*vds) <= vdsat) {
              if ( (model->MOS9maxDriftVel > 0.0) ||
                   (model->MOS9alpha == 0.0) ||
                   (ckt->CKTbadMos3)                 ) goto line700;
              else {
                 arga = (here->MOS9mode*vds)/vdsat;
                 delxl = sqrt(model->MOS9kappa*model->MOS9alpha*vdsat/8);
                 dldvd = 4*delxl*arga*arga*arga/vdsat;
                 arga *= arga;
                 arga *= arga;
                 delxl *= arga;
                 ddldvg = 0.0;
                 ddldvd = -dldvd;
                 ddldvb = 0.0;
                 goto line520;
              }
            }
            if ( model->MOS9maxDriftVel <= 0.0 ) goto line510;
            if (model->MOS9alpha == 0.0) goto line700;
            cdsat = cdrain;
            gdsat = cdsat*(1.0-fdrain)*onvdsc;
            gdsat = MAX(1.0e-12,gdsat);
            gdoncd = gdsat/cdsat;
            gdonfd = gdsat/(1.0-fdrain);
            gdonfg = gdsat*onfg;
            dgdvg = gdoncd*here->MOS9gm-gdonfd*dfddvg+gdonfg*dfgdvg;
            dgdvd = gdoncd*here->MOS9gds-gdonfd*dfddvd+gdonfg*dfgdvd;
            dgdvb = gdoncd*here->MOS9gmbs-gdonfd*dfddvb+gdonfg*dfgdvb;

	    if (ckt->CKTbadMos3)
		    emax = cdsat*oneoverxl/gdsat;
	    else
		    emax = model->MOS9kappa * cdsat*oneoverxl/gdsat;
            emoncd = emax/cdsat;
            emongd = emax/gdsat;
            demdvg = emoncd*here->MOS9gm-emongd*dgdvg;
            demdvd = emoncd*here->MOS9gds-emongd*dgdvd;
            demdvb = emoncd*here->MOS9gmbs-emongd*dgdvb;

            arga = 0.5*emax*model->MOS9alpha;
            argc = model->MOS9kappa*model->MOS9alpha;
            argb = sqrt(arga*arga+argc*((here->MOS9mode*vds)-vdsat));
            delxl = argb-arga;
            dldvd = argc/(argb+argb);
            dldem = 0.5*(arga/argb-1.0)*model->MOS9alpha;
            ddldvg = dldem*demdvg;
            ddldvd = dldem*demdvd-dldvd;
            ddldvb = dldem*demdvb;
            goto line520;
line510:
	    if (ckt->CKTbadMos3) {
            delxl = sqrt(model->MOS9kappa*((here->MOS9mode*vds)-vdsat)*
                model->MOS9alpha);
            dldvd = 0.5*delxl/((here->MOS9mode*vds)-vdsat);
            } else {
               delxl = sqrt(model->MOS9kappa*model->MOS9alpha*
                                      ((here->MOS9mode*vds)-vdsat+(vdsat/8)));
               dldvd = 0.5*delxl/((here->MOS9mode*vds)-vdsat+(vdsat/8));
            }
            ddldvg = 0.0;
            ddldvd = -dldvd;
            ddldvb = 0.0;
            /*
             *.....punch through approximation
             */
line520:
            if ( delxl > (0.5*EffectiveLength) ) {
                delxl = EffectiveLength-(EffectiveLength*EffectiveLength/
                    (4.0*delxl));
                arga = 4.0*(EffectiveLength-delxl)*(EffectiveLength-delxl)/
                    (EffectiveLength*EffectiveLength);
                ddldvg = ddldvg*arga;
                ddldvd = ddldvd*arga;
                ddldvb = ddldvb*arga;
                dldvd =  dldvd*arga;
            }
            /*
             *.....saturation region
             */
            dlonxl = delxl*oneoverxl;
            xlfact = 1.0/(1.0-dlonxl);
            cd1 = cdrain;
            cdrain = cdrain*xlfact;
            diddl = cdrain/(EffectiveLength-delxl);
            here->MOS9gm = here->MOS9gm*xlfact+diddl*ddldvg;
            here->MOS9gmbs = here->MOS9gmbs*xlfact+diddl*ddldvb;
            gds0 = diddl*ddldvd;
            here->MOS9gm = here->MOS9gm+gds0*dvsdvg;
            here->MOS9gmbs = here->MOS9gmbs+gds0*dvsdvb;
            here->MOS9gds = here->MOS9gds*xlfact+diddl*dldvd+gds0*dvsdvd;
/*          here->MOS9gds = (here->MOS9gds*xlfact)+gds0*dvsdvd-
                    (cd1*ddldvd/(EffectiveLength*(1-2*dlonxl+dlonxl*dlonxl)));*/

            /*
             *.....finish strong inversion case
             */
line700:
            if ( (here->MOS9mode==1?vgs:vgd) < von ) {
                /*
                 *.....weak inversion
                 */
                onxn = 1.0/xn;
                ondvt = onxn/vt;
                wfact = exp( ((here->MOS9mode==1?vgs:vgd)-von)*ondvt );
                cdrain = cdrain*wfact;
                gms = here->MOS9gm*wfact;
                gmw = cdrain*ondvt;
                here->MOS9gm = gmw;
                if ((here->MOS9mode*vds) > vdsat) {
                    here->MOS9gm = here->MOS9gm+gds0*dvsdvg*wfact;
                }
                here->MOS9gds = here->MOS9gds*wfact+(gms-gmw)*dvodvd;
                here->MOS9gmbs = here->MOS9gmbs*wfact+(gms-gmw)*dvodvb-gmw*
                    ((here->MOS9mode==1?vgs:vgd)-von)*onxn*dxndvb;
            }
            /*
             *.....charge computation
             */
            goto innerline1000;
            /*
             *.....special case of vds = 0.0d0
             */
line900:
            Beta = Beta*fgate;
            cdrain = 0.0;
            here->MOS9gm = 0.0;
            here->MOS9gds = Beta*(vgsx-vth);
            here->MOS9gmbs = 0.0;
            if ( (model->MOS9fastSurfaceStateDensity != 0.0) && 
                    ((here->MOS9mode==1?vgs:vgd) < von) ) {
                here->MOS9gds *=exp(((here->MOS9mode==1?vgs:vgd)-von)/(vt*xn));
            }
innerline1000:;
            /* 
             *.....done
             */
            }


            /* now deal with n vs p polarity */

            here->MOS9von = model->MOS9type * von;
            here->MOS9vdsat = model->MOS9type * vdsat;
            /* line 490 */
            /*
             *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
             */
            here->MOS9cd=here->MOS9mode * cdrain - here->MOS9cbd;

            if (ckt->CKTmode & (MODETRAN | MODETRANOP | MODEINITSMSIG)) {
                /* 
                 * now we do the hard part of the bulk-drain and bulk-source
                 * diode - we evaluate the non-linear capacitance and
                 * charge
                 *
                 * the basic equations are not hard, but the implementation
                 * is somewhat long in an attempt to avoid log/exponential
                 * evaluations
                 */
                /*
                 *  charge storage elements
                 *
                 *.. bulk-drain and bulk-source depletion capacitances
                 */
#ifdef CAPBYPASS
                if(((ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) ||
                        fabs(delvbs) >= ckt->CKTreltol * MAX(fabs(vbs),
                        fabs(*(ckt->CKTstate0+here->MOS9vbs)))+
                        ckt->CKTvoltTol)|| senflag )
#endif /*CAPBYPASS*/
                {
                    /* can't bypass the diode capacitance calculations */
                    if(here->MOS9Cbs != 0 || here->MOS9Cbssw != 0 ) {
                    if (vbs < here->MOS9tDepCap){
                        arg=1-vbs/here->MOS9tBulkPot;
                        /*
                         * the following block looks somewhat long and messy,
                         * but since most users use the default grading
                         * coefficients of .5, and sqrt is MUCH faster than an
                         * exp(log()) we use this special case code to buy time.
                         * (as much as 10% of total job time!)
                         */
                        if(model->MOS9bulkJctBotGradingCoeff ==
                                model->MOS9bulkJctSideGradingCoeff) {
                            if(model->MOS9bulkJctBotGradingCoeff == .5) {
                                sarg = sargsw = 1/sqrt(arg);
                            } else {
                                sarg = sargsw =
                                        exp(-model->MOS9bulkJctBotGradingCoeff*
                                        log(arg));
                            }
                        } else {
                            if(model->MOS9bulkJctBotGradingCoeff == .5) {
                                sarg = 1/sqrt(arg);
                            } else {
                                sarg = exp(-model->MOS9bulkJctBotGradingCoeff*
                                        log(arg));
                            }
                            if(model->MOS9bulkJctSideGradingCoeff == .5) {
                                sargsw = 1/sqrt(arg);
                            } else {
                                sargsw =exp(-model->MOS9bulkJctSideGradingCoeff*
                                        log(arg));
                            }
                        }
                        *(ckt->CKTstate0 + here->MOS9qbs) =
                            here->MOS9tBulkPot*(here->MOS9Cbs*
                            (1-arg*sarg)/(1-model->MOS9bulkJctBotGradingCoeff)
                            +here->MOS9Cbssw*
                            (1-arg*sargsw)/
                            (1-model->MOS9bulkJctSideGradingCoeff));
                        here->MOS9capbs=here->MOS9Cbs*sarg+
                                here->MOS9Cbssw*sargsw;
                    } else {
                        *(ckt->CKTstate0 + here->MOS9qbs) = here->MOS9f4s +
                                vbs*(here->MOS9f2s+vbs*(here->MOS9f3s/2));
                        here->MOS9capbs=here->MOS9f2s+here->MOS9f3s*vbs;
                    }
                    } else {
                        *(ckt->CKTstate0 + here->MOS9qbs) = 0;
                        here->MOS9capbs=0;
                    }
                }
#ifdef CAPBYPASS
                if(((ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) ||
                        fabs(delvbd) >= ckt->CKTreltol * MAX(fabs(vbd),
                        fabs(*(ckt->CKTstate0+here->MOS9vbd)))+
                        ckt->CKTvoltTol)|| senflag )
#endif /*CAPBYPASS*/
                    /* can't bypass the diode capacitance calculations */
                {
                    if(here->MOS9Cbd != 0 || here->MOS9Cbdsw != 0 ) {
                    if (vbd < here->MOS9tDepCap) {
                        arg=1-vbd/here->MOS9tBulkPot;
                        /*
                         * the following block looks somewhat long and messy,
                         * but since most users use the default grading
                         * coefficients of .5, and sqrt is MUCH faster than an
                         * exp(log()) we use this special case code to buy time.
                         * (as much as 10% of total job time!)
                         */
                        if(model->MOS9bulkJctBotGradingCoeff == .5 &&
                                model->MOS9bulkJctSideGradingCoeff == .5) {
                            sarg = sargsw = 1/sqrt(arg);
                        } else {
                            if(model->MOS9bulkJctBotGradingCoeff == .5) {
                                sarg = 1/sqrt(arg);
                            } else {
                                sarg = exp(-model->MOS9bulkJctBotGradingCoeff*
                                        log(arg));
                            }
                            if(model->MOS9bulkJctSideGradingCoeff == .5) {
                                sargsw = 1/sqrt(arg);
                            } else {
                                sargsw =exp(-model->MOS9bulkJctSideGradingCoeff*
                                        log(arg));
                            }
                        }
                        *(ckt->CKTstate0 + here->MOS9qbd) =
                            here->MOS9tBulkPot*(here->MOS9Cbd*
                            (1-arg*sarg)
                            /(1-model->MOS9bulkJctBotGradingCoeff)
                            +here->MOS9Cbdsw*
                            (1-arg*sargsw)
                            /(1-model->MOS9bulkJctSideGradingCoeff));
                        here->MOS9capbd=here->MOS9Cbd*sarg+
                                here->MOS9Cbdsw*sargsw;
                    } else {
                        *(ckt->CKTstate0 + here->MOS9qbd) = here->MOS9f4d +
                                vbd * (here->MOS9f2d + vbd * here->MOS9f3d/2);
                        here->MOS9capbd=here->MOS9f2d + vbd * here->MOS9f3d;
                    }
                } else {
                    *(ckt->CKTstate0 + here->MOS9qbd) = 0;
                    here->MOS9capbd = 0;
                }
                }

                if(SenCond && (ckt->CKTsenInfo->SENmode==TRANSEN)) goto next2;

                if ( ckt->CKTmode & MODETRAN ) {
                    /* (above only excludes tranop, since we're only at this
                     * point if tran or tranop )
                     */

                    /*
                     *    calculate equivalent conductances and currents for
                     *    depletion capacitors
                     */

                    /* integrate the capacitors and save results */

                    error = NIintegrate(ckt,&geq,&ceq,here->MOS9capbd,
                            here->MOS9qbd);
                    if(error) return(error);
                    here->MOS9gbd += geq;
                    here->MOS9cbd += *(ckt->CKTstate0 + here->MOS9cqbd);
                    here->MOS9cd -= *(ckt->CKTstate0 + here->MOS9cqbd);
                    error = NIintegrate(ckt,&geq,&ceq,here->MOS9capbs,
                            here->MOS9qbs);
                    if(error) return(error);
                    here->MOS9gbs += geq;
                    here->MOS9cbs += *(ckt->CKTstate0 + here->MOS9cqbs);
                }
            }

            if(SenCond) goto next2;

            /*
             *  check convergence
             */
            if ( (here->MOS9off == 0)  || 
                    (!(ckt->CKTmode & (MODEINITFIX|MODEINITSMSIG))) ){
                if (Check == 1) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }



            /* save things away for next time */

next2:      *(ckt->CKTstate0 + here->MOS9vbs) = vbs;
            *(ckt->CKTstate0 + here->MOS9vbd) = vbd;
            *(ckt->CKTstate0 + here->MOS9vgs) = vgs;
            *(ckt->CKTstate0 + here->MOS9vds) = vds;



            /*
             *     meyer's capacitor model
             */
            if ( ckt->CKTmode & (MODETRAN | MODETRANOP | MODEINITSMSIG) ) {
                /*
                 *     calculate meyer's capacitors
                 */
                /* 
                 * new cmeyer - this just evaluates at the current time,
                 * expects you to remember values from previous time
                 * returns 1/2 of non-constant portion of capacitance
                 * you must add in the other half from previous time
                 * and the constant part
                 */
                if (here->MOS9mode > 0){
                    DEVqmeyer (vgs,vgd,vgb,von,vdsat,
                        (ckt->CKTstate0 + here->MOS9capgs),
                        (ckt->CKTstate0 + here->MOS9capgd),
                        (ckt->CKTstate0 + here->MOS9capgb),
                        here->MOS9tPhi,OxideCap);
                } else {
                    DEVqmeyer (vgd,vgs,vgb,von,vdsat,
                        (ckt->CKTstate0 + here->MOS9capgd),
                        (ckt->CKTstate0 + here->MOS9capgs),
                        (ckt->CKTstate0 + here->MOS9capgb),
                        here->MOS9tPhi,OxideCap);
                }
                vgs1 = *(ckt->CKTstate1 + here->MOS9vgs);
                vgd1 = vgs1 - *(ckt->CKTstate1 + here->MOS9vds);
                vgb1 = vgs1 - *(ckt->CKTstate1 + here->MOS9vbs);
                if(ckt->CKTmode & MODETRANOP) {
                    capgs =  2 * *(ckt->CKTstate0+here->MOS9capgs)+ 
                              GateSourceOverlapCap ;
                    capgd =  2 * *(ckt->CKTstate0+here->MOS9capgd)+ 
                              GateDrainOverlapCap ;
                    capgb =  2 * *(ckt->CKTstate0+here->MOS9capgb)+ 
                              GateBulkOverlapCap ;
                } else {
                    capgs = ( *(ckt->CKTstate0+here->MOS9capgs)+ 
                              *(ckt->CKTstate1+here->MOS9capgs) +
                              GateSourceOverlapCap );
                    capgd = ( *(ckt->CKTstate0+here->MOS9capgd)+ 
                              *(ckt->CKTstate1+here->MOS9capgd) +
                              GateDrainOverlapCap );
                    capgb = ( *(ckt->CKTstate0+here->MOS9capgb)+ 
                              *(ckt->CKTstate1+here->MOS9capgb) +
                              GateBulkOverlapCap );
                }
                if(ckt->CKTsenInfo){
                    here->MOS9cgs = capgs;
                    here->MOS9cgd = capgd;
                    here->MOS9cgb = capgb;
                }


                /*
                 *     store small-signal parameters (for meyer's model)
                 *  all parameters already stored, so done...
                 */


                if(SenCond){
                    if(ckt->CKTsenInfo->SENmode & (DCSEN|ACSEN)) {
                        continue;
                    }
                }
#ifndef PREDICTOR
                if (ckt->CKTmode & (MODEINITPRED | MODEINITTRAN) ) {
                    *(ckt->CKTstate0 + here->MOS9qgs) =
                        (1+xfact) * *(ckt->CKTstate1 + here->MOS9qgs)
                        - xfact * *(ckt->CKTstate2 + here->MOS9qgs);
                    *(ckt->CKTstate0 + here->MOS9qgd) =
                        (1+xfact) * *(ckt->CKTstate1 + here->MOS9qgd)
                        - xfact * *(ckt->CKTstate2 + here->MOS9qgd);
                    *(ckt->CKTstate0 + here->MOS9qgb) =
                        (1+xfact) * *(ckt->CKTstate1 + here->MOS9qgb)
                        - xfact * *(ckt->CKTstate2 + here->MOS9qgb);
                } else {
#endif /*PREDICTOR*/
                    if(ckt->CKTmode & MODETRAN) {
                        *(ckt->CKTstate0 + here->MOS9qgs) = (vgs-vgs1)*capgs +
                            *(ckt->CKTstate1 + here->MOS9qgs) ;
                        *(ckt->CKTstate0 + here->MOS9qgd) = (vgd-vgd1)*capgd +
                            *(ckt->CKTstate1 + here->MOS9qgd) ;
                        *(ckt->CKTstate0 + here->MOS9qgb) = (vgb-vgb1)*capgb +
                            *(ckt->CKTstate1 + here->MOS9qgb) ;
                    } else {
                        /* TRANOP only */
                        *(ckt->CKTstate0 + here->MOS9qgs) = vgs*capgs;
                        *(ckt->CKTstate0 + here->MOS9qgd) = vgd*capgd;
                        *(ckt->CKTstate0 + here->MOS9qgb) = vgb*capgb;
                    }
#ifndef PREDICTOR
                }
#endif /*PREDICTOR*/
            }
#ifndef NOBYPASS
bypass:
#endif
            if(SenCond) continue;


            if ( (ckt->CKTmode & (MODEINITTRAN)) || 
                    (! (ckt->CKTmode & (MODETRAN)) )  ) {
                /*
                 *  initialize to zero charge conductances 
                 *  and current
                 */
                gcgs=0;
                ceqgs=0;
                gcgd=0;
                ceqgd=0;
                gcgb=0;
                ceqgb=0;
            } else {
                if(capgs == 0) *(ckt->CKTstate0 + here->MOS9cqgs) =0;
                if(capgd == 0) *(ckt->CKTstate0 + here->MOS9cqgd) =0;
                if(capgb == 0) *(ckt->CKTstate0 + here->MOS9cqgb) =0;
                /*
                 *    calculate equivalent conductances and currents for
                 *    meyer"s capacitors
                 */
                error = NIintegrate(ckt,&gcgs,&ceqgs,capgs,here->MOS9qgs);
                if(error) return(error);
                error = NIintegrate(ckt,&gcgd,&ceqgd,capgd,here->MOS9qgd);
                if(error) return(error);
                error = NIintegrate(ckt,&gcgb,&ceqgb,capgb,here->MOS9qgb);
                if(error) return(error);
                ceqgs=ceqgs-gcgs*vgs+ckt->CKTag[0]* 
                        *(ckt->CKTstate0 + here->MOS9qgs);
                ceqgd=ceqgd-gcgd*vgd+ckt->CKTag[0]*
                        *(ckt->CKTstate0 + here->MOS9qgd);
                ceqgb=ceqgb-gcgb*vgb+ckt->CKTag[0]*
                        *(ckt->CKTstate0 + here->MOS9qgb);
            }
            /*
             *     store charge storage info for meyer's cap in lx table
             */


            /*
             *  load current vector
             */
            ceqbs = model->MOS9type * 
                    (here->MOS9cbs-(here->MOS9gbs)*vbs);
            ceqbd = model->MOS9type * 
                    (here->MOS9cbd-(here->MOS9gbd)*vbd);
            if (here->MOS9mode >= 0) {
                xnrm=1;
                xrev=0;
                cdreq=model->MOS9type*(cdrain-here->MOS9gds*vds-
                        here->MOS9gm*vgs-here->MOS9gmbs*vbs);
            } else {
                xnrm=0;
                xrev=1;
                cdreq = -(model->MOS9type)*(cdrain-here->MOS9gds*(-vds)-
                        here->MOS9gm*vgd-here->MOS9gmbs*vbd);
            }
            *(ckt->CKTrhs + here->MOS9gNode) -= 
                (model->MOS9type * (ceqgs + ceqgb + ceqgd));
            *(ckt->CKTrhs + here->MOS9bNode) -=
                (ceqbs + ceqbd - model->MOS9type * ceqgb);
            *(ckt->CKTrhs + here->MOS9dNodePrime) +=
                    (ceqbd - cdreq + model->MOS9type * ceqgd);
            *(ckt->CKTrhs + here->MOS9sNodePrime) += 
                    cdreq + ceqbs + model->MOS9type * ceqgs;
            /*
             *  load y matrix
             */
/*printf(" loading %s at time %g\n",here->MOS9name,ckt->CKTtime);*/
/*printf("%g %g %g %g %g\n", here->MOS9drainConductance,gcgd+gcgs+gcgb,
        here->MOS9sourceConductance,here->MOS9gbd,here->MOS9gbs);*/
/*printf("%g %g %g %g %g\n",-gcgb,0.0,0.0,here->MOS9gds,here->MOS9gm);*/
/*printf("%g %g %g %g %g\n", here->MOS9gds,here->MOS9gmbs,gcgd,-gcgs,-gcgd);*/
/*printf("%g %g %g %g %g\n", -gcgs,-gcgd,0.0,-gcgs,0.0);*/

            *(here->MOS9DdPtr) += (here->MOS9drainConductance);
            *(here->MOS9GgPtr) += ((gcgd+gcgs+gcgb));
            *(here->MOS9SsPtr) += (here->MOS9sourceConductance);
            *(here->MOS9BbPtr) += (here->MOS9gbd+here->MOS9gbs+gcgb);
            *(here->MOS9DPdpPtr) += 
                (here->MOS9drainConductance+here->MOS9gds+
                here->MOS9gbd+xrev*(here->MOS9gm+here->MOS9gmbs)+gcgd);
            *(here->MOS9SPspPtr) += 
                (here->MOS9sourceConductance+here->MOS9gds+
                here->MOS9gbs+xnrm*(here->MOS9gm+here->MOS9gmbs)+gcgs);
            *(here->MOS9DdpPtr) += (-here->MOS9drainConductance);
            *(here->MOS9GbPtr) -= gcgb;
            *(here->MOS9GdpPtr) -= gcgd;
            *(here->MOS9GspPtr) -= gcgs;
            *(here->MOS9SspPtr) += (-here->MOS9sourceConductance);
            *(here->MOS9BgPtr) -= gcgb;
            *(here->MOS9BdpPtr) -= here->MOS9gbd;
            *(here->MOS9BspPtr) -= here->MOS9gbs;
            *(here->MOS9DPdPtr) += (-here->MOS9drainConductance);
            *(here->MOS9DPgPtr) += ((xnrm-xrev)*here->MOS9gm-gcgd);
            *(here->MOS9DPbPtr) += (-here->MOS9gbd+(xnrm-xrev)*here->MOS9gmbs);
            *(here->MOS9DPspPtr) += (-here->MOS9gds-
                    xnrm*(here->MOS9gm+here->MOS9gmbs));
            *(here->MOS9SPgPtr) += (-(xnrm-xrev)*here->MOS9gm-gcgs);
            *(here->MOS9SPsPtr) += (-here->MOS9sourceConductance);
            *(here->MOS9SPbPtr) += (-here->MOS9gbs-(xnrm-xrev)*here->MOS9gmbs);
            *(here->MOS9SPdpPtr) += (-here->MOS9gds-
                    xrev*(here->MOS9gm+here->MOS9gmbs));
        }
    }
    return(OK);
}
