/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

/*
 * This is the function called each iteration to evaluate the
 * BJT2s in the circuit and load them into the matrix as appropriate
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "bjt2defs.h"
#include "const.h"
#include "trandefs.h"
#include "sperror.h"
#include "devdefs.h"
#include "suffix.h"

int
BJT2load(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current resistance value into the 
         * sparse matrix previously provided 
         */
{
    BJT2model *model = (BJT2model*)inModel;
    BJT2instance *here;
    double arg1;
    double arg2;
    double arg3;
    double arg;
    double argtf;
    double c2;
    double c4;
    double capbc;
    double capbe;
    double capbx=0;
    double capsub=0;
    double cb;
    double cbc;
    double cbcn;
    double cbe;
    double cben;
    double cbhat;
    double cc;
    double cchat;
    double cdis;
    double ceq;
    double ceqbc;
    double ceqbe;
    double ceqbx;
    double geqsub;
    double ceqsub;
    double cex;
    double csat;
    double csubsat;
    double ctot;
    double czbc;
    double czbcf2;
    double czbe;
    double czbef2;
    double czbx;
    double czbxf2;
    double czsub;
    double delvbc;
    double delvbe;
    double denom;
    double dqbdvc;
    double dqbdve;
    double evbc;
    double evbcn;
    double evbe;
    double evben;
    double f1;
    double f2;
    double f3;
    double fcpc;
    double fcpe;
    double gbc;
    double gbcn;
    double gbe;
    double gben;
    double gcsub;
    double gcpr;
    double gepr;
    double geq;
    double geqbx;
    double geqcb;
    double gex;
    double gm;
    double gmu;
    double go;
    double gpi;
    double gx;
    double oik;
    double oikr;
    double ovtf;
    double pc;
    double pe;
    double ps;
    double q1;
    double q2;
    double qb;
    double rbpi;
    double rbpr;
    double sarg;
    double sqarg;
    double td;
    double temp;
    double tf;
    double tr;
    double vbc;
    double vbe;
    double vbx = 0.0;
    double vce;
    double vsub = 0.0;
    double vt;
    double vtc;
    double vte;
    double vtn;
    double xfact;
    double xjrb;
    double xjtf;
    double xmc;
    double xme;
    double xms;
    double xtf;
    double evsub;
    double gdsub;
    double cdsub;
    int icheck;
    int ichk1;
    int error;
    int SenCond=0;
    double m;
    
    /*  loop through all the models */
    for( ; model != NULL; model = model->BJT2nextModel ) {

        /* loop through all the instances of the model */
        for (here = model->BJT2instances; here != NULL ;
                here=here->BJT2nextInstance) {
            if (here->BJT2owner != ARCHme) continue;
           
	    vt = here->BJT2temp * CONSTKoverQ;

            if(ckt->CKTsenInfo){
#ifdef SENSDEBUG
                printf("BJT2load \n");
#endif /* SENSDEBUG */

                if((ckt->CKTsenInfo->SENstatus == PERTURBATION)&&
                    (here->BJT2senPertFlag == OFF)) continue;
                SenCond = here->BJT2senPertFlag;
            }


            gcsub=0;
            ceqsub=0;
            geqbx=0;
            ceqbx=0;
            geqcb=0;
            /*
             *   dc model paramters
             */
            csat=here->BJT2tSatCur*here->BJT2area;
            csubsat=here->BJT2tSubSatCur*here->BJT2area;
            rbpr=here->BJT2tMinBaseResist/here->BJT2area;
            rbpi=here->BJT2tBaseResist/here->BJT2area-rbpr;
            gcpr=here->BJT2tCollectorConduct*here->BJT2area;
            gepr=here->BJT2tEmitterConduct*here->BJT2area;
            oik=model->BJT2invRollOffF/here->BJT2area;
            c2=here->BJT2tBEleakCur*here->BJT2area;
            vte=model->BJT2leakBEemissionCoeff*vt;
            oikr=model->BJT2invRollOffR/here->BJT2area;
	    
	    if (model->BJT2subs == VERTICAL)
                c4=here->BJT2tBCleakCur * here->BJT2areab;
            else
	        c4=here->BJT2tBCleakCur * here->BJT2areac;
	    
	    vtc=model->BJT2leakBCemissionCoeff*vt;
            td=model->BJT2excessPhaseFactor;
            xjrb=model->BJT2baseCurrentHalfResist*here->BJT2area;

            if(SenCond){
#ifdef SENSDEBUG
                printf("BJT2senPertFlag = ON \n");
#endif /* SENSDEBUG */

                if((ckt->CKTsenInfo->SENmode == TRANSEN)&&
                    (ckt->CKTmode & MODEINITTRAN)) {
                    vbe = *(ckt->CKTstate1 + here->BJT2vbe);
                    vbc = *(ckt->CKTstate1 + here->BJT2vbc);
                    vbx=model->BJT2type*(
                        *(ckt->CKTrhsOp+here->BJT2baseNode)-
                        *(ckt->CKTrhsOp+here->BJT2colPrimeNode));
                    vsub=model->BJT2type*model->BJT2subs*(
                      *(ckt->CKTrhsOp+here->BJT2substNode)-
                      *(ckt->CKTrhsOp+here->BJT2substConNode));
                }
                else{
                    vbe = *(ckt->CKTstate0 + here->BJT2vbe);
                    vbc = *(ckt->CKTstate0 + here->BJT2vbc);
                    if((ckt->CKTsenInfo->SENmode == DCSEN)||
                        (ckt->CKTsenInfo->SENmode == TRANSEN)){
                        vbx=model->BJT2type*(
                            *(ckt->CKTrhsOld+here->BJT2baseNode)-
                            *(ckt->CKTrhsOld+here->BJT2colPrimeNode));
                        vsub=model->BJT2type*model->BJT2subs*(
                            *(ckt->CKTrhsOld+here->BJT2substNode)-
                            *(ckt->CKTrhsOld+here->BJT2substConNode));
                    }
                    if(ckt->CKTsenInfo->SENmode == ACSEN){
                        vbx=model->BJT2type*(
                            *(ckt->CKTrhsOp+here->BJT2baseNode)-
                            *(ckt->CKTrhsOp+here->BJT2colPrimeNode));
                        vsub=model->BJT2type*model->BJT2subs*(
                            *(ckt->CKTrhsOp+here->BJT2substNode)-
                            *(ckt->CKTrhsOp+here->BJT2substConNode));
                    }
                }
                goto next1;
            }

            /*
             *   initialization
             */
            icheck=1;
            if(ckt->CKTmode & MODEINITSMSIG) {
                vbe= *(ckt->CKTstate0 + here->BJT2vbe);
                vbc= *(ckt->CKTstate0 + here->BJT2vbc);
                vbx=model->BJT2type*(
                    *(ckt->CKTrhsOld+here->BJT2baseNode)-
                    *(ckt->CKTrhsOld+here->BJT2colPrimeNode));
                vsub=model->BJT2type*model->BJT2subs*(
                    *(ckt->CKTrhsOld+here->BJT2substNode)-
                    *(ckt->CKTrhsOld+here->BJT2substConNode));
            } else if(ckt->CKTmode & MODEINITTRAN) {
                vbe = *(ckt->CKTstate1 + here->BJT2vbe);
                vbc = *(ckt->CKTstate1 + here->BJT2vbc);
                vbx=model->BJT2type*(
                    *(ckt->CKTrhsOld+here->BJT2baseNode)-
                    *(ckt->CKTrhsOld+here->BJT2colPrimeNode));
                vsub=model->BJT2type*model->BJT2subs*(
                    *(ckt->CKTrhsOld+here->BJT2substNode)-
                    *(ckt->CKTrhsOld+here->BJT2substConNode));
                if( (ckt->CKTmode & MODETRAN) && (ckt->CKTmode & MODEUIC) ) {
                    vbx=model->BJT2type*(here->BJT2icVBE-here->BJT2icVCE);
                    vsub=0;
                }
            } else if((ckt->CKTmode & MODEINITJCT) && 
                    (ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)){
                vbe=model->BJT2type*here->BJT2icVBE;
                vce=model->BJT2type*here->BJT2icVCE;
                vbc=vbe-vce;
                vbx=vbc;
                vsub=0;
            } else if((ckt->CKTmode & MODEINITJCT) && (here->BJT2off==0)) {
                vbe=here->BJT2tVcrit;
                vbc=0;
                /* ERROR:  need to initialize VCS, VBX here */
                vsub=vbx=0;
            } else if((ckt->CKTmode & MODEINITJCT) ||
                    ( (ckt->CKTmode & MODEINITFIX) && (here->BJT2off!=0))) {
                vbe=0;
                vbc=0;
                /* ERROR:  need to initialize VCS, VBX here */
                vsub=vbx=0;
            } else {
#ifndef PREDICTOR
                if(ckt->CKTmode & MODEINITPRED) {
                    xfact = ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    *(ckt->CKTstate0 + here->BJT2vbe) = 
                            *(ckt->CKTstate1 + here->BJT2vbe);
                    vbe = (1+xfact)**(ckt->CKTstate1 + here->BJT2vbe)-
                            xfact* *(ckt->CKTstate2 + here->BJT2vbe);
                    *(ckt->CKTstate0 + here->BJT2vbc) = 
                            *(ckt->CKTstate1 + here->BJT2vbc);
                    vbc = (1+xfact)**(ckt->CKTstate1 + here->BJT2vbc)-
                            xfact* *(ckt->CKTstate2 + here->BJT2vbc);
                    *(ckt->CKTstate0 + here->BJT2cc) = 
                            *(ckt->CKTstate1 + here->BJT2cc);
                    *(ckt->CKTstate0 + here->BJT2cb) = 
                            *(ckt->CKTstate1 + here->BJT2cb);
                    *(ckt->CKTstate0 + here->BJT2gpi) = 
                            *(ckt->CKTstate1 + here->BJT2gpi);
                    *(ckt->CKTstate0 + here->BJT2gmu) = 
                            *(ckt->CKTstate1 + here->BJT2gmu);
                    *(ckt->CKTstate0 + here->BJT2gm) = 
                            *(ckt->CKTstate1 + here->BJT2gm);
                    *(ckt->CKTstate0 + here->BJT2go) = 
                            *(ckt->CKTstate1 + here->BJT2go);
                    *(ckt->CKTstate0 + here->BJT2gx) = 
                            *(ckt->CKTstate1 + here->BJT2gx);
                    *(ckt->CKTstate0 + here->BJT2vsub) = 
                            *(ckt->CKTstate1 + here->BJT2vsub);
                    vsub = (1+xfact)**(ckt->CKTstate1 + here->BJT2vsub)-
                            xfact* *(ckt->CKTstate2 + here->BJT2vsub);
                } else {
#endif /* PREDICTOR */
                    /*
                     *   compute new nonlinear branch voltages
                     */
                    vbe=model->BJT2type*(
                        *(ckt->CKTrhsOld+here->BJT2basePrimeNode)-
                        *(ckt->CKTrhsOld+here->BJT2emitPrimeNode));
                    vbc=model->BJT2type*(
                        *(ckt->CKTrhsOld+here->BJT2basePrimeNode)-
                        *(ckt->CKTrhsOld+here->BJT2colPrimeNode));
                    vsub=model->BJT2type*model->BJT2subs*(
                        *(ckt->CKTrhsOld+here->BJT2substNode)-
                        *(ckt->CKTrhsOld+here->BJT2substConNode));
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */
                delvbe=vbe- *(ckt->CKTstate0 + here->BJT2vbe);
                delvbc=vbc- *(ckt->CKTstate0 + here->BJT2vbc);
                vbx=model->BJT2type*(
                    *(ckt->CKTrhsOld+here->BJT2baseNode)-
                    *(ckt->CKTrhsOld+here->BJT2colPrimeNode));
                vsub=model->BJT2type*model->BJT2subs*(
                    *(ckt->CKTrhsOld+here->BJT2substNode)-
                    *(ckt->CKTrhsOld+here->BJT2substConNode));
                cchat= *(ckt->CKTstate0 + here->BJT2cc)+(*(ckt->CKTstate0 + 
                        here->BJT2gm)+ *(ckt->CKTstate0 + here->BJT2go))*delvbe-
                        (*(ckt->CKTstate0 + here->BJT2go)+*(ckt->CKTstate0 +
                        here->BJT2gmu))*delvbc;
                cbhat= *(ckt->CKTstate0 + here->BJT2cb)+ *(ckt->CKTstate0 + 
                        here->BJT2gpi)*delvbe+ *(ckt->CKTstate0 + here->BJT2gmu)*
                        delvbc;
#ifndef NOBYPASS
                /*
                 *    bypass if solution has not changed
                 */
                /* the following collections of if's would be just one
                 * if the average compiler could handle it, but many
                 * find the expression too complicated, thus the split.
                 */
                if( (ckt->CKTbypass) &&
                        (!(ckt->CKTmode & MODEINITPRED)) &&
                        (fabs(delvbe) < (ckt->CKTreltol*MAX(fabs(vbe),
                            fabs(*(ckt->CKTstate0 + here->BJT2vbe)))+
                            ckt->CKTvoltTol)) )
                    if( (fabs(delvbc) < ckt->CKTreltol*MAX(fabs(vbc),
                            fabs(*(ckt->CKTstate0 + here->BJT2vbc)))+
                            ckt->CKTvoltTol) )
                    if( (fabs(cchat-*(ckt->CKTstate0 + here->BJT2cc)) < 
                            ckt->CKTreltol* MAX(fabs(cchat),
                            fabs(*(ckt->CKTstate0 + here->BJT2cc)))+
                            ckt->CKTabstol) )
                    if( (fabs(cbhat-*(ckt->CKTstate0 + here->BJT2cb)) < 
                            ckt->CKTreltol* MAX(fabs(cbhat),
                            fabs(*(ckt->CKTstate0 + here->BJT2cb)))+
                            ckt->CKTabstol) ) {
                    /*
                     * bypassing....
                     */
                    vbe = *(ckt->CKTstate0 + here->BJT2vbe);
                    vbc = *(ckt->CKTstate0 + here->BJT2vbc);
                    cc = *(ckt->CKTstate0 + here->BJT2cc);
                    cb = *(ckt->CKTstate0 + here->BJT2cb);
                    gpi = *(ckt->CKTstate0 + here->BJT2gpi);
                    gmu = *(ckt->CKTstate0 + here->BJT2gmu);
                    gm = *(ckt->CKTstate0 + here->BJT2gm);
                    go = *(ckt->CKTstate0 + here->BJT2go);
                    gx = *(ckt->CKTstate0 + here->BJT2gx);
                    geqcb = *(ckt->CKTstate0 + here->BJT2geqcb);
                    gcsub = *(ckt->CKTstate0 + here->BJT2gcsub);
                    geqbx = *(ckt->CKTstate0 + here->BJT2geqbx);
                    vsub = *(ckt->CKTstate0 + here->BJT2vsub);
                    gdsub = *(ckt->CKTstate0 + here->BJT2gdsub);
                    cdsub = *(ckt->CKTstate0 + here->BJT2cdsub);
                    goto load;
                }
#endif /*NOBYPASS*/
                /*
                 *   limit nonlinear branch voltages
                 */
                ichk1=1;
                vbe = DEVpnjlim(vbe,*(ckt->CKTstate0 + here->BJT2vbe),vt,
                        here->BJT2tVcrit,&icheck);
                vbc = DEVpnjlim(vbc,*(ckt->CKTstate0 + here->BJT2vbc),vt,
                        here->BJT2tVcrit,&ichk1);
                if (ichk1 == 1) icheck=1;
                vsub = DEVpnjlim(vsub,*(ckt->CKTstate0 + here->BJT2vsub),vt,
                        here->BJT2tSubVcrit,&ichk1);
                if (ichk1 == 1) icheck=1;
            }
            /*
             *   determine dc current and derivitives
             */
next1:      vtn=vt*model->BJT2emissionCoeffF;
            if(vbe >= -3*vtn){
                evbe=exp(vbe/vtn);
                cbe=csat*(evbe-1);
                gbe=csat*evbe/vtn;
            } else {
                arg=3*vtn/(vbe*CONSTe);
                arg = arg * arg * arg;
                cbe = -csat*(1+arg);
                gbe = csat*3*arg/vbe;
            }
            if (c2 == 0) {
                cben=0;
                gben=0;
            } else {
                if(vbe >= -3*vte){
                    evben=exp(vbe/vte);
                    cben=c2*(evben-1);
                    gben=c2*evben/vte;
                } else {
                    arg=3*vte/(vbe*CONSTe);
                    arg = arg * arg * arg;
                    cben = -c2*(1+arg);
                    gben = c2*3*arg/vbe;
                }
            }
            gben+=ckt->CKTgmin;
            cben+=ckt->CKTgmin*vbe;
            vtn=vt*model->BJT2emissionCoeffR;
            if(vbc >= -3*vtn) {
                evbc=exp(vbc/vtn);
                cbc=csat*(evbc-1);
                gbc=csat*evbc/vtn;
            } else {
                arg=3*vtn/(vbc*CONSTe);
                arg = arg * arg * arg;
                cbc = -csat*(1+arg);
                gbc = csat*3*arg/vbc;
            }
            if (c4 == 0) {
                cbcn=0;
                gbcn=0;
            } else {
                if(vbc >= -3*vtc) {
                    evbcn=exp(vbc/vtc);
                    cbcn=c4*(evbcn-1);
                    gbcn=c4*evbcn/vtc;
                } else {
                    arg=3*vtc/(vbc*CONSTe);
                    arg = arg * arg * arg;
                    cbcn = -c4*(1+arg);
                    gbcn = c4*3*arg/vbc;
                }
            }
            gbcn+=ckt->CKTgmin;
            cbcn+=ckt->CKTgmin*vbc;
            if(vsub <= -3*vt) {
                arg=3*vt/(vsub*CONSTe);
                arg = arg * arg * arg;
                gdsub = csubsat*3*arg/vsub+ckt->CKTgmin;
                cdsub = -csubsat*(1+arg)+ckt->CKTgmin*vsub;
            } else {
                evsub = exp(MIN(MAX_EXP_ARG,vsub/vt));
                gdsub = csubsat*evsub/vt + ckt->CKTgmin;
                cdsub = csubsat*(evsub-1) + ckt->CKTgmin*vsub;
            }
            /*
             *   determine base charge terms
             */
            q1=1/(1-model->BJT2invEarlyVoltF*vbc-model->BJT2invEarlyVoltR*vbe);
            if(oik == 0 && oikr == 0) {
                qb=q1;
                dqbdve=q1*qb*model->BJT2invEarlyVoltR;
                dqbdvc=q1*qb*model->BJT2invEarlyVoltF;
            } else {
                q2=oik*cbe+oikr*cbc;
                arg=MAX(0,1+4*q2);
                sqarg=1;
                if(arg != 0) sqarg=sqrt(arg);
                qb=q1*(1+sqarg)/2;
                dqbdve=q1*(qb*model->BJT2invEarlyVoltR+oik*gbe/sqarg);
                dqbdvc=q1*(qb*model->BJT2invEarlyVoltF+oikr*gbc/sqarg);
            }
            /*
             *   weil's approx. for excess phase applied with backward-
             *   euler integration
             */
            cc=0;
            cex=cbe;
            gex=gbe;
            if(ckt->CKTmode & (MODETRAN | MODEAC) && td != 0) {
                arg1=ckt->CKTdelta/td;
                arg2=3*arg1;
                arg1=arg2*arg1;
                denom=1+arg1+arg2;
                arg3=arg1/denom;
                if(ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate1 + here->BJT2cexbc)=cbe/qb;
                    *(ckt->CKTstate2 + here->BJT2cexbc)=
                            *(ckt->CKTstate1 + here->BJT2cexbc);
                }
                cc=(*(ckt->CKTstate1 + here->BJT2cexbc)*(1+ckt->CKTdelta/
                        ckt->CKTdeltaOld[1]+arg2)-
                        *(ckt->CKTstate2 + here->BJT2cexbc)*ckt->CKTdelta/
                        ckt->CKTdeltaOld[1])/denom;
                cex=cbe*arg3;
                gex=gbe*arg3;
                *(ckt->CKTstate0 + here->BJT2cexbc)=cc+cex/qb;
            }
            /*
             *   determine dc incremental conductances
             */
            cc=cc+(cex-cbc)/qb-cbc/here->BJT2tBetaR-cbcn;
            cb=cbe/here->BJT2tBetaF+cben+cbc/here->BJT2tBetaR+cbcn;
            gx=rbpr+rbpi/qb;
            if(xjrb != 0) {
                arg1=MAX(cb/xjrb,1e-9);
                arg2=(-1+sqrt(1+14.59025*arg1))/2.4317/sqrt(arg1);
                arg1=tan(arg2);
                gx=rbpr+3*rbpi*(arg1-arg2)/arg2/arg1/arg1;
            }
            if(gx != 0) gx=1/gx;
            gpi=gbe/here->BJT2tBetaF+gben;
            gmu=gbc/here->BJT2tBetaR+gbcn;
            go=(gbc+(cex-cbc)*dqbdvc/qb)/qb;
            gm=(gex-(cex-cbc)*dqbdve/qb)/qb-go;
            if( (ckt->CKTmode & (MODETRAN | MODEAC)) ||
                    ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) ||
                    (ckt->CKTmode & MODEINITSMSIG)) {
                /*
                 *   charge storage elements
                 */
                tf=model->BJT2transitTimeF;
                tr=model->BJT2transitTimeR;
                czbe=here->BJT2tBEcap*here->BJT2area;
                pe=here->BJT2tBEpot;
                xme=model->BJT2junctionExpBE;
                cdis=model->BJT2baseFractionBCcap;
		
		if (model->BJT2subs == VERTICAL)
		    ctot=here->BJT2tBCcap*here->BJT2areab;
		else    
                    ctot=here->BJT2tBCcap*here->BJT2areac;
		    
		czbc=ctot*cdis;
                czbx=ctot-czbc;
                pc=here->BJT2tBCpot;
                xmc=model->BJT2junctionExpBC;
                fcpe=here->BJT2tDepCap;
		
		if (model->BJT2subs == VERTICAL)
                    czsub=here->BJT2tSubcap*here->BJT2areac;
                else
		    czsub=here->BJT2tSubcap*here->BJT2areab;
		    
		ps=here->BJT2tSubpot;
                xms=model->BJT2exponentialSubstrate;
                xtf=model->BJT2transitTimeBiasCoeffF;
                ovtf=model->BJT2transitTimeVBCFactor;
                xjtf=model->BJT2transitTimeHighCurrentF*here->BJT2area;
                if(tf != 0 && vbe >0) {
                    argtf=0;
                    arg2=0;
                    arg3=0;
                    if(xtf != 0){
                        argtf=xtf;
                        if(ovtf != 0) {
                            argtf=argtf*exp(vbc*ovtf);
                        }
                        arg2=argtf;
                        if(xjtf != 0) {
                            temp=cbe/(cbe+xjtf);
                            argtf=argtf*temp*temp;
                            arg2=argtf*(3-temp-temp);
                        }
                        arg3=cbe*argtf*ovtf;
                    }
                    cbe=cbe*(1+argtf)/qb;
                    gbe=(gbe*(1+arg2)-cbe*dqbdve)/qb;
                    geqcb=tf*(arg3-cbe*dqbdvc)/qb;
                }
                if (vbe < fcpe) {
                    arg=1-vbe/pe;
                    sarg=exp(-xme*log(arg));
                    *(ckt->CKTstate0 + here->BJT2qbe)=tf*cbe+pe*czbe*
                            (1-arg*sarg)/(1-xme);
                    capbe=tf*gbe+czbe*sarg;
                } else {
                    f1=here->BJT2tf1;
                    f2=model->BJT2f2;
                    f3=model->BJT2f3;
                    czbef2=czbe/f2;
                    *(ckt->CKTstate0 + here->BJT2qbe) = tf*cbe+czbe*f1+czbef2*
                            (f3*(vbe-fcpe) +(xme/(pe+pe))*(vbe*vbe-fcpe*fcpe));
                    capbe=tf*gbe+czbef2*(f3+xme*vbe/pe);
                }
                fcpc=here->BJT2tf4;
                f1=here->BJT2tf5;
                f2=model->BJT2f6;
                f3=model->BJT2f7;
                if (vbc < fcpc) {
                    arg=1-vbc/pc;
                    sarg=exp(-xmc*log(arg));
                    *(ckt->CKTstate0 + here->BJT2qbc) = tr*cbc+pc*czbc*(
                            1-arg*sarg)/(1-xmc);
                    capbc=tr*gbc+czbc*sarg;
                } else {
                    czbcf2=czbc/f2;
                    *(ckt->CKTstate0 + here->BJT2qbc) = tr*cbc+czbc*f1+czbcf2*
                            (f3*(vbc-fcpc) +(xmc/(pc+pc))*(vbc*vbc-fcpc*fcpc));
                    capbc=tr*gbc+czbcf2*(f3+xmc*vbc/pc);
                }
                if(vbx < fcpc) {
                    arg=1-vbx/pc;
                    sarg=exp(-xmc*log(arg));
                    *(ckt->CKTstate0 + here->BJT2qbx)= 
                        pc*czbx* (1-arg*sarg)/(1-xmc);
                    capbx=czbx*sarg;
                } else {
                    czbxf2=czbx/f2;
                    *(ckt->CKTstate0 + here->BJT2qbx)=czbx*f1+czbxf2*
                            (f3*(vbx-fcpc)+(xmc/(pc+pc))*(vbx*vbx-fcpc*fcpc));
                    capbx=czbxf2*(f3+xmc*vbx/pc);
                }
                if(vsub < 0){
                    arg=1-vsub/ps;
                    sarg=exp(-xms*log(arg));
                    *(ckt->CKTstate0 + here->BJT2qsub) = ps*czsub*(1-arg*sarg)/
                            (1-xms);
                    capsub=czsub*sarg;
                } else {
                    *(ckt->CKTstate0 + here->BJT2qsub) = vsub*czsub*(1+xms*vsub/
                            (2*ps));
                    capsub=czsub*(1+xms*vsub/ps);
		}
		here->BJT2capbe = capbe;
		here->BJT2capbc = capbc;
		here->BJT2capsub = capsub;
		here->BJT2capbx = capbx;

                /*
                 *   store small-signal parameters
                 */
                if ( (!(ckt->CKTmode & MODETRANOP))||
                        (!(ckt->CKTmode & MODEUIC)) ) {
                    if(ckt->CKTmode & MODEINITSMSIG) {
                        *(ckt->CKTstate0 + here->BJT2cqbe) = capbe;
                        *(ckt->CKTstate0 + here->BJT2cqbc) = capbc;
                        *(ckt->CKTstate0 + here->BJT2cqsub) = capsub;
                        *(ckt->CKTstate0 + here->BJT2cqbx) = capbx;
                        *(ckt->CKTstate0 + here->BJT2cexbc) = geqcb;
                        if(SenCond){
                            *(ckt->CKTstate0 + here->BJT2cc) = cc;
                            *(ckt->CKTstate0 + here->BJT2cb) = cb;
                            *(ckt->CKTstate0 + here->BJT2gpi) = gpi;
                            *(ckt->CKTstate0 + here->BJT2gmu) = gmu;
                            *(ckt->CKTstate0 + here->BJT2gm) = gm;
                            *(ckt->CKTstate0 + here->BJT2go) = go;
                            *(ckt->CKTstate0 + here->BJT2gx) = gx;
                            *(ckt->CKTstate0 + here->BJT2gcsub) = gcsub;
                            *(ckt->CKTstate0 + here->BJT2geqbx) = geqbx;
                        }
#ifdef SENSDEBUG
                        printf("storing small signal parameters for op\n");
                        printf("capbe = %.7e ,capbc = %.7e\n",capbe,capbc);
                        printf("capsub = %.7e ,capbx = %.7e\n",capsub,capbx);
                        printf("geqcb = %.7e ,gpi = %.7e\n",geqcb,gpi);
                        printf("gmu = %.7e ,gm = %.7e\n",gmu,gm);
                        printf("go = %.7e ,gx = %.7e\n",go,gx);
                        printf("gcsub = %.7e ,geqbx = %.7e\n",gcsub,geqbx);
                        printf("cc = %.7e ,cb = %.7e\n",cc,cb);
#endif /* SENSDEBUG */
                        continue; /* go to 1000 */
                    }
                    /*
                     *   transient analysis
                     */
                    if(SenCond && ckt->CKTsenInfo->SENmode == TRANSEN){
                        *(ckt->CKTstate0 + here->BJT2cc) = cc;
                        *(ckt->CKTstate0 + here->BJT2cb) = cb;
                        *(ckt->CKTstate0 + here->BJT2gx) = gx;
                        continue;
                    }

                    if(ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->BJT2qbe) =
                                *(ckt->CKTstate0 + here->BJT2qbe) ;
                        *(ckt->CKTstate1 + here->BJT2qbc) =
                                *(ckt->CKTstate0 + here->BJT2qbc) ;
                        *(ckt->CKTstate1 + here->BJT2qbx) =
                                *(ckt->CKTstate0 + here->BJT2qbx) ;
                        *(ckt->CKTstate1 + here->BJT2qsub) =
                                *(ckt->CKTstate0 + here->BJT2qsub) ;
                    }
                    error = NIintegrate(ckt,&geq,&ceq,capbe,here->BJT2qbe);
                    if(error) return(error);
                    geqcb=geqcb*ckt->CKTag[0];
                    gpi=gpi+geq;
                    cb=cb+*(ckt->CKTstate0 + here->BJT2cqbe);
                    error = NIintegrate(ckt,&geq,&ceq,capbc,here->BJT2qbc);
                    if(error) return(error);
                    gmu=gmu+geq;
                    cb=cb+*(ckt->CKTstate0 + here->BJT2cqbc);
                    cc=cc-*(ckt->CKTstate0 + here->BJT2cqbc);
                    if(ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->BJT2cqbe) =
                                *(ckt->CKTstate0 + here->BJT2cqbe);
                        *(ckt->CKTstate1 + here->BJT2cqbc) =
                                *(ckt->CKTstate0 + here->BJT2cqbc);
                    }
                }
            }

            if(SenCond) goto next2;

            /*
             *   check convergence
             */
            if ( (!(ckt->CKTmode & MODEINITFIX))||(!(here->BJT2off))) {
                if (icheck == 1) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }

            /*
             *      charge storage for c-s and b-x junctions
             */
            if(ckt->CKTmode & (MODETRAN | MODEAC)) {
                error = NIintegrate(ckt,&gcsub,&ceq,capsub,here->BJT2qsub);
                if(error) return(error);
                error = NIintegrate(ckt,&geqbx,&ceq,capbx,here->BJT2qbx);
                if(error) return(error);
                if(ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate1 + here->BJT2cqbx) =
                            *(ckt->CKTstate0 + here->BJT2cqbx);
                    *(ckt->CKTstate1 + here->BJT2cqsub) =
                            *(ckt->CKTstate0 + here->BJT2cqsub);
                }
            }
next2:
            *(ckt->CKTstate0 + here->BJT2vbe) = vbe;
            *(ckt->CKTstate0 + here->BJT2vbc) = vbc;
            *(ckt->CKTstate0 + here->BJT2cc) = cc;
            *(ckt->CKTstate0 + here->BJT2cb) = cb;
            *(ckt->CKTstate0 + here->BJT2gpi) = gpi;
            *(ckt->CKTstate0 + here->BJT2gmu) = gmu;
            *(ckt->CKTstate0 + here->BJT2gm) = gm;
            *(ckt->CKTstate0 + here->BJT2go) = go;
            *(ckt->CKTstate0 + here->BJT2gx) = gx;
            *(ckt->CKTstate0 + here->BJT2geqcb) = geqcb;
            *(ckt->CKTstate0 + here->BJT2gcsub) = gcsub;
            *(ckt->CKTstate0 + here->BJT2geqbx) = geqbx;
            *(ckt->CKTstate0 + here->BJT2vsub) = vsub;
            *(ckt->CKTstate0 + here->BJT2gdsub) = gdsub;
            *(ckt->CKTstate0 + here->BJT2cdsub) = cdsub;


            /* Do not load the Jacobian and the rhs if
               perturbation is being carried out */

            if(SenCond)continue;
load:
            m = here->BJT2m;
            /*
             *  load current excitation vector
             */
            geqsub = gcsub + gdsub;
            ceqsub=model->BJT2type * model->BJT2subs *
                   (*(ckt->CKTstate0 + here->BJT2cqsub) + cdsub - vsub*geqsub);
/*
            ceqsub=model->BJT2type * (*(ckt->CKTstate0 + here->BJT2cqsub) + 
                                      model->BJT2subs*cdsub - vsub*geqsub);
*/
            ceqbx=model->BJT2type * (*(ckt->CKTstate0 + here->BJT2cqbx) -
                    vbx * geqbx);
            ceqbe=model->BJT2type * (cc + cb - vbe * (gm + go + gpi) + vbc * 
                    (go - geqcb));
            ceqbc=model->BJT2type * (-cc + vbe * (gm + go) - vbc * (gmu + go));

            *(ckt->CKTrhs + here->BJT2baseNode) +=  m * (-ceqbx);
            *(ckt->CKTrhs + here->BJT2colPrimeNode) += 
                     m * (ceqbx+ceqbc);
            *(ckt->CKTrhs + here->BJT2substConNode) += m * ceqsub;
            *(ckt->CKTrhs + here->BJT2basePrimeNode) += 
                    m * (-ceqbe-ceqbc);
            *(ckt->CKTrhs + here->BJT2emitPrimeNode) += m * (ceqbe);
            *(ckt->CKTrhs + here->BJT2substNode) +=  m * (-ceqsub);
            
            /*
             *  load y matrix
             */
            *(here->BJT2colColPtr) +=               m * (gcpr);
            *(here->BJT2baseBasePtr) +=             m * (gx+geqbx);
            *(here->BJT2emitEmitPtr) +=             m * (gepr);
            *(here->BJT2colPrimeColPrimePtr) +=     m * (gmu+go+gcpr+geqbx);
            *(here->BJT2substConSubstConPtr) +=     m * (geqsub);
            *(here->BJT2basePrimeBasePrimePtr) +=   m * (gx +gpi+gmu+geqcb);
            *(here->BJT2emitPrimeEmitPrimePtr) +=   m * (gpi+gepr+gm+go);
            *(here->BJT2colColPrimePtr) +=          m * (-gcpr);
            *(here->BJT2baseBasePrimePtr) +=        m * (-gx);
            *(here->BJT2emitEmitPrimePtr) +=        m * (-gepr);
            *(here->BJT2colPrimeColPtr) +=          m * (-gcpr);
            *(here->BJT2colPrimeBasePrimePtr) +=    m * (-gmu+gm);
            *(here->BJT2colPrimeEmitPrimePtr) +=    m * (-gm-go);
            *(here->BJT2basePrimeBasePtr) +=        m * (-gx);
            *(here->BJT2basePrimeColPrimePtr) +=    m * (-gmu-geqcb);
            *(here->BJT2basePrimeEmitPrimePtr) +=   m * (-gpi);
            *(here->BJT2emitPrimeEmitPtr) +=        m * (-gepr);
            *(here->BJT2emitPrimeColPrimePtr) +=    m * (-go+geqcb);
            *(here->BJT2emitPrimeBasePrimePtr) +=   m * (-gpi-gm-geqcb);
            *(here->BJT2substSubstPtr) +=           m * (geqsub);
            *(here->BJT2substConSubstPtr) +=        m * (-geqsub);
            *(here->BJT2substSubstConPtr) +=        m * (-geqsub);
            *(here->BJT2baseColPrimePtr) +=         m * (-geqbx);
            *(here->BJT2colPrimeBasePtr) +=         m * (-geqbx);
        }
    }
    return(OK);
}
