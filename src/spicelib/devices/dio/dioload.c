/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
Modified by Paolo Nenzi 2003, Dietmar Warning 2012 and Ste Kulov 2021
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/const.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
DIOload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current resistance value into the
         * sparse matrix previously provided
         */
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;
    double arg;
    double capd;
    double cd;
    double cdeq;
    double cdhat;
    double ceq;
    double csat;    /* area-scaled saturation current */
    double czero;
    double czof2;
    double delvd;   /* change in diode voltage temporary */
    double evd;
    double evrev;
    double gd;
    double geq;
    double gspr;    /* area-scaled conductance */
    double sarg;
#ifndef NOBYPASS
    double tol;     /* temporary for tolerence calculations */
#endif
    double vd;      /* current diode voltage */
    double vdtemp;
    double vt;      /* K t / Q */
    double vte, vtebrk;
    int Check=0;
    int error;
    int SenCond=0;    /* sensitivity condition */
    double diffcharge, deplcharge, diffcap, deplcap;

    double tt;
    double vp;

    /*  loop through all the diode models */
    for( ; model != NULL; model = DIOnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ;
                here=DIOnextInstance(here)) {

            /*
             *     this routine loads diodes for dc and transient analyses.
             */

            if(ckt->CKTsenInfo){
                if((ckt->CKTsenInfo->SENstatus == PERTURBATION)
                        && (here->DIOsenPertFlag == OFF))continue;
                SenCond = here->DIOsenPertFlag;

#ifdef SENSDEBUG
                printf("DIOload \n");
#endif /* SENSDEBUG */

            }
            csat = here->DIOtSatCur;
            gspr = here->DIOtConductance;
            vt = CONSTKoverQ * here->DIOtemp;
            vte = model->DIOemissionCoeff * vt;
            vtebrk = model->DIObrkdEmissionCoeff * vt;
            tt = here->DIOtTransitTime;
            vp = model->DIOsoftRevRecParam;

            /*
             *   initialization
             */

            if(SenCond){

#ifdef SENSDEBUG
                printf("DIOsenPertFlag = ON \n");
#endif /* SENSDEBUG */

                if((ckt->CKTsenInfo->SENmode == TRANSEN)&&
                        (ckt->CKTmode & MODEINITTRAN)) {
                    vd = *(ckt->CKTstate1 + here->DIOvoltage);
                } else{
                    vd = *(ckt->CKTstate0 + here->DIOvoltage);
                }

#ifdef SENSDEBUG
                printf("vd = %.7e \n",vd);
#endif /* SENSDEBUG */
                goto next1;
            }

            Check=1;
            if(ckt->CKTmode & MODEINITSMSIG) {
                vd= *(ckt->CKTstate0 + here->DIOvoltage);
            } else if (ckt->CKTmode & MODEINITTRAN) {
                vd= *(ckt->CKTstate1 + here->DIOvoltage);
            } else if ( (ckt->CKTmode & MODEINITJCT) &&
                    (ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC) ) {
                vd=here->DIOinitCond;
            } else if ( (ckt->CKTmode & MODEINITJCT) && here->DIOoff) {
                vd=0;
            } else if ( ckt->CKTmode & MODEINITJCT) {
                vd=here->DIOtVcrit;
            } else if ( ckt->CKTmode & MODEINITFIX && here->DIOoff) {
                vd=0;
            } else {
#ifndef PREDICTOR
                if (ckt->CKTmode & MODEINITPRED) {
                    *(ckt->CKTstate0 + here->DIOvoltage) =
                            *(ckt->CKTstate1 + here->DIOvoltage);
                    vd = DEVpred(ckt,here->DIOvoltage);
                    *(ckt->CKTstate0 + here->DIOcurrent) =
                            *(ckt->CKTstate1 + here->DIOcurrent);
                    *(ckt->CKTstate0 + here->DIOconduct) =
                            *(ckt->CKTstate1 + here->DIOconduct);
                } else {
#endif /* PREDICTOR */
                    vd = *(ckt->CKTrhsOld+here->DIOposPrimeNode)-
                            *(ckt->CKTrhsOld + here->DIOnegNode);
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */
                delvd=vd- *(ckt->CKTstate0 + here->DIOvoltage);
                cdhat= *(ckt->CKTstate0 + here->DIOcurrent) +
                        *(ckt->CKTstate0 + here->DIOconduct) * delvd;
                /*
                 *   bypass if solution has not changed
                 */
#ifndef NOBYPASS
                if ((!(ckt->CKTmode & MODEINITPRED)) && (ckt->CKTbypass)) {
                    tol=ckt->CKTvoltTol + ckt->CKTreltol*
                        MAX(fabs(vd),fabs(*(ckt->CKTstate0 +here->DIOvoltage)));
                    if (fabs(delvd) < tol){
                        tol=ckt->CKTreltol* MAX(fabs(cdhat),
                                fabs(*(ckt->CKTstate0 + here->DIOcurrent)))+
                                ckt->CKTabstol;
                        if (fabs(cdhat- *(ckt->CKTstate0 + here->DIOcurrent))
                                < tol) {
                            vd= *(ckt->CKTstate0 + here->DIOvoltage);
                            cd= *(ckt->CKTstate0 + here->DIOcurrent);
                            gd= *(ckt->CKTstate0 + here->DIOconduct);
                            goto load;
                        }
                    }
                }
#endif /* NOBYPASS */
                /*
                 *   limit new junction voltage
                 */
                if ( (model->DIObreakdownVoltageGiven) &&
                        (vd < MIN(0,-here->DIOtBrkdwnV+10*vtebrk))) {
                    vdtemp = -(vd+here->DIOtBrkdwnV);
                    vdtemp = DEVpnjlim(vdtemp,
                            -(*(ckt->CKTstate0 + here->DIOvoltage) +
                            here->DIOtBrkdwnV),vtebrk,
                            here->DIOtVcrit,&Check);
                    vd = -(vdtemp+here->DIOtBrkdwnV);
                } else {
                    vd = DEVpnjlim(vd,*(ckt->CKTstate0 + here->DIOvoltage),
                            vte,here->DIOtVcrit,&Check);
                }
            }
            /*
             *   compute dc current and derivitives
             */
next1:
            if (vd >= -3*vte) {                 /* forward */

                evd = exp(vd/vte);
                cd = csat*(evd-1) + ckt->CKTgmin*vd;
                gd = csat*evd/vte + ckt->CKTgmin;

            } else if((!(model->DIObreakdownVoltageGiven)) ||
                    vd >= -here->DIOtBrkdwnV) { /* reverse */

                arg = 3*vte/(vd*CONSTe);
                arg = arg * arg * arg;
                cd = -csat*(1+arg) + ckt->CKTgmin*vd;
                gd = csat*3*arg/vd + ckt->CKTgmin;

            } else {                            /* breakdown */

                evrev = exp(-(here->DIOtBrkdwnV+vd)/vtebrk);
                cd = -csat*evrev + ckt->CKTgmin*vd;
                gd = csat*evrev/vtebrk + ckt->CKTgmin;

            }

            if ((ckt->CKTmode & (MODEDCTRANCURVE | MODETRAN | MODEAC | MODEINITSMSIG)) ||
                     ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC))) {
              /*
               *   charge storage elements
               */
                czero=here->DIOtJctCap;
                if (vd < here->DIOtDepCap){
                    arg=1-vd/here->DIOtJctPot;
                    sarg=exp(-here->DIOtGradingCoeff*log(arg));
                    deplcharge = here->DIOtJctPot*czero*(1-arg*sarg)/(1-here->DIOtGradingCoeff);
                    deplcap = czero*sarg;
                } else {
                    czof2=czero/here->DIOtF2;
                    deplcharge = czero*here->DIOtF1+czof2*(here->DIOtF3*(vd-here->DIOtDepCap)+
                                 (here->DIOtGradingCoeff/(here->DIOtJctPot+here->DIOtJctPot))*(vd*vd-here->DIOtDepCap*here->DIOtDepCap));
                    deplcap = czof2*(here->DIOtF3+here->DIOtGradingCoeff*vd/here->DIOtJctPot);
                }

                if (model->DIOsoftRevRecParamGiven) {

                    if (ckt->CKTmode & MODEINITTRAN) {
                        diffcharge = tt * cd;
                        diffcap = tt * gd;
                        *(ckt->CKTstate2 + here->DIOdiffCharge) = diffcharge;
                        *(ckt->CKTstate2 + here->DIOdiffCap) = diffcap;
                        *(ckt->CKTstate1 + here->DIOdiffCharge) = diffcharge;
                        *(ckt->CKTstate1 + here->DIOdiffCap) = diffcap;
                        *(ckt->CKTstate1 + here->DIOoldCurr) = cd;
                        *(ckt->CKTstate1 + here->DIOoldCond) = gd;
                    }
                    else {
                        double dt = ckt->CKTdelta;

                        //Backward Euler
                        // Qk+1 = tt (VP Qk + h Ik+1) / (tt VP + h)
                        //diffcharge = tt*((vp*(*(ckt->CKTstate1 + here->DIOdiffCharge)) + dt*cd) / (tt*vp + dt));
                        //diffcap = tt*((vp*(*(ckt->CKTstate1 + here->DIOdiffCap)) + dt*gd) / (tt*vp + dt));

                        //Trap
                        // Qk+1 = (Qk (2 tt VP - h) + h tt (Ik + Ik+1)) / (2 tt VP + h)
                        diffcharge = ((*(ckt->CKTstate1 + here->DIOdiffCharge))*(2.0*tt*vp - dt) + dt*tt*((*(ckt->CKTstate1 + here->DIOoldCurr)) + cd)) / (2.0*tt*vp + dt);
                        diffcap = ((*(ckt->CKTstate1 + here->DIOdiffCap))*(2.0*tt*vp - dt) + dt*tt*((*(ckt->CKTstate1 + here->DIOoldCond)) + gd)) / (2.0*tt*vp + dt);

                        //Gear
                        // Qk+1 = tt ((VP (4 Qk - Qk-1) + 2 h Ik+1) / (3 tt VP + 2 h))
                        //diffcharge = tt*((vp*(4.0*(*(ckt->CKTstate1 + here->DIOdiffCharge)) - (*(ckt->CKTstate2 + here->DIOdiffCharge))) + 2.0*dt*cd) / (3.0*tt*vp + 2.0*dt));
                        //diffcap = tt*((vp*(4.0*(*(ckt->CKTstate1 + here->DIOdiffCap)) - (*(ckt->CKTstate2 + here->DIOdiffCap))) + 2.0*dt*gd) / (3.0*tt*vp + 2.0*dt));

                    }

                    //printf("time: %e vd: %e\n", ckt->CKTtime, vd);
                    //printf("%e %e %e %e %e %e %e\n", ckt->CKTtime, ckt->CKTdelta, diffcharge, diffcap, vd, cd, gd);

                } else {

                    diffcharge = tt*cd;
                    diffcap = tt*gd;

                }

                *(ckt->CKTstate0 + here->DIOdiffCharge) = diffcharge;
                *(ckt->CKTstate0 + here->DIOdiffCap) = diffcap;
                *(ckt->CKTstate0 + here->DIOcapCharge) = deplcharge + diffcharge;
                *(ckt->CKTstate0 + here->DIOoldCurr) = cd;
                *(ckt->CKTstate0 + here->DIOoldCond) = gd;

                capd = deplcap + diffcap;

                here->DIOcap = capd;

                /*
                 *   store small-signal parameters
                 */
                if( (!(ckt->CKTmode & MODETRANOP)) ||
                        (!(ckt->CKTmode & MODEUIC)) ) {
                    if (ckt->CKTmode & MODEINITSMSIG){
                        *(ckt->CKTstate0 + here->DIOcapCurrent) = capd;

                        if(SenCond){
                            *(ckt->CKTstate0 + here->DIOcurrent) = cd;
                            *(ckt->CKTstate0 + here->DIOconduct) = gd;
#ifdef SENSDEBUG
                            printf("storing small signal parameters\n");
                            printf("cd = %.7e,vd = %.7e\n",cd,vd);
                            printf("capd = %.7e ,gd = %.7e \n",capd,gd);
#endif /* SENSDEBUG */
                        }
                        continue;
                    }

                    /*
                     *   transient analysis
                     */
                    if(SenCond && (ckt->CKTsenInfo->SENmode == TRANSEN)){
                        *(ckt->CKTstate0 + here->DIOcurrent) = cd;
#ifdef SENSDEBUG
                        printf("storing parameters for transient sensitivity\n"
                                );
                        printf("qd = %.7e, capd = %.7e,cd = %.7e\n",
                                *(ckt->CKTstate0 + here->DIOcapCharge),capd,cd);
#endif /* SENSDEBUG */
                        continue;
                    }

                    if (ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->DIOcapCharge) =
                                *(ckt->CKTstate0 + here->DIOcapCharge);
                    }
                    error = NIintegrate(ckt,&geq,&ceq,capd,here->DIOcapCharge);
                    if(error) return(error);
                    gd=gd+geq;
                    cd=cd+*(ckt->CKTstate0 + here->DIOcapCurrent);
                    if (ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->DIOcapCurrent) =
                                *(ckt->CKTstate0 + here->DIOcapCurrent);
                    }
                }
            }

            if(SenCond) goto next2;

            /*
             *   check convergence
             */
            if ( (!(ckt->CKTmode & MODEINITFIX)) || (!(here->DIOoff))  ) {
                if (Check == 1)  {
                    ckt->CKTnoncon++;
                    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }
next2:      *(ckt->CKTstate0 + here->DIOvoltage) = vd;
            *(ckt->CKTstate0 + here->DIOcurrent) = cd;
            *(ckt->CKTstate0 + here->DIOconduct) = gd;

            if(SenCond)  continue;

#ifndef NOBYPASS
            load:
#endif
            /*
             *   load current vector
             */
            cdeq=cd-gd*vd;
            *(ckt->CKTrhs + here->DIOnegNode) += cdeq;
            *(ckt->CKTrhs + here->DIOposPrimeNode) -= cdeq;
            /*
             *   load matrix
             */
            *(here->DIOposPosPtr) += gspr;
            *(here->DIOnegNegPtr) += gd;
            *(here->DIOposPrimePosPrimePtr) += (gd + gspr);
            *(here->DIOposPosPrimePtr) -= gspr;
            *(here->DIOnegPosPrimePtr) -= gd;
            *(here->DIOposPrimePosPtr) -= gspr;
            *(here->DIOposPrimeNegPtr) -= gd;
        }
    }
    return(OK);
}
