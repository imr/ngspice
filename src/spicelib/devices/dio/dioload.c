/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
Modified by Paolo Nenzi 2003 and Dietmar Warning 2012
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
    double argsw;
    double capd;
    double cd, cdsw=0.0;
    double cdeq;
    double cdhat;
    double ceq;
    double csat;    /* area-scaled saturation current */
    double csatsw;  /* perimeter-scaled saturation current */
    double czero;
    double czof2;
    double argSW;
    double czeroSW;
    double czof2SW;
    double sargSW;
    double sqrt_ikr;
    double sqrt_ikf;
    double ikf_area_m;
    double ikr_area_m;

    double delvd;   /* change in diode voltage temporary */
    double evd;
    double evdsw;
    double evrev;
    double gd, gdsw=0.0;
    double geq;
    double gspr;    /* area-scaled conductance */
    double sarg;
#ifndef NOBYPASS
    double tol;     /* temporary for tolerence calculations */
#endif
    double vd;      /* current diode voltage */
    double vdtemp;
    double vt;      /* K t / Q */
    double vte, vtesw, vtetun;
    double vtebrk;
    int Check;
    int error;
    int SenCond=0;    /* sensitivity condition */

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->DIOnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->DIOinstances; here != NULL ;
                here=here->DIOnextInstance) {
            if (here->DIOowner != ARCHme) continue;

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
            csatsw = here->DIOtSatSWCur;
            gspr = here->DIOtConductance * here->DIOarea;
            vt = CONSTKoverQ * here->DIOtemp;
            vte = model->DIOemissionCoeff * vt;
            vtebrk = model->DIObrkdEmissionCoeff * vt;
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
next1:      if (model->DIOsatSWCurGiven) { /* consider sidewall currents */

                if (model->DIOswEmissionCoeffGiven) { /* sidewall currents with own characteristic */

                    vtesw = model->DIOswEmissionCoeff * vt;

                    if (vd >= -3*vtesw) {   /* sidewall forward */

                        evdsw = exp(vd/vtesw);
                        cdsw = csatsw*(evdsw-1);
                        gdsw = csatsw*evdsw/vtesw;
                        if (model->DIOtunSatSWCurGiven) {
                            vtetun = model->DIOtunEmissionCoeff * vt;
                            evd = exp(-vd/vtetun);
                            cdsw = cdsw - here->DIOtTunSatSWCur * (evd - 1);
                            gdsw = gdsw + here->DIOtTunSatSWCur * evd / vtetun;
                        }

                    } else if((!(model->DIObreakdownVoltageGiven)) ||   /* sidewall reverse */
                            vd >= -here->DIOtBrkdwnV) {

                        if (model->DIOtunSatSWCurGiven) {
                            evdsw = exp(vd/vtesw);
                            cdsw = csatsw*(evdsw-1);
                            gdsw = csatsw*evdsw/vtesw;
                            vtetun = model->DIOtunEmissionCoeff * vt;
                            evdsw = exp(-vd/vtetun);
                            cdsw = cdsw - here->DIOtTunSatSWCur * (evdsw - 1);
                            gdsw = gdsw + here->DIOtTunSatSWCur * evdsw / vtetun;
                        } else {
                            argsw = 3*vtesw/(vd*CONSTe);
                            argsw = argsw * argsw * argsw;
                            cdsw = -csatsw*(1+argsw);
                            gdsw = csatsw*3*argsw/vd;
                        }

                    } else {    /* sidewall breakdown */

                        evrev = exp(-(here->DIOtBrkdwnV+vd)/vtebrk);
                        cdsw = -csatsw*evrev;
                        gdsw = csatsw*evrev/vtebrk;
                        if (model->DIOtunSatSWCurGiven) {
                            evd = exp(vd/vte);
                            cdsw = cdsw - csatsw*(evd-1);
                            gdsw = gdsw - csatsw*evd/vte;
                            vtetun = model->DIOtunEmissionCoeff * vt;
                            evd = exp(-vd/vtetun);
                            cdsw = cdsw - here->DIOtTunSatSWCur * (evd - 1);
                            gdsw = gdsw + here->DIOtTunSatSWCur * evd / vtetun;
                        }

                    }

                } else { /* merge the current densities and use same characteristic as bottom diode */

                    csat = csat + csatsw;

                }

            }

            if (vd >= -3*vte) {   /* bottom forward */

                evd = exp(vd/vte);
                cd = csat*(evd-1);
                gd = csat*evd/vte;
                if (model->DIOtunSatCurGiven) {
                    vtetun = model->DIOtunEmissionCoeff * vt;
                    evd = exp(-vd/vtetun);
                    cd = cd - here->DIOtTunSatCur * (evd - 1);
                    gd = gd + here->DIOtTunSatCur * evd / vtetun;
                }

            } else if((!(model->DIObreakdownVoltageGiven)) ||   /* bottom reverse */
                    vd >= -here->DIOtBrkdwnV) {

                if (model->DIOtunSatCurGiven) {
                    evd = exp(vd/vte);
                    cd = csat*(evd-1);
                    gd = csat*evd/vte;
                    vtetun = model->DIOtunEmissionCoeff * vt;
                    evd = exp(-vd/vtetun);
                    cd = cd - here->DIOtTunSatCur * (evd - 1);
                    gd = gd + here->DIOtTunSatCur * evd / vtetun;
                } else {
                    arg = 3*vte/(vd*CONSTe);
                    arg = arg * arg * arg;
                    cd = -csat*(1+arg);
                    gd = csat*3*arg/vd;
                }

            } else {    /* bottom breakdown */

                evrev = exp(-(here->DIOtBrkdwnV+vd)/vtebrk);
                cd = -csat*evrev;
                gd = csat*evrev/vtebrk;
                if (model->DIOtunSatCurGiven) {
                    evd = exp(vd/vte);
                    cd = cd - csat*(evd-1);
                    gd = gd - csat*evd/vte;
                    vtetun = model->DIOtunEmissionCoeff * vt;
                    evd = exp(-vd/vtetun);
                    cd = cd - here->DIOtTunSatCur * (evd - 1);
                    gd = gd + here->DIOtTunSatCur * evd / vtetun;
                }

            }

            if (model->DIOsatSWCurGiven) {

                if (model->DIOswEmissionCoeffGiven) {

                    cd = cdsw + cd;
                    gd = gdsw + gd;

                }

            }

            if (vd >= -3*vte) {   /* limit forward */

                if( (model->DIOforwardKneeCurrent > 0.0) && (cd > 1.0e-18) ) {
                    ikf_area_m = here->DIOforwardKneeCurrent;
                    sqrt_ikf = sqrt(cd/ikf_area_m);
                    gd = ((1+sqrt_ikf)*gd - cd*gd/(2*sqrt_ikf*ikf_area_m))/(1+2*sqrt_ikf + cd/ikf_area_m) + ckt->CKTgmin*vd;
                    cd = cd/(1+sqrt_ikf) + ckt->CKTgmin;
                }

            } else {    /* limit reverse */

                if( (model->DIOreverseKneeCurrent > 0.0) && (cd < -1.0e-18) ) {
                    ikr_area_m = here->DIOreverseKneeCurrent;
                    sqrt_ikr = sqrt(cd/(-ikr_area_m));
                    gd = ((1+sqrt_ikr)*gd + cd*gd/(2*sqrt_ikr*ikr_area_m))/(1+2*sqrt_ikr - cd/ikr_area_m) + ckt->CKTgmin*vd;
                    cd = cd/(1+sqrt_ikr) + ckt->CKTgmin;
                }

            }

            if ((ckt->CKTmode & (MODETRAN | MODEAC | MODEINITSMSIG)) ||
                     ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC))) {
              /*
               *   charge storage elements
               */
                czero=here->DIOtJctCap;
                czeroSW=here->DIOtJctSWCap;
                if (vd < here->DIOtDepCap){
                    arg=1-vd/here->DIOtJctPot;
                    argSW=1-vd/here->DIOtJctSWPot;
                    sarg=exp(-here->DIOtGradingCoeff*log(arg));
                    sargSW=exp(-model->DIOgradingSWCoeff*log(argSW));
                    *(ckt->CKTstate0 + here->DIOcapCharge) =
                            here->DIOtTransitTime*cd+
                            here->DIOtJctPot*czero*(1-arg*sarg)/(1-here->DIOtGradingCoeff)+
                            here->DIOtJctSWPot*czeroSW*(1-argSW*sargSW)/(1-model->DIOgradingSWCoeff);
                    capd=here->DIOtTransitTime*gd+czero*sarg+czeroSW*sargSW;
                } else {
                    czof2=czero/here->DIOtF2;
                    czof2SW=czeroSW/here->DIOtF2SW;
                    *(ckt->CKTstate0 + here->DIOcapCharge) =
                            here->DIOtTransitTime*cd+czero*here->DIOtF1+
                            czof2*(here->DIOtF3*(vd-here->DIOtDepCap)+(here->DIOtGradingCoeff/(here->DIOtJctPot+here->DIOtJctPot))*(vd*vd-here->DIOtDepCap*here->DIOtDepCap))+
                            czof2SW*(here->DIOtF3SW*(vd-here->DIOtDepCap)+(model->DIOgradingSWCoeff/(here->DIOtJctSWPot+here->DIOtJctSWPot))*(vd*vd-here->DIOtDepCap*here->DIOtDepCap));
                    capd=here->DIOtTransitTime*gd+
                            czof2*(here->DIOtF3+here->DIOtGradingCoeff*vd/here->DIOtJctPot)+
                            czof2SW*(here->DIOtF3SW+model->DIOgradingSWCoeff*vd/here->DIOtJctSWPot);
                }
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
