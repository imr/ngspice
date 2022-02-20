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
    double cd, cdb, cdsw, cdb_dT, cdsw_dT;
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
    double evrev;
    double gd, gdb, gdsw, gen_fac, gen_fac_vd;
    double t1, evd_rec, cdb_rec, gdb_rec, cdb_rec_dT;
    double geq;
    double gspr;    /* area-scaled conductance */
    double sarg;
#ifndef NOBYPASS
    double tol;     /* temporary for tolerence calculations */
#endif
    double vd;      /* current diode voltage */
    double vdtemp;
    double vt;      /* K t / Q */
    double vte, vtesw, vtetun, vtebrk;
    int Check_dio=0, Check_th;
    int error;
    int SenCond=0;    /* sensitivity condition */
    double diffcharge, deplcharge, deplchargeSW, diffcap, deplcap, deplcapSW;

    double deldelTemp, delTemp, Temp;
    double ceqqth=0.0, Ith=0.0, gcTt=0.0, vrs=0.0;
    double dIdio_dT, dIth_dVdio=0.0, dIrs_dT=0.0, dIth_dVrs=0.0, dIth_dT=0.0;
    double argsw_dT, csat_dT, csatsw_dT;

    /*  loop through all the diode models */
    for( ; model != NULL; model = DIOnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ;
                here=DIOnextInstance(here)) {

            int selfheat = ((here->DIOtempNode > 0) && (here->DIOthermal) && (model->DIOrth0Given));

            /*
             *     this routine loads diodes for dc and transient analyses.
             */

            if (selfheat)
                Check_th = 1;
            else
                Check_th = 0;

            if(ckt->CKTsenInfo){
                if((ckt->CKTsenInfo->SENstatus == PERTURBATION)
                        && (here->DIOsenPertFlag == OFF))continue;
                SenCond = here->DIOsenPertFlag;

#ifdef SENSDEBUG
                printf("DIOload \n");
#endif /* SENSDEBUG */

            }
            cdsw = 0.0;
            cdsw_dT = 0.0;
            gdsw = 0.0;
            delTemp = 0.0;
            vt = CONSTKoverQ * here->DIOtemp;
            vte = model->DIOemissionCoeff * vt;
            vtebrk = model->DIObrkdEmissionCoeff * vt;
            gspr = here->DIOtConductance;
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
                    delTemp = *(ckt->CKTstate1 + here->DIOdeltemp);
                } else{
                    vd = *(ckt->CKTstate0 + here->DIOvoltage);
                    delTemp = *(ckt->CKTstate0 + here->DIOdeltemp);
                }

#ifdef SENSDEBUG
                printf("vd = %.7e \n",vd);
#endif /* SENSDEBUG */
                goto next1;
            }

            Check_dio=1;
            if(ckt->CKTmode & MODEINITSMSIG) {
                vd= *(ckt->CKTstate0 + here->DIOvoltage);
                delTemp = *(ckt->CKTstate0 + here->DIOdeltemp);
            } else if (ckt->CKTmode & MODEINITTRAN) {
                vd= *(ckt->CKTstate1 + here->DIOvoltage);
                delTemp = *(ckt->CKTstate1 + here->DIOdeltemp);
            } else if ( (ckt->CKTmode & MODEINITJCT) &&
                    (ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC) ) {
                vd=here->DIOinitCond;
            } else if ( (ckt->CKTmode & MODEINITJCT) && here->DIOoff) {
                vd=0;
                delTemp = 0.0;
            } else if ( ckt->CKTmode & MODEINITJCT) {
                vd=here->DIOtVcrit;
                delTemp = 0.0;
            } else if ( ckt->CKTmode & MODEINITFIX && here->DIOoff) {
                vd=0;
                delTemp = 0.0;
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
                    *(ckt->CKTstate0 + here->DIOdeltemp) =
                            *(ckt->CKTstate1 + here->DIOdeltemp);
                    delTemp = DEVpred(ckt,here->DIOdeltemp);
                    *(ckt->CKTstate0 + here->DIOdIdio_dT) =
                            *(ckt->CKTstate1 + here->DIOdIdio_dT);
                    *(ckt->CKTstate0+here->DIOqth) =
                            *(ckt->CKTstate1+here->DIOqth);
                } else {
#endif /* PREDICTOR */
                    vd = *(ckt->CKTrhsOld+here->DIOposPrimeNode)-
                            *(ckt->CKTrhsOld + here->DIOnegNode);
                    if (selfheat)
                        delTemp = *(ckt->CKTrhsOld + here->DIOtempNode);
                    else
                        delTemp = 0.0;
                    *(ckt->CKTstate0+here->DIOqth) = model->DIOcth0 * delTemp;
                    if((ckt->CKTmode & MODEINITTRAN)) {
                        *(ckt->CKTstate1+here->DIOqth) =
                            *(ckt->CKTstate0+here->DIOqth);
                    }
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */
                delvd=vd- *(ckt->CKTstate0 + here->DIOvoltage);
                deldelTemp = delTemp - *(ckt->CKTstate0 + here->DIOdeltemp);
                cdhat= *(ckt->CKTstate0 + here->DIOcurrent) +
                        *(ckt->CKTstate0 + here->DIOconduct) * delvd +
                        *(ckt->CKTstate0 + here->DIOdIdio_dT) * deldelTemp;
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
                            if ((here->DIOtempNode == 0) ||
                                (fabs(deldelTemp) < (ckt->CKTreltol * MAX(fabs(delTemp),
                                      fabs(*(ckt->CKTstate0+here->DIOdeltemp)))+
                                      ckt->CKTvoltTol*1e4))) {
                                vd= *(ckt->CKTstate0 + here->DIOvoltage);
                                cd= *(ckt->CKTstate0 + here->DIOcurrent);
                                gd= *(ckt->CKTstate0 + here->DIOconduct);
                                delTemp = *(ckt->CKTstate0 + here->DIOdeltemp);
                                dIdio_dT= *(ckt->CKTstate0 + here->DIOdIdio_dT);
                                goto load;
                            }
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
                            here->DIOtVcrit,&Check_dio);
                    vd = -(vdtemp+here->DIOtBrkdwnV);
                } else {
                    vd = DEVpnjlim(vd,*(ckt->CKTstate0 + here->DIOvoltage),
                            vte,here->DIOtVcrit,&Check_dio);
                }
                if (selfheat)
                    delTemp = DEVlimitlog(delTemp,
                        *(ckt->CKTstate0 + here->DIOdeltemp), 100, &Check_th);
                else
                    delTemp = 0.0;
            }
            /*
             *   compute dc current and derivitives
             */
next1:
            if (selfheat) {
                Temp = here->DIOtemp + delTemp;
                DIOtempUpdate(model, here, Temp, ckt);
                vt = CONSTKoverQ * Temp;
                vte = model->DIOemissionCoeff * vt;
                vtebrk = model->DIObrkdEmissionCoeff * vt;
            } else {
                Temp = here->DIOtemp;
            }

            csat = here->DIOtSatCur;
            csat_dT = here->DIOtSatCur_dT;
            csatsw = here->DIOtSatSWCur;
            csatsw_dT = here->DIOtSatSWCur_dT;
            gspr = here->DIOtConductance;

            if (model->DIOsatSWCurGiven) {              /* sidewall current */

                if (model->DIOswEmissionCoeffGiven) {   /* current with own characteristic */

                    vtesw = model->DIOswEmissionCoeff * vt;

                    if (vd >= -3*vtesw) {               /* forward */

                        evd = exp(vd/vtesw);
                        cdsw = csatsw*(evd-1);
                        gdsw = csatsw*evd/vtesw;
                        cdsw_dT = csatsw_dT * (evd - 1) - csatsw * vd * evd / (vtesw * Temp);

                    } else if((!(model->DIObreakdownVoltageGiven)) ||
                            vd >= -here->DIOtBrkdwnV) { /* reverse */

                        argsw = 3*vtesw/(vd*CONSTe);
                        argsw = argsw * argsw * argsw;
                        argsw_dT = 3 * argsw / Temp;
                        cdsw = -csatsw*(1+argsw);
                        gdsw = csatsw*3*argsw/vd;
                        cdsw_dT = -csatsw_dT - (csatsw_dT*argsw + csatsw*argsw_dT);

                    } else {                            /* breakdown */
                        double evrev_dT;

                        evrev = exp(-(here->DIOtBrkdwnV+vd)/vtebrk);
                        evrev_dT = (here->DIOtBrkdwnV+vd)*evrev/(vtebrk*Temp);
                        cdsw = -csatsw*evrev;
                        gdsw = csatsw*evrev/vtebrk;
                        cdsw_dT = -(csatsw_dT*evrev + csatsw*evrev_dT);

                    }

                } else { /* merge saturation currents and use same characteristic as bottom diode */

                    csat = csat + csatsw;
                    csat_dT = csat_dT + csatsw_dT;
                    cdsw_dT = 0.0;

                }

            }

            /*
             *   temperature dependent diode saturation current and derivative
             */

            if (vd >= -3*vte) {                 /* bottom current forward */

                evd = exp(vd/vte);
                cdb = csat*(evd-1);
                gdb = csat*evd/vte;
                cdb_dT = csat_dT * (evd - 1) - csat * vd * evd / (vte * Temp);
                if (model->DIOrecSatCurGiven) { /* recombination current */
                    double vterec = model->DIOrecEmissionCoeff*vt;
                    evd_rec = exp(vd/(vterec));
                    cdb_rec = here->DIOtRecSatCur*(evd_rec-1);
                    gdb_rec = here->DIOtRecSatCur*evd_rec/vterec;
                    cdb_rec_dT = here->DIOtRecSatCur_dT * (evd_rec - 1)
                                -here->DIOtRecSatCur * vd * evd_rec / (vterec*Temp);
                    t1 = pow((1-vd/here->DIOtJctPot), 2) + 0.005;
                    gen_fac = pow(t1, here->DIOtGradingCoeff/2);
                    gen_fac_vd = -here->DIOtGradingCoeff * (1-vd/here->DIOtJctPot)
                                                         * pow(t1, (here->DIOtGradingCoeff/2-1));
                    cdb_rec = cdb_rec * gen_fac;
                    gdb_rec = gdb_rec * gen_fac + cdb_rec * gen_fac_vd;
                    cdb = cdb + cdb_rec;
                    gdb = gdb + gdb_rec;
                    cdb_dT = cdb_dT + cdb_rec_dT*gen_fac;
                }

            } else if((!(model->DIObreakdownVoltageGiven)) ||
                    vd >= -here->DIOtBrkdwnV) { /* reverse */

                double darg_dT;

                arg = 3*vte/(vd*CONSTe);
                arg = arg * arg * arg;
                darg_dT = 3 * arg / Temp;
                cdb = -csat*(1+arg);
                gdb = csat*3*arg/vd;
                cdb_dT = -csat_dT - (csat_dT*arg + csat*darg_dT);

            } else {                            /* breakdown */
                double evrev_dT;

                evrev = exp(-(here->DIOtBrkdwnV+vd)/vtebrk);
                evrev_dT = (here->DIOtBrkdwnV+vd)*evrev/(vtebrk*Temp);
                cdb = -csat*evrev;
                gdb = csat*evrev/vtebrk;
                cdb_dT = -(csat_dT*evrev + csat*evrev_dT);

            }

            if (model->DIOtunSatSWCurGiven) {    /* tunnel sidewall current */

                vtetun = model->DIOtunEmissionCoeff * vt;
                evd = exp(-vd/vtetun);

                cdsw = cdsw - here->DIOtTunSatSWCur * (evd - 1);
                gdsw = gdsw + here->DIOtTunSatSWCur * evd / vtetun;
                cdsw_dT = cdsw_dT - here->DIOtTunSatSWCur_dT * (evd - 1)
                                  - here->DIOtTunSatSWCur * vd * evd / (vtetun * Temp);

            }

            if (model->DIOtunSatCurGiven) {      /* tunnel bottom current */

                vtetun = model->DIOtunEmissionCoeff * vt;
                evd = exp(-vd/vtetun);

                cdb = cdb - here->DIOtTunSatCur * (evd - 1);
                gdb = gdb + here->DIOtTunSatCur * evd / vtetun;
                cdb_dT = cdb_dT - here->DIOtTunSatCur_dT * (evd - 1)
                                - here->DIOtTunSatCur * vd * evd / (vtetun * Temp);

            }

            cd = cdb + cdsw;
            gd = gdb + gdsw;
            dIdio_dT = cdb_dT + cdsw_dT;

            if (vd >= -3*vte) { /* limit forward */

                if( (model->DIOforwardKneeCurrentGiven) && (cd > 1.0e-18) ) {
                    ikf_area_m = here->DIOforwardKneeCurrent;
                    sqrt_ikf = sqrt(cd/ikf_area_m);
                    gd = ((1+sqrt_ikf)*gd - cd*gd/(2*sqrt_ikf*ikf_area_m))/(1+2*sqrt_ikf + cd/ikf_area_m) + ckt->CKTgmin;
                    cd = cd/(1+sqrt_ikf) + ckt->CKTgmin*vd;
                } else {
                    gd = gd + ckt->CKTgmin;
                    cd = cd + ckt->CKTgmin*vd;
                }

            } else {            /* limit reverse */

                if( (model->DIOreverseKneeCurrentGiven) && (cd < -1.0e-18) ) {
                    ikr_area_m = here->DIOreverseKneeCurrent;
                    sqrt_ikr = sqrt(cd/(-ikr_area_m));
                    gd = ((1+sqrt_ikr)*gd + cd*gd/(2*sqrt_ikr*ikr_area_m))/(1+2*sqrt_ikr - cd/ikr_area_m) + ckt->CKTgmin;
                    cd = cd/(1+sqrt_ikr) + ckt->CKTgmin*vd;
                } else {
                    gd = gd + ckt->CKTgmin;
                    cd = cd + ckt->CKTgmin*vd;
                }

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
                czeroSW=here->DIOtJctSWCap;
                if (vd < here->DIOtDepSWCap){
                    argSW=1-vd/here->DIOtJctSWPot;
                    sargSW=exp(-model->DIOgradingSWCoeff*log(argSW));
                    deplchargeSW = here->DIOtJctSWPot*czeroSW*(1-argSW*sargSW)/(1-model->DIOgradingSWCoeff);
                    deplcapSW = czeroSW*sargSW;
                } else {
                    czof2SW=czeroSW/here->DIOtF2SW;
                    deplchargeSW = czeroSW*here->DIOtF1+czof2SW*(here->DIOtF3SW*(vd-here->DIOtDepSWCap)+
                                   (model->DIOgradingSWCoeff/(here->DIOtJctSWPot+here->DIOtJctSWPot))*(vd*vd-here->DIOtDepSWCap*here->DIOtDepSWCap));
                    deplcapSW = czof2SW*(here->DIOtF3SW+model->DIOgradingSWCoeff*vd/here->DIOtJctSWPot);
                }

                diffcharge = here->DIOtTransitTime*cd;
                *(ckt->CKTstate0 + here->DIOcapCharge) =
                        diffcharge + deplcharge + deplchargeSW;

                diffcap = here->DIOtTransitTime*gd;

                capd = diffcap + deplcap + deplcapSW + here->DIOcmetal + here->DIOcpoly;

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
                            *(ckt->CKTstate0 + here->DIOdIdio_dT) = dIdio_dT;
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
                    if (selfheat)
                    {
                        error = NIintegrate(ckt, &gcTt, &ceqqth, model->DIOcth0, here->DIOqth);
                        if (error) return(error);
                        if (ckt->CKTmode & MODEINITTRAN) {
                            *(ckt->CKTstate1 + here->DIOcqth) =
                                    *(ckt->CKTstate0 + here->DIOcqth);
                        }
                    }
                }
            }

            if(SenCond) goto next2;

            /*
             *   check convergence
             */
            if ( (!(ckt->CKTmode & MODEINITFIX)) || (!(here->DIOoff))  ) {
                if ((Check_th == 1) || (Check_dio == 1)) {
                    ckt->CKTnoncon++;
                    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }
next2:      *(ckt->CKTstate0 + here->DIOvoltage) = vd;
            *(ckt->CKTstate0 + here->DIOcurrent) = cd;
            *(ckt->CKTstate0 + here->DIOconduct) = gd;
            *(ckt->CKTstate0 + here->DIOdeltemp) = delTemp;
            *(ckt->CKTstate0 + here->DIOdIdio_dT) = dIdio_dT;

            if(SenCond)  continue;

#ifndef NOBYPASS
            load:
#endif
            if (selfheat) {
                double dIrs_dVrs, dIrs_dgspr, dIth_dIrs;
                vrs = *(ckt->CKTrhsOld + here->DIOposNode) - *(ckt->CKTrhsOld + here->DIOposPrimeNode);
                Ith = vd*cd + vrs*vrs*gspr; /* Diode dissipated power */
                dIrs_dVrs = gspr;
                dIrs_dgspr = vrs;
                dIrs_dT = dIrs_dgspr * here->DIOtConductance_dT;
                dIth_dVrs = vrs*gspr;
                dIth_dIrs = vrs;
                dIth_dVrs = dIth_dVrs + dIth_dIrs*dIrs_dVrs;
                dIth_dT = dIth_dIrs*dIrs_dT + dIdio_dT*vd;
                dIth_dVdio = cd + vd*gd;
                here->DIOdIth_dVrs = dIth_dVrs;
                here->DIOdIth_dVdio = dIth_dVdio;
                here->DIOdIth_dT = dIth_dT;
                here->DIOgcTt = gcTt;
                here->DIOdIrs_dT = dIrs_dT;
            }
            /*
             *   load current vector
             */
            cdeq=cd-gd*vd;
            *(ckt->CKTrhs + here->DIOnegNode) += cdeq;
            *(ckt->CKTrhs + here->DIOposPrimeNode) -= cdeq;
            if (selfheat) {
                *(ckt->CKTrhs + here->DIOposNode)      +=  dIrs_dT*delTemp;
                *(ckt->CKTrhs + here->DIOposPrimeNode) +=  dIdio_dT*delTemp - dIrs_dT*delTemp;
                *(ckt->CKTrhs + here->DIOnegNode)      += -dIdio_dT*delTemp;
                *(ckt->CKTrhs + here->DIOtempNode)     +=  Ith - dIth_dVdio*vd - dIth_dVrs*vrs - dIth_dT*delTemp - ceqqth;
            }
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
            if (selfheat) {
                (*(here->DIOtempPosPtr)      += -dIth_dVrs);
                (*(here->DIOtempPosPrimePtr) += -dIth_dVdio + dIth_dVrs);
                (*(here->DIOtempNegPtr)      +=  dIth_dVdio);
                (*(here->DIOtempTempPtr)     += -dIth_dT + 1/model->DIOrth0 + gcTt);
                (*(here->DIOposTempPtr)      +=  dIrs_dT);
                (*(here->DIOposPrimeTempPtr) +=  dIdio_dT - dIrs_dT);
                (*(here->DIOnegTempPtr)      += -dIdio_dT);
            }
        }
    }
    return(OK);
}
