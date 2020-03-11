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

/* DIOlimitlog(deltemp, deltemp_old, LIM_TOL, check)
 * Logarithmic damping the per-iteration change of deltemp beyond LIM_TOL.
 */
static double
DIOlimitlog(
    double deltemp,
    double deltemp_old,
    double LIM_TOL,
    int *check)
{
    *check = 0;
    if (isnan (deltemp) || isnan (deltemp_old))
    {
        fprintf(stderr, "Alberto says:  YOU TURKEY!  The limiting function received NaN.\n");
        fprintf(stderr, "New prediction returns to 0.0!\n");
        deltemp = 0.0;
        *check = 1;
    }
    /* Logarithmic damping of deltemp beyond LIM_TOL */
    if (deltemp > deltemp_old + LIM_TOL) {
        deltemp = deltemp_old + LIM_TOL + log10((deltemp-deltemp_old)/LIM_TOL);
        *check = 1;
    }
    else if (deltemp < deltemp_old - LIM_TOL) {
        deltemp = deltemp_old - LIM_TOL - log10((deltemp_old-deltemp)/LIM_TOL);
        *check = 1;
    }
    return deltemp;
}

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
    double cd, cdb, cdsw;
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

    double evd;
    double delvd;   /* change in diode voltage temporary */
    double evrev;
    double gd, gdb, gdsw, gen_fac, gen_fac_vd;
    double t1, evd_rec, cdb_rec, gdb_rec;
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
    int Check_dio, Check_th;
    int error;
    int SenCond=0;    /* sensitivity condition */
    double diffcharge, diffchargeSW, deplcharge, deplchargeSW, diffcap, diffcapSW, deplcap, deplcapSW;

    register int selfheat;
    double deldelTemp, delTemp, Temp;
    double ceqqth, Ith=0.0;
    double dIdio_dT, dIth_dVdio=0.0, dIth_dT=0.0;
    double arg1, darg1_dT, arg2, darg2_dT, csat_dT;
    double vbrknp, gcTt;

    /*  loop through all the diode models */
    for( ; model != NULL; model = DIOnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ;
                here=DIOnextInstance(here)) {

            /*
             *     this routine loads diodes for dc and transient analyses.
             */

            selfheat = ((model->DIOshMod == 1) && (model->DIOrth0Given));
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
            cd = 0.0;
            cdb = 0.0;
            cdsw = 0.0;
            gd = 0.0;
            gdb = 0.0;
            gdsw = 0.0;
            delTemp = 0.0;
            csat = here->DIOtSatCur;
            csatsw = here->DIOtSatSWCur;
            gspr = here->DIOtConductance * here->DIOarea;
            vt = CONSTKoverQ * here->DIOtemp;
            vte = model->DIOemissionCoeff * vt;
            vtebrk = model->DIObrkdEmissionCoeff * vt;
            vbrknp = here->DIOtBrkdwnV;
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
                } else {
#endif /* PREDICTOR */
                    vd = *(ckt->CKTrhsOld+here->DIOposPrimeNode)-
                            *(ckt->CKTrhsOld + here->DIOnegNode);
                    if (selfheat)
                        delTemp = *(ckt->CKTrhsOld + here->DIOtempNode);
                    else
                        delTemp = 0.0;
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
                                delTemp = *(ckt->CKTstate0 + here->DIOdeltemp);
                                cd= *(ckt->CKTstate0 + here->DIOcurrent);
                                gd= *(ckt->CKTstate0 + here->DIOconduct);
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
                    delTemp = DIOlimitlog(delTemp,
                        *(ckt->CKTstate0 + here->DIOdeltemp), 100, &Check_th);
                else
                    delTemp = 0.0;
            }
            /*
             *   compute dc current and derivitives
             */
next1:      if (model->DIOsatSWCurGiven) {              /* sidewall current */

                if (model->DIOswEmissionCoeffGiven) {   /* current with own characteristic */

                    vtesw = model->DIOswEmissionCoeff * vt;

                    if (vd >= -3*vtesw) {               /* forward */

                        evd = exp(vd/vtesw);
                        cdsw = csatsw*(evd-1);
                        gdsw = csatsw*evd/vtesw;

                    } else if((!(model->DIObreakdownVoltageGiven)) ||
                            vd >= -here->DIOtBrkdwnV) { /* reverse */

                        argsw = 3*vtesw/(vd*CONSTe);
                        argsw = argsw * argsw * argsw;
                        cdsw = -csatsw*(1+argsw);
                        gdsw = csatsw*3*argsw/vd;

                    } else {                            /* breakdown */

                        evrev = exp(-(here->DIOtBrkdwnV+vd)/vtebrk);
                        cdsw = -csatsw*evrev;
                        gdsw = csatsw*evrev/vtebrk;

                    }

                } else { /* merge saturation currents and use same characteristic as bottom diode */

                    csat = csat + csatsw;

                }

            }

            Temp = here->DIOtemp + delTemp;
            /*
            *   temperature dependent diode saturation current and derivative
            */
            arg1 = ((Temp / model->DIOnomTemp) - 1) * model->DIOactivationEnergy / (model->DIOemissionCoeff*vt);
            darg1_dT = model->DIOactivationEnergy / (vte*model->DIOnomTemp)
                      - model->DIOactivationEnergy*(Temp/model->DIOnomTemp -1)/(vte*Temp);

            arg2 = model->DIOsaturationCurrentExp / model->DIOemissionCoeff * log(Temp / model->DIOnomTemp);
            darg2_dT = model->DIOsaturationCurrentExp / model->DIOemissionCoeff / Temp;

            csat = here->DIOm * model->DIOsatCur * exp(arg1 + arg2);
            csat_dT = here->DIOm * model->DIOsatCur * exp(arg1 + arg2) * (darg1_dT + darg2_dT);

            if (vd >= -3*vte) {                 /* bottom current forward */

                evd = exp(vd/vte);
                cdb = csat*(evd-1);
                dIdio_dT = csat_dT * (evd - 1) - csat * vd * evd / (vte * Temp);
                gdb = csat*evd/vte;
                if (model->DIOrecSatCurGiven) { /* recombination current */
                    evd_rec = exp(vd/(model->DIOrecEmissionCoeff*vt));
                    cdb_rec = here->DIOtRecSatCur*(evd_rec-1);
                    gdb_rec = here->DIOtRecSatCur*evd_rec/vt;
                    t1 = pow((1-vd/here->DIOtJctPot), 2) + 0.005;
                    gen_fac = pow(t1, here->DIOtGradingCoeff/2);
                    gen_fac_vd = here->DIOtGradingCoeff * (1-vd/here->DIOtJctPot) * pow(t1, (here->DIOtGradingCoeff/2-1));
                    cdb_rec = cdb_rec * gen_fac;
                    gdb_rec = gdb_rec * gen_fac + cdb_rec * gen_fac_vd;
                    cdb = cdb + cdb_rec;
                    gdb = gdb + gdb_rec;
                }

            } else if((!(model->DIObreakdownVoltageGiven)) ||
                    vd >= -here->DIOtBrkdwnV) { /* reverse */
                double arg3, darg3_dT;

                arg = 3*vte/(vd*CONSTe);
                arg = arg * arg * arg;
                arg3 = arg * arg * arg;
                darg3_dT = 3 * arg3 / Temp; 
                cdb = -csat*(1+arg);
                dIdio_dT = -csat_dT * (arg3 + 1) - csat * darg3_dT;
                gdb = csat*3*arg/vd;

            } else {                            /* breakdown */

                evrev = exp(-(here->DIOtBrkdwnV+vd)/vtebrk);
                cdb = -csat*evrev;
                dIdio_dT = csat * (-vbrknp-vd) * evrev / vtebrk / Temp - csat_dT * evrev;
                gdb = csat*evrev/vtebrk;

            }

            if (model->DIOtunSatSWCurGiven) {    /* tunnel sidewall current */

                vtetun = model->DIOtunEmissionCoeff * vt;
                evd = exp(-vd/vtetun);

                cdsw = cdsw - here->DIOtTunSatSWCur * (evd - 1);
                gdsw = gdsw + here->DIOtTunSatSWCur * evd / vtetun;

            }

            if (model->DIOtunSatCurGiven) {      /* tunnel bottom current */

                vtetun = model->DIOtunEmissionCoeff * vt;
                evd = exp(-vd/vtetun);

                cdb = cdb - here->DIOtTunSatCur * (evd - 1);
                gdb = gdb + here->DIOtTunSatCur * evd / vtetun;

            }

            cd = cdb + cdsw;
            gd = gdb + gdsw;

            if (vd >= -3*vte) { /* limit forward */

                if( (model->DIOforwardKneeCurrent > 0.0) && (cd > 1.0e-18) ) {
                    ikf_area_m = here->DIOforwardKneeCurrent;
                    sqrt_ikf = sqrt(cd/ikf_area_m);
                    gd = ((1+sqrt_ikf)*gd - cd*gd/(2*sqrt_ikf*ikf_area_m))/(1+2*sqrt_ikf + cd/ikf_area_m) + ckt->CKTgmin;
                    cd = cd/(1+sqrt_ikf) + ckt->CKTgmin*vd;
                } else {
                    gd = gd + ckt->CKTgmin;
                    cd = cd + ckt->CKTgmin*vd;
                }

            } else {            /* limit reverse */

                if( (model->DIOreverseKneeCurrent > 0.0) && (cd < -1.0e-18) ) {
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

                diffcharge = here->DIOtTransitTime*cdb;
                diffchargeSW = here->DIOtTransitTime*cdsw;
                *(ckt->CKTstate0 + here->DIOcapCharge) =
                        diffcharge + diffchargeSW + deplcharge + deplchargeSW;

                diffcap = here->DIOtTransitTime*gdb;
                diffcapSW = here->DIOtTransitTime*gdsw;
                capd = diffcap + diffcapSW + deplcap + deplcapSW;

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
                    if (selfheat)
                    {
                        error = NIintegrate(ckt, &gcTt, &ceqqth, 0.0, here->DIOqth);
                        gcTt = model->DIOcth0 * ckt->CKTag[0];
                        ceqqth = *(ckt->CKTstate0 + here->DIOcqth) - gcTt * delTemp;
                    }
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
                if ((Check_th == 1) || (Check_dio == 1)) {
                    ckt->CKTnoncon++;
                    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }
next2:      *(ckt->CKTstate0 + here->DIOvoltage) = vd;
            *(ckt->CKTstate0 + here->DIOcurrent) = cd;
            *(ckt->CKTstate0 + here->DIOconduct) = gd;
            *(ckt->CKTstate0 + here->DIOdIdio_dT) = dIdio_dT;
            *(ckt->CKTstate0 + here->DIOdeltemp) = delTemp;

            if(SenCond)  continue;

#ifndef NOBYPASS
            load:
#endif
            if (selfheat) {
                dIth_dVdio = cd + vd*gd;
                here->DIOdIth_dVdio = dIth_dVdio;
                Ith = vd*cd;
                dIth_dT = dIdio_dT*vd;
            }
            /*
             *   load current vector
             */
            cdeq=cd-gd*vd;
            *(ckt->CKTrhs + here->DIOnegNode) += cdeq;
            *(ckt->CKTrhs + here->DIOposPrimeNode) -= cdeq;
            if (selfheat) {
                *(ckt->CKTrhs + here->DIOnegNode)      += dIdio_dT*delTemp;
                *(ckt->CKTrhs + here->DIOposPrimeNode) -= dIdio_dT*delTemp;
                *(ckt->CKTrhs + here->DIOtempNode)     += Ith - dIth_dVdio*vd - dIth_dT*delTemp; /* Diode dissipated power */
//printf("pdio: %g rhs: %g delTemp: %g\n", Ith, *(ckt->CKTrhs + here->DIOtempNode), delTemp);
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
                (*(here->DIOtempTempPtr)     += -dIth_dT + 1/model->DIOrth0);
                (*(here->DIOtempPosPrimePtr) += -dIth_dVdio);
                (*(here->DIOtempNegPtr)      +=  dIth_dVdio);
                (*(here->DIOposPrimeTempPtr) += -dIdio_dT);
                (*(here->DIOnegTempPtr)      +=  dIdio_dT);
            }
        }
    }
    return(OK);
}
