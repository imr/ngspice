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
    double capd, capdsw=0.0;
    double cd, cdb, cdsw, cdb_dT, cdsw_dT;
    double cdeq;
    double cdhat, cdhatsw=0.0;
    double ceq;
    double csat;    /* area-scaled saturation current */
    double csatsw;  /* perimeter-scaled saturation current */
    double czero;
    double czof2;
    double argSW;
    double czeroSW;
    double czof2SW;
    double sargSW;
    double sqrt_ikx;

    double delvd, delvdsw=0.0; /* change in diode voltage temporary */
    double evd;
    double evrev;
    double gd, gdb, gdsw, gen_fac, gen_fac_vd;
    double t1, evd_rec, cdb_rec, gdb_rec, cdb_rec_dT;
    double geq;
    double gspr;    /* area-scaled conductance */
    double gsprsw;  /* perim-scaled conductance */
    double sarg;
#ifndef NOBYPASS
    double tol;     /* temporary for tolerence calculations */
#endif
    double vd, vdsw=0.0; /* current diode voltage */
    double vdtemp;
    double vt;      /* K t / Q */
    double vte, vtesw, vtetun, vtebrk;
    int Check_dio=0, Check_dio_sw=0, Check_th;
    int error;
    int SenCond=0;    /* sensitivity condition */
    double diffcharge, deplcharge, deplchargeSW, diffcap, deplcap, deplcapSW;

    double deldelTemp, delTemp, Temp;
    double ceqqth=0.0, Ith=0.0, gcTt=0.0, vrs=0.0, vrssw=0.0;
    double dIdio_dT, dIth_dVdio=0.0, dIrs_dT=0.0, dIth_dVrs=0.0, dIth_dT=0.0;
    double dIdioSw_dT=0.0, dIth_dVdioSw=0.0, dIth_dVrssw=0.0, dIrssw_dT=0.0;
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
            vtesw = model->DIOswEmissionCoeff * vt;
            vtebrk = model->DIObrkdEmissionCoeff * vt;
            gspr = here->DIOtConductance;
            gsprsw = here->DIOtConductanceSW;
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
                    if (model->DIOresistSWGiven) vdsw = *(ckt->CKTstate1 + here->DIOvoltageSW);
                    delTemp = *(ckt->CKTstate1 + here->DIOdeltemp);
                } else{
                    vd = *(ckt->CKTstate0 + here->DIOvoltage);
                    if (model->DIOresistSWGiven) vdsw = *(ckt->CKTstate0 + here->DIOvoltageSW);
                    delTemp = *(ckt->CKTstate0 + here->DIOdeltemp);
                }

#ifdef SENSDEBUG
                printf("vd = %.7e \n",vd);
#endif /* SENSDEBUG */
                goto next1;
            }

            Check_dio=1; Check_dio_sw=1;
            if(ckt->CKTmode & MODEINITSMSIG) {
                vd= *(ckt->CKTstate0 + here->DIOvoltage);
                if (model->DIOresistSWGiven) vdsw = *(ckt->CKTstate0 + here->DIOvoltageSW);
                delTemp = *(ckt->CKTstate0 + here->DIOdeltemp);
            } else if (ckt->CKTmode & MODEINITTRAN) {
                vd= *(ckt->CKTstate1 + here->DIOvoltage);
                if (model->DIOresistSWGiven) vdsw = *(ckt->CKTstate1 + here->DIOvoltageSW);
                delTemp = *(ckt->CKTstate1 + here->DIOdeltemp);
            } else if ( (ckt->CKTmode & MODEINITJCT) &&
                    (ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC) ) {
                vd=here->DIOinitCond;
                if (model->DIOresistSWGiven) vdsw = here->DIOinitCond;
            } else if ( (ckt->CKTmode & MODEINITJCT) && here->DIOoff) {
                vd=vdsw=0;
                delTemp = 0.0;
            } else if ( ckt->CKTmode & MODEINITJCT) {
                vd=here->DIOtVcrit;
                vdsw=here->DIOtVcritSW;
                delTemp = 0.0;
            } else if ( ckt->CKTmode & MODEINITFIX && here->DIOoff) {
                vd=vdsw=0;
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
                    if (model->DIOresistSWGiven) {
                        vdsw = DEVpred(ckt,here->DIOvoltageSW);
                        *(ckt->CKTstate0 + here->DIOdIdioSW_dT) =
                                *(ckt->CKTstate1 + here->DIOdIdioSW_dT);
                    }
                } else {
#endif /* PREDICTOR */
                    vd = *(ckt->CKTrhsOld+here->DIOposPrimeNode)-
                            *(ckt->CKTrhsOld + here->DIOnegNode);
                    if (model->DIOresistSWGiven) vdsw = *(ckt->CKTrhsOld+here->DIOposSwPrimeNode)-
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
                if (model->DIOresistSWGiven) {
                    delvdsw=vdsw - *(ckt->CKTstate0 + here->DIOvoltageSW);
                    cdhatsw = *(ckt->CKTstate0 + here->DIOconductSW) * delvdsw +
                              *(ckt->CKTstate0 + here->DIOdIdioSW_dT) * deldelTemp;
                }
                /*
                 *   bypass if solution has not changed
                 */
#ifndef NOBYPASS
                if ((!(ckt->CKTmode & MODEINITPRED)) && (ckt->CKTbypass)) {
                    tol=ckt->CKTvoltTol + ckt->CKTreltol*
                        MAX(fabs(vd),fabs(*(ckt->CKTstate0 +here->DIOvoltage)));
                    if (fabs(delvd) < tol) {
                        tol=ckt->CKTreltol* MAX(fabs(cdhat),
                                fabs(*(ckt->CKTstate0 + here->DIOcurrent)))+
                                ckt->CKTabstol;
                        if (fabs(cdhat- *(ckt->CKTstate0 + here->DIOcurrent))
                                < tol) {
                            if ((here->DIOtempNode == 0) ||
                                (fabs(deldelTemp) < (ckt->CKTreltol * MAX(fabs(delTemp),
                                      fabs(*(ckt->CKTstate0+here->DIOdeltemp)))+
                                      ckt->CKTvoltTol*1e4))) {
                                if ((!model->DIOresistSWGiven) || (fabs(delvdsw) < ckt->CKTvoltTol + ckt->CKTreltol *
                                                         MAX(fabs(vdsw),fabs(*(ckt->CKTstate0+here->DIOvoltageSW))))) {
                                    if ((!model->DIOresistSWGiven) || (fabs(cdhatsw- *(ckt->CKTstate0 + here->DIOcurrentSW))
                                                             < ckt->CKTreltol* MAX(fabs(cdhatsw),
                                                             fabs(*(ckt->CKTstate0 + here->DIOcurrentSW)))+ckt->CKTabstol)) {
                                        vd= *(ckt->CKTstate0 + here->DIOvoltage);
                                        cd= *(ckt->CKTstate0 + here->DIOcurrent);
                                        gd= *(ckt->CKTstate0 + here->DIOconduct);
                                        delTemp = *(ckt->CKTstate0 + here->DIOdeltemp);
                                        dIdio_dT= *(ckt->CKTstate0 + here->DIOdIdio_dT);
                                        if (model->DIOresistSWGiven) {
                                            vdsw= *(ckt->CKTstate0 + here->DIOvoltageSW);
                                            cdsw= *(ckt->CKTstate0 + here->DIOcurrentSW);
                                            gdsw= *(ckt->CKTstate0 + here->DIOconductSW);
                                            dIdioSw_dT= *(ckt->CKTstate0 + here->DIOdIdioSW_dT);
                                        }
                                        goto load;
                                    }
                                }
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
                if (model->DIOresistSWGiven) {
                    if ( (model->DIObreakdownVoltageGiven) &&
                            (vdsw < MIN(0,-here->DIOtBrkdwnV+10*vtebrk))) {
                        vdtemp = -(vdsw+here->DIOtBrkdwnV);
                        vdtemp = DEVpnjlim(vdtemp,
                                -(*(ckt->CKTstate0 + here->DIOvoltageSW) +
                                here->DIOtBrkdwnV),vtebrk,
                                here->DIOtVcritSW,&Check_dio_sw);
                        vdsw = -(vdtemp+here->DIOtBrkdwnV);
                    } else {
                        vdsw = DEVpnjlim(vdsw,*(ckt->CKTstate0 + here->DIOvoltageSW),
                                  vtesw,here->DIOtVcritSW,&Check_dio_sw);
                    }
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
            gsprsw = here->DIOtConductanceSW;

            if (model->DIOsatSWCurGiven) {               /* sidewall current */
                double vds;
                if (model->DIOresistSWGiven) 
                    vds = vdsw;                          /* sidewall voltage used */
                else
                    vds = vd;                            /* common voltage used */

                if (model->DIOswEmissionCoeffGiven) {    /* with own characteristic */

                    if (vds >= -3*vtesw) {               /* forward */

                        evd = exp(vds/vtesw);
                        cdsw = csatsw*(evd-1);
                        gdsw = csatsw*evd/vtesw;
                        cdsw_dT = csatsw_dT * (evd - 1) - csatsw * vds * evd / (vtesw * Temp);

                    } else if ((!(model->DIObreakdownVoltageGiven)) ||
                            vds >= -here->DIOtBrkdwnV) { /* reverse */

                        argsw = 3*vtesw/(vds*CONSTe);
                        argsw = argsw * argsw * argsw;
                        argsw_dT = 3 * argsw / Temp;
                        cdsw = -csatsw*(1+argsw);
                        gdsw = csatsw*3*argsw/vds;
                        cdsw_dT = -csatsw_dT - (csatsw_dT*argsw + csatsw*argsw_dT);

                    } else if (!model->DIOresistSWGiven){ /* breakdown, but not for separate sidewall diode */
                        double evrev_dT;

                        evrev = exp(-(here->DIOtBrkdwnV+vds)/vtebrk);
                        evrev_dT = (here->DIOtBrkdwnV+vds)*evrev/(vtebrk*Temp);
                        cdsw = -csatsw*evrev;
                        gdsw = csatsw*evrev/vtebrk;
                        cdsw_dT = -(csatsw_dT*evrev + csatsw*evrev_dT);

                    }

                }

            }

            /*
             *   temperature dependent diode saturation current and derivative
             */

            if (vd >= -3*vte) {      /* bottom and sidewall current forward with common voltage */
                                     /* and with common characteristic */
                evd = exp(vd/vte);
                cdb = csat*(evd-1);
                gdb = csat*evd/vte;
                cdb_dT = csat_dT * (evd - 1) - csat * vd * evd / (vte * Temp);
                if ((model->DIOsatSWCurGiven)&&(!model->DIOswEmissionCoeffGiven)) {
                    cdsw = csatsw*(evd-1);
                    gdsw = csatsw*evd/vte;
                    cdsw_dT = csatsw_dT * (evd - 1) - csatsw * vd * evd / (vte * Temp);
                }
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

            } else if ((!(model->DIObreakdownVoltageGiven)) ||
                    vd >= -here->DIOtBrkdwnV) { /* reverse */

                double darg_dT;

                arg = 3*vte/(vd*CONSTe);
                arg = arg * arg * arg;
                darg_dT = 3 * arg / Temp;
                cdb = -csat*(1+arg);
                gdb = csat*3*arg/vd;
                cdb_dT = -csat_dT - (csat_dT*arg + csat*darg_dT);
                if ((model->DIOsatSWCurGiven)&&(!model->DIOswEmissionCoeffGiven)) {
                    cdsw = -csatsw*(1+arg);
                    gdsw = csatsw*3*arg/vd;
                    cdsw_dT = -csatsw_dT - (csatsw_dT*arg + csatsw*darg_dT);
                }

            } else {                            /* breakdown */

                double evrev_dT;

                evrev = exp(-(here->DIOtBrkdwnV+vd)/vtebrk);
                evrev_dT = (here->DIOtBrkdwnV+vd)*evrev/(vtebrk*Temp);
                cdb = -csat*evrev;
                gdb = csat*evrev/vtebrk;
                cdb_dT = -(csat_dT*evrev + csat*evrev_dT);
                if ((model->DIOsatSWCurGiven)
                    &&(!model->DIOresistSWGiven) /* no breakdown for separate sidewall diode */
                    &&(!model->DIOswEmissionCoeffGiven)) {
                    evrev = exp(-(here->DIOtBrkdwnV+vdsw)/vtebrk);
                    evrev_dT = (here->DIOtBrkdwnV+vdsw)*evrev/(vtebrk*Temp);
                    cdsw = -csatsw*evrev;
                    gdsw = csatsw*evrev/vtebrk;
                    cdsw_dT = -(csatsw_dT*evrev + csatsw*evrev_dT);
                }

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

            if (vd >= -3*vte) { /* limit forward */

                if ( (model->DIOforwardKneeCurrentGiven) && (cdb > 1.0e-18) ) {
                    double ikf_area_m = here->DIOforwardKneeCurrent;
                    sqrt_ikx = sqrt(cdb/ikf_area_m);
                    gdb = ((1+sqrt_ikx)*gdb - cdb*gdb/(2*sqrt_ikx*ikf_area_m))/(1+2*sqrt_ikx + cdb/ikf_area_m);
                    cdb = cdb/(1+sqrt_ikx);
                }

            } else {            /* limit reverse */

                if ( (model->DIOreverseKneeCurrentGiven) && (cdb < -1.0e-18) ) {
                    double ikr_area_m = here->DIOreverseKneeCurrent;
                    sqrt_ikx = sqrt(cdb/(-ikr_area_m));
                    gdb = ((1+sqrt_ikx)*gdb + cdb*gdb/(2*sqrt_ikx*ikr_area_m))/(1+2*sqrt_ikx - cdb/ikr_area_m);
                    cdb = cdb/(1+sqrt_ikx);
                }

            }

            if ( (model->DIOforwardSWKneeCurrentGiven) && (cdsw > 1.0e-18) ) {
                double ikp_peri_m = here->DIOforwardSWKneeCurrent;
                sqrt_ikx = sqrt(cdsw/ikp_peri_m);
                gdsw = ((1+sqrt_ikx)*gdsw - cdsw*gdsw/(2*sqrt_ikx*ikp_peri_m))/(1+2*sqrt_ikx + cdsw/ikp_peri_m);
                cdsw = cdsw/(1+sqrt_ikx);
            }

            if (!model->DIOresistSWGiven) {
                cd = cdb + cdsw + ckt->CKTgmin*vd;
                gd = gdb + gdsw + ckt->CKTgmin;
                dIdio_dT = cdb_dT + cdsw_dT;
            } else {
                cd = cdb + ckt->CKTgmin*vd;
                gd = gdb + ckt->CKTgmin;
                cdsw = cdsw + ckt->CKTgmin*vdsw;
                gdsw = gdsw + ckt->CKTgmin;
                dIdio_dT = cdb_dT;
                dIdioSw_dT = cdsw_dT;
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
                double vdx;
                if (model->DIOresistSWGiven)
                    vdx = vdsw;
                else
                    vdx = vd;
                if (vdx < here->DIOtDepSWCap){
                    argSW=1-vdx/here->DIOtJctSWPot;
                    sargSW=exp(-model->DIOgradingSWCoeff*log(argSW));
                    deplchargeSW = here->DIOtJctSWPot*czeroSW*(1-argSW*sargSW)/(1-model->DIOgradingSWCoeff);
                    deplcapSW = czeroSW*sargSW;
                } else {
                    czof2SW=czeroSW/here->DIOtF2SW;
                    deplchargeSW = czeroSW*here->DIOtF1+czof2SW*(here->DIOtF3SW*(vdx-here->DIOtDepSWCap)+
                                   (model->DIOgradingSWCoeff/(here->DIOtJctSWPot+here->DIOtJctSWPot))*(vdx*vdx-here->DIOtDepSWCap*here->DIOtDepSWCap));
                    deplcapSW = czof2SW*(here->DIOtF3SW+model->DIOgradingSWCoeff*vdx/here->DIOtJctSWPot);
                }

                diffcharge = here->DIOtTransitTime*cd;
                diffcap = here->DIOtTransitTime*gd;
                if (!model->DIOresistSWGiven) {
                    *(ckt->CKTstate0 + here->DIOcapCharge) =
                            diffcharge + deplcharge + deplchargeSW + (here->DIOcmetal + here->DIOcpoly)*vd;
                    capd = diffcap + deplcap + deplcapSW + here->DIOcmetal + here->DIOcpoly;
                    here->DIOcap = capd;
                } else {
                    *(ckt->CKTstate0 + here->DIOcapCharge) =
                            diffcharge + deplcharge + (here->DIOcmetal + here->DIOcpoly)*vd;
                    capd = diffcap + deplcap + here->DIOcmetal + here->DIOcpoly;
                    here->DIOcap = capd;
                    *(ckt->CKTstate0 + here->DIOcapChargeSW) =
                            deplcapSW;
                    capdsw = deplcapSW;
                    here->DIOcapSW = capdsw;
                }
                /*
                 *   store small-signal parameters
                 */
                if( (!(ckt->CKTmode & MODETRANOP)) ||
                        (!(ckt->CKTmode & MODEUIC)) ) {
                    if (ckt->CKTmode & MODEINITSMSIG){
                        *(ckt->CKTstate0 + here->DIOcapCurrent) = capd;
                        if (model->DIOresistSWGiven) {
                            *(ckt->CKTstate0 + here->DIOcapCurrentSW) = capdsw;
                        }
                        if(SenCond){
                            *(ckt->CKTstate0 + here->DIOcurrent) = cd;
                            *(ckt->CKTstate0 + here->DIOconduct) = gd;
                            *(ckt->CKTstate0 + here->DIOdIdio_dT) = dIdio_dT;
                            if (model->DIOresistSWGiven) {
                                *(ckt->CKTstate0 + here->DIOcurrentSW) = cdsw;
                                *(ckt->CKTstate0 + here->DIOconductSW) = gdsw;
                                *(ckt->CKTstate0 + here->DIOdIdioSW_dT) = dIdioSw_dT;
                            }
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
                        if (model->DIOresistSWGiven)
                            *(ckt->CKTstate0 + here->DIOcurrentSW) = cdsw;
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
                        if (model->DIOresistSWGiven)
                            *(ckt->CKTstate1 + here->DIOcapChargeSW) =
                                    *(ckt->CKTstate0 + here->DIOcapChargeSW);
                    }
                    error = NIintegrate(ckt,&geq,&ceq,capd,here->DIOcapCharge);
                    if(error) return(error);
                    gd=gd+geq;
                    cd=cd+*(ckt->CKTstate0 + here->DIOcapCurrent);
                    if (model->DIOresistSWGiven) {
                        error = NIintegrate(ckt,&geq,&ceq,capdsw,here->DIOcapChargeSW);
                        if(error) return(error);
                        gdsw=gdsw+geq;
                        cdsw=cdsw+*(ckt->CKTstate0 + here->DIOcapCurrentSW);
                    }
                    if (ckt->CKTmode & MODEINITTRAN) {
                        *(ckt->CKTstate1 + here->DIOcapCurrent) =
                                *(ckt->CKTstate0 + here->DIOcapCurrent);
                        if (model->DIOresistSWGiven)
                            *(ckt->CKTstate1 + here->DIOcapCurrentSW) =
                                    *(ckt->CKTstate0 + here->DIOcapCurrentSW);
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
                if (!model->DIOresistSWGiven) {
                    if ((Check_th == 1) || (Check_dio == 1)) {
                        ckt->CKTnoncon++;
                        ckt->CKTtroubleElt = (GENinstance *) here;
                    }
                } else {
                    if ((Check_th == 1) || (Check_dio == 1) || (Check_dio_sw == 1)) {
                        ckt->CKTnoncon++;
                        ckt->CKTtroubleElt = (GENinstance *) here;
                    }
                }
            }
next2:      *(ckt->CKTstate0 + here->DIOvoltage) = vd;
            *(ckt->CKTstate0 + here->DIOcurrent) = cd;
            *(ckt->CKTstate0 + here->DIOconduct) = gd;
            *(ckt->CKTstate0 + here->DIOdeltemp) = delTemp;
            *(ckt->CKTstate0 + here->DIOdIdio_dT) = dIdio_dT;
            if (model->DIOresistSWGiven) {
                *(ckt->CKTstate0 + here->DIOvoltageSW) = vdsw;
                *(ckt->CKTstate0 + here->DIOcurrentSW) = cdsw;
                *(ckt->CKTstate0 + here->DIOconductSW) = gdsw;
                *(ckt->CKTstate0 + here->DIOdIdioSW_dT) = dIdioSw_dT;
            }
            if(SenCond)  continue;

#ifndef NOBYPASS
            load:
#endif
            if (selfheat) {
                double dIrs_dVrs, dIrs_dgspr, dIth_dIrs;
                vrs = *(ckt->CKTrhsOld + here->DIOposNode) - *(ckt->CKTrhsOld + here->DIOposPrimeNode);
                dIrs_dVrs = gspr;
                dIrs_dgspr = vrs;
                dIrs_dT = dIrs_dgspr * here->DIOtConductance_dT;
                Ith = vd*cd + vrs*vrs*gspr; /* Diode dissipated power */
                dIth_dVrs = vrs*gspr;
                dIth_dIrs = vrs;
                dIth_dVrs = dIth_dVrs + dIth_dIrs*dIrs_dVrs;
                dIth_dT = dIth_dIrs*dIrs_dT + dIdio_dT*vd;
                dIth_dVdio = cd + vd*gd;
                here->DIOdIth_dVrs = dIth_dVrs;
                here->DIOgcTt = gcTt;
                here->DIOdIrs_dT = dIrs_dT;
                here->DIOdIth_dVdio = dIth_dVdio;
                here->DIOdIth_dT = dIth_dT;
                if (model->DIOresistSWGiven) {
                    double dIrssw_dVrssw, dIrssw_dgsprsw, dIth_dIrssw;
                    vrssw = *(ckt->CKTrhsOld + here->DIOposNode) - *(ckt->CKTrhsOld + here->DIOposSwPrimeNode);
                    dIrssw_dVrssw = gsprsw;
                    dIrssw_dgsprsw = vrssw;
                    dIrssw_dT = dIrssw_dgsprsw * here->DIOtConductanceSW_dT;
                    Ith = Ith + vdsw*cdsw + vrssw*vrssw*gsprsw; /* Diode dissipated power */
                    dIth_dVrssw = vrssw*gsprsw;
                    dIth_dIrssw = vrssw;
                    dIth_dVrssw = dIth_dVrssw + dIth_dIrssw*dIrssw_dVrssw;
                    dIth_dT = dIth_dT + dIth_dIrssw*dIrssw_dT + dIdioSw_dT*vdsw;
                    dIth_dVdioSw = cdsw + vdsw*gdsw;
                    here->DIOdIth_dVrssw = dIth_dVrssw;
                    here->DIOdIth_dVdio = dIth_dVdioSw;
                    here->DIOdIth_dT = dIth_dT;
                    here->DIOdIrssw_dT = dIrssw_dT;
                }
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
            if (model->DIOresistSWGiven) {
                cdeq=cdsw-gdsw*vdsw;
                *(ckt->CKTrhs + here->DIOnegNode) += cdeq;
                *(ckt->CKTrhs + here->DIOposSwPrimeNode) -= cdeq;
                if (selfheat) {
                    *(ckt->CKTrhs + here->DIOposNode)        +=  dIrssw_dT*delTemp;
                    *(ckt->CKTrhs + here->DIOposSwPrimeNode) +=  dIdioSw_dT*delTemp - dIrssw_dT*delTemp;
                    *(ckt->CKTrhs + here->DIOnegNode)        += -dIdioSw_dT*delTemp;
                    *(ckt->CKTrhs + here->DIOtempNode)       += -dIth_dVdioSw*vdsw - dIth_dVrssw*vrssw;
                }
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
            if (model->DIOresistSWGiven) {
                *(here->DIOposPosPtr) += gsprsw;
                *(here->DIOnegNegPtr) += gdsw;
                *(here->DIOposSwPrimePosSwPrimePtr) += (gdsw + gsprsw);
                *(here->DIOposPosSwPrimePtr) -= gsprsw;
                *(here->DIOnegPosSwPrimePtr) -= gdsw;
                *(here->DIOposSwPrimePosPtr) -= gsprsw;
                *(here->DIOposSwPrimeNegPtr) -= gdsw;

                if (selfheat) {
                    (*(here->DIOtempPosPtr)        += -dIth_dVrssw);
                    (*(here->DIOtempPosSwPrimePtr) += -dIth_dVdioSw + dIth_dVrssw);
                    (*(here->DIOtempNegPtr)        +=  dIth_dVdioSw);
                    (*(here->DIOposTempPtr)        +=  dIrssw_dT);
                    (*(here->DIOposSwPrimeTempPtr) +=  dIdioSw_dT - dIrssw_dT);
                    (*(here->DIOnegTempPtr)        += -dIdioSw_dT);
                }
            }
        }
    }
    return(OK);
}
