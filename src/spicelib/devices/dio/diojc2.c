/**********
 * Diode level 6: JUNCAP2 200.6 junction model
 *
 * The equations are in devices/juncap2.c (shared with BSIM4, juncap=2).
 * Geometry: AB = area [m^2], LS = pj [m], LG = pgate [m] from the instance,
 * else from the model card, else the Verilog-A defaults; given values are
 * scaled with the option scale.  M multiplies current, charge and noise.  As in other simulators: the reference
 * temperature is TRJ, default 21 C (TNOM and TREF have no effect), and DTA
 * is the default of the instance DTEMP (not added to it).  TYPE = 1: anode p, cathode n; TYPE = -1: the
 * reverse.  The series resistance RS (ohm, divided by M) is kept from the
 * diode model; self-heating, sidewall resistance and reverse recovery of
 * the other levels are not available.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/const.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* temperature and geometry dependent values of one instance */
void
DIOjc2temp(DIOmodel *model, DIOinstance *here)
{
    JC2tdata *td = &here->DIOjc2td;
    double vt, isat, temp;

    /* DTA only if the instance has neither dtemp nor temp */
    temp = here->DIOtemp;
    if (!here->DIOdtempGiven && !here->DIOtempGiven)
        temp += model->DIOjc2dta;
    JC2temp(&model->DIOjc2, td, temp, DIO_JC2_TRDEF);
    JC2initInst(td, 0, &here->DIOjc2i, here->DIOjc2ab, here->DIOjc2ls, here->DIOjc2lg,
                here->DIOjc2ifactor, here->DIOjc2cfactor);

    here->DIOtConductance = model->DIOconductance * here->DIOm;
    here->DIOtConductance_dT = 0.0;
    here->DIOtConductanceSW = 0.0;
    here->DIOtConductanceSW_dT = 0.0;

    /* critical voltage for the junction voltage limiting, from the ideal
       saturation current */
    vt = td->g.phitd;
    isat = here->DIOm * here->DIOjc2ifactor *
           (td->t[0].idsat[JC2_BOT] * here->DIOjc2ab +
            td->t[0].idsat[JC2_STI] * here->DIOjc2ls +
            td->t[0].idsat[JC2_GAT] * here->DIOjc2lg);
    if (isat > 0.0)
        here->DIOtVcrit = vt * log(vt / (CONSTroot2 * isat));
    else
        here->DIOtVcrit = 1.0;
    here->DIOtVcritSW = here->DIOtVcrit;
    here->DIOtTransitTime = 0.0;
}

/* DC and transient load of one instance */
int
DIOjc2load(DIOmodel *model, DIOinstance *here, CKTcircuit *ckt)
{
    double type = model->DIOjc2type;
    double m = here->DIOm;
    double vte = here->DIOjc2td.g.phitd;
    double gspr = here->DIOtConductance;
    double vd, vak, cd, gd, ij, gj, qj, cj, capd, geq, ceq, cdeq;
    int Check = 0, error;

    if (ckt->CKTmode & MODEINITSMSIG) {
        vd = *(ckt->CKTstate0 + here->DIOvoltage);
    } else if (ckt->CKTmode & MODEINITTRAN) {
        vd = *(ckt->CKTstate1 + here->DIOvoltage);
    } else if ((ckt->CKTmode & MODEINITJCT) &&
               (ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC)) {
        vd = here->DIOinitCond;
    } else if ((ckt->CKTmode & MODEINITJCT) && here->DIOoff) {
        vd = 0.0;
    } else if (ckt->CKTmode & MODEINITJCT) {
        vd = type * here->DIOtVcrit;
    } else if ((ckt->CKTmode & MODEINITFIX) && here->DIOoff) {
        vd = 0.0;
    } else {
        double vakold;
#ifndef PREDICTOR
        if (ckt->CKTmode & MODEINITPRED) {
            *(ckt->CKTstate0 + here->DIOvoltage) = *(ckt->CKTstate1 + here->DIOvoltage);
            vd = DEVpred(ckt, here->DIOvoltage);
            *(ckt->CKTstate0 + here->DIOcurrent) = *(ckt->CKTstate1 + here->DIOcurrent);
            *(ckt->CKTstate0 + here->DIOconduct) = *(ckt->CKTstate1 + here->DIOconduct);
        } else {
#endif /* PREDICTOR */
            vd = *(ckt->CKTrhsOld + here->DIOposPrimeNode) - *(ckt->CKTrhsOld + here->DIOnegNode);
#ifndef PREDICTOR
        }
#endif /* PREDICTOR */

#ifndef NOBYPASS
        /* bypass if the solution has not changed */
        if ((!(ckt->CKTmode & MODEINITPRED)) && (ckt->CKTbypass)) {
            double delvd = vd - *(ckt->CKTstate0 + here->DIOvoltage);
            double cdhat = *(ckt->CKTstate0 + here->DIOcurrent) +
                           *(ckt->CKTstate0 + here->DIOconduct) * delvd;
            double tol = ckt->CKTvoltTol + ckt->CKTreltol *
                         MAX(fabs(vd), fabs(*(ckt->CKTstate0 + here->DIOvoltage)));
            if (fabs(delvd) < tol) {
                tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(*(ckt->CKTstate0 + here->DIOcurrent)))
                    + ckt->CKTabstol;
                if (fabs(cdhat - *(ckt->CKTstate0 + here->DIOcurrent)) < tol) {
                    vd = *(ckt->CKTstate0 + here->DIOvoltage);
                    cd = *(ckt->CKTstate0 + here->DIOcurrent);
                    gd = *(ckt->CKTstate0 + here->DIOconduct);
                    goto load;
                }
            }
        }
#endif /* NOBYPASS */

        /* limit the new junction voltage (forward direction of the junction) */
        vakold = type * *(ckt->CKTstate0 + here->DIOvoltage);
        vak = DEVpnjlim(type * vd, vakold, vte, here->DIOtVcrit, &Check);
        vd = type * vak;
    }

    vak = type * vd;
    JC2eval(&here->DIOjc2td, 0, &here->DIOjc2i, vak, &ij, &gj, &qj, &cj);
    cd = type * m * ij + ckt->CKTgmin * vd;
    gd = m * gj + ckt->CKTgmin;

    if ((ckt->CKTmode & (MODEDCTRANCURVE | MODETRAN | MODEAC | MODEINITSMSIG)) ||
        ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC))) {

        capd = m * cj;
        *(ckt->CKTstate0 + here->DIOcapCharge) = type * m * qj;
        here->DIOcap = capd;
        here->DIOcapSW = 0.0;

        if ((!(ckt->CKTmode & MODETRANOP)) || (!(ckt->CKTmode & MODEUIC))) {
            if (ckt->CKTmode & MODEINITSMSIG) {
                *(ckt->CKTstate0 + here->DIOcapCurrent) = capd;
                return OK;
            }
            if (ckt->CKTmode & MODEINITTRAN)
                *(ckt->CKTstate1 + here->DIOcapCharge) = *(ckt->CKTstate0 + here->DIOcapCharge);
            /* no integration, if dc sweep, but keep evaluating capacitances */
            if (!(ckt->CKTmode & MODEDCTRANCURVE)) {
                error = NIintegrate(ckt, &geq, &ceq, capd, here->DIOcapCharge);
                if (error)
                    return error;
                gd += geq;
                cd += *(ckt->CKTstate0 + here->DIOcapCurrent);
                if (ckt->CKTmode & MODEINITTRAN)
                    *(ckt->CKTstate1 + here->DIOcapCurrent) = *(ckt->CKTstate0 + here->DIOcapCurrent);
            }
        }
    }

    /* check convergence */
    if ((!(ckt->CKTmode & MODEINITFIX)) || (!(here->DIOoff))) {
        if (Check == 1) {
            ckt->CKTnoncon++;
            ckt->CKTtroubleElt = (GENinstance *) here;
        }
    }

    *(ckt->CKTstate0 + here->DIOvoltage) = vd;
    *(ckt->CKTstate0 + here->DIOcurrent) = cd;
    *(ckt->CKTstate0 + here->DIOconduct) = gd;
    *(ckt->CKTstate0 + here->DIOdeltemp) = 0.0;
    *(ckt->CKTstate0 + here->DIOdIdio_dT) = 0.0;
    *(ckt->CKTstate0 + here->DIOqp) = 0.0;
    *(ckt->CKTstate0 + here->DIOresCurrent) = 0.0;
    *(ckt->CKTstate0 + here->DIOresConduct) = 0.0;
    *(ckt->CKTstate0 + here->DIOsrcapCurrent) = 0.0;
    *(ckt->CKTstate0 + here->DIOgqcsr) = 0.0;

#ifndef NOBYPASS
load:
#endif
    cdeq = cd - gd * vd;
    *(ckt->CKTrhs + here->DIOnegNode) += cdeq;
    *(ckt->CKTrhs + here->DIOposPrimeNode) -= cdeq;

    *(here->DIOposPosPtr) += gspr;
    *(here->DIOnegNegPtr) += gd;
    *(here->DIOposPrimePosPrimePtr) += (gd + gspr);
    *(here->DIOposPosPrimePtr) -= gspr;
    *(here->DIOnegPosPrimePtr) -= gd;
    *(here->DIOposPrimePosPtr) -= gspr;
    *(here->DIOposPrimeNegPtr) -= gd;

    return OK;
}
