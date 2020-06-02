/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1990 Michael Schröter TU Dresden
Spice3 Implementation: 2019 Dietmar Warning
**********/

/*
 * This routine performs the device convergence test for
 * HICUMs in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hicum2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
HICUMconvTest(GENmodel *inModel, CKTcircuit *ckt)
{
    HICUMinstance *here;
    HICUMmodel *model = (HICUMmodel *) inModel;
    double tol;
    double delvbiei;
    double delvbici;
    double delvbpei;
    double delvbpbi;
    double delvbpci;
    double delvsici;
    double delvrth;

    double ibieihat;
    double ibicihat;
    double icieihat;
    double ibpeihat;
    double ibpbihat;
    double ibpcihat;
    double ibpsihat;
    double isicihat;
    double volatile ithhat;

    double Vbiei, Vbici, Vciei, Vbpei, Vbpbi, Vbpci, Vbci, Vsici, Vrth;
    double Ibiei, Ibici, Iciei, Ibpei, Ibpbi, Ibpci, Ibpsi, Isici, Ith;

    for( ; model != NULL; model = HICUMnextModel(model)) {
        for(here=HICUMinstances(model);here!=NULL;here = HICUMnextInstance(here)) {

            Vbci = model->HICUMtype*(
                  *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                  *(ckt->CKTrhsOld+here->HICUMcollCINode));
            Vbiei= model->HICUMtype*(
                   *(ckt->CKTrhsOld+here->HICUMbaseBINode)-
                   *(ckt->CKTrhsOld+here->HICUMemitEINode));
            Vbici = model->HICUMtype*(
                   *(ckt->CKTrhsOld+here->HICUMbaseBINode)-
                   *(ckt->CKTrhsOld+here->HICUMcollCINode));
            Vbpei = model->HICUMtype*(
                   *(ckt->CKTrhsOld+here->HICUMbaseBPNode)-
                   *(ckt->CKTrhsOld+here->HICUMemitEINode));
            Vbpbi = model->HICUMtype*(
                   *(ckt->CKTrhsOld+here->HICUMbaseBPNode)-
                   *(ckt->CKTrhsOld+here->HICUMbaseBINode));
            Vbpci = model->HICUMtype*(
                   *(ckt->CKTrhsOld+here->HICUMbaseBPNode)-
                   *(ckt->CKTrhsOld+here->HICUMcollCINode));
            Vsici = model->HICUMtype*(
                   *(ckt->CKTrhsOld+here->HICUMsubsSINode)-
                   *(ckt->CKTrhsOld+here->HICUMcollCINode));
            Vciei = Vbiei - Vbici;
            Vrth  = model->HICUMtype*(*(ckt->CKTrhsOld+here->HICUMtempNode));

            delvbiei = Vbiei - *(ckt->CKTstate0 + here->HICUMvbiei);
            delvbici = Vbici - *(ckt->CKTstate0 + here->HICUMvbici);
            delvbpei = Vbpei - *(ckt->CKTstate0 + here->HICUMvbpei);
            delvbpbi = Vbpbi - *(ckt->CKTstate0 + here->HICUMvbpbi);
            delvbpci = Vbpci - *(ckt->CKTstate0 + here->HICUMvbpci);
            delvsici = Vsici - *(ckt->CKTstate0 + here->HICUMvsici);
            delvrth  = Vrth  - *(ckt->CKTstate0 + here->HICUMvrth);

            ibieihat = *(ckt->CKTstate0 + here->HICUMibiei) +
                       *(ckt->CKTstate0 + here->HICUMibiei_Vbiei)*delvbiei + 
                       *(ckt->CKTstate0 + here->HICUMibiei_Vbici)*delvbici;
            ibicihat = *(ckt->CKTstate0 + here->HICUMibici) +
                       *(ckt->CKTstate0 + here->HICUMibici_Vbici)*delvbici+
                       *(ckt->CKTstate0 + here->HICUMibici_Vbiei)*delvbiei;
            icieihat = *(ckt->CKTstate0 + here->HICUMiciei) +
                       *(ckt->CKTstate0 + here->HICUMiciei_Vbiei)*delvbiei +
                       *(ckt->CKTstate0 + here->HICUMiciei_Vbici)*delvbici;
            ibpeihat = *(ckt->CKTstate0 + here->HICUMibpei) +
                       *(ckt->CKTstate0 + here->HICUMibpei_Vbpei)*delvbpei;
            ibpbihat = *(ckt->CKTstate0 + here->HICUMibpbi) +
                       *(ckt->CKTstate0 + here->HICUMibpbi_Vbiei)*delvbiei +
                       *(ckt->CKTstate0 + here->HICUMibpbi_Vbici)*delvbici;
            ibpcihat = *(ckt->CKTstate0 + here->HICUMibpci) +
                       *(ckt->CKTstate0 + here->HICUMibpci_Vbpci)*delvbici;
            ibpsihat = *(ckt->CKTstate0 + here->HICUMibpsi) +
                       *(ckt->CKTstate0 + here->HICUMibpsi_Vbpci)*delvbpci +
                       *(ckt->CKTstate0 + here->HICUMibpsi_Vsici)*delvsici;
            isicihat = *(ckt->CKTstate0 + here->HICUMisici) +
                       *(ckt->CKTstate0 + here->HICUMisici_Vsici)*delvsici;
            ithhat   = *(ckt->CKTstate0 + here->HICUMith) +
                       *(ckt->CKTstate0 + here->HICUMith_Vrth)*delvrth;

            Ibiei       = *(ckt->CKTstate0 + here->HICUMibiei);
            Ibici       = *(ckt->CKTstate0 + here->HICUMibici);
            Iciei       = *(ckt->CKTstate0 + here->HICUMiciei);
            Ibpei       = *(ckt->CKTstate0 + here->HICUMibpei);
            Ibpbi       = *(ckt->CKTstate0 + here->HICUMibpbi);
            Ibpci       = *(ckt->CKTstate0 + here->HICUMibpci);
            Ibpsi       = *(ckt->CKTstate0 + here->HICUMibpsi);
            Isici       = *(ckt->CKTstate0 + here->HICUMisici);
            Ith         = *(ckt->CKTstate0 + here->HICUMith);

            /*
             *   check convergence
             */
            tol=ckt->CKTreltol*MAX(fabs(ibieihat),fabs(Ibiei))+ckt->CKTabstol;
            if (fabs(ibieihat-Ibiei) > tol) {
                ckt->CKTnoncon++;
                ckt->CKTtroubleElt = (GENinstance *) here;
                return(OK); /* no reason to continue - we've failed... */
            } else {
                tol=ckt->CKTreltol*MAX(fabs(ibicihat),fabs(Ibici))+ckt->CKTabstol;
                if (fabs(ibicihat-Ibici) > tol) {
                    ckt->CKTnoncon++;
                    ckt->CKTtroubleElt = (GENinstance *) here;
                    return(OK); /* no reason to continue - we've failed... */
                } else {
                    tol=ckt->CKTreltol*MAX(fabs(icieihat),fabs(Iciei))+ckt->CKTabstol;
                    if (fabs(icieihat-Iciei) > tol) {
                        ckt->CKTnoncon++;
                        ckt->CKTtroubleElt = (GENinstance *) here;
                        return(OK); /* no reason to continue - we've failed... */
                    } else {
                        tol=ckt->CKTreltol*MAX(fabs(ibpeihat),fabs(Ibpei))+ckt->CKTabstol;
                        if (fabs(ibpeihat-Ibpei) > tol) {
                            ckt->CKTnoncon++;
                            ckt->CKTtroubleElt = (GENinstance *) here;
                            return(OK); /* no reason to continue - we've failed... */
                        } else {
                            tol=ckt->CKTreltol*MAX(fabs(ibpbihat),fabs(Ibpbi))+ckt->CKTabstol;
                            if (fabs(ibpbihat-Ibpbi) > tol) {
                                ckt->CKTnoncon++;
                                ckt->CKTtroubleElt = (GENinstance *) here;
                                return(OK); /* no reason to continue - we've failed... */
                            } else {
                                tol=ckt->CKTreltol*MAX(fabs(ibpcihat),fabs(Ibpci))+ckt->CKTabstol;
                                if (fabs(ibpcihat-Ibpci) > tol) {
                                    ckt->CKTnoncon++;
                                    ckt->CKTtroubleElt = (GENinstance *) here;
                                    return(OK); /* no reason to continue - we've failed... */
                                } else {
                                    tol=ckt->CKTreltol*MAX(fabs(ibpsihat),fabs(Ibpsi))+ckt->CKTabstol;
                                    if (fabs(ibpsihat-Ibpsi) > tol) {
                                        ckt->CKTnoncon++;
                                        ckt->CKTtroubleElt = (GENinstance *) here;
                                        return(OK); /* no reason to continue - we've failed... */
                                    } else {
                                        tol=ckt->CKTreltol*MAX(fabs(isicihat),fabs(Isici))+ckt->CKTabstol;
                                        if (fabs(isicihat-Isici) > tol) {
                                            ckt->CKTnoncon++;
                                            ckt->CKTtroubleElt = (GENinstance *) here;
                                            return(OK); /* no reason to continue - we've failed... */
                                        } else {
                                            tol=ckt->CKTreltol*MAX(fabs(ithhat),fabs(Ith))+ckt->CKTabstol;
                                            if (fabs(ithhat-Ith) > tol) {
                                                ckt->CKTnoncon++;
                                                ckt->CKTtroubleElt = (GENinstance *) here;
                                                return(OK); /* no reason to continue - we've failed... */
                                            } 
                                        }

                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return(OK);
}
