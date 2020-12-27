/**********
License              : 3-clause BSD
Spice3 Implementation: 2019-2020 Dietmar Warning, Markus Müller, Mario Krattenmacher
Model Author         : 1990 Michael Schröter TU Dresden
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
    double delvciei;
    double delvcic;
    double delvbbp;
    double delveie;
    double delvxf;

    double ibieihat;
    double ibicihat;
    double icieihat;
    double ibpeihat;
    double ibpbihat;
    double ibpcihat;
    double ibpsihat;
    double isicihat;
    double volatile ithhat;


    double Vbiei, Vbici, Vbpei, Vbpbi, Vbpci, Vsici, Vrth, Vcic, Vbbp, Veie, Vxf;
    double Ibiei, Ibici, Iciei, Ibpei, Ibpbi, Ibpci, Ibpsi, Isici, Ith;

    for( ; model != NULL; model = HICUMnextModel(model)) {
        for(here=HICUMinstances(model);here!=NULL;here = HICUMnextInstance(here)) {
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
            Vcic  = model->HICUMtype*(
                   *(ckt->CKTrhsOld+here->HICUMcollCINode)-
                   *(ckt->CKTrhsOld+here->HICUMcollNode));
            Vbbp  = model->HICUMtype*(
                   *(ckt->CKTrhsOld+here->HICUMbaseNode)-
                   *(ckt->CKTrhsOld+here->HICUMbaseBPNode));
            Veie  = model->HICUMtype*(
                   *(ckt->CKTrhsOld+here->HICUMemitNode)-
                   *(ckt->CKTrhsOld+here->HICUMemitEINode));
            Vxf   = *(ckt->CKTrhsOld+here->HICUMxfNode);
 
            Vrth  = model->HICUMtype*(*(ckt->CKTrhsOld+here->HICUMtempNode));

            delvrth  = Vrth  - *(ckt->CKTstate0 + here->HICUMvrth);
            delvbiei = Vbiei - *(ckt->CKTstate0 + here->HICUMvbiei);
            delvbici = Vbici - *(ckt->CKTstate0 + here->HICUMvbici);
            delvbpei = Vbpei - *(ckt->CKTstate0 + here->HICUMvbpei);
            delvbpbi = Vbpbi - *(ckt->CKTstate0 + here->HICUMvbpbi);
            delvbpci = Vbpci - *(ckt->CKTstate0 + here->HICUMvbpci);
            delvsici = Vsici - *(ckt->CKTstate0 + here->HICUMvsici);
            delvciei = delvbiei-delvbici;
            delvcic  = Vcic  - *(ckt->CKTstate0 + here->HICUMvcic);
            delvbbp  = Vbbp  - *(ckt->CKTstate0 + here->HICUMvbbp);
            delveie  = Veie  - *(ckt->CKTstate0 + here->HICUMveie);
            delvxf   = Vxf   - *(ckt->CKTstate0 + here->HICUMvxf);

            //todo: maybe add ibiei_Vxf
            ibieihat = *(ckt->CKTstate0 + here->HICUMibiei) +
                       *(ckt->CKTstate0 + here->HICUMibiei_Vbiei)*delvbiei + 
                       *(ckt->CKTstate0 + here->HICUMibiei_Vrth)*delvrth + 
                       *(ckt->CKTstate0 + here->HICUMibiei_Vbici)*delvbici +
                       *(ckt->CKTstate0 + here->HICUMibiei_Vxf)*delvxf;
            ibicihat = *(ckt->CKTstate0 + here->HICUMibici) +
                       *(ckt->CKTstate0 + here->HICUMibici_Vbici)*delvbici+
                       *(ckt->CKTstate0 + here->HICUMibici_Vrth)*delvrth+
                       *(ckt->CKTstate0 + here->HICUMibici_Vbiei)*delvbiei;
            icieihat = *(ckt->CKTstate0 + here->HICUMiciei) +
                       *(ckt->CKTstate0 + here->HICUMiciei_Vbiei)*delvbiei +
                       *(ckt->CKTstate0 + here->HICUMiciei_Vrth)*delvrth +
                       *(ckt->CKTstate0 + here->HICUMiciei_Vbici)*delvbici;
            ibpeihat = *(ckt->CKTstate0 + here->HICUMibpei) +
                       *(ckt->CKTstate0 + here->HICUMibpei_Vrth)*delvrth+
                       *(ckt->CKTstate0 + here->HICUMibpei_Vbpei)*delvbpei;
            ibpbihat = *(ckt->CKTstate0 + here->HICUMibpbi) +
                       *(ckt->CKTstate0 + here->HICUMibpbi_Vbiei)*delvbiei +
                       *(ckt->CKTstate0 + here->HICUMibpbi_Vrth)*delvrth +
                       *(ckt->CKTstate0 + here->HICUMibpbi_Vbici)*delvbici;
            ibpcihat = *(ckt->CKTstate0 + here->HICUMibpci) +
                       *(ckt->CKTstate0 + here->HICUMibpci_Vrth)*delvrth+
                       *(ckt->CKTstate0 + here->HICUMibpci_Vbpci)*delvbici;
            ibpsihat = *(ckt->CKTstate0 + here->HICUMibpsi) +
                       *(ckt->CKTstate0 + here->HICUMibpsi_Vbpci)*delvbpci +
                       *(ckt->CKTstate0 + here->HICUMibpsi_Vrth)*delvrth +
                       *(ckt->CKTstate0 + here->HICUMibpsi_Vsici)*delvsici;
            isicihat = *(ckt->CKTstate0 + here->HICUMisici) +
                       *(ckt->CKTstate0 + here->HICUMisici_Vrth)*delvrth+
                       *(ckt->CKTstate0 + here->HICUMisici_Vsici)*delvsici;
            ithhat   = *(ckt->CKTstate0 + here->HICUMith) +
                       *(ckt->CKTstate0 + here->HICUMith_Vrth)*delvrth+
                       *(ckt->CKTstate0 + here->HICUMith_Vbiei)*delvbiei+
                       *(ckt->CKTstate0 + here->HICUMith_Vbici)*delvbici+
                       *(ckt->CKTstate0 + here->HICUMith_Vbpbi)*delvbpbi+
                       *(ckt->CKTstate0 + here->HICUMith_Vbpci)*delvbpci+
                       *(ckt->CKTstate0 + here->HICUMith_Vbpei)*delvbpei+
                       *(ckt->CKTstate0 + here->HICUMith_Vciei)*delvciei+
                       *(ckt->CKTstate0 + here->HICUMith_Vsici)*delvsici+
                       *(ckt->CKTstate0 + here->HICUMith_Vcic)*delvcic+
                       *(ckt->CKTstate0 + here->HICUMith_Vbbp)*delvbbp+
                       *(ckt->CKTstate0 + here->HICUMith_Veie)*delveie;

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
