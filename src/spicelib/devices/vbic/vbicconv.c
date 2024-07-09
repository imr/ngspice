/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * This routine performs the device convergence test for
 * VBICs in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vbicdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
VBICconvTest(GENmodel *inModel, CKTcircuit *ckt)
{
    VBICinstance *here;
    VBICmodel *model = (VBICmodel *) inModel;
    double tol;
    double delvbei;
    double delvbex;
    double delvbci;
    double delvbcx;
    double delvbep;
    double delvrci;
    double delvrbi;
    double delvrbp;
    double delvbcp;
    double ibehat;
    double ibexhat;
    double itzfhat;
    double itzrhat;
    double ibchat;
    double ibephat;
    double ircihat;
    double irbihat;
    double irbphat;
    double ibcphat;
    double iccphat;
    double Vbei, Vbex, Vbci, Vbcx, Vbep, Vrci, Vrbi, Vrbp, Vbcp;
    double Ibe, Ibex, Itzf, Itzr, Ibc, Ibep, Irci, Irbi, Irbp, Ibcp, Iccp;

    for( ; model != NULL; model = VBICnextModel(model)) {
        for(here=VBICinstances(model);here!=NULL;here = VBICnextInstance(here)) {

            Vbei=model->VBICtype*(
                *(ckt->CKTrhsOld+here->VBICbaseBINode)-
                *(ckt->CKTrhsOld+here->VBICemitEINode));
            Vbex=model->VBICtype*(
                *(ckt->CKTrhsOld+here->VBICbaseBXNode)-
                *(ckt->CKTrhsOld+here->VBICemitEINode));
            Vbci =model->VBICtype*(
                *(ckt->CKTrhsOld+here->VBICbaseBINode)-
                *(ckt->CKTrhsOld+here->VBICcollCINode));
            Vbcx =model->VBICtype*(
                *(ckt->CKTrhsOld+here->VBICbaseBINode)-
                *(ckt->CKTrhsOld+here->VBICcollCXNode));
            Vbep =model->VBICtype*(
                *(ckt->CKTrhsOld+here->VBICbaseBXNode)-
                *(ckt->CKTrhsOld+here->VBICbaseBPNode));
            Vrci =model->VBICtype*(
                *(ckt->CKTrhsOld+here->VBICcollCXNode)-
                *(ckt->CKTrhsOld+here->VBICcollCINode));
            Vrbi =model->VBICtype*(
                *(ckt->CKTrhsOld+here->VBICbaseBXNode)-
                *(ckt->CKTrhsOld+here->VBICbaseBINode));
            Vrbp =model->VBICtype*(
                *(ckt->CKTrhsOld+here->VBICbaseBPNode)-
                *(ckt->CKTrhsOld+here->VBICcollCXNode));
            Vbcp =model->VBICtype*(
                *(ckt->CKTrhsOld+here->VBICsubsSINode)-
                *(ckt->CKTrhsOld+here->VBICbaseBPNode));
            delvbei = Vbei - *(ckt->CKTstate0 + here->VBICvbei);
            delvbex = Vbex - *(ckt->CKTstate0 + here->VBICvbex);
            delvbci = Vbci - *(ckt->CKTstate0 + here->VBICvbci);
            delvbcx = Vbcx - *(ckt->CKTstate0 + here->VBICvbcx);
            delvbep = Vbep - *(ckt->CKTstate0 + here->VBICvbep);
            delvrci = Vrci - *(ckt->CKTstate0 + here->VBICvrci);
            delvrbi = Vrbi - *(ckt->CKTstate0 + here->VBICvrbi);
            delvrbp = Vrbp - *(ckt->CKTstate0 + here->VBICvrbp);
            delvbcp = Vbcp - *(ckt->CKTstate0 + here->VBICvbcp);
            ibehat = *(ckt->CKTstate0 + here->VBICibe) + 
                     *(ckt->CKTstate0 + here->VBICibe_Vbei)*delvbei;
            ibexhat = *(ckt->CKTstate0 + here->VBICibex) + 
                     *(ckt->CKTstate0 + here->VBICibex_Vbex)*delvbex;
            itzfhat = *(ckt->CKTstate0 + here->VBICitzf) + 
                     *(ckt->CKTstate0 + here->VBICitzf_Vbei)*delvbei + *(ckt->CKTstate0 + here->VBICitzf_Vbci)*delvbci;
            itzrhat = *(ckt->CKTstate0 + here->VBICitzr) + 
                     *(ckt->CKTstate0 + here->VBICitzr_Vbei)*delvbei + *(ckt->CKTstate0 + here->VBICitzr_Vbci)*delvbci;
            ibchat = *(ckt->CKTstate0 + here->VBICibc) + 
                     *(ckt->CKTstate0 + here->VBICibc_Vbei)*delvbei + *(ckt->CKTstate0 + here->VBICibc_Vbci)*delvbci;
            ibephat = *(ckt->CKTstate0 + here->VBICibep) + 
                     *(ckt->CKTstate0 + here->VBICibep_Vbep)*delvbep;
            ircihat = *(ckt->CKTstate0 + here->VBICirci) + *(ckt->CKTstate0 + here->VBICirci_Vrci)*delvrci +
                     *(ckt->CKTstate0 + here->VBICirci_Vbcx)*delvbcx + *(ckt->CKTstate0 + here->VBICirci_Vbci)*delvbci;
            irbihat = *(ckt->CKTstate0 + here->VBICirbi) + *(ckt->CKTstate0 + here->VBICirbi_Vrbi)*delvrbi +
                     *(ckt->CKTstate0 + here->VBICirbi_Vbei)*delvbei + *(ckt->CKTstate0 + here->VBICirbi_Vbci)*delvbci;
            irbphat = *(ckt->CKTstate0 + here->VBICirbp) + *(ckt->CKTstate0 + here->VBICirbp_Vrbp)*delvrbp +
                     *(ckt->CKTstate0 + here->VBICirbp_Vbep)*delvbep + *(ckt->CKTstate0 + here->VBICirbp_Vbci)*delvbci;
            ibcphat = *(ckt->CKTstate0 + here->VBICibcp) + 
                     *(ckt->CKTstate0 + here->VBICibcp_Vbcp)*delvbcp;
            iccphat = *(ckt->CKTstate0 + here->VBICiccp) + *(ckt->CKTstate0 + here->VBICiccp_Vbep)*delvbep + 
                     *(ckt->CKTstate0 + here->VBICiccp_Vbci)*delvbci + *(ckt->CKTstate0 + here->VBICiccp_Vbcp)*delvbcp;
            Ibe  = *(ckt->CKTstate0 + here->VBICibe);
            Ibex = *(ckt->CKTstate0 + here->VBICibex);
            Itzf = *(ckt->CKTstate0 + here->VBICitzf);
            Itzr = *(ckt->CKTstate0 + here->VBICitzr);
            Ibc  = *(ckt->CKTstate0 + here->VBICibc);
            Ibep = *(ckt->CKTstate0 + here->VBICibep);
            Irci = *(ckt->CKTstate0 + here->VBICirci);
            Irbi = *(ckt->CKTstate0 + here->VBICirbi);
            Irbp = *(ckt->CKTstate0 + here->VBICirbp);
            Ibcp = *(ckt->CKTstate0 + here->VBICibcp);
            Iccp = *(ckt->CKTstate0 + here->VBICiccp);
            /*
             *   check convergence
             */
            tol=ckt->CKTreltol*MAX(fabs(ibehat),fabs(Ibe))+ckt->CKTabstol;
            if (fabs(ibehat-Ibe) > tol) {
                ckt->CKTnoncon++;
                ckt->CKTtroubleElt = (GENinstance *) here;
                return(OK); /* no reason to continue - we've failed... */
            } else {
                tol=ckt->CKTreltol*MAX(fabs(ibexhat),fabs(Ibex))+ckt->CKTabstol;
                if (fabs(ibexhat-Ibex) > tol) {
                    ckt->CKTnoncon++;
                    ckt->CKTtroubleElt = (GENinstance *) here;
                    return(OK); /* no reason to continue - we've failed... */
                } else {
                    tol=ckt->CKTreltol*MAX(fabs(itzfhat),fabs(Itzf))+ckt->CKTabstol;
                    if (fabs(itzfhat-Itzf) > tol) {
                        ckt->CKTnoncon++;
                        ckt->CKTtroubleElt = (GENinstance *) here;
                        return(OK); /* no reason to continue - we've failed... */
                    } else {
                        tol=ckt->CKTreltol*MAX(fabs(itzrhat),fabs(Itzr))+ckt->CKTabstol;
                        if (fabs(itzrhat-Itzr) > tol) {
                            ckt->CKTnoncon++;
                            ckt->CKTtroubleElt = (GENinstance *) here;
                            return(OK); /* no reason to continue - we've failed... */
                        } else {
                            tol=ckt->CKTreltol*MAX(fabs(ibchat),fabs(Ibc))+ckt->CKTabstol;
                            if (fabs(ibchat-Ibc) > tol) {
                                ckt->CKTnoncon++;
                                ckt->CKTtroubleElt = (GENinstance *) here;
                                return(OK); /* no reason to continue - we've failed... */
                            } else {
                                tol=ckt->CKTreltol*MAX(fabs(ibephat),fabs(Ibep))+ckt->CKTabstol;
                                if (fabs(ibephat-Ibep) > tol) {
                                    ckt->CKTnoncon++;
                                    ckt->CKTtroubleElt = (GENinstance *) here;
                                    return(OK); /* no reason to continue - we've failed... */
                                } else {
                                    tol=ckt->CKTreltol*MAX(fabs(ircihat),fabs(Irci))+ckt->CKTabstol;
                                    if (fabs(ircihat-Irci) > tol) {
                                        ckt->CKTnoncon++;
                                        ckt->CKTtroubleElt = (GENinstance *) here;
                                        return(OK); /* no reason to continue - we've failed... */
                                    } else {
                                        tol=ckt->CKTreltol*MAX(fabs(irbihat),fabs(Irbi))+ckt->CKTabstol;
                                        if (fabs(irbihat-Irbi) > tol) {
                                            ckt->CKTnoncon++;
                                            ckt->CKTtroubleElt = (GENinstance *) here;
                                            return(OK); /* no reason to continue - we've failed... */
                                        } else {
                                            tol=ckt->CKTreltol*MAX(fabs(irbphat),fabs(Irbp))+ckt->CKTabstol;
                                            if (fabs(irbphat-Irbp) > tol) {
                                                ckt->CKTnoncon++;
                                                ckt->CKTtroubleElt = (GENinstance *) here;
                                                return(OK); /* no reason to continue - we've failed... */
                                            } else {
                                                tol=ckt->CKTreltol*MAX(fabs(ibcphat),fabs(Ibcp))+ckt->CKTabstol;
                                                if (fabs(ibcphat-Ibcp) > tol) {
                                                    ckt->CKTnoncon++;
                                                    ckt->CKTtroubleElt = (GENinstance *) here;
                                                    return(OK); /* no reason to continue - we've failed... */
                                                } else {
                                                    tol=ckt->CKTreltol*MAX(fabs(iccphat),fabs(Iccp))+ckt->CKTabstol;
                                                    if (fabs(iccphat-Iccp) > tol) {
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
        }
    }
    return(OK);
}
