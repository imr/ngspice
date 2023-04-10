/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * Function to load the COMPLEX circuit matrix using the
 * small signal parameters saved during a previous DC operating
 * point analysis.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vbicdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
VBICacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    VBICinstance *here;
    VBICmodel *model = (VBICmodel*)inModel;
    double Ibe_Vbei,Ibex_Vbex
    ,Itzf_Vbei,Itzf_Vbci,Itzr_Vbci,Itzr_Vbei,Ibc_Vbci
    ,Ibc_Vbei,Ibep_Vbep,Ircx_Vrcx,Irci_Vrci
    ,Irci_Vbci,Irci_Vbcx,Irbx_Vrbx,Irbi_Vrbi,Irbi_Vbei
    ,Irbi_Vbci,Ire_Vre,Irbp_Vrbp,Irbp_Vbep,Irbp_Vbci
    ,Ibcp_Vbcp,Iccp_Vbep,Irs_Vrs,Iccp_Vbci,Iccp_Vbcp;
    double XQbe_Vbei, XQbe_Vbci, XQbex_Vbex, XQbc_Vbci,
           XQbcx_Vbcx, XQbep_Vbep, XQbep_Vbci,
           XQbcp_Vbcp, XQbeo_Vbe, XQbco_Vbc;

    /*  loop through all the models */
    for( ; model != NULL; model = VBICnextModel(model)) {

        /* loop through all the instances of the model */
        for( here = VBICinstances(model); here!= NULL; 
                here = VBICnextInstance(here)) {

            Ibe_Vbei  = *(ckt->CKTstate0 + here->VBICibe_Vbei);
            Ibex_Vbex = *(ckt->CKTstate0 + here->VBICibex_Vbex);
            Itzf_Vbei = *(ckt->CKTstate0 + here->VBICitzf_Vbei);
            Itzf_Vbci = *(ckt->CKTstate0 + here->VBICitzf_Vbci);
            Itzr_Vbci = *(ckt->CKTstate0 + here->VBICitzr_Vbci);
            Itzr_Vbei = *(ckt->CKTstate0 + here->VBICitzr_Vbei);
            Ibc_Vbci  = *(ckt->CKTstate0 + here->VBICibc_Vbci);
            Ibc_Vbei  = *(ckt->CKTstate0 + here->VBICibc_Vbei);
            Ibep_Vbep = *(ckt->CKTstate0 + here->VBICibep_Vbep);
            Irci_Vrci = *(ckt->CKTstate0 + here->VBICirci_Vrci);
            Irci_Vbci = *(ckt->CKTstate0 + here->VBICirci_Vbci);
            Irci_Vbcx = *(ckt->CKTstate0 + here->VBICirci_Vbcx);
            Irbi_Vrbi = *(ckt->CKTstate0 + here->VBICirbi_Vrbi);
            Irbi_Vbei = *(ckt->CKTstate0 + here->VBICirbi_Vbei);
            Irbi_Vbci = *(ckt->CKTstate0 + here->VBICirbi_Vbci);
            Irbp_Vrbp = *(ckt->CKTstate0 + here->VBICirbp_Vrbp);
            Irbp_Vbep = *(ckt->CKTstate0 + here->VBICirbp_Vbep);
            Irbp_Vbci = *(ckt->CKTstate0 + here->VBICirbp_Vbci);
            Ibcp_Vbcp = *(ckt->CKTstate0 + here->VBICibcp_Vbcp);
            Iccp_Vbep = *(ckt->CKTstate0 + here->VBICiccp_Vbep);
            Iccp_Vbci = *(ckt->CKTstate0 + here->VBICiccp_Vbci);
            Iccp_Vbcp = *(ckt->CKTstate0 + here->VBICiccp_Vbcp);
            Ircx_Vrcx = *(ckt->CKTstate0 + here->VBICircx_Vrcx);
            Irbx_Vrbx = *(ckt->CKTstate0 + here->VBICirbx_Vrbx);
            Irs_Vrs   = *(ckt->CKTstate0 + here->VBICirs_Vrs);
            Ire_Vre   = *(ckt->CKTstate0 + here->VBICire_Vre);

/*
c           The real part
*/
/*
c           Stamp element: Ibe
*/
            *(here->VBICbaseBIBaseBIPtr) +=  Ibe_Vbei;
            *(here->VBICbaseBIEmitEIPtr) += -Ibe_Vbei;
            *(here->VBICemitEIBaseBIPtr) += -Ibe_Vbei;
            *(here->VBICemitEIEmitEIPtr) +=  Ibe_Vbei;
/*
c           Stamp element: Ibex
*/
            *(here->VBICbaseBXBaseBXPtr) +=  Ibex_Vbex;
            *(here->VBICbaseBXEmitEIPtr) += -Ibex_Vbex;
            *(here->VBICemitEIBaseBXPtr) += -Ibex_Vbex;
            *(here->VBICemitEIEmitEIPtr) +=  Ibex_Vbex;
/*
c           Stamp element: Itzf
*/
            *(here->VBICcollCIBaseBIPtr) +=  Itzf_Vbei;
            *(here->VBICcollCIEmitEIPtr) += -Itzf_Vbei;
            *(here->VBICcollCIBaseBIPtr) +=  Itzf_Vbci;
            *(here->VBICcollCICollCIPtr) += -Itzf_Vbci;
            *(here->VBICemitEIBaseBIPtr) += -Itzf_Vbei;
            *(here->VBICemitEIEmitEIPtr) +=  Itzf_Vbei;
            *(here->VBICemitEIBaseBIPtr) += -Itzf_Vbci;
            *(here->VBICemitEICollCIPtr) +=  Itzf_Vbci;
/*
c           Stamp element: Itzr
*/
            *(here->VBICemitEIBaseBIPtr) +=  Itzr_Vbei;
            *(here->VBICemitEIEmitEIPtr) += -Itzr_Vbei;
            *(here->VBICemitEIBaseBIPtr) +=  Itzr_Vbci;
            *(here->VBICemitEICollCIPtr) += -Itzr_Vbci;
            *(here->VBICcollCIBaseBIPtr) += -Itzr_Vbei;
            *(here->VBICcollCIEmitEIPtr) +=  Itzr_Vbei;
            *(here->VBICcollCIBaseBIPtr) += -Itzr_Vbci;
            *(here->VBICcollCICollCIPtr) +=  Itzr_Vbci;
/*
c           Stamp element: Ibc
*/
            *(here->VBICbaseBIBaseBIPtr) +=  Ibc_Vbci;
            *(here->VBICbaseBICollCIPtr) += -Ibc_Vbci;
            *(here->VBICbaseBIBaseBIPtr) +=  Ibc_Vbei;
            *(here->VBICbaseBIEmitEIPtr) += -Ibc_Vbei;
            *(here->VBICcollCIBaseBIPtr) += -Ibc_Vbci;
            *(here->VBICcollCICollCIPtr) +=  Ibc_Vbci;
            *(here->VBICcollCIBaseBIPtr) += -Ibc_Vbei;
            *(here->VBICcollCIEmitEIPtr) +=  Ibc_Vbei;
/*
c           Stamp element: Ibep
*/
            *(here->VBICbaseBXBaseBXPtr) +=  Ibep_Vbep;
            *(here->VBICbaseBXBaseBPPtr) += -Ibep_Vbep;
            *(here->VBICbaseBPBaseBXPtr) += -Ibep_Vbep;
            *(here->VBICbaseBPBaseBPPtr) +=  Ibep_Vbep;
/*
c           Stamp element: Rcx
*/
            *(here->VBICcollCollPtr)     +=  Ircx_Vrcx;
            *(here->VBICcollCXCollCXPtr) +=  Ircx_Vrcx;
            *(here->VBICcollCXCollPtr)   += -Ircx_Vrcx;
            *(here->VBICcollCollCXPtr)   += -Ircx_Vrcx;
/*
c           Stamp element: Irci
*/
            *(here->VBICcollCXCollCXPtr) +=  Irci_Vrci;
            *(here->VBICcollCXCollCIPtr) += -Irci_Vrci;
            *(here->VBICcollCXBaseBIPtr) +=  Irci_Vbci;
            *(here->VBICcollCXCollCIPtr) += -Irci_Vbci;
            *(here->VBICcollCXBaseBIPtr) +=  Irci_Vbcx;
            *(here->VBICcollCXCollCXPtr) += -Irci_Vbcx;
            *(here->VBICcollCICollCXPtr) += -Irci_Vrci;
            *(here->VBICcollCICollCIPtr) +=  Irci_Vrci;
            *(here->VBICcollCIBaseBIPtr) += -Irci_Vbci;
            *(here->VBICcollCICollCIPtr) +=  Irci_Vbci;
            *(here->VBICcollCIBaseBIPtr) += -Irci_Vbcx;
            *(here->VBICcollCICollCXPtr) +=  Irci_Vbcx;
/*
c           Stamp element: Rbx
*/
            *(here->VBICbaseBasePtr)     +=  Irbx_Vrbx;
            *(here->VBICbaseBXBaseBXPtr) +=  Irbx_Vrbx;
            *(here->VBICbaseBXBasePtr)   += -Irbx_Vrbx;
            *(here->VBICbaseBaseBXPtr)   += -Irbx_Vrbx;
/*
c           Stamp element: Irbi
*/
            *(here->VBICbaseBXBaseBXPtr) +=  Irbi_Vrbi;
            *(here->VBICbaseBXBaseBIPtr) += -Irbi_Vrbi;
            *(here->VBICbaseBXBaseBIPtr) +=  Irbi_Vbei;
            *(here->VBICbaseBXEmitEIPtr) += -Irbi_Vbei;
            *(here->VBICbaseBXBaseBIPtr) +=  Irbi_Vbci;
            *(here->VBICbaseBXCollCIPtr) += -Irbi_Vbci;
            *(here->VBICbaseBIBaseBXPtr) += -Irbi_Vrbi;
            *(here->VBICbaseBIBaseBIPtr) +=  Irbi_Vrbi;
            *(here->VBICbaseBIBaseBIPtr) += -Irbi_Vbei;
            *(here->VBICbaseBIEmitEIPtr) +=  Irbi_Vbei;
            *(here->VBICbaseBIBaseBIPtr) += -Irbi_Vbci;
            *(here->VBICbaseBICollCIPtr) +=  Irbi_Vbci;
/*
c           Stamp element: Re
*/
            *(here->VBICemitEmitPtr)     +=  Ire_Vre;
            *(here->VBICemitEIEmitEIPtr) +=  Ire_Vre;
            *(here->VBICemitEIEmitPtr)   += -Ire_Vre;
            *(here->VBICemitEmitEIPtr)   += -Ire_Vre;
/*
c           Stamp element: Irbp
*/
            *(here->VBICbaseBPBaseBPPtr) +=  Irbp_Vrbp;
            *(here->VBICbaseBPCollCXPtr) += -Irbp_Vrbp;
            *(here->VBICbaseBPBaseBXPtr) +=  Irbp_Vbep;
            *(here->VBICbaseBPBaseBPPtr) += -Irbp_Vbep;
            *(here->VBICbaseBPBaseBIPtr) +=  Irbp_Vbci;
            *(here->VBICbaseBPCollCIPtr) += -Irbp_Vbci;
            *(here->VBICcollCXBaseBPPtr) += -Irbp_Vrbp;
            *(here->VBICcollCXCollCXPtr) +=  Irbp_Vrbp;
            *(here->VBICcollCXBaseBXPtr) += -Irbp_Vbep;
            *(here->VBICcollCXBaseBPPtr) +=  Irbp_Vbep;
            *(here->VBICcollCXBaseBIPtr) += -Irbp_Vbci;
            *(here->VBICcollCXCollCIPtr) +=  Irbp_Vbci;
/*
c           Stamp element: Ibcp
*/
            *(here->VBICsubsSISubsSIPtr) +=  Ibcp_Vbcp;
            *(here->VBICsubsSIBaseBPPtr) += -Ibcp_Vbcp;
            *(here->VBICbaseBPSubsSIPtr) += -Ibcp_Vbcp;
            *(here->VBICbaseBPBaseBPPtr) +=  Ibcp_Vbcp;
/*
c           Stamp element: Iccp
*/
            *(here->VBICbaseBXBaseBXPtr) +=  Iccp_Vbep;
            *(here->VBICbaseBXBaseBPPtr) += -Iccp_Vbep;
            *(here->VBICbaseBXBaseBIPtr) +=  Iccp_Vbci;
            *(here->VBICbaseBXCollCIPtr) += -Iccp_Vbci;
            *(here->VBICbaseBXSubsSIPtr) +=  Iccp_Vbcp;
            *(here->VBICbaseBXBaseBPPtr) += -Iccp_Vbcp;
            *(here->VBICsubsSIBaseBXPtr) += -Iccp_Vbep;
            *(here->VBICsubsSIBaseBPPtr) +=  Iccp_Vbep;
            *(here->VBICsubsSIBaseBIPtr) += -Iccp_Vbci;
            *(here->VBICsubsSICollCIPtr) +=  Iccp_Vbci;
            *(here->VBICsubsSISubsSIPtr) += -Iccp_Vbcp;
            *(here->VBICsubsSIBaseBPPtr) +=  Iccp_Vbcp;
/*
c           Stamp element: Rs
*/
            *(here->VBICsubsSubsPtr)     +=  Irs_Vrs;
            *(here->VBICsubsSISubsSIPtr) +=  Irs_Vrs;
            *(here->VBICsubsSISubsPtr)   += -Irs_Vrs;
            *(here->VBICsubsSubsSIPtr)   += -Irs_Vrs;
/*
c           The complex part
*/
            XQbe_Vbei  = *(ckt->CKTstate0 + here->VBICcqbe) * ckt->CKTomega;
            XQbe_Vbci  = *(ckt->CKTstate0 + here->VBICcqbeci) * ckt->CKTomega;
            XQbex_Vbex = *(ckt->CKTstate0 + here->VBICcqbex) * ckt->CKTomega;
            XQbc_Vbci  = *(ckt->CKTstate0 + here->VBICcqbc) * ckt->CKTomega;
            XQbcx_Vbcx = *(ckt->CKTstate0 + here->VBICcqbcx) * ckt->CKTomega;
            XQbep_Vbep = *(ckt->CKTstate0 + here->VBICcqbep) * ckt->CKTomega;
            XQbep_Vbci = *(ckt->CKTstate0 + here->VBICcqbepci) * ckt->CKTomega;
            XQbcp_Vbcp = *(ckt->CKTstate0 + here->VBICcqbcp) * ckt->CKTomega;
            XQbeo_Vbe  = *(ckt->CKTstate0 + here->VBICcqbeo) * ckt->CKTomega;
            XQbco_Vbc  = *(ckt->CKTstate0 + here->VBICcqbco) * ckt->CKTomega;
/*
c   Stamp element: Qbe
*/
            *(here->VBICbaseBIBaseBIPtr + 1) +=  XQbe_Vbei;
            *(here->VBICbaseBIEmitEIPtr + 1) += -XQbe_Vbei;
            *(here->VBICbaseBIBaseBIPtr + 1) +=  XQbe_Vbci;
            *(here->VBICbaseBICollCIPtr + 1) += -XQbe_Vbci;
            *(here->VBICemitEIBaseBIPtr + 1) += -XQbe_Vbei;
            *(here->VBICemitEIEmitEIPtr + 1) +=  XQbe_Vbei;
            *(here->VBICemitEIBaseBIPtr + 1) += -XQbe_Vbci;
            *(here->VBICemitEICollCIPtr + 1) +=  XQbe_Vbci;
/*
c   Stamp element: Qbex
*/
            *(here->VBICbaseBXBaseBXPtr + 1) +=  XQbex_Vbex;
            *(here->VBICbaseBXEmitEIPtr + 1) += -XQbex_Vbex;
            *(here->VBICemitEIBaseBXPtr + 1) += -XQbex_Vbex;
            *(here->VBICemitEIEmitEIPtr + 1) +=  XQbex_Vbex;
/*
c   Stamp element: Qbc
*/
            *(here->VBICbaseBIBaseBIPtr + 1) +=  XQbc_Vbci;
            *(here->VBICbaseBICollCIPtr + 1) += -XQbc_Vbci;
            *(here->VBICcollCIBaseBIPtr + 1) += -XQbc_Vbci;
            *(here->VBICcollCICollCIPtr + 1) +=  XQbc_Vbci;
/*
c   Stamp element: Qbcx
*/
            *(here->VBICbaseBIBaseBIPtr + 1) +=  XQbcx_Vbcx;
            *(here->VBICbaseBICollCXPtr + 1) += -XQbcx_Vbcx;
            *(here->VBICcollCXBaseBIPtr + 1) += -XQbcx_Vbcx;
            *(here->VBICcollCXCollCXPtr + 1) +=  XQbcx_Vbcx;
/*
c   Stamp element: Qbep
*/
            *(here->VBICbaseBXBaseBXPtr + 1) +=  XQbep_Vbep;
            *(here->VBICbaseBXBaseBPPtr + 1) += -XQbep_Vbep;
            *(here->VBICbaseBXBaseBIPtr + 1) +=  XQbep_Vbci;
            *(here->VBICbaseBXCollCIPtr + 1) += -XQbep_Vbci;
            *(here->VBICbaseBPBaseBXPtr + 1) += -XQbep_Vbep;
            *(here->VBICbaseBPBaseBPPtr + 1) +=  XQbep_Vbep;
            *(here->VBICbaseBPBaseBIPtr + 1) += -XQbep_Vbci;
            *(here->VBICbaseBPCollCIPtr + 1) +=  XQbep_Vbci;
/*
c   Stamp element: Qbcp
*/
            *(here->VBICsubsSISubsSIPtr + 1) +=  XQbcp_Vbcp;
            *(here->VBICsubsSIBaseBPPtr + 1) += -XQbcp_Vbcp;
            *(here->VBICbaseBPSubsSIPtr + 1) += -XQbcp_Vbcp;
            *(here->VBICbaseBPBaseBPPtr + 1) +=  XQbcp_Vbcp;
/*
c   Stamp element: Qbeo
*/
            *(here->VBICbaseBasePtr + 1) +=  XQbeo_Vbe;
            *(here->VBICemitEmitPtr + 1) +=  XQbeo_Vbe;
            *(here->VBICbaseEmitPtr + 1) += -XQbeo_Vbe;
            *(here->VBICemitBasePtr + 1) += -XQbeo_Vbe;
/*
c   Stamp element: Qbco
*/
            *(here->VBICbaseBasePtr + 1) +=  XQbco_Vbc;
            *(here->VBICcollCollPtr + 1) +=  XQbco_Vbc;
            *(here->VBICbaseCollPtr + 1) += -XQbco_Vbc;
            *(here->VBICcollBasePtr + 1) += -XQbco_Vbc;


        }
    }
    return(OK);
}
