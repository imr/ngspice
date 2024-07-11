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
VBICpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
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

    double Ibe_Vrth, Ibex_Vrth, Itzf_Vrth=0.0, Itzr_Vrth, Ibc_Vrth, Ibep_Vrth,
           Ircx_Vrth, Irci_Vrth, Irbx_Vrth, Irbi_Vrth, Ire_Vrth, Irbp_Vrth,
           Ibcp_Vrth, Iccp_Vrth, Irs_Vrth, Irth_Vrth, Ith_Vrth,
           Ith_Vbei, Ith_Vbci, Ith_Vcei, Ith_Vbex, Ith_Vbep, Ith_Vbcp, Ith_Vcep,
           Ith_Vrci, Ith_Vbcx, Ith_Vrbi, Ith_Vrbp, Ith_Vrcx, Ith_Vrbx, Ith_Vre, Ith_Vrs;
    double XQcth_Vrth, XQbe_Vrth, XQbex_Vrth, XQbc_Vrth, XQbcx_Vrth, XQbep_Vrth, XQbcp_Vrth;

    //NQS
    double Itxf_Vrxf, Ibc_Vrxf, Ith_Vrxf, Ixzf_Vrth, Ixxf_Vrxf, XQcxf_Vcxf;
    double Ixzf_Vbei, Ixzf_Vbci, Xl;

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

            if (!here->VBIC_excessPhase) {
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
            }
/*
c           Stamp element: Itzr
*/
            *(here->VBICemitEIBaseBIPtr) +=  Itzr_Vbci;
            *(here->VBICemitEICollCIPtr) += -Itzr_Vbci;
            *(here->VBICemitEIBaseBIPtr) +=  Itzr_Vbei;
            *(here->VBICemitEIEmitEIPtr) += -Itzr_Vbei;
            *(here->VBICcollCIBaseBIPtr) += -Itzr_Vbci;
            *(here->VBICcollCICollCIPtr) +=  Itzr_Vbci;
            *(here->VBICcollCIBaseBIPtr) += -Itzr_Vbei;
            *(here->VBICcollCIEmitEIPtr) +=  Itzr_Vbei;
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
c           Stamp element: Ircx
*/
            *(here->VBICcollCollPtr)     +=  Ircx_Vrcx;
            *(here->VBICcollCXCollCXPtr) +=  Ircx_Vrcx;
            *(here->VBICcollCXCollPtr)   +=  -Ircx_Vrcx;
            *(here->VBICcollCollCXPtr)   +=  -Ircx_Vrcx;
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
c           Stamp element: Irbx
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
c           Stamp element: Ire
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
c           Stamp element: Irs
*/
            *(here->VBICsubsSubsPtr)     +=  Irs_Vrs;
            *(here->VBICsubsSISubsSIPtr) +=  Irs_Vrs;
            *(here->VBICsubsSISubsPtr)   += -Irs_Vrs;
            *(here->VBICsubsSubsSIPtr)   += -Irs_Vrs;

            if (here->VBIC_selfheat) {

                Ibe_Vrth   = here->VBICibe_Vrth;
                Ibex_Vrth  = here->VBICibex_Vrth;
                if (!here->VBIC_excessPhase)
                    Itzf_Vrth  = here->VBICitzf_vrth;
                Itzr_Vrth  = here->VBICitzr_Vrth;
                Ibc_Vrth   = here->VBICibc_Vrth;
                Ibep_Vrth  = here->VBICibep_Vrth;
                Ircx_Vrth  = here->VBICircx_Vrth;
                Irci_Vrth  = here->VBICirci_Vrth;
                Irbx_Vrth  = here->VBICirbx_Vrth;
                Irbi_Vrth  = here->VBICirbi_Vrth;
                Ire_Vrth   = here->VBICire_Vrth;
                Irbp_Vrth  = here->VBICirbp_Vrth;
                Ibcp_Vrth  = here->VBICibcp_Vrth;
                Iccp_Vrth  = here->VBICiccp_Vrth;
                Irs_Vrth   = here->VBICirs_Vrth;
                Irth_Vrth  = here->VBICirth_Vrth;
                Ith_Vrth   = here->VBICith_Vrth;
                Ith_Vbei   = here->VBICith_Vbei;
                Ith_Vbci   = here->VBICith_Vbci;
                Ith_Vcei   = here->VBICith_Vcei;
                Ith_Vbex   = here->VBICith_Vbex;
                Ith_Vbep   = here->VBICith_Vbep;
                Ith_Vbcp   = here->VBICith_Vbcp;
                Ith_Vcep   = here->VBICith_Vcep;
                Ith_Vrci   = here->VBICith_Vrci;
                Ith_Vbcx   = here->VBICith_Vbcx;
                Ith_Vrbi   = here->VBICith_Vrbi;
                Ith_Vrbp   = here->VBICith_Vrbp;
                Ith_Vrcx   = here->VBICith_Vrcx;
                Ith_Vrbx   = here->VBICith_Vrbx;
                Ith_Vre    = here->VBICith_Vre;
                Ith_Vrs    = here->VBICith_Vrs;

/*
c               Stamp element: Ibe
*/
                *(here->VBICbaseBItempPtr) +=  Ibe_Vrth;
                *(here->VBICemitEItempPtr) += -Ibe_Vrth;
/*
c               Stamp element: Ibex
*/
                *(here->VBICbaseBXtempPtr) +=  Ibex_Vrth;
                *(here->VBICemitEItempPtr) += -Ibex_Vrth;

                if (!here->VBIC_excessPhase) {
/*
c               Stamp element: Itzf
*/
                    *(here->VBICcollCItempPtr) +=  Itzf_Vrth;
                    *(here->VBICemitEItempPtr) += -Itzf_Vrth;
                }
/*
c               Stamp element: Itzr
*/
                *(here->VBICemitEItempPtr) +=  Itzr_Vrth;
                *(here->VBICcollCItempPtr) += -Itzr_Vrth;
/*
c               Stamp element: Ibc
*/
                *(here->VBICbaseBItempPtr) +=  Ibc_Vrth;
                *(here->VBICcollCItempPtr) += -Ibc_Vrth;
/*
c               Stamp element: Ibep
*/
                *(here->VBICbaseBXtempPtr) +=  Ibep_Vrth;
                *(here->VBICbaseBPtempPtr) += -Ibep_Vrth;
/*
c               Stamp element: Rcx
*/
                *(here->VBICcollTempPtr)   +=  Ircx_Vrth;
                *(here->VBICcollCXtempPtr) += -Ircx_Vrth;
/*
c               Stamp element: Irci
*/
                *(here->VBICcollCXtempPtr) +=  Irci_Vrth;
                *(here->VBICcollCItempPtr) += -Irci_Vrth;
/*
c               Stamp element: Rbx
*/
                *(here->VBICbaseTempPtr)   +=  Irbx_Vrth;
                *(here->VBICbaseBXtempPtr) += -Irbx_Vrth;
/*
c               Stamp element: Irbi
*/
                *(here->VBICbaseBXtempPtr) +=  Irbi_Vrth;
                *(here->VBICbaseBItempPtr) += -Irbi_Vrth;
/*
c               Stamp element: Re
*/
                *(here->VBICemitTempPtr)   +=  Ire_Vrth;
                *(here->VBICemitEItempPtr) += -Ire_Vrth;
/*
c               Stamp element: Irbp
*/
                *(here->VBICbaseBPtempPtr) +=  Irbp_Vrth;
                *(here->VBICcollCXtempPtr) += -Irbp_Vrth;
/*
c               Stamp element: Ibcp
*/
                *(here->VBICsubsSItempPtr) +=  Ibcp_Vrth;
                *(here->VBICbaseBPtempPtr) += -Ibcp_Vrth;
/*
c               Stamp element: Iccp
*/
                *(here->VBICbaseBXtempPtr) +=  Iccp_Vrth;
                *(here->VBICsubsSItempPtr) += -Iccp_Vrth;
/*
c               Stamp element: Rs
*/
                *(here->VBICsubsTempPtr)   +=  Irs_Vrth;
                *(here->VBICsubsSItempPtr) += -Irs_Vrth;
/*
c               Stamp element: Rth
*/
                *(here->VBICtempTempPtr) +=  Irth_Vrth;
/*
c               Stamp element: Ith
*/
                *(here->VBICtempTempPtr)   += -Ith_Vrth;

                *(here->VBICtempBaseBIPtr) += -Ith_Vbei;
                *(here->VBICtempEmitEIPtr) += +Ith_Vbei;
                *(here->VBICtempBaseBIPtr) += -Ith_Vbci;
                *(here->VBICtempCollCIPtr) += +Ith_Vbci;
                *(here->VBICtempCollCIPtr) += -Ith_Vcei;
                *(here->VBICtempEmitEIPtr) += +Ith_Vcei;
                *(here->VBICtempBaseBXPtr) += -Ith_Vbex;
                *(here->VBICtempEmitEIPtr) += +Ith_Vbex;
                *(here->VBICtempBaseBXPtr) += -Ith_Vbep;
                *(here->VBICtempBaseBPPtr) += +Ith_Vbep;
                *(here->VBICtempSubsPtr)   += -Ith_Vbcp;
                *(here->VBICtempBaseBPPtr) += +Ith_Vbcp;
                *(here->VBICtempBaseBXPtr) += -Ith_Vcep;
                *(here->VBICtempSubsPtr)   += +Ith_Vcep;
                *(here->VBICtempCollCXPtr) += -Ith_Vrci;
                *(here->VBICtempCollCIPtr) += +Ith_Vrci;
                *(here->VBICtempBaseBIPtr) += -Ith_Vbcx;
                *(here->VBICtempCollCXPtr) += +Ith_Vbcx;
                *(here->VBICtempBaseBXPtr) += -Ith_Vrbi;
                *(here->VBICtempBaseBIPtr) += +Ith_Vrbi;
                *(here->VBICtempBaseBPPtr) += -Ith_Vrbp;
                *(here->VBICtempCollCXPtr) += +Ith_Vrbp;
                *(here->VBICtempCollPtr)   += -Ith_Vrcx;
                *(here->VBICtempCollCXPtr) += +Ith_Vrcx;
                *(here->VBICtempBasePtr)   += -Ith_Vrbx;
                *(here->VBICtempBaseBXPtr) += +Ith_Vrbx;
                *(here->VBICtempEmitPtr)   += -Ith_Vre;
                *(here->VBICtempEmitEIPtr) += +Ith_Vre;
                *(here->VBICtempSubsPtr)   += -Ith_Vrs;
                *(here->VBICtempSubsSIPtr) += +Ith_Vrs;
                if (here->VBIC_excessPhase) {
                    Ith_Vrxf = *(ckt->CKTstate0 + here->VBICith_Vrxf);
                    *(here->VBICtempXf2Ptr) += +Ith_Vrxf;
                }
            }

            if (here->VBIC_excessPhase) {
                Itxf_Vrxf = *(ckt->CKTstate0 + here->VBICitxf_Vrxf);
                Ibc_Vrxf  = *(ckt->CKTstate0 + here->VBICibc_Vrxf);
                Ixzf_Vbei = *(ckt->CKTstate0 + here->VBICixzf_Vbei);
                Ixzf_Vbci = *(ckt->CKTstate0 + here->VBICixzf_Vbci);
                Ixxf_Vrxf = *(ckt->CKTstate0 + here->VBICixxf_Vrxf);
/*
c           Stamp element: Itxf
*/
                *(here->VBICcollCIXf2Ptr)     +=  Itxf_Vrxf;
                *(here->VBICemitEIXf2Ptr)     += -Itxf_Vrxf;
/*
c           Stamp element: Ibc
*/
                *(here->VBICbaseBIXf2Ptr)     +=  Ibc_Vrxf;
                *(here->VBICcollCIXf2Ptr)     += -Ibc_Vrxf;
/*
c               Stamp element: Ixzf, Branch: xf1-ground
*/
                *(here->VBICxf1BaseBIPtr)     += +Ixzf_Vbei;
                *(here->VBICxf1EmitEIPtr)     += -Ixzf_Vbei;
                *(here->VBICxf1BaseBIPtr)     += +Ixzf_Vbci;
                *(here->VBICxf1CollCIPtr)     += -Ixzf_Vbci;
                if (here->VBIC_selfheat) {
                    Ixzf_Vrth = *(ckt->CKTstate0 + here->VBICixzf_Vrth);
                    *(here->VBICxf1TempPtr)   +=  Ixzf_Vrth;
                }
/*
c               Stamp element: Ixxf, Branch: xf2-ground
*/
                *(here->VBICxf2Xf2Ptr)        += +Ixxf_Vrxf;

            }

/*
c           The complex part
*/
            XQbe_Vbei  = *(ckt->CKTstate0 + here->VBICcqbe);
            XQbe_Vbci  = *(ckt->CKTstate0 + here->VBICcqbeci);
            XQbex_Vbex = *(ckt->CKTstate0 + here->VBICcqbex);
            XQbc_Vbci  = *(ckt->CKTstate0 + here->VBICcqbc);
            XQbcx_Vbcx = *(ckt->CKTstate0 + here->VBICcqbcx);
            XQbep_Vbep = *(ckt->CKTstate0 + here->VBICcqbep);
            XQbep_Vbci = *(ckt->CKTstate0 + here->VBICcqbepci);
            XQbcp_Vbcp = *(ckt->CKTstate0 + here->VBICcqbcp);
            XQbeo_Vbe  = *(ckt->CKTstate0 + here->VBICcqbeo);
            XQbco_Vbc  = *(ckt->CKTstate0 + here->VBICcqbco);
/*
c	Stamp element: Qbe
*/
            *(here->VBICbaseBIBaseBIPtr)     +=  XQbe_Vbei * (s->real);
            *(here->VBICbaseBIBaseBIPtr + 1) +=  XQbe_Vbei * (s->imag);
            *(here->VBICbaseBIEmitEIPtr)     += -XQbe_Vbei * (s->real);
            *(here->VBICbaseBIEmitEIPtr + 1) += -XQbe_Vbei * (s->imag);
            *(here->VBICbaseBIBaseBIPtr)     +=  XQbe_Vbci * (s->real);
            *(here->VBICbaseBIBaseBIPtr + 1) +=  XQbe_Vbci * (s->imag);
            *(here->VBICbaseBICollCIPtr)     += -XQbe_Vbci * (s->real);
            *(here->VBICbaseBICollCIPtr + 1) += -XQbe_Vbci * (s->imag);
            *(here->VBICemitEIBaseBIPtr)     += -XQbe_Vbei * (s->real);
            *(here->VBICemitEIBaseBIPtr + 1) += -XQbe_Vbei * (s->imag);
            *(here->VBICemitEIEmitEIPtr)     +=  XQbe_Vbei * (s->real);
            *(here->VBICemitEIEmitEIPtr + 1) +=  XQbe_Vbei * (s->imag);
            *(here->VBICemitEIBaseBIPtr)     += -XQbe_Vbci * (s->real);
            *(here->VBICemitEIBaseBIPtr + 1) += -XQbe_Vbci * (s->imag);
            *(here->VBICemitEICollCIPtr)     +=  XQbe_Vbci * (s->real);
            *(here->VBICemitEICollCIPtr + 1) +=  XQbe_Vbci * (s->imag);
/*
c	Stamp element: Qbex
*/
            *(here->VBICbaseBXBaseBXPtr)     +=  XQbex_Vbex * (s->real);
            *(here->VBICbaseBXBaseBXPtr + 1) +=  XQbex_Vbex * (s->imag);
            *(here->VBICbaseBXEmitEIPtr)     += -XQbex_Vbex * (s->real);
            *(here->VBICbaseBXEmitEIPtr + 1) += -XQbex_Vbex * (s->imag);
            *(here->VBICemitEIBaseBXPtr)     += -XQbex_Vbex * (s->real);
            *(here->VBICemitEIBaseBXPtr + 1) += -XQbex_Vbex * (s->imag);
            *(here->VBICemitEIEmitEIPtr )    +=  XQbex_Vbex * (s->real);
            *(here->VBICemitEIEmitEIPtr + 1) +=  XQbex_Vbex * (s->imag);
/*
c	Stamp element: Qbc
*/
            *(here->VBICbaseBIBaseBIPtr)     +=  XQbc_Vbci * (s->real);
            *(here->VBICbaseBIBaseBIPtr + 1) +=  XQbc_Vbci * (s->imag);
            *(here->VBICbaseBICollCIPtr)     += -XQbc_Vbci * (s->real);
            *(here->VBICbaseBICollCIPtr + 1) += -XQbc_Vbci * (s->imag);
            *(here->VBICcollCIBaseBIPtr)     += -XQbc_Vbci * (s->real);
            *(here->VBICcollCIBaseBIPtr + 1) += -XQbc_Vbci * (s->imag);
            *(here->VBICcollCICollCIPtr)     +=  XQbc_Vbci * (s->real);
            *(here->VBICcollCICollCIPtr + 1) +=  XQbc_Vbci * (s->imag);
/*
c	Stamp element: Qbcx
*/
            *(here->VBICbaseBIBaseBIPtr)     +=  XQbcx_Vbcx * (s->real);
            *(here->VBICbaseBIBaseBIPtr + 1) +=  XQbcx_Vbcx * (s->imag);
            *(here->VBICbaseBICollCXPtr)     += -XQbcx_Vbcx * (s->real);
            *(here->VBICbaseBICollCXPtr + 1) += -XQbcx_Vbcx * (s->imag);
            *(here->VBICcollCXBaseBIPtr)     += -XQbcx_Vbcx * (s->real);
            *(here->VBICcollCXBaseBIPtr + 1) += -XQbcx_Vbcx * (s->imag);
            *(here->VBICcollCXCollCXPtr)     +=  XQbcx_Vbcx * (s->real);
            *(here->VBICcollCXCollCXPtr + 1) +=  XQbcx_Vbcx * (s->imag);
/*
c	Stamp element: Qbep
*/
            *(here->VBICbaseBXBaseBXPtr)     +=  XQbep_Vbep * (s->real);
            *(here->VBICbaseBXBaseBXPtr + 1) +=  XQbep_Vbep * (s->imag);
            *(here->VBICbaseBXBaseBPPtr)     += -XQbep_Vbep * (s->real);
            *(here->VBICbaseBXBaseBPPtr + 1) += -XQbep_Vbep * (s->imag);
            *(here->VBICbaseBXBaseBIPtr)     +=  XQbep_Vbci * (s->real);
            *(here->VBICbaseBXBaseBIPtr + 1) +=  XQbep_Vbci * (s->imag);
            *(here->VBICbaseBXCollCIPtr)     += -XQbep_Vbci * (s->real);
            *(here->VBICbaseBXCollCIPtr + 1) += -XQbep_Vbci * (s->imag);
            *(here->VBICbaseBPBaseBXPtr)     += -XQbep_Vbep * (s->real);
            *(here->VBICbaseBPBaseBXPtr + 1) += -XQbep_Vbep * (s->imag);
            *(here->VBICbaseBPBaseBPPtr)     +=  XQbep_Vbep * (s->real);
            *(here->VBICbaseBPBaseBPPtr + 1) +=  XQbep_Vbep * (s->imag);
            *(here->VBICbaseBPBaseBIPtr)     += -XQbep_Vbci * (s->real);
            *(here->VBICbaseBPBaseBIPtr + 1) += -XQbep_Vbci * (s->imag);
            *(here->VBICbaseBPCollCIPtr)     +=  XQbep_Vbci * (s->real);
            *(here->VBICbaseBPCollCIPtr + 1) +=  XQbep_Vbci * (s->imag);
/*
c	Stamp element: Qbcp
*/
            *(here->VBICsubsSISubsSIPtr)     +=  XQbcp_Vbcp * (s->real);
            *(here->VBICsubsSISubsSIPtr + 1) +=  XQbcp_Vbcp * (s->imag);
            *(here->VBICsubsSIBaseBPPtr)     += -XQbcp_Vbcp * (s->real);
            *(here->VBICsubsSIBaseBPPtr + 1) += -XQbcp_Vbcp * (s->imag);
            *(here->VBICbaseBPSubsSIPtr)     += -XQbcp_Vbcp * (s->real);
            *(here->VBICbaseBPSubsSIPtr + 1) += -XQbcp_Vbcp * (s->imag);
            *(here->VBICbaseBPBaseBPPtr)     +=  XQbcp_Vbcp * (s->real);
            *(here->VBICbaseBPBaseBPPtr + 1) +=  XQbcp_Vbcp * (s->imag);

/*
c   Stamp element: Qbeo
*/
            *(here->VBICbaseBasePtr    ) +=  XQbeo_Vbe * (s->real);
            *(here->VBICbaseBasePtr + 1) +=  XQbeo_Vbe * (s->imag);
            *(here->VBICemitEmitPtr    ) +=  XQbeo_Vbe * (s->real);
            *(here->VBICemitEmitPtr + 1) +=  XQbeo_Vbe * (s->imag);
            *(here->VBICbaseEmitPtr    ) += -XQbeo_Vbe * (s->real);
            *(here->VBICbaseEmitPtr + 1) += -XQbeo_Vbe * (s->imag);
            *(here->VBICemitBasePtr    ) += -XQbeo_Vbe * (s->real);
            *(here->VBICemitBasePtr + 1) += -XQbeo_Vbe * (s->imag);
/*
c   Stamp element: Qbco
*/
            *(here->VBICbaseBasePtr    ) +=  XQbco_Vbc * (s->real);
            *(here->VBICbaseBasePtr + 1) +=  XQbco_Vbc * (s->imag);
            *(here->VBICcollCollPtr    ) +=  XQbco_Vbc * (s->real);
            *(here->VBICcollCollPtr + 1) +=  XQbco_Vbc * (s->imag);
            *(here->VBICbaseCollPtr    ) += -XQbco_Vbc * (s->real);
            *(here->VBICbaseCollPtr + 1) += -XQbco_Vbc * (s->imag);
            *(here->VBICcollBasePtr    ) += -XQbco_Vbc * (s->real);
            *(here->VBICcollBasePtr + 1) += -XQbco_Vbc * (s->imag);

            if (here->VBIC_selfheat) {
                XQcth_Vrth = here->VBICcapcth;
                XQbe_Vrth  = here->VBICcapqbeth;
                XQbex_Vrth = here->VBICcapqbexth;
                XQbc_Vrth  = here->VBICcapqbcth;
                XQbcx_Vrth = here->VBICcapqbcxth;
                XQbep_Vrth = here->VBICcapqbepth;
                XQbcp_Vrth = here->VBICcapqbcpth;

                *(here->VBICtempTempPtr    )   +=  XQcth_Vrth * (s->real);
                *(here->VBICtempTempPtr + 1)   +=  XQcth_Vrth * (s->imag);

                *(here->VBICbaseBItempPtr    ) +=  XQbe_Vrth  * (s->real);
                *(here->VBICbaseBItempPtr + 1) +=  XQbe_Vrth  * (s->imag);
                *(here->VBICemitEItempPtr    ) += -XQbe_Vrth  * (s->real);
                *(here->VBICemitEItempPtr + 1) += -XQbe_Vrth  * (s->imag);
                *(here->VBICbaseBXtempPtr    ) +=  XQbex_Vrth * (s->real);
                *(here->VBICbaseBXtempPtr + 1) +=  XQbex_Vrth * (s->imag);
                *(here->VBICemitEItempPtr    ) += -XQbex_Vrth * (s->real);
                *(here->VBICemitEItempPtr + 1) += -XQbex_Vrth * (s->imag);
                *(here->VBICbaseBItempPtr    ) +=  XQbc_Vrth  * (s->real);
                *(here->VBICbaseBItempPtr + 1) +=  XQbc_Vrth  * (s->imag);
                *(here->VBICcollCItempPtr    ) += -XQbc_Vrth  * (s->real);
                *(here->VBICcollCItempPtr + 1) += -XQbc_Vrth  * (s->imag);
                *(here->VBICbaseBItempPtr    ) +=  XQbcx_Vrth * (s->real);
                *(here->VBICbaseBItempPtr + 1) +=  XQbcx_Vrth * (s->imag);
                *(here->VBICcollCXtempPtr    ) += -XQbcx_Vrth * (s->real);
                *(here->VBICcollCXtempPtr + 1) += -XQbcx_Vrth * (s->imag);
                *(here->VBICbaseBXtempPtr    ) +=  XQbep_Vrth * (s->real);
                *(here->VBICbaseBXtempPtr + 1) +=  XQbep_Vrth * (s->imag);
                *(here->VBICbaseBPtempPtr    ) += -XQbep_Vrth * (s->real);
                *(here->VBICbaseBPtempPtr + 1) += -XQbep_Vrth * (s->imag);
                *(here->VBICsubsSItempPtr    ) +=  XQbcp_Vrth * (s->real);
                *(here->VBICsubsSItempPtr + 1) +=  XQbcp_Vrth * (s->imag);
                *(here->VBICbaseBPtempPtr    ) += -XQbcp_Vrth * (s->real);
                *(here->VBICbaseBPtempPtr + 1) += -XQbcp_Vrth * (s->imag);

            }
            if (here->VBIC_excessPhase) {
/*
c               Stamp element: Qcxf
*/
                XQcxf_Vcxf = here->VBICcapQcxf;
                *(here->VBICxf1Xf1Ptr    ) +=  XQcxf_Vcxf * s->real;
                *(here->VBICxf1Xf1Ptr + 1) +=  XQcxf_Vcxf * s->imag;
/*
c               Stamp element: L = TD/3
*/
                Xl = here->VBICindInduct;

                *(here->VBICxf1IbrPtr) +=  1;
                *(here->VBICxf2IbrPtr) -=  1;
                *(here->VBICibrXf1Ptr) +=  1;
                *(here->VBICibrXf2Ptr) -=  1;
                *(here->VBICibrIbrPtr    ) -= Xl * s->real;
                *(here->VBICibrIbrPtr + 1) -= Xl * s->imag;

            }
        }
    }
    return(OK);
}
