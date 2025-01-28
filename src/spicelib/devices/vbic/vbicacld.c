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
    ,Iciei_Vbei,Iciei_Vbci,Ibc_Vbci
    ,Ibc_Vbei,Ibep_Vbep,Ircx_Vrcx,Irci_Vrci
    ,Irci_Vbci,Irci_Vbcx,Irbx_Vrbx,Irbi_Vrbi,Irbi_Vbei
    ,Irbi_Vbci,Ire_Vre,Irbp_Vrbp,Irbp_Vbep,Irbp_Vbci
    ,Ibcp_Vbcp,Iccp_Vbep,Irs_Vrs,Iccp_Vbci,Iccp_Vbcp;
    double XQbe_Vbei, XQbe_Vbci, XQbex_Vbex, XQbc_Vbci,
           XQbcx_Vbcx, XQbep_Vbep, XQbep_Vbci,
           XQbcp_Vbcp, XQbeo_Vbe, XQbco_Vbc;

    double Ibe_Vrth, Ibex_Vrth, Iciei_Vrth, Iciei_Vxf2, Ibc_Vrth, Ibep_Vrth,
           Ircx_Vrth, Irci_Vrth, Irbx_Vrth, Irbi_Vrth, Ire_Vrth, Irbp_Vrth,
           Ibcp_Vrth, Iccp_Vrth, Irs_Vrth, Irth_Vrth, Ith_Vrth,
           Ith_Vbei, Ith_Vbci, Ith_Vcei, Ith_Vbex, Ith_Vbep, Ith_Vbcp, Ith_Vcep,
           Ith_Vrci, Ith_Vbcx, Ith_Vrbi, Ith_Vrbp, Ith_Vrcx, Ith_Vrbx, Ith_Vre, Ith_Vrs;
    double XQcth_Vrth, XQbe_Vrth, XQbex_Vrth, XQbc_Vrth, XQbcx_Vrth, XQbep_Vrth, XQbcp_Vrth;

    //NQS
    double XQxf1_Vxf1;
    double XQxf2_Vxf2;

    double Ixf1_Vbei;
    double Ixf1_Vbci;
    double Ixf1_Vxf2;
    double Ixf1_Vxf1;
    double Ixf1_Vrth;

    double Ixf2_Vbei;
    double Ixf2_Vbci;
    double Ixf2_Vxf2;
    double Ixf2_Vxf1;
    double Ixf2_Vrth;

    /*  loop through all the models */
    for( ; model != NULL; model = VBICnextModel(model)) {

        /* loop through all the instances of the model */
        for( here = VBICinstances(model); here!= NULL; 
                here = VBICnextInstance(here)) {

            Ibe_Vbei   = *(ckt->CKTstate0 + here->VBICibe_Vbei);
            Ibex_Vbex  = *(ckt->CKTstate0 + here->VBICibex_Vbex);
            Iciei_Vbei = *(ckt->CKTstate0 + here->VBICiciei_Vbei);
            Iciei_Vbci = *(ckt->CKTstate0 + here->VBICiciei_Vbci);
            Iciei_Vrth = *(ckt->CKTstate0 + here->VBICiciei_Vrth);
            Iciei_Vxf2 = *(ckt->CKTstate0 + here->VBICiciei_Vxf2);
            Ibc_Vbci   = *(ckt->CKTstate0 + here->VBICibc_Vbci);
            Ibc_Vbei   = *(ckt->CKTstate0 + here->VBICibc_Vbei);
            Ibep_Vbep  = *(ckt->CKTstate0 + here->VBICibep_Vbep);
            Irci_Vrci  = *(ckt->CKTstate0 + here->VBICirci_Vrci);
            Irci_Vbci  = *(ckt->CKTstate0 + here->VBICirci_Vbci);
            Irci_Vbcx  = *(ckt->CKTstate0 + here->VBICirci_Vbcx);
            Irbi_Vrbi  = *(ckt->CKTstate0 + here->VBICirbi_Vrbi);
            Irbi_Vbei  = *(ckt->CKTstate0 + here->VBICirbi_Vbei);
            Irbi_Vbci  = *(ckt->CKTstate0 + here->VBICirbi_Vbci);
            Irbp_Vrbp  = *(ckt->CKTstate0 + here->VBICirbp_Vrbp);
            Irbp_Vbep  = *(ckt->CKTstate0 + here->VBICirbp_Vbep);
            Irbp_Vbci  = *(ckt->CKTstate0 + here->VBICirbp_Vbci);
            Ibcp_Vbcp  = *(ckt->CKTstate0 + here->VBICibcp_Vbcp);
            Iccp_Vbep  = *(ckt->CKTstate0 + here->VBICiccp_Vbep);
            Iccp_Vbci  = *(ckt->CKTstate0 + here->VBICiccp_Vbci);
            Iccp_Vbcp  = *(ckt->CKTstate0 + here->VBICiccp_Vbcp);
            Ircx_Vrcx  = *(ckt->CKTstate0 + here->VBICircx_Vrcx);
            Irbx_Vrbx  = *(ckt->CKTstate0 + here->VBICirbx_Vrbx);
            Irs_Vrs    = *(ckt->CKTstate0 + here->VBICirs_Vrs);
            Ire_Vre    = *(ckt->CKTstate0 + here->VBICire_Vre);

            Ixf1_Vbei  = *(ckt->CKTstate0 + here->VBICixf1_Vbei);
            Ixf1_Vbci  = *(ckt->CKTstate0 + here->VBICixf1_Vbci);
            Ixf1_Vxf2  = *(ckt->CKTstate0 + here->VBICixf1_Vxf2);
            Ixf1_Vxf1  = *(ckt->CKTstate0 + here->VBICixf1_Vxf1);
            Ixf1_Vrth  = *(ckt->CKTstate0 + here->VBICixf1_Vrth);

            Ixf2_Vbei  = *(ckt->CKTstate0 + here->VBICixf2_Vbei);
            Ixf2_Vbci  = *(ckt->CKTstate0 + here->VBICixf2_Vbci);
            Ixf2_Vxf2  = *(ckt->CKTstate0 + here->VBICixf2_Vxf2);
            Ixf2_Vxf1  = *(ckt->CKTstate0 + here->VBICixf2_Vxf1);
            Ixf2_Vrth  = *(ckt->CKTstate0 + here->VBICixf2_Vrth);

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
c           Stamp element: Iciei
*/
            *(here->VBICcollCIBaseBIPtr) +=  Iciei_Vbei;
            *(here->VBICcollCIEmitEIPtr) += -Iciei_Vbei;
            *(here->VBICcollCIBaseBIPtr) +=  Iciei_Vbci;
            *(here->VBICcollCICollCIPtr) += -Iciei_Vbci;
            *(here->VBICemitEIBaseBIPtr) += -Iciei_Vbei;
            *(here->VBICemitEIEmitEIPtr) +=  Iciei_Vbei;
            *(here->VBICemitEIBaseBIPtr) += -Iciei_Vbci;
            *(here->VBICemitEICollCIPtr) +=  Iciei_Vbci;
            if (here->VBIC_excessPhase) { 
                *(here->VBICcollCIXf2Ptr) +=  Iciei_Vxf2;
                *(here->VBICemitEIXf2Ptr) += -Iciei_Vxf2;
            }
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
c           Stamp element: Rci
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
c           Stamp element: Rbi
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
c           Stamp element: Rbp
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

            if (here->VBIC_selfheat) {

                Ibe_Vrth   = here->VBICibe_Vrth;
                Ibex_Vrth  = here->VBICibex_Vrth;
                Iciei_Vrth = here->VBICiciei_Vrth;
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
/*
c               Stamp element: Iciei
*/
                *(here->VBICcollCItempPtr) +=  Iciei_Vrth;
                *(here->VBICemitEItempPtr) += -Iciei_Vrth;
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
            }
            if (here->VBIC_excessPhase) { 
                //Ixf1
                *(here->VBICxf1BaseBIPtr)  += +Ixf1_Vbei;
                *(here->VBICxf1EmitEIPtr)  += -Ixf1_Vbei;
                *(here->VBICxf1BaseBIPtr)  += +Ixf1_Vbci;
                *(here->VBICxf1CollCIPtr)  += -Ixf1_Vbci;
                *(here->VBICxf1Xf2Ptr)     += +Ixf1_Vxf2;
                *(here->VBICxf1Xf1Ptr)     += +Ixf1_Vxf1;
                //Ixf2
                *(here->VBICxf2BaseBIPtr)  += +Ixf2_Vbei;
                *(here->VBICxf2EmitEIPtr)  += -Ixf2_Vbei;
                *(here->VBICxf2BaseBIPtr)  += +Ixf2_Vbci;
                *(here->VBICxf2CollCIPtr)  += -Ixf2_Vbci;
                *(here->VBICxf2Xf2Ptr)     += +Ixf2_Vxf2;
                *(here->VBICxf2Xf1Ptr)     += +Ixf2_Vxf1;
            }

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
            XQxf1_Vxf1 = *(ckt->CKTstate0 + here->VBICcqxf1) * ckt->CKTomega;
            XQxf2_Vxf2 = *(ckt->CKTstate0 + here->VBICcqxf2) * ckt->CKTomega;
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

            if (here->VBIC_excessPhase) {
                //Qxf1
                *(here->VBICxf1Xf1Ptr + 1) += +XQxf1_Vxf1;
                //Qxf2
                *(here->VBICxf2Xf2Ptr + 1) += +XQxf2_Vxf2;
            }

            if (here->VBIC_selfheat) {
                XQcth_Vrth = here->VBICcapcth * ckt->CKTomega;
                XQbe_Vrth  = here->VBICcapqbeth * ckt->CKTomega;
                XQbex_Vrth = here->VBICcapqbexth * ckt->CKTomega;
                XQbc_Vrth  = here->VBICcapqbcth * ckt->CKTomega;
                XQbcx_Vrth = here->VBICcapqbcxth * ckt->CKTomega;
                XQbep_Vrth = here->VBICcapqbepth * ckt->CKTomega;
                XQbcp_Vrth = here->VBICcapqbcpth * ckt->CKTomega;

                *(here->VBICtempTempPtr + 1)   +=  XQcth_Vrth;

                *(here->VBICbaseBItempPtr + 1) +=  XQbe_Vrth;
                *(here->VBICemitEItempPtr + 1) += -XQbe_Vrth;
                *(here->VBICbaseBXtempPtr + 1) +=  XQbex_Vrth;
                *(here->VBICemitEItempPtr + 1) += -XQbex_Vrth;
                *(here->VBICbaseBItempPtr + 1) +=  XQbc_Vrth;
                *(here->VBICcollCItempPtr + 1) += -XQbc_Vrth;
                *(here->VBICbaseBItempPtr + 1) +=  XQbcx_Vrth;
                *(here->VBICcollCXtempPtr + 1) += -XQbcx_Vrth;
                *(here->VBICbaseBXtempPtr + 1) +=  XQbep_Vrth;
                *(here->VBICbaseBPtempPtr + 1) += -XQbep_Vrth;
                *(here->VBICsubsSItempPtr + 1) +=  XQbcp_Vrth;
                *(here->VBICbaseBPtempPtr + 1) += -XQbcp_Vrth;
                if (here->VBIC_excessPhase) { 
//                  Stamp element: Ixf1    f_xf1 = +   
                    *(here->VBICxf1TempPtr)    +=  Ixf1_Vrth;
//                  Stamp element: Ixf2    f_xf2 = +   
                    *(here->VBICxf2TempPtr)    +=  Ixf2_Vrth;
                }
            }

        }
    }
    return(OK);
}
