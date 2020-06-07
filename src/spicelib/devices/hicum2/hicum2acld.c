/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1990 Michael Schröter TU Dresden
Spice3 Implementation: 2019 Dietmar Warning
**********/

/*
 * Function to load the COMPLEX circuit matrix using the
 * small signal parameters saved during a previous DC operating
 * point analysis.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hicum2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
HICUMacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    HICUMinstance *here;
    HICUMmodel *model = (HICUMmodel*)inModel;
    double Ibpei_Vbpei;
    double Ibpei_Vrth;

    double Ibiei_Vbiei;
    double Ibiei_Vbici;
    double Ibiei_Vrth;

    double Ibici_Vbici;
    double Ibici_Vbiei;
    double Ibici_Vrth;

    double Ibpci_Vbpci;
    double Ibpci_Vrth;

    double Isici_Vsici;
    double Isici_Vrth;

    double Iciei_Vbiei;
    double Iciei_Vbici;
    double Iciei_Vrth;


    double Ibpbi_Vbpbi; 
    double Ibpbi_Vbici;
    double Ibpbi_Vbiei;
    double Ibpbi_Vrth;

    double Isis_Vsis, Isis_Vrth;
    double Icic_Vcic, Icic_Vrth;
    double Ieie_Veie, Ieie_Vrth;
    double Ibbp_Vbbp, Ibbp_Vrth;

    double Ibpsi_Vbpci; 
    double Ibpsi_Vsici;
    double Ibpsi_Vrth;

    double XQrbi_Vbpbi;
//    double XQrbi_Vbiei;
//    double XQrbi_Vbici;
    double XQdeix_Vbiei;
    double XQjei_Vbiei;
    double XQdci_Vbici;
    double XQjci_Vbici;
    double XQjep_Vbpei;
    double Xqjcx0_t_i_Vbci;
    double Xqjcx0_t_ii_Vbpci;
    double XQdsu_Vbpci;
//    double XQdsu_Vsici;
    double XQjs_Vsici;
    double XQscp_Vsc;
    double XQbepar1_Vbe;
    double XQbepar2_Vbpe;
    double XQbcpar1_Vbci;
    double XQbcpar2_Vbpci;
    double XQsu_Vsis;

    /*  loop through all the models */
    for( ; model != NULL; model = HICUMnextModel(model)) {

        /* loop through all the instances of the model */
        for( here = HICUMinstances(model); here!= NULL;
                here = HICUMnextInstance(here)) {

            // get all derivatives of branch DC currents
            Ibbp_Vbbp    = 1/here->HICUMrbx_t.rpart;
            Icic_Vcic    = 1/here->HICUMrcx_t.rpart;
            Ieie_Veie    = 1/here->HICUMre_t.rpart;
            Isis_Vsis    = 1/model->HICUMrsu;

            Ibbp_Vrth    = -*(ckt->CKTstate0 + here->HICUMvbbp)/here->HICUMrbx_t.rpart/here->HICUMrbx_t.rpart*here->HICUMrbx_t.dpart;
            Icic_Vrth    = -*(ckt->CKTstate0 + here->HICUMvcic)/here->HICUMrcx_t.rpart/here->HICUMrcx_t.rpart*here->HICUMrcx_t.dpart;
            Icic_Vrth    = -*(ckt->CKTstate0 + here->HICUMveie)/here->HICUMre_t.rpart/here->HICUMre_t.rpart*here->HICUMre_t.dpart;
            Isis_Vrth    = 0.0;

            Ibiei_Vbiei = *(ckt->CKTstate0 + here->HICUMibiei_Vbiei);
            Ibiei_Vbici = *(ckt->CKTstate0 + here->HICUMibiei_Vbici);
            Ibiei_Vrth  = *(ckt->CKTstate0 + here->HICUMibiei_Vrth);

            Ibpei_Vbpei = *(ckt->CKTstate0 + here->HICUMibpei_Vbpei);
            Ibpei_Vrth  = *(ckt->CKTstate0 + here->HICUMibpei_Vrth);

            Iciei_Vbiei = *(ckt->CKTstate0 + here->HICUMiciei_Vbiei);
            Iciei_Vbici = *(ckt->CKTstate0 + here->HICUMiciei_Vbici);
            Iciei_Vrth  = *(ckt->CKTstate0 + here->HICUMiciei_Vrth);

            Ibici_Vbici = *(ckt->CKTstate0 + here->HICUMibici_Vbici);
            Ibici_Vbiei = *(ckt->CKTstate0 + here->HICUMibici_Vbiei);
            Ibici_Vrth  = *(ckt->CKTstate0 + here->HICUMibici_Vrth);

            Ibpbi_Vbpbi = *(ckt->CKTstate0 + here->HICUMibpbi_Vbpbi);
            Ibpbi_Vbiei = *(ckt->CKTstate0 + here->HICUMibpbi_Vbiei);
            Ibpbi_Vbici = *(ckt->CKTstate0 + here->HICUMibpbi_Vbici);
            Ibpbi_Vrth  = *(ckt->CKTstate0 + here->HICUMibpbi_Vrth);

            Ibpci_Vbpci = *(ckt->CKTstate0 + here->HICUMibpci_Vbpci);
            Ibpci_Vrth  = *(ckt->CKTstate0 + here->HICUMibpci_Vrth);

            Isici_Vsici = *(ckt->CKTstate0 + here->HICUMisici_Vsici);
            Isici_Vrth  = *(ckt->CKTstate0 + here->HICUMisici_Vrth);

            Ibpsi_Vbpci = *(ckt->CKTstate0 + here->HICUMibpsi_Vbpci);
            Ibpsi_Vsici = *(ckt->CKTstate0 + here->HICUMibpsi_Vsici);
            Ibpsi_Vrth  = *(ckt->CKTstate0 + here->HICUMibpsi_Vrth);

////////////////////////////////////
//////////  The real part  /////////
////////////////////////////////////

//          Stamp element: Ibiei
            *(here->HICUMbaseBIBaseBIPtr)          +=  Ibiei_Vbiei;
            *(here->HICUMemitEIEmitEIPtr)          +=  Ibiei_Vbiei;
            *(here->HICUMbaseBIEmitEIPtr)          += -Ibiei_Vbiei;
            *(here->HICUMemitEIBaseBIPtr)          += -Ibiei_Vbiei;
            *(here->HICUMbaseBIBaseBIPtr)          +=  Ibiei_Vbici;
            *(here->HICUMemitEICollCIPtr)          +=  Ibiei_Vbici;
            *(here->HICUMbaseBICollCIPtr)          += -Ibiei_Vbici;
            *(here->HICUMemitEIBaseBIPtr)          += -Ibiei_Vbici;

//          Stamp element: Ibpei
            *(here->HICUMbaseBPBaseBPPtr)          +=  Ibpei_Vbpei;
            *(here->HICUMemitEIEmitEIPtr)          +=  Ibpei_Vbpei;
            *(here->HICUMbaseBPEmitEIPtr)          += -Ibpei_Vbpei;
            *(here->HICUMemitEIBaseBPPtr)          += -Ibpei_Vbpei;;

//          Stamp element: Iciei
            *(here->HICUMcollCIBaseBIPtr)          +=  Iciei_Vbiei;
            *(here->HICUMemitEIEmitEIPtr)          +=  Iciei_Vbiei;
            *(here->HICUMcollCIEmitEIPtr)          += -Iciei_Vbiei;
            *(here->HICUMemitEIBaseBIPtr)          += -Iciei_Vbiei;
            *(here->HICUMcollCIBaseBIPtr)          +=  Iciei_Vbici; 
            *(here->HICUMemitEICollCIPtr)          +=  Iciei_Vbici;
            *(here->HICUMcollCICollCIPtr)          += -Iciei_Vbici;
            *(here->HICUMemitEIBaseBIPtr)          += -Iciei_Vbici;

//          Stamp element: Ibici (missing Ibici_Vbiei)
            *(here->HICUMbaseBIBaseBIPtr)          +=  Ibici_Vbici;
            *(here->HICUMcollCICollCIPtr)          +=  Ibici_Vbici;
            *(here->HICUMcollCIBaseBIPtr)          += -Ibici_Vbici;
            *(here->HICUMbaseBICollCIPtr)          += -Ibici_Vbici;
            *(here->HICUMbaseBIBaseBIPtr)          +=  Ibici_Vbiei;
            *(here->HICUMcollCIEmitEIPtr)          +=  Ibici_Vbiei;
            *(here->HICUMcollCIBaseBIPtr)          += -Ibici_Vbiei;
            *(here->HICUMbaseBIEmitEIPtr)          += -Ibici_Vbiei;

//          Stamp element: Ibpci
            *(here->HICUMbaseBPBaseBPPtr)          +=  Ibpci_Vbpci;
            *(here->HICUMcollCICollCIPtr)          +=  Ibpci_Vbpci;
            *(here->HICUMbaseBPCollCIPtr)          += -Ibpci_Vbpci;
            *(here->HICUMcollCIBaseBPPtr)          += -Ibpci_Vbpci;

//          Stamp element: Rcx
            *(here->HICUMcollCollPtr)              +=  Icic_Vcic;
            *(here->HICUMcollCICollCIPtr)          +=  Icic_Vcic;
            *(here->HICUMcollCICollPtr)            += -Icic_Vcic;
            *(here->HICUMcollCollCIPtr)            += -Icic_Vcic;

//          Stamp element: Rbx
            *(here->HICUMbaseBasePtr)              +=  Ibbp_Vbbp;
            *(here->HICUMbaseBPBaseBPPtr)          +=  Ibbp_Vbbp;
            *(here->HICUMbaseBPBasePtr)            += -Ibbp_Vbbp;
            *(here->HICUMbaseBaseBPPtr)            += -Ibbp_Vbbp;

//          Stamp element: Ibpbi
            *(here->HICUMbaseBPBaseBPPtr)          +=  Ibpbi_Vbpbi; 
            *(here->HICUMbaseBIBaseBIPtr)          +=  Ibpbi_Vbpbi;
            *(here->HICUMbaseBPBaseBIPtr)          += -Ibpbi_Vbpbi;
            *(here->HICUMbaseBIBaseBPPtr)          += -Ibpbi_Vbpbi;
            *(here->HICUMbaseBPBaseBIPtr)          +=  Ibpbi_Vbiei; 
            *(here->HICUMbaseBIEmitEIPtr)          +=  Ibpbi_Vbiei;
            *(here->HICUMbaseBPEmitEIPtr)          += -Ibpbi_Vbiei;
            *(here->HICUMbaseBIBaseBIPtr)          += -Ibpbi_Vbiei;
            *(here->HICUMbaseBPBaseBIPtr)          +=  Ibpbi_Vbici; 
            *(here->HICUMbaseBICollCIPtr)          +=  Ibpbi_Vbici;
            *(here->HICUMbaseBPCollCIPtr)          += -Ibpbi_Vbici;
            *(here->HICUMbaseBIBaseBIPtr)          += -Ibpbi_Vbici

//          Stamp element: Re
            *(here->HICUMemitEmitPtr)              +=  Ieie_Veie;
            *(here->HICUMemitEIEmitEIPtr)          +=  Ieie_Veie;
            *(here->HICUMemitEIEmitPtr)            += -Ieie_Veie;
            *(here->HICUMemitEmitEIPtr)            += -Ieie_Veie;

//          Stamp element: Isici
            *(here->HICUMsubsSISubsSIPtr)          +=  Isici_Vsici;
            *(here->HICUMcollCICollCIPtr)          +=  Isici_Vsici;
            *(here->HICUMsubsSICollCIPtr)          += -Isici_Vsici;
            *(here->HICUMcollCISubsSIPtr)          += -Isici_Vsici;;

//          Stamp element: Ibpsi
            *(here->HICUMbaseBPSubsSIPtr)          +=  Ibpsi_Vsici;
            *(here->HICUMsubsSICollCIPtr)          +=  Ibpsi_Vsici;
            *(here->HICUMbaseBPCollCIPtr)          += -Ibpsi_Vsici;
            *(here->HICUMsubsSISubsSIPtr)          += -Ibpsi_Vsici;
            *(here->HICUMbaseBPBaseBPPtr)          +=  Ibpsi_Vbpci;
            *(here->HICUMsubsSICollCIPtr)          +=  Ibpsi_Vbpci;
            *(here->HICUMbaseBPCollCIPtr)          += -Ibpsi_Vbpci;
            *(here->HICUMsubsSIBaseBPPtr)          += -Ibpsi_Vbpci;;

//          Stamp element: Rs
            *(here->HICUMsubsSubsPtr)              +=  Isis_Vsis;
            *(here->HICUMsubsSISubsSIPtr)          +=  Isis_Vsis;
            *(here->HICUMsubsSISubsPtr)            += -Isis_Vsis;
            *(here->HICUMsubsSubsSIPtr)            += -Isis_Vsis;;

////////////////////////////////////
//////////  The complex part  //////
////////////////////////////////////
//todo: Complete with partial dervatives e.g. Qjs_Vsici, Qrbi_Vbici

            XQrbi_Vbpbi       = *(ckt->CKTstate0 + here->HICUMcqrbi)      * ckt->CKTomega;
            XQdeix_Vbiei      = *(ckt->CKTstate0 + here->HICUMcqdeix)     * ckt->CKTomega;
            XQjei_Vbiei       = *(ckt->CKTstate0 + here->HICUMcqjei)      * ckt->CKTomega;
            XQdci_Vbici       = *(ckt->CKTstate0 + here->HICUMcqdci)      * ckt->CKTomega;
            XQjci_Vbici       = *(ckt->CKTstate0 + here->HICUMcqjci)      * ckt->CKTomega;
            XQjep_Vbpei       = *(ckt->CKTstate0 + here->HICUMcqjep)      * ckt->CKTomega;
            Xqjcx0_t_i_Vbci   = *(ckt->CKTstate0 + here->HICUMcqcx0_t_i)  * ckt->CKTomega;
            Xqjcx0_t_ii_Vbpci = *(ckt->CKTstate0 + here->HICUMcqcx0_t_ii) * ckt->CKTomega;
            XQdsu_Vbpci       = *(ckt->CKTstate0 + here->HICUMcqdsu)      * ckt->CKTomega;
            XQjs_Vsici        = *(ckt->CKTstate0 + here->HICUMcqjs)       * ckt->CKTomega;
            XQscp_Vsc         = *(ckt->CKTstate0 + here->HICUMcqscp)      * ckt->CKTomega;
            XQbepar1_Vbe      = *(ckt->CKTstate0 + here->HICUMcqbepar1)   * ckt->CKTomega;
            XQbepar2_Vbpe     = *(ckt->CKTstate0 + here->HICUMcqbepar2)   * ckt->CKTomega;
            XQbcpar1_Vbci     = *(ckt->CKTstate0 + here->HICUMcqbcpar1)   * ckt->CKTomega;
            XQbcpar2_Vbpci    = *(ckt->CKTstate0 + here->HICUMcqbcpar2)   * ckt->CKTomega;
            XQsu_Vsis         = *(ckt->CKTstate0 + here->HICUMcqsu)       * ckt->CKTomega;
/*
c           Stamp element: Qbepar1
*/
            *(here->HICUMbaseBasePtr + 1)     +=  XQbepar1_Vbe;
            *(here->HICUMbaseEmitPtr + 1)     += -XQbepar1_Vbe;
            *(here->HICUMemitBasePtr + 1)     += -XQbepar1_Vbe;
            *(here->HICUMemitEmitPtr + 1)     +=  XQbepar1_Vbe;
/*
c           Stamp element: Qbepar2
*/
            *(here->HICUMbaseBPBaseBPPtr + 1) +=  XQbepar2_Vbpe;
            *(here->HICUMemitBaseBPPtr + 1)   += -XQbepar2_Vbpe;
            *(here->HICUMemitEmitPtr + 1)     += -XQbepar2_Vbpe;
            *(here->HICUMbaseBPEmitPtr + 1)   +=  XQbepar2_Vbpe;
/*
c           Stamp element: Qdeix, Qjei
*/
            *(here->HICUMbaseBIBaseBIPtr + 1) +=  XQdeix_Vbiei;
            *(here->HICUMbaseBIEmitEIPtr + 1) += -XQdeix_Vbiei;
            *(here->HICUMemitEIBaseBIPtr + 1) += -XQdeix_Vbiei;
            *(here->HICUMemitEIEmitEIPtr + 1) +=  XQdeix_Vbiei;
            *(here->HICUMbaseBIBaseBIPtr + 1) +=  XQjei_Vbiei;
            *(here->HICUMbaseBIEmitEIPtr + 1) += -XQjei_Vbiei;
            *(here->HICUMemitEIBaseBIPtr + 1) += -XQjei_Vbiei;
            *(here->HICUMemitEIEmitEIPtr + 1) +=  XQjei_Vbiei;
/*
c           Stamp element: Qjep
*/
            *(here->HICUMbaseBPBaseBPPtr + 1) +=  XQjep_Vbpei;
            *(here->HICUMbaseBPEmitEIPtr + 1) += -XQjep_Vbpei;
            *(here->HICUMemitEIBaseBPPtr + 1) += -XQjep_Vbpei;
            *(here->HICUMemitEIEmitEIPtr + 1) +=  XQjep_Vbpei;

/*
c           Stamp element: Qdci, Qjci
*/
            *(here->HICUMbaseBIBaseBIPtr + 1) +=  XQdci_Vbici;
            *(here->HICUMbaseBICollCIPtr + 1) += -XQdci_Vbici;
            *(here->HICUMcollCIBaseBIPtr + 1) += -XQdci_Vbici;
            *(here->HICUMcollCICollCIPtr + 1) +=  XQdci_Vbici;
            *(here->HICUMbaseBIBaseBIPtr + 1) +=  XQjci_Vbici;
            *(here->HICUMbaseBICollCIPtr + 1) += -XQjci_Vbici;
            *(here->HICUMcollCIBaseBIPtr + 1) += -XQjci_Vbici;
            *(here->HICUMcollCICollCIPtr + 1) +=  XQjci_Vbici;
/*
c           Stamp element: Qbcpar1, qjcx0_i
*/
            *(here->HICUMbaseBasePtr + 1)     +=  XQbcpar1_Vbci;
            *(here->HICUMbaseCollCIPtr + 1)   += -XQbcpar1_Vbci;
            *(here->HICUMcollCIBasePtr + 1)   += -XQbcpar1_Vbci;
            *(here->HICUMcollCICollCIPtr + 1) +=  XQbcpar1_Vbci;
            *(here->HICUMbaseBasePtr + 1)     +=  Xqjcx0_t_i_Vbci;
            *(here->HICUMbaseCollCIPtr + 1)   += -Xqjcx0_t_i_Vbci;
            *(here->HICUMcollCIBasePtr + 1)   += -Xqjcx0_t_i_Vbci;
            *(here->HICUMcollCICollCIPtr + 1) +=  Xqjcx0_t_i_Vbci;
/*
c           Stamp element: Qbcpar2, qjcx0_ii, Qdsu
*/
            *(here->HICUMbaseBPBaseBPPtr + 1) +=  XQbcpar2_Vbpci;
            *(here->HICUMcollCICollCIPtr + 1) +=  XQbcpar2_Vbpci;
            *(here->HICUMbaseBPCollCIPtr + 1) += -XQbcpar2_Vbpci;
            *(here->HICUMcollCIBaseBPPtr + 1) += -XQbcpar2_Vbpci;
            *(here->HICUMbaseBPCollCIPtr + 1) +=  Xqjcx0_t_ii_Vbpci;
            *(here->HICUMbaseBPBaseBPPtr + 1) += -Xqjcx0_t_ii_Vbpci;
            *(here->HICUMcollCIBaseBPPtr + 1) += -Xqjcx0_t_ii_Vbpci;
            *(here->HICUMcollCICollCIPtr + 1) +=  Xqjcx0_t_ii_Vbpci;
            *(here->HICUMbaseBPCollCIPtr + 1) +=  XQdsu_Vbpci;
            *(here->HICUMbaseBPBaseBPPtr + 1) += -XQdsu_Vbpci;
            *(here->HICUMcollCIBaseBPPtr + 1) += -XQdsu_Vbpci;
            *(here->HICUMcollCICollCIPtr + 1) +=  XQdsu_Vbpci;
/*
c           Stamp element: Qrbi
*/
            *(here->HICUMbaseBPBaseBPPtr + 1) +=  XQrbi_Vbpbi;
            *(here->HICUMbaseBPBaseBIPtr + 1) += -XQrbi_Vbpbi;
            *(here->HICUMbaseBIBaseBPPtr + 1) += -XQrbi_Vbpbi;
            *(here->HICUMbaseBIBaseBIPtr + 1) +=  XQrbi_Vbpbi;
//todo:
//            *(here->HICUMbaseBPBaseBIPtr + 1) +=  XQrbi_Vbiei;
//            *(here->HICUMbaseBPEmitEIPtr + 1) += -XQrbi_Vbiei;
//            *(here->HICUMbaseBIBaseBIPtr + 1) += -XQrbi_Vbiei;
//            *(here->HICUMbaseBIEmitEIPtr + 1) +=  XQrbi_Vbiei;
//            *(here->HICUMbaseBPCollCIPtr + 1) +=  XQrbi_Vbici;
//            *(here->HICUMbaseBPEmitEIPtr + 1) += -XQrbi_Vbici;
//            *(here->HICUMbaseBICollCIPtr + 1) += -XQrbi_Vbici;
//            *(here->HICUMbaseBIEmitEIPtr + 1) +=  XQrbi_Vbici;
/*
c           Stamp element: Cscp
*/
            *(here->HICUMsubsSubsPtr + 1)     +=  XQscp_Vsc;
            *(here->HICUMcollSubsPtr + 1)     += -XQscp_Vsc;
            *(here->HICUMcollCollPtr + 1)     += -XQscp_Vsc;
            *(here->HICUMsubsCollPtr + 1)     +=  XQscp_Vsc;
/*
c           Stamp element: Cjs
*/
            *(here->HICUMsubsSISubsSIPtr + 1) +=  XQjs_Vsici;
            *(here->HICUMsubsSICollCIPtr + 1) += -XQjs_Vsici;
            *(here->HICUMcollCISubsSIPtr + 1) += -XQjs_Vsici;
            *(here->HICUMcollCICollCIPtr + 1) +=  XQjs_Vsici;
/*
c           Stamp element: Csu
*/
            *(here->HICUMsubsSubsPtr + 1)     +=  XQsu_Vsis;
            *(here->HICUMsubsSISubsPtr + 1)   += -XQsu_Vsis;
            *(here->HICUMsubsSubsSIPtr + 1)   += -XQsu_Vsis;
            *(here->HICUMsubsSISubsSIPtr + 1) +=  XQsu_Vsis;

        }
    }
    return(OK);
}
