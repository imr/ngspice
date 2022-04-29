/**********
License              : 3-clause BSD
Spice3 Implementation: 2019-2020 Dietmar Warning, Markus Müller, Mario Krattenmacher
Model Author         : 1990 Michael Schröter TU Dresden
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
    double Ibiei_Vxf;
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
    double Iciei_Vxf2;


    double Ibpbi_Vbpbi; 
    double Ibpbi_Vbici;
    double Ibpbi_Vbiei;
    double Ibpbi_Vrth;

    double Isis_Vsis;
    double Irth_Vrth;
    double Icic_Vcic, Icic_Vrth;
    double Ieie_Veie, Ieie_Vrth;
    double Ibbp_Vbbp, Ibbp_Vrth;

    double Ibpsi_Vbpci; 
    double Ibpsi_Vsici;
    double Ibpsi_Vrth;

    double XQrbi_Vbpbi;
    double XQrbi_Vbiei;
    double XQrbi_Vbici;
    double XQrbi_Vrth;
    double XQjei_Vbiei;
    double XQjei_Vrth;
    double XQjci_Vbici;
    double XQjci_Vrth;
    double XQjep_Vbpei;
    double XQjep_Vrth;
    double volatile Xqjcx0_t_i_Vbci;
    double Xqjcx0_t_i_Vrth;
    double volatile Xqjcx0_t_ii_Vbpci;
    double Xqjcx0_t_ii_Vrth;
    double XQdsu_Vbpci;
    double XQdsu_Vsici;
    double XQdsu_Vrth;
    double XQjs_Vsici;
    double XQjs_Vrth;
    double XQscp_Vsc;
    double XQscp_Vrth;
    double XQbepar1_Vbe;
    double XQbepar2_Vbpe;
    double XQbcpar1_Vbci;
    double XQbcpar2_Vbpci;
    double XQsu_Vsis;
    double XQcth_Vrth;
    double XQf_Vbiei;
    double XQf_Vbici;
    double XQf_Vrth;
    double XQf_Vxf;
    double XQr_Vrth;
    double XQr_Vbiei;
    double XQr_Vbici;

    double XQxf_Vxf;
    double XQxf1_Vxf1;
    double XQxf2_Vxf2;

    double Ixf1_Vbiei;
    double Ixf1_Vbici;
    double Ixf1_Vxf2 ;
    double Ixf1_Vxf1 ;
    double Ixf1_Vrth ;

    double Ixf2_Vbiei;
    double Ixf2_Vbici;
    double Ixf2_Vxf2 ;
    double Ixf2_Vxf1 ;
    double Ixf2_Vrth ;

    double Ixf_Vbiei ;
    double Ixf_Vbici ;
    double Ixf_Vxf ;
    double Ixf_Vrth ;

    double Ith_Vrth, Ith_Vbiei, Ith_Vbici, Ith_Vbpbi, Ith_Vbpci, Ith_Vbpei, Ith_Vciei, Ith_Vsici, Ith_Vcic, Ith_Vbbp, Ith_Veie;

    /*  loop through all the models */
    for( ; model != NULL; model = HICUMnextModel(model)) {

        /* loop through all the instances of the model */
        for( here = HICUMinstances(model); here!= NULL;
                here = HICUMnextInstance(here)) {


            // get all derivatives of branch DC currents
            if(model->HICUMrcxGiven && model->HICUMrcx != 0) {
                Icic_Vcic    = 1/here->HICUMrcx_t.rpart;
                Icic_Vrth    = -*(ckt->CKTstate0 + here->HICUMvcic)/here->HICUMrcx_t.rpart/here->HICUMrcx_t.rpart*here->HICUMrcx_t.dpart;
            } else {
                Icic_Vcic    = 0.0;
                Icic_Vrth    = 0.0;
            }
            if(model->HICUMrbxGiven && model->HICUMrbx != 0) {
                Ibbp_Vbbp    = 1/here->HICUMrbx_t.rpart;
                Ibbp_Vrth    = -*(ckt->CKTstate0 + here->HICUMvbbp)/here->HICUMrbx_t.rpart/here->HICUMrbx_t.rpart*here->HICUMrbx_t.dpart;
            } else {
                Ibbp_Vbbp    = 0.0;
                Ibbp_Vrth    = 0.0;
            }
            if(model->HICUMreGiven && model->HICUMre != 0) {
                Ieie_Veie    = 1/here->HICUMre_t.rpart;
                Ieie_Vrth    = -*(ckt->CKTstate0 + here->HICUMveie)/here->HICUMre_t.rpart/here->HICUMre_t.rpart*here->HICUMre_t.dpart;
            } else {
                Ieie_Veie    = 0.0;
                Ieie_Vrth    = 0.0;
            }
            if(model->HICUMrsuGiven && model->HICUMrsu != 0) {
                Isis_Vsis    = 1/model->HICUMrsu*here->HICUMm;
            } else {
                Isis_Vsis    = 0.0;
            }
            if(model->HICUMselfheat) {
                Irth_Vrth    = (1/here->HICUMrth_t.rpart - *(ckt->CKTstate0 + here->HICUMvrth)/(here->HICUMrth_t.rpart*here->HICUMrth_t.rpart) * here->HICUMrth_t.dpart);
            } else {
                Irth_Vrth    = 0.0;
            }



            Ibiei_Vbiei = *(ckt->CKTstate0 + here->HICUMibiei_Vbiei);
            Ibiei_Vxf   = *(ckt->CKTstate0 + here->HICUMibiei_Vxf);
            Ibiei_Vbici = *(ckt->CKTstate0 + here->HICUMibiei_Vbici);
            Ibiei_Vrth  = *(ckt->CKTstate0 + here->HICUMibiei_Vrth);

            Ibpei_Vbpei = *(ckt->CKTstate0 + here->HICUMibpei_Vbpei);
            Ibpei_Vrth  = *(ckt->CKTstate0 + here->HICUMibpei_Vrth);

            Iciei_Vbiei = *(ckt->CKTstate0 + here->HICUMiciei_Vbiei);
            Iciei_Vbici = *(ckt->CKTstate0 + here->HICUMiciei_Vbici);
            Iciei_Vrth  = *(ckt->CKTstate0 + here->HICUMiciei_Vrth);
            Iciei_Vxf2  = *(ckt->CKTstate0 + here->HICUMiciei_Vxf2);

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

            Ith_Vrth    = *(ckt->CKTstate0 + here->HICUMith_Vrth);
            Ith_Vbiei   = *(ckt->CKTstate0 + here->HICUMith_Vbiei);
            Ith_Vbici   = *(ckt->CKTstate0 + here->HICUMith_Vbici);
            Ith_Vbpbi   = *(ckt->CKTstate0 + here->HICUMith_Vbpbi);
            Ith_Vbpci   = *(ckt->CKTstate0 + here->HICUMith_Vbpci);
            Ith_Vbpei   = *(ckt->CKTstate0 + here->HICUMith_Vbpei);
            Ith_Vciei   = *(ckt->CKTstate0 + here->HICUMith_Vciei);
            Ith_Vsici   = *(ckt->CKTstate0 + here->HICUMith_Vsici);
            Ith_Vcic    = *(ckt->CKTstate0 + here->HICUMith_Vcic);
            Ith_Vbbp    = *(ckt->CKTstate0 + here->HICUMith_Vbbp);
            Ith_Veie    = *(ckt->CKTstate0 + here->HICUMith_Veie);

            Ixf1_Vbiei  = *(ckt->CKTstate0 + here->HICUMixf1_Vbiei);
            Ixf1_Vbici  = *(ckt->CKTstate0 + here->HICUMixf1_Vbici);
            Ixf1_Vxf2   = *(ckt->CKTstate0 + here->HICUMixf1_Vxf2);
            Ixf1_Vxf1   = *(ckt->CKTstate0 + here->HICUMixf1_Vxf1);
            Ixf1_Vrth   = *(ckt->CKTstate0 + here->HICUMixf1_Vrth);

            Ixf2_Vbiei  = *(ckt->CKTstate0 + here->HICUMixf2_Vbiei);
            Ixf2_Vbici  = *(ckt->CKTstate0 + here->HICUMixf2_Vbici);
            Ixf2_Vxf2   = *(ckt->CKTstate0 + here->HICUMixf2_Vxf2);
            Ixf2_Vxf1   = *(ckt->CKTstate0 + here->HICUMixf2_Vxf1);
            Ixf2_Vrth   = *(ckt->CKTstate0 + here->HICUMixf2_Vrth);

            Ixf_Vbiei   = *(ckt->CKTstate0 + here->HICUMixf_Vbiei);
            Ixf_Vbici   = *(ckt->CKTstate0 + here->HICUMixf_Vbici);
            Ixf_Vxf     = *(ckt->CKTstate0 + here->HICUMixf_Vxf);
            Ixf_Vrth    = *(ckt->CKTstate0 + here->HICUMixf_Vrth);

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
            if (model->HICUMnqs) {
                *(here->HICUMbaseBIXfPtr)          +=  Ibiei_Vxf;
                *(here->HICUMemitEIXfPtr)          += -Ibiei_Vxf;
            }

//          Stamp element: Ibpei
            *(here->HICUMbaseBPBaseBPPtr)          +=  Ibpei_Vbpei;
            *(here->HICUMemitEIEmitEIPtr)          +=  Ibpei_Vbpei;
            *(here->HICUMbaseBPEmitEIPtr)          += -Ibpei_Vbpei;
            *(here->HICUMemitEIBaseBPPtr)          += -Ibpei_Vbpei;;

//          Stamp element: Ibici 
            *(here->HICUMbaseBIBaseBIPtr)          +=  Ibici_Vbici;
            *(here->HICUMcollCICollCIPtr)          +=  Ibici_Vbici;
            *(here->HICUMcollCIBaseBIPtr)          += -Ibici_Vbici;
            *(here->HICUMbaseBICollCIPtr)          += -Ibici_Vbici;
            *(here->HICUMbaseBIBaseBIPtr)          +=  Ibici_Vbiei;
            *(here->HICUMcollCIEmitEIPtr)          +=  Ibici_Vbiei;
            *(here->HICUMcollCIBaseBIPtr)          += -Ibici_Vbiei;
            *(here->HICUMbaseBIEmitEIPtr)          += -Ibici_Vbiei;

//          Stamp element: Iciei
            *(here->HICUMcollCIBaseBIPtr)          +=  Iciei_Vbiei;
            *(here->HICUMemitEIEmitEIPtr)          +=  Iciei_Vbiei;
            *(here->HICUMcollCIEmitEIPtr)          += -Iciei_Vbiei;
            *(here->HICUMemitEIBaseBIPtr)          += -Iciei_Vbiei;
            *(here->HICUMcollCIBaseBIPtr)          +=  Iciei_Vbici; 
            *(here->HICUMemitEICollCIPtr)          +=  Iciei_Vbici;
            *(here->HICUMcollCICollCIPtr)          += -Iciei_Vbici;
            *(here->HICUMemitEIBaseBIPtr)          += -Iciei_Vbici;
            if (model->HICUMnqs) { 
                *(here->HICUMcollCIXf2Ptr)             +=  Iciei_Vxf2;
                *(here->HICUMemitEIXf2Ptr)             += -Iciei_Vxf2;
            }


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

//          Stamp element: Re
            *(here->HICUMemitEmitPtr)              +=  Ieie_Veie;
            *(here->HICUMemitEIEmitEIPtr)          +=  Ieie_Veie;
            *(here->HICUMemitEIEmitPtr)            += -Ieie_Veie;
            *(here->HICUMemitEmitEIPtr)            += -Ieie_Veie;

//          Stamp element: Ibpbi
            if (here->HICUMrbi>0.0) {
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
                *(here->HICUMbaseBIBaseBIPtr)          += -Ibpbi_Vbici;
            };

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

//          Stamp element: Rsu
            *(here->HICUMsubsSubsPtr)              +=  Isis_Vsis;
            *(here->HICUMsubsSISubsSIPtr)          +=  Isis_Vsis;
            *(here->HICUMsubsSISubsPtr)            += -Isis_Vsis;
            *(here->HICUMsubsSubsSIPtr)            += -Isis_Vsis;

            if (model->HICUMnqs) { 
                //Ixf1
                *(here->HICUMxf1BaseBIPtr)             += +Ixf1_Vbiei;
                *(here->HICUMxf1EmitEIPtr)             += -Ixf1_Vbiei;
                *(here->HICUMxf1BaseBIPtr)             += +Ixf1_Vbici;
                *(here->HICUMxf1CollCIPtr)             += -Ixf1_Vbici;
                *(here->HICUMxf1Xf2Ptr)                += +Ixf1_Vxf2;
                *(here->HICUMxf1Xf1Ptr)                += +Ixf1_Vxf1;
                //Ixf2
                *(here->HICUMxf2BaseBIPtr)             += +Ixf2_Vbiei;
                *(here->HICUMxf2EmitEIPtr)             += -Ixf2_Vbiei;
                *(here->HICUMxf2BaseBIPtr)             += +Ixf2_Vbici;
                *(here->HICUMxf2CollCIPtr)             += -Ixf2_Vbici;
                *(here->HICUMxf2Xf2Ptr)                += +Ixf2_Vxf2;
                *(here->HICUMxf2Xf1Ptr)                += +Ixf2_Vxf1;
                //Ixf
                *(here->HICUMxfBaseBIPtr)              += +Ixf_Vbiei;
                *(here->HICUMxfEmitEIPtr)              += -Ixf_Vbiei;
                *(here->HICUMxfBaseBIPtr)              += +Ixf_Vbici;
                *(here->HICUMxfCollCIPtr)              += -Ixf_Vbici;
                *(here->HICUMxfXfPtr)                  += +Ixf_Vxf;
            }

////////////////////////////////////
//////////  The complex part  //////
////////////////////////////////////

            //Qrbi
            XQrbi_Vbpbi       = *(ckt->CKTstate0 + here->HICUMcqrbi)      * ckt->CKTomega;
            XQrbi_Vbiei       = here->HICUMqrbi_Vbiei * ckt->CKTomega;
            XQrbi_Vbici       = here->HICUMqrbi_Vbici * ckt->CKTomega;
            XQrbi_Vrth        = here->HICUMqrbi_Vrth  * ckt->CKTomega;
            //Qjei
            XQjei_Vbiei       = *(ckt->CKTstate0 + here->HICUMcqjei)      * ckt->CKTomega;
            XQjei_Vrth        = here->HICUMqjei_Vrth  * ckt->CKTomega;
            //Qf
            XQf_Vbiei         = *(ckt->CKTstate0 + here->HICUMcqf)        * ckt->CKTomega;
            XQf_Vbici         = here->HICUMqf_Vbici   * ckt->CKTomega;
            XQf_Vrth          = here->HICUMqf_Vrth    * ckt->CKTomega;
            XQf_Vxf           = here->HICUMqf_Vxf     * ckt->CKTomega;
            //Qr
            XQr_Vbici         = *(ckt->CKTstate0 + here->HICUMcqr)        * ckt->CKTomega;
            XQr_Vbiei         = here->HICUMqr_Vbiei   * ckt->CKTomega;
            XQr_Vrth          = here->HICUMqr_Vrth    * ckt->CKTomega;
            //Qjci
            XQjci_Vbici       = *(ckt->CKTstate0 + here->HICUMcqjci)      * ckt->CKTomega;
            XQjci_Vrth        = here->HICUMqjci_Vrth  * ckt->CKTomega;
            //Qjep
            XQjep_Vbpei       = *(ckt->CKTstate0 + here->HICUMcqjep)      * ckt->CKTomega;
            XQjep_Vrth        = here->HICUMqjep_Vrth  * ckt->CKTomega;
            //Qjcx_i
            Xqjcx0_t_i_Vbci   = *(ckt->CKTstate0 + here->HICUMcqcx0_t_i)  * ckt->CKTomega;
            Xqjcx0_t_i_Vrth   = here->HICUMqjcx0_i_Vrth  * ckt->CKTomega;
            //Qjcx_ii
            Xqjcx0_t_ii_Vbpci = *(ckt->CKTstate0 + here->HICUMcqcx0_t_ii) * ckt->CKTomega;
            Xqjcx0_t_ii_Vrth  = here->HICUMqjcx0_ii_Vrth * ckt->CKTomega;
            //Qdsu
            XQdsu_Vbpci       = *(ckt->CKTstate0 + here->HICUMcqdsu)      * ckt->CKTomega;
            XQdsu_Vsici       = here->HICUMqdsu_Vsici * ckt->CKTomega;
            XQdsu_Vrth        = here->HICUMqdsu_Vrth  * ckt->CKTomega;
            //Qjs
            XQjs_Vsici        = *(ckt->CKTstate0 + here->HICUMcqjs)       * ckt->CKTomega;
            XQjs_Vrth         = here->HICUMqjs_Vrth   * ckt->CKTomega;
            //Qscp
            XQscp_Vsc         = *(ckt->CKTstate0 + here->HICUMcqscp)      * ckt->CKTomega;
            XQscp_Vrth        = here->HICUMqscp_Vrth  * ckt->CKTomega;
            //Qbepar1
            XQbepar1_Vbe      = *(ckt->CKTstate0 + here->HICUMcqbepar1)   * ckt->CKTomega;
            //Qbepar2
            XQbepar2_Vbpe     = *(ckt->CKTstate0 + here->HICUMcqbepar2)   * ckt->CKTomega;
            //Qbcpar1
            XQbcpar1_Vbci     = *(ckt->CKTstate0 + here->HICUMcqbcpar1)   * ckt->CKTomega;
            //Qbcpar2
            XQbcpar2_Vbpci    = *(ckt->CKTstate0 + here->HICUMcqbcpar2)   * ckt->CKTomega;
            //Qsu
            XQsu_Vsis         = *(ckt->CKTstate0 + here->HICUMcqsu)       * ckt->CKTomega;
            //Qcth
            XQcth_Vrth        = *(ckt->CKTstate0 + here->HICUMcqcth)      * ckt->CKTomega;
            //Qxf
            XQxf_Vxf          = *(ckt->CKTstate0 + here->HICUMcqxf)       * ckt->CKTomega;
            XQxf1_Vxf1        = *(ckt->CKTstate0 + here->HICUMcqxf1)      * ckt->CKTomega;
            XQxf2_Vxf2        = *(ckt->CKTstate0 + here->HICUMcqxf2)      * ckt->CKTomega;

            //Qrbi f_bp=+ f_bi=-
            if (here->HICUMrbi>0.0) {
                *(here->HICUMbaseBPBaseBPPtr + 1)          +=  XQrbi_Vbpbi; 
                *(here->HICUMbaseBIBaseBIPtr + 1)          +=  XQrbi_Vbpbi;
                *(here->HICUMbaseBPBaseBIPtr + 1)          += -XQrbi_Vbpbi;
                *(here->HICUMbaseBIBaseBPPtr + 1)          += -XQrbi_Vbpbi;
                *(here->HICUMbaseBPBaseBIPtr + 1)          +=  XQrbi_Vbiei; 
                *(here->HICUMbaseBIEmitEIPtr + 1)          +=  XQrbi_Vbiei;
                *(here->HICUMbaseBPEmitEIPtr + 1)          += -XQrbi_Vbiei;
                *(here->HICUMbaseBIBaseBIPtr + 1)          += -XQrbi_Vbiei;
                *(here->HICUMbaseBPBaseBIPtr + 1)          +=  XQrbi_Vbici; 
                *(here->HICUMbaseBICollCIPtr + 1)          +=  XQrbi_Vbici;
                *(here->HICUMbaseBPCollCIPtr + 1)          += -XQrbi_Vbici;
                *(here->HICUMbaseBIBaseBIPtr + 1)          += -XQrbi_Vbici;
            }
            //Qjei
            *(here->HICUMbaseBIBaseBIPtr + 1)          +=  XQjei_Vbiei;
            *(here->HICUMemitEIEmitEIPtr + 1)          +=  XQjei_Vbiei;
            *(here->HICUMbaseBIEmitEIPtr + 1)          += -XQjei_Vbiei;
            *(here->HICUMemitEIBaseBIPtr + 1)          += -XQjei_Vbiei;
            //Qf  f_Bi=+ f_Ei =-
            *(here->HICUMbaseBIBaseBIPtr +1)           +=  XQf_Vbiei;
            *(here->HICUMemitEIEmitEIPtr +1)           +=  XQf_Vbiei;
            *(here->HICUMbaseBIEmitEIPtr +1)           += -XQf_Vbiei;
            *(here->HICUMemitEIBaseBIPtr +1)           += -XQf_Vbiei;
            *(here->HICUMbaseBIBaseBIPtr +1)           +=  XQf_Vbici;
            *(here->HICUMemitEICollCIPtr +1)           +=  XQf_Vbici;
            *(here->HICUMbaseBICollCIPtr +1)           += -XQf_Vbici;
            *(here->HICUMemitEIBaseBIPtr +1)           += -XQf_Vbici;
            if (model->HICUMnqs) { 
                *(here->HICUMbaseBIXfPtr    +1)            +=  XQf_Vxf;
                *(here->HICUMemitEIXfPtr    +1)            += -XQf_Vxf;
            }
            //Qjci
            *(here->HICUMbaseBIBaseBIPtr +1)           +=  XQjci_Vbici;
            *(here->HICUMcollCICollCIPtr +1)           +=  XQjci_Vbici;
            *(here->HICUMcollCIBaseBIPtr +1)           += -XQjci_Vbici;
            *(here->HICUMbaseBICollCIPtr +1)           += -XQjci_Vbici;
            //Qr f_bi = + f_ci=-
            *(here->HICUMbaseBIBaseBIPtr +1)           +=  XQr_Vbici;
            *(here->HICUMcollCICollCIPtr +1)           +=  XQr_Vbici;
            *(here->HICUMcollCIBaseBIPtr +1)           += -XQr_Vbici;
            *(here->HICUMbaseBICollCIPtr +1)           += -XQr_Vbici;
            *(here->HICUMbaseBIBaseBIPtr +1)           +=  XQr_Vbiei;
            *(here->HICUMcollCIEmitEIPtr +1)           +=  XQr_Vbiei;
            *(here->HICUMcollCIBaseBIPtr +1)           += -XQr_Vbiei;
            *(here->HICUMbaseBIEmitEIPtr +1)           += -XQr_Vbiei;
            //Qjep
            *(here->HICUMbaseBPBaseBPPtr +1)           +=  XQjep_Vbpei;
            *(here->HICUMemitEIEmitEIPtr +1)           +=  XQjep_Vbpei;
            *(here->HICUMbaseBPEmitEIPtr +1)           += -XQjep_Vbpei;
            *(here->HICUMemitEIBaseBPPtr +1)           += -XQjep_Vbpei;
            //Qjcx_i
            *(here->HICUMbaseBasePtr +1)               +=  Xqjcx0_t_i_Vbci;
            *(here->HICUMcollCICollCIPtr +1)           +=  Xqjcx0_t_i_Vbci;
            *(here->HICUMbaseCollCIPtr +1)             += -Xqjcx0_t_i_Vbci;
            *(here->HICUMcollCIBasePtr +1)             += -Xqjcx0_t_i_Vbci;
            //Qjcx_ii
            *(here->HICUMbaseBPBaseBPPtr +1)           +=  Xqjcx0_t_ii_Vbpci;
            *(here->HICUMcollCICollCIPtr +1)           +=  Xqjcx0_t_ii_Vbpci;
            *(here->HICUMbaseBPCollCIPtr +1)           += -Xqjcx0_t_ii_Vbpci;
            *(here->HICUMcollCIBaseBPPtr +1)           += -Xqjcx0_t_ii_Vbpci;
            //Qdsu f_bp=+ f_ci=-
            *(here->HICUMbaseBPBaseBPPtr +1)           +=  XQdsu_Vbpci;
            *(here->HICUMcollCICollCIPtr +1)           +=  XQdsu_Vbpci;
            *(here->HICUMbaseBPCollCIPtr +1)           += -XQdsu_Vbpci;
            *(here->HICUMcollCIBaseBPPtr +1)           += -XQdsu_Vbpci;
            *(here->HICUMbaseBPSubsSIPtr +1)           +=  XQdsu_Vsici;
            *(here->HICUMcollCICollCIPtr +1)           +=  XQdsu_Vsici;
            *(here->HICUMbaseBPCollCIPtr +1)           +=  -XQdsu_Vsici;
            *(here->HICUMcollCISubsSIPtr +1)           +=  -XQdsu_Vsici;
            //Qjs
            *(here->HICUMsubsSISubsSIPtr +1)          +=  XQjs_Vsici;
            *(here->HICUMcollCICollCIPtr +1)          +=  XQjs_Vsici;
            *(here->HICUMsubsSICollCIPtr +1)          += -XQjs_Vsici;
            *(here->HICUMcollCISubsSIPtr +1)          += -XQjs_Vsici;
            //Qscp
            *(here->HICUMsubsSubsPtr + 1)             +=  XQscp_Vsc;
            *(here->HICUMcollCollPtr + 1)             +=  XQscp_Vsc;
            *(here->HICUMcollSubsPtr + 1)             += -XQscp_Vsc;
            *(here->HICUMsubsCollPtr + 1)             += -XQscp_Vsc;
            //Qbepar1
            *(here->HICUMbaseBasePtr + 1)             +=  XQbepar1_Vbe;
            *(here->HICUMemitEmitPtr + 1)             +=  XQbepar1_Vbe;
            *(here->HICUMbaseEmitPtr + 1)             += -XQbepar1_Vbe;
            *(here->HICUMemitBasePtr + 1)             += -XQbepar1_Vbe;
            //Qbepar2
            *(here->HICUMbaseBPBaseBPPtr + 1)         +=  XQbepar2_Vbpe;
            *(here->HICUMemitEmitPtr + 1)             +=  XQbepar2_Vbpe;
            *(here->HICUMemitBaseBPPtr + 1)           += -XQbepar2_Vbpe;
            *(here->HICUMbaseBPEmitPtr + 1)           += -XQbepar2_Vbpe;
            //Qbcpar1
            *(here->HICUMbaseBasePtr + 1)             +=  XQbcpar1_Vbci;
            *(here->HICUMcollCICollCIPtr + 1)         +=  XQbcpar1_Vbci;
            *(here->HICUMbaseCollCIPtr + 1)           += -XQbcpar1_Vbci;
            *(here->HICUMcollCIBasePtr + 1)           += -XQbcpar1_Vbci;
            //Qbcpar2
            *(here->HICUMbaseBPBaseBPPtr +1)          +=  XQbcpar2_Vbpci;
            *(here->HICUMcollCICollCIPtr +1)          +=  XQbcpar2_Vbpci;
            *(here->HICUMbaseBPCollCIPtr +1)          += -XQbcpar2_Vbpci;
            *(here->HICUMcollCIBaseBPPtr +1)          += -XQbcpar2_Vbpci;
            //Qsu
            *(here->HICUMsubsSubsPtr + 1)             +=  XQsu_Vsis;
            *(here->HICUMsubsSISubsSIPtr + 1)         +=  XQsu_Vsis;
            *(here->HICUMsubsSISubsPtr + 1)           += -XQsu_Vsis;
            *(here->HICUMsubsSubsSIPtr + 1)           += -XQsu_Vsis;
            if (model->HICUMnqs) { 
                //Qxf1
                *(here->HICUMxf1Xf1Ptr + 1)               += +XQxf1_Vxf1;
                //Qxf2
                *(here->HICUMxf2Xf2Ptr + 1)               += +XQxf2_Vxf2;
                //Qxf
                *(here->HICUMxfXfPtr + 1)                 += +XQxf_Vxf;
            }

            // Stamps with SH
            if (model->HICUMselfheat) { 
//              Stamp element: Ibiei  f_Bi = +   f_Ei = -
                *(here->HICUMbaseBItempPtr)            +=  Ibiei_Vrth;
                *(here->HICUMemitEItempPtr)            += -Ibiei_Vrth;
//              Stamp element: Ibpei  f_Bp = +   f_Ei = -
                // with respect to Potential Vrth
                *(here->HICUMbaseBPtempPtr)            +=  Ibpei_Vrth;
                *(here->HICUMemitEItempPtr)            += -Ibpei_Vrth;
//              Stamp element: Ibici  f_Bi = +   f_Ci = -
                *(here->HICUMbaseBItempPtr)            +=  Ibici_Vrth;
                *(here->HICUMcollCItempPtr)            += -Ibici_Vrth;
//              Stamp element: Iciei  f_Ci = +   f_Ei = -
                *(here->HICUMcollCItempPtr)            +=  Iciei_Vrth;
                *(here->HICUMemitEItempPtr)            += -Iciei_Vrth;
//              Stamp element: Ibpci  f_Bp = +   f_Ci = -
                *(here->HICUMbaseBPtempPtr)            +=  Ibpci_Vrth;
                *(here->HICUMcollCItempPtr)            += -Ibpci_Vrth;
//              Stamp element: Rcx  f_Ci = +   f_C = -
                *(here->HICUMcollCItempPtr)            +=  Icic_Vrth;
                *(here->HICUMcollTempPtr)              += -Icic_Vrth;
//              Stamp element: Rbx  f_B = +   f_Bp = -
                *(here->HICUMbaseTempPtr)              +=  Ibbp_Vrth;
                *(here->HICUMbaseBPtempPtr)            += -Ibbp_Vrth;
//              Stamp element: Re   f_Ei = +   f_E = -
                *(here->HICUMemitEItempPtr)            +=  Ieie_Vrth;
                *(here->HICUMemitTempPtr)              += -Ieie_Vrth;
//              Stamp element: Rbi    f_Bp = +   f_Bi = -
                *(here->HICUMbaseBPtempPtr)            +=  Ibpbi_Vrth;
                *(here->HICUMbaseBItempPtr)            += -Ibpbi_Vrth;
//              Stamp element: Isici   f_Si = +   f_Ci = -
                *(here->HICUMsubsSItempPtr)            +=  Isici_Vrth;
                *(here->HICUMcollCItempPtr)            += -Isici_Vrth;
//              Branch: bpsi, Stamp element: Its
                *(here->HICUMbaseBPtempPtr)            +=  Ibpsi_Vrth;
                *(here->HICUMsubsSItempPtr)            += -Ibpsi_Vrth;
                if (model->HICUMnqs) { 
    //              Stamp element: Ixf    f_xf = +   
                    *(here->HICUMxfTempPtr)                +=  Ixf_Vrth;
    //              Stamp element: Ixf1    f_xf1 = +   
                    *(here->HICUMxf1TempPtr)               +=  Ixf1_Vrth;
    //              Stamp element: Ixf2    f_xf2 = +   
                    *(here->HICUMxf2TempPtr)               +=  Ixf2_Vrth;
                }

//              Stamp element: Rth   f_T = +
                *(here->HICUMtempTempPtr)   +=  Irth_Vrth;

//              Stamp element:    Ith f_T = - Ith 
                // with respect to Potential Vrth
                *(here->HICUMtempTempPtr)   += -Ith_Vrth;
                // with respect to Potential Vbiei
                *(here->HICUMtempBaseBIPtr) += -Ith_Vbiei;
                *(here->HICUMtempEmitEIPtr) += +Ith_Vbiei;
                // with respect to Potential Vbici
                *(here->HICUMtempBaseBIPtr) += -Ith_Vbici;
                *(here->HICUMtempCollCIPtr) += +Ith_Vbici;
                // with respect to Potential Vciei
                *(here->HICUMtempCollCIPtr) += -Ith_Vciei;
                *(here->HICUMtempEmitEIPtr) += +Ith_Vciei;
                // with respect to Potential Vbpei
                *(here->HICUMtempBaseBPPtr) += -Ith_Vbpei;
                *(here->HICUMtempEmitEIPtr) += +Ith_Vbpei;
                // with respect to Potential Vbpci
                *(here->HICUMtempBaseBPPtr) += -Ith_Vbpci;
                *(here->HICUMtempCollCIPtr) += +Ith_Vbpci;
                // with respect to Potential Vsici
                *(here->HICUMtempSubsSIPtr) += -Ith_Vsici;
                *(here->HICUMtempCollCIPtr) += +Ith_Vsici;
                // with respect to Potential Vbpbi
                *(here->HICUMtempBaseBPPtr) += -Ith_Vbpbi;
                *(here->HICUMtempBaseBIPtr) += +Ith_Vbpbi;
                // with respect to Potential Vcic
                *(here->HICUMtempCollCIPtr) += -Ith_Vcic;
                *(here->HICUMtempCollPtr)   += +Ith_Vcic;
                // with respect to Potential Vbbp
                *(here->HICUMtempBasePtr)   += -Ith_Vbbp;
                *(here->HICUMtempBaseBPPtr) += +Ith_Vbbp;
                // with respect to Potential Veie
                *(here->HICUMtempEmitEIPtr) += -Ith_Veie;
                *(here->HICUMtempEmitPtr)   += +Ith_Veie;

                //the charges
                *(here->HICUMbaseBItempPtr + 1) += +XQrbi_Vrth;
                *(here->HICUMbaseBPtempPtr + 1) += -XQrbi_Vrth;
                *(here->HICUMbaseBItempPtr + 1) += +XQjei_Vrth;
                *(here->HICUMemitEItempPtr + 1) += -XQjei_Vrth;
                *(here->HICUMbaseBItempPtr + 1) += +XQf_Vrth;
                *(here->HICUMemitEItempPtr + 1) += -XQf_Vrth;
                *(here->HICUMbaseBItempPtr + 1) += +XQr_Vrth;
                *(here->HICUMcollCItempPtr + 1) += -XQr_Vrth;
                *(here->HICUMbaseBItempPtr + 1) += +XQjci_Vrth;
                *(here->HICUMcollCItempPtr + 1) += -XQjci_Vrth;
                *(here->HICUMbaseBPtempPtr + 1) += +XQjep_Vrth;
                *(here->HICUMemitEItempPtr + 1) += -XQjep_Vrth;
                *(here->HICUMbaseTempPtr   + 1) += +Xqjcx0_t_i_Vrth;
                *(here->HICUMcollCItempPtr + 1) += -Xqjcx0_t_i_Vrth;
                *(here->HICUMbaseBPtempPtr + 1) += +Xqjcx0_t_ii_Vrth;
                *(here->HICUMcollCItempPtr + 1) += -Xqjcx0_t_ii_Vrth;
                *(here->HICUMbaseBPtempPtr + 1) += +XQdsu_Vrth;
                *(here->HICUMcollCItempPtr + 1) += -XQdsu_Vrth;
                *(here->HICUMsubsSItempPtr + 1) += +XQjs_Vrth;
                *(here->HICUMcollCItempPtr + 1) += -XQjs_Vrth;
                *(here->HICUMsubsTempPtr   + 1) += +XQscp_Vrth;
                *(here->HICUMcollTempPtr   + 1) += -XQscp_Vrth;
                *(here->HICUMtempTempPtr   + 1) += +XQcth_Vrth;

            } 

        }
    }
    return(OK);
}
