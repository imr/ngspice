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
HICUMpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
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
    double Icic_Vcic, Icic_Vrth;
    double Ieie_Veie, Ieie_Vrth;
    double Ibbp_Vbbp, Ibbp_Vrth;
    double Irth_Vrth;

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
            XQrbi_Vbpbi       = *(ckt->CKTstate0 + here->HICUMcqrbi);
            XQrbi_Vbiei       = here->HICUMqrbi_Vbiei;
            XQrbi_Vbici       = here->HICUMqrbi_Vbici;
            XQrbi_Vrth        = here->HICUMqrbi_Vrth;
            //Qjei
            XQjei_Vbiei       = *(ckt->CKTstate0 + here->HICUMcqjei);
            XQjei_Vrth        = here->HICUMqjei_Vrth;
            //Qf
            XQf_Vbiei         = *(ckt->CKTstate0 + here->HICUMcqf);
            XQf_Vbici         = here->HICUMqf_Vbici;
            XQf_Vrth          = here->HICUMqf_Vrth;
            XQf_Vxf           = here->HICUMqf_Vxf;
            //Qr
            XQr_Vbici         = *(ckt->CKTstate0 + here->HICUMcqr);
            XQr_Vbiei         = here->HICUMqr_Vbiei;
            XQr_Vrth          = here->HICUMqr_Vrth;
            //Qjci
            XQjci_Vbici       = *(ckt->CKTstate0 + here->HICUMcqjci);
            XQjci_Vrth        = here->HICUMqjci_Vrth;
            //Qjep
            XQjep_Vbpei       = *(ckt->CKTstate0 + here->HICUMcqjep);
            XQjep_Vrth        = here->HICUMqjep_Vrth;
            //Qjcx_i
            Xqjcx0_t_i_Vbci   = *(ckt->CKTstate0 + here->HICUMcqcx0_t_i);
            Xqjcx0_t_i_Vrth   = here->HICUMqjcx0_i_Vrth;
            //Qjcx_ii
            Xqjcx0_t_ii_Vbpci = *(ckt->CKTstate0 + here->HICUMcqcx0_t_ii);
            Xqjcx0_t_ii_Vrth  = here->HICUMqjcx0_ii_Vrth;
            //Qdsu
            XQdsu_Vbpci       = *(ckt->CKTstate0 + here->HICUMcqdsu);
            XQdsu_Vsici       = here->HICUMqdsu_Vsici;
            XQdsu_Vrth        = here->HICUMqdsu_Vrth;
            //Qjs
            XQjs_Vsici        = *(ckt->CKTstate0 + here->HICUMcqjs);
            XQjs_Vrth         = here->HICUMqjs_Vrth;
            //Qscp
            XQscp_Vsc         = *(ckt->CKTstate0 + here->HICUMcqscp);
            XQscp_Vrth        = here->HICUMqscp_Vrth;
            //Qbepar1
            XQbepar1_Vbe      = *(ckt->CKTstate0 + here->HICUMcqbepar1);
            //Qbepar2
            XQbepar2_Vbpe     = *(ckt->CKTstate0 + here->HICUMcqbepar2);
            //Qbcpar1
            XQbcpar1_Vbci     = *(ckt->CKTstate0 + here->HICUMcqbcpar1);
            //Qbcpar2
            XQbcpar2_Vbpci    = *(ckt->CKTstate0 + here->HICUMcqbcpar2);
            //Qsu
            XQsu_Vsis         = *(ckt->CKTstate0 + here->HICUMcqsu);
            //Qcth
            XQcth_Vrth        = *(ckt->CKTstate0 + here->HICUMcqcth);
            //Qxf
            XQxf_Vxf          = *(ckt->CKTstate0 + here->HICUMcqxf);
            XQxf1_Vxf1        = *(ckt->CKTstate0 + here->HICUMcqxf1);
            XQxf2_Vxf2        = *(ckt->CKTstate0 + here->HICUMcqxf2);

            //Qrbi f_bp=+ f_bi=-
            if (here->HICUMrbi>0.0) {
                *(here->HICUMbaseBPBaseBPPtr + 1)          +=  XQrbi_Vbpbi*(s->imag); 
                *(here->HICUMbaseBPBaseBPPtr )             +=  XQrbi_Vbpbi*(s->real); 
                *(here->HICUMbaseBIBaseBIPtr + 1)          +=  XQrbi_Vbpbi*(s->imag);
                *(here->HICUMbaseBIBaseBIPtr )             +=  XQrbi_Vbpbi*(s->real);
                *(here->HICUMbaseBPBaseBIPtr + 1)          += -XQrbi_Vbpbi*(s->imag);
                *(here->HICUMbaseBPBaseBIPtr )             += -XQrbi_Vbpbi*(s->real);
                *(here->HICUMbaseBIBaseBPPtr + 1)          += -XQrbi_Vbpbi*(s->imag);
                *(here->HICUMbaseBIBaseBPPtr )             += -XQrbi_Vbpbi*(s->real);
                *(here->HICUMbaseBPBaseBIPtr + 1)          +=  XQrbi_Vbiei*(s->imag); 
                *(here->HICUMbaseBPBaseBIPtr )             +=  XQrbi_Vbiei*(s->real); 
                *(here->HICUMbaseBIEmitEIPtr + 1)          +=  XQrbi_Vbiei*(s->imag);
                *(here->HICUMbaseBIEmitEIPtr )             +=  XQrbi_Vbiei*(s->real);
                *(here->HICUMbaseBPEmitEIPtr + 1)          += -XQrbi_Vbiei*(s->imag);
                *(here->HICUMbaseBPEmitEIPtr )             += -XQrbi_Vbiei*(s->real);
                *(here->HICUMbaseBIBaseBIPtr + 1)          += -XQrbi_Vbiei*(s->imag);
                *(here->HICUMbaseBIBaseBIPtr )             += -XQrbi_Vbiei*(s->real);
                *(here->HICUMbaseBPBaseBIPtr + 1)          +=  XQrbi_Vbici*(s->imag); 
                *(here->HICUMbaseBPBaseBIPtr )             +=  XQrbi_Vbici*(s->real); 
                *(here->HICUMbaseBICollCIPtr + 1)          +=  XQrbi_Vbici*(s->imag);
                *(here->HICUMbaseBICollCIPtr )             +=  XQrbi_Vbici*(s->real);
                *(here->HICUMbaseBPCollCIPtr + 1)          += -XQrbi_Vbici*(s->imag);
                *(here->HICUMbaseBPCollCIPtr )             += -XQrbi_Vbici*(s->real);
                *(here->HICUMbaseBIBaseBIPtr + 1)          += -XQrbi_Vbici*(s->imag);
                *(here->HICUMbaseBIBaseBIPtr )             += -XQrbi_Vbici*(s->real);
            };
            //Qjei
            *(here->HICUMbaseBIBaseBIPtr + 1)          +=  XQjei_Vbiei*(s->imag);
            *(here->HICUMbaseBIBaseBIPtr )             +=  XQjei_Vbiei*(s->real);
            *(here->HICUMemitEIEmitEIPtr + 1)          +=  XQjei_Vbiei*(s->imag);
            *(here->HICUMemitEIEmitEIPtr )             +=  XQjei_Vbiei*(s->real);
            *(here->HICUMbaseBIEmitEIPtr + 1)          += -XQjei_Vbiei*(s->imag);
            *(here->HICUMbaseBIEmitEIPtr )             += -XQjei_Vbiei*(s->real);
            *(here->HICUMemitEIBaseBIPtr + 1)          += -XQjei_Vbiei*(s->imag);
            *(here->HICUMemitEIBaseBIPtr )             += -XQjei_Vbiei*(s->real);
            //Qf  f_Bi=+ f_Ei =-
            *(here->HICUMbaseBIBaseBIPtr +1)           +=  XQf_Vbiei*(s->imag);
            *(here->HICUMbaseBIBaseBIPtr)              +=  XQf_Vbiei*(s->real);
            *(here->HICUMemitEIEmitEIPtr +1)           +=  XQf_Vbiei*(s->imag);
            *(here->HICUMemitEIEmitEIPtr)              +=  XQf_Vbiei*(s->real);
            *(here->HICUMbaseBIEmitEIPtr +1)           += -XQf_Vbiei*(s->imag);
            *(here->HICUMbaseBIEmitEIPtr)              += -XQf_Vbiei*(s->real);
            *(here->HICUMemitEIBaseBIPtr +1)           += -XQf_Vbiei*(s->imag);
            *(here->HICUMemitEIBaseBIPtr)              += -XQf_Vbiei*(s->real);
            *(here->HICUMbaseBIBaseBIPtr +1)           +=  XQf_Vbici*(s->imag);
            *(here->HICUMbaseBIBaseBIPtr)              +=  XQf_Vbici*(s->real);
            *(here->HICUMemitEICollCIPtr +1)           +=  XQf_Vbici*(s->imag);
            *(here->HICUMemitEICollCIPtr)              +=  XQf_Vbici*(s->real);
            *(here->HICUMbaseBICollCIPtr +1)           += -XQf_Vbici*(s->imag);
            *(here->HICUMbaseBICollCIPtr)              += -XQf_Vbici*(s->real);
            *(here->HICUMemitEIBaseBIPtr +1)           += -XQf_Vbici*(s->imag);
            *(here->HICUMemitEIBaseBIPtr)              += -XQf_Vbici*(s->real);
            if (model->HICUMnqs) { 
                *(here->HICUMbaseBIXfPtr    +1)            +=  XQf_Vxf*(s->imag);
                *(here->HICUMbaseBIXfPtr   )               +=  XQf_Vxf*(s->real);
                *(here->HICUMemitEIXfPtr    +1)            += -XQf_Vxf*(s->imag);
                *(here->HICUMemitEIXfPtr   )               += -XQf_Vxf*(s->real);
            }
            //Qjci
            *(here->HICUMbaseBIBaseBIPtr +1)           +=  XQjci_Vbici*(s->imag);
            *(here->HICUMbaseBIBaseBIPtr)              +=  XQjci_Vbici*(s->real);
            *(here->HICUMcollCICollCIPtr +1)           +=  XQjci_Vbici*(s->imag);
            *(here->HICUMcollCICollCIPtr)              +=  XQjci_Vbici*(s->real);
            *(here->HICUMcollCIBaseBIPtr +1)           += -XQjci_Vbici*(s->imag);
            *(here->HICUMcollCIBaseBIPtr)              += -XQjci_Vbici*(s->real);
            *(here->HICUMbaseBICollCIPtr +1)           += -XQjci_Vbici*(s->imag);
            *(here->HICUMbaseBICollCIPtr)              += -XQjci_Vbici*(s->real);
            //Qr f_bi = + f_ci=-
            *(here->HICUMbaseBIBaseBIPtr +1)           +=  XQr_Vbici*(s->imag);
            *(here->HICUMbaseBIBaseBIPtr)              +=  XQr_Vbici*(s->real);
            *(here->HICUMcollCICollCIPtr +1)           +=  XQr_Vbici*(s->imag);
            *(here->HICUMcollCICollCIPtr)              +=  XQr_Vbici*(s->real);
            *(here->HICUMcollCIBaseBIPtr +1)           += -XQr_Vbici*(s->imag);
            *(here->HICUMcollCIBaseBIPtr)              += -XQr_Vbici*(s->real);
            *(here->HICUMbaseBICollCIPtr +1)           += -XQr_Vbici*(s->imag);
            *(here->HICUMbaseBICollCIPtr)              += -XQr_Vbici*(s->real);
            *(here->HICUMbaseBIBaseBIPtr +1)           +=  XQr_Vbiei*(s->imag);
            *(here->HICUMbaseBIBaseBIPtr)              +=  XQr_Vbiei*(s->real);
            *(here->HICUMcollCIEmitEIPtr +1)           +=  XQr_Vbiei*(s->imag);
            *(here->HICUMcollCIEmitEIPtr)              +=  XQr_Vbiei*(s->real);
            *(here->HICUMcollCIBaseBIPtr +1)           += -XQr_Vbiei*(s->imag);
            *(here->HICUMcollCIBaseBIPtr)              += -XQr_Vbiei*(s->real);
            *(here->HICUMbaseBIEmitEIPtr +1)           += -XQr_Vbiei*(s->imag);
            *(here->HICUMbaseBIEmitEIPtr)              += -XQr_Vbiei*(s->real);
            //Qjep
            *(here->HICUMbaseBPBaseBPPtr +1)           +=  XQjep_Vbpei*(s->imag);
            *(here->HICUMbaseBPBaseBPPtr)              +=  XQjep_Vbpei*(s->real);
            *(here->HICUMemitEIEmitEIPtr +1)           +=  XQjep_Vbpei*(s->imag);
            *(here->HICUMemitEIEmitEIPtr)              +=  XQjep_Vbpei*(s->real);
            *(here->HICUMbaseBPEmitEIPtr +1)           += -XQjep_Vbpei*(s->imag);
            *(here->HICUMbaseBPEmitEIPtr)              += -XQjep_Vbpei*(s->real);
            *(here->HICUMemitEIBaseBPPtr +1)           += -XQjep_Vbpei*(s->imag);
            *(here->HICUMemitEIBaseBPPtr)              += -XQjep_Vbpei*(s->real);
            //Qjcx_i
            *(here->HICUMbaseBasePtr +1)               +=  Xqjcx0_t_i_Vbci*(s->imag);
            *(here->HICUMbaseBasePtr)                  +=  Xqjcx0_t_i_Vbci*(s->real);
            *(here->HICUMcollCICollCIPtr +1)           +=  Xqjcx0_t_i_Vbci*(s->imag);
            *(here->HICUMcollCICollCIPtr)              +=  Xqjcx0_t_i_Vbci*(s->real);
            *(here->HICUMbaseCollCIPtr +1)             += -Xqjcx0_t_i_Vbci*(s->imag);
            *(here->HICUMbaseCollCIPtr)                += -Xqjcx0_t_i_Vbci*(s->real);
            *(here->HICUMcollCIBasePtr +1)             += -Xqjcx0_t_i_Vbci*(s->imag);
            *(here->HICUMcollCIBasePtr)                += -Xqjcx0_t_i_Vbci*(s->real);
            //Qjcx_ii
            *(here->HICUMbaseBPBaseBPPtr +1)           +=  Xqjcx0_t_ii_Vbpci*(s->imag);
            *(here->HICUMbaseBPBaseBPPtr)              +=  Xqjcx0_t_ii_Vbpci*(s->real);
            *(here->HICUMcollCICollCIPtr +1)           +=  Xqjcx0_t_ii_Vbpci*(s->imag);
            *(here->HICUMcollCICollCIPtr)              +=  Xqjcx0_t_ii_Vbpci*(s->real);
            *(here->HICUMbaseBPCollCIPtr +1)           += -Xqjcx0_t_ii_Vbpci*(s->imag);
            *(here->HICUMbaseBPCollCIPtr)              += -Xqjcx0_t_ii_Vbpci*(s->real);
            *(here->HICUMcollCIBaseBPPtr +1)           += -Xqjcx0_t_ii_Vbpci*(s->imag);
            *(here->HICUMcollCIBaseBPPtr)              += -Xqjcx0_t_ii_Vbpci*(s->real);
            //Qdsu f_bp=+ f_ci=-
            *(here->HICUMbaseBPBaseBPPtr +1)           +=  XQdsu_Vbpci*(s->imag);
            *(here->HICUMbaseBPBaseBPPtr)              +=  XQdsu_Vbpci*(s->real);
            *(here->HICUMcollCICollCIPtr +1)           +=  XQdsu_Vbpci*(s->imag);
            *(here->HICUMcollCICollCIPtr)              +=  XQdsu_Vbpci*(s->real);
            *(here->HICUMbaseBPCollCIPtr +1)           += -XQdsu_Vbpci*(s->imag);
            *(here->HICUMbaseBPCollCIPtr)              += -XQdsu_Vbpci*(s->real);
            *(here->HICUMcollCIBaseBPPtr +1)           += -XQdsu_Vbpci*(s->imag);
            *(here->HICUMcollCIBaseBPPtr)              += -XQdsu_Vbpci*(s->real);
            *(here->HICUMbaseBPSubsSIPtr +1)           +=  XQdsu_Vsici*(s->imag);
            *(here->HICUMbaseBPSubsSIPtr)              +=  XQdsu_Vsici*(s->real);
            *(here->HICUMcollCICollCIPtr +1)           +=  XQdsu_Vsici*(s->imag);
            *(here->HICUMcollCICollCIPtr)              +=  XQdsu_Vsici*(s->real);
            *(here->HICUMbaseBPCollCIPtr +1)           +=  -XQdsu_Vsici*(s->imag);
            *(here->HICUMbaseBPCollCIPtr)              +=  -XQdsu_Vsici*(s->real);
            *(here->HICUMcollCISubsSIPtr +1)           +=  -XQdsu_Vsici*(s->imag);
            *(here->HICUMcollCISubsSIPtr)              +=  -XQdsu_Vsici*(s->real);
            //Qjs
            *(here->HICUMsubsSISubsSIPtr +1)          +=  XQjs_Vsici*(s->imag);
            *(here->HICUMsubsSISubsSIPtr)             +=  XQjs_Vsici*(s->real);
            *(here->HICUMcollCICollCIPtr +1)          +=  XQjs_Vsici*(s->imag);
            *(here->HICUMcollCICollCIPtr)             +=  XQjs_Vsici*(s->real);
            *(here->HICUMsubsSICollCIPtr +1)          += -XQjs_Vsici*(s->imag);
            *(here->HICUMsubsSICollCIPtr)             += -XQjs_Vsici*(s->real);
            *(here->HICUMcollCISubsSIPtr +1)          += -XQjs_Vsici*(s->imag);
            *(here->HICUMcollCISubsSIPtr)             += -XQjs_Vsici*(s->real);
            //Qscp
            *(here->HICUMsubsSubsPtr + 1)             +=  XQscp_Vsc*(s->imag);
            *(here->HICUMsubsSubsPtr )                +=  XQscp_Vsc*(s->real);
            *(here->HICUMcollCollPtr + 1)             +=  XQscp_Vsc*(s->imag);
            *(here->HICUMcollCollPtr )                +=  XQscp_Vsc*(s->real);
            *(here->HICUMcollSubsPtr + 1)             += -XQscp_Vsc*(s->imag);
            *(here->HICUMcollSubsPtr )                += -XQscp_Vsc*(s->real);
            *(here->HICUMsubsCollPtr + 1)             += -XQscp_Vsc*(s->imag);
            *(here->HICUMsubsCollPtr )                += -XQscp_Vsc*(s->real);
            //Qbepar1
            *(here->HICUMbaseBasePtr + 1)             +=  XQbepar1_Vbe*(s->imag);
            *(here->HICUMbaseBasePtr )                +=  XQbepar1_Vbe*(s->real);
            *(here->HICUMemitEmitPtr + 1)             +=  XQbepar1_Vbe*(s->imag);
            *(here->HICUMemitEmitPtr )                +=  XQbepar1_Vbe*(s->real);
            *(here->HICUMbaseEmitPtr + 1)             += -XQbepar1_Vbe*(s->imag);
            *(here->HICUMbaseEmitPtr )                += -XQbepar1_Vbe*(s->real);
            *(here->HICUMemitBasePtr + 1)             += -XQbepar1_Vbe*(s->imag);
            *(here->HICUMemitBasePtr )                += -XQbepar1_Vbe*(s->real);
            //Qbepar2
            *(here->HICUMbaseBPBaseBPPtr + 1)         +=  XQbepar2_Vbpe*(s->imag);
            *(here->HICUMbaseBPBaseBPPtr )            +=  XQbepar2_Vbpe*(s->real);
            *(here->HICUMemitEmitPtr + 1)             +=  XQbepar2_Vbpe*(s->imag);
            *(here->HICUMemitEmitPtr )                +=  XQbepar2_Vbpe*(s->real);
            *(here->HICUMemitBaseBPPtr + 1)           += -XQbepar2_Vbpe*(s->imag);
            *(here->HICUMemitBaseBPPtr )              += -XQbepar2_Vbpe*(s->real);
            *(here->HICUMbaseBPEmitPtr + 1)           += -XQbepar2_Vbpe*(s->imag);
            *(here->HICUMbaseBPEmitPtr )              += -XQbepar2_Vbpe*(s->real);
            //Qbcpar1
            *(here->HICUMbaseBasePtr + 1)             +=  XQbcpar1_Vbci*(s->imag);
            *(here->HICUMbaseBasePtr )                +=  XQbcpar1_Vbci*(s->real);
            *(here->HICUMcollCICollCIPtr + 1)         +=  XQbcpar1_Vbci*(s->imag);
            *(here->HICUMcollCICollCIPtr )            +=  XQbcpar1_Vbci*(s->real);
            *(here->HICUMbaseCollCIPtr + 1)           += -XQbcpar1_Vbci*(s->imag);
            *(here->HICUMbaseCollCIPtr )              += -XQbcpar1_Vbci*(s->real);
            *(here->HICUMcollCIBasePtr + 1)           += -XQbcpar1_Vbci*(s->imag);
            *(here->HICUMcollCIBasePtr )              += -XQbcpar1_Vbci*(s->real);
            //Qbcpar2
            *(here->HICUMbaseBPBaseBPPtr +1)          +=  XQbcpar2_Vbpci*(s->imag);
            *(here->HICUMbaseBPBaseBPPtr)             +=  XQbcpar2_Vbpci*(s->real);
            *(here->HICUMcollCICollCIPtr +1)          +=  XQbcpar2_Vbpci*(s->imag);
            *(here->HICUMcollCICollCIPtr)             +=  XQbcpar2_Vbpci*(s->real);
            *(here->HICUMbaseBPCollCIPtr +1)          += -XQbcpar2_Vbpci*(s->imag);
            *(here->HICUMbaseBPCollCIPtr)             += -XQbcpar2_Vbpci*(s->real);
            *(here->HICUMcollCIBaseBPPtr +1)          += -XQbcpar2_Vbpci*(s->imag);
            *(here->HICUMcollCIBaseBPPtr)             += -XQbcpar2_Vbpci*(s->real);
            //Qsu
            *(here->HICUMsubsSubsPtr + 1)             +=  XQsu_Vsis*(s->imag);
            *(here->HICUMsubsSubsPtr )                +=  XQsu_Vsis*(s->real);
            *(here->HICUMsubsSISubsSIPtr + 1)         +=  XQsu_Vsis*(s->imag);
            *(here->HICUMsubsSISubsSIPtr )            +=  XQsu_Vsis*(s->real);
            *(here->HICUMsubsSISubsPtr + 1)           += -XQsu_Vsis*(s->imag);
            *(here->HICUMsubsSISubsPtr )              += -XQsu_Vsis*(s->real);
            *(here->HICUMsubsSubsSIPtr + 1)           += -XQsu_Vsis*(s->imag);
            *(here->HICUMsubsSubsSIPtr )              += -XQsu_Vsis*(s->real);
            if (model->HICUMnqs) { 
                //Qxf1
                *(here->HICUMxf1Xf1Ptr + 1)               += +XQxf1_Vxf1*(s->imag);
                *(here->HICUMxf1Xf1Ptr)                   += +XQxf1_Vxf1*(s->real);
                //Qxf2
                *(here->HICUMxf2Xf2Ptr + 1)               += +XQxf2_Vxf2*(s->imag);
                *(here->HICUMxf2Xf2Ptr )                  += +XQxf2_Vxf2*(s->real);
                //Qxf
                *(here->HICUMxfXfPtr + 1)                 += +XQxf_Vxf*(s->imag);
                *(here->HICUMxfXfPtr )                    += +XQxf_Vxf*(s->real);
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
                if (here->HICUMrbi>0.0) {
//                  Stamp element: Rbi    f_Bp = +   f_Bi = -
                    *(here->HICUMbaseBPtempPtr)            +=  Ibpbi_Vrth;
                    *(here->HICUMbaseBItempPtr)            += -Ibpbi_Vrth;
                };
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
                *(here->HICUMbaseBItempPtr + 1) += +XQrbi_Vrth*(s->imag);
                *(here->HICUMbaseBItempPtr )    += +XQrbi_Vrth*(s->real);
                *(here->HICUMbaseBPtempPtr + 1) += -XQrbi_Vrth*(s->imag);
                *(here->HICUMbaseBPtempPtr )    += -XQrbi_Vrth*(s->real);
                *(here->HICUMbaseBItempPtr + 1) += +XQjei_Vrth*(s->imag);
                *(here->HICUMbaseBItempPtr )    += +XQjei_Vrth*(s->real);
                *(here->HICUMemitEItempPtr + 1) += -XQjei_Vrth*(s->imag);
                *(here->HICUMemitEItempPtr )    += -XQjei_Vrth*(s->real);
                *(here->HICUMbaseBItempPtr + 1) += +XQf_Vrth*(s->imag);
                *(here->HICUMbaseBItempPtr )    += +XQf_Vrth*(s->real);
                *(here->HICUMemitEItempPtr + 1) += -XQf_Vrth*(s->imag);
                *(here->HICUMemitEItempPtr )    += -XQf_Vrth*(s->real);
                *(here->HICUMbaseBItempPtr + 1) += +XQr_Vrth*(s->imag);
                *(here->HICUMbaseBItempPtr )    += +XQr_Vrth*(s->real);
                *(here->HICUMcollCItempPtr + 1) += -XQr_Vrth*(s->imag);
                *(here->HICUMcollCItempPtr )    += -XQr_Vrth*(s->real);
                *(here->HICUMbaseBItempPtr + 1) += +XQjci_Vrth*(s->imag);
                *(here->HICUMbaseBItempPtr )    += +XQjci_Vrth*(s->real);
                *(here->HICUMcollCItempPtr + 1) += -XQjci_Vrth*(s->imag);
                *(here->HICUMcollCItempPtr )    += -XQjci_Vrth*(s->real);
                *(here->HICUMbaseBPtempPtr + 1) += +XQjep_Vrth*(s->imag);
                *(here->HICUMbaseBPtempPtr )    += +XQjep_Vrth*(s->real);
                *(here->HICUMemitEItempPtr + 1) += -XQjep_Vrth*(s->imag);
                *(here->HICUMemitEItempPtr )    += -XQjep_Vrth*(s->real);
                *(here->HICUMbaseTempPtr   + 1) += +Xqjcx0_t_i_Vrth*(s->imag);
                *(here->HICUMbaseTempPtr   )    += +Xqjcx0_t_i_Vrth*(s->real);
                *(here->HICUMcollCItempPtr + 1) += -Xqjcx0_t_i_Vrth*(s->imag);
                *(here->HICUMcollCItempPtr )    += -Xqjcx0_t_i_Vrth*(s->real);
                *(here->HICUMbaseBPtempPtr + 1) += +Xqjcx0_t_ii_Vrth*(s->imag);
                *(here->HICUMbaseBPtempPtr )    += +Xqjcx0_t_ii_Vrth*(s->real);
                *(here->HICUMcollCItempPtr + 1) += -Xqjcx0_t_ii_Vrth*(s->imag);
                *(here->HICUMcollCItempPtr )    += -Xqjcx0_t_ii_Vrth*(s->real);
                *(here->HICUMbaseBPtempPtr + 1) += +XQdsu_Vrth*(s->imag);
                *(here->HICUMbaseBPtempPtr )    += +XQdsu_Vrth*(s->real);
                *(here->HICUMcollCItempPtr + 1) += -XQdsu_Vrth*(s->imag);
                *(here->HICUMcollCItempPtr )    += -XQdsu_Vrth*(s->real);
                *(here->HICUMsubsSItempPtr + 1) += +XQjs_Vrth*(s->imag);
                *(here->HICUMsubsSItempPtr )    += +XQjs_Vrth*(s->real);
                *(here->HICUMcollCItempPtr + 1) += -XQjs_Vrth*(s->imag);
                *(here->HICUMcollCItempPtr )    += -XQjs_Vrth*(s->real);
                *(here->HICUMsubsTempPtr   + 1) += +XQscp_Vrth*(s->imag);
                *(here->HICUMsubsTempPtr   )    += +XQscp_Vrth*(s->real);
                *(here->HICUMcollTempPtr   + 1) += -XQscp_Vrth*(s->imag);
                *(here->HICUMcollTempPtr   )    += -XQscp_Vrth*(s->real);
                *(here->HICUMtempTempPtr   + 1) += +XQcth_Vrth*(s->imag);
                *(here->HICUMtempTempPtr   )    += +XQcth_Vrth*(s->real);

            } 

        }
    }
    return(OK);
}
