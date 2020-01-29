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
#include "hicumdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
HICUMpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    HICUMinstance *here;
    HICUMmodel *model = (HICUMmodel*)inModel;
    double Ibpei_Vbpei;
    double Ibiei_Vbiei;
    double Ibici_Vbici;
    double Ibpci_Vbpci;
    double Isici_Vsici;
    double Iciei_Vbiei;
    double Iciei_Vbici;
    double Ibbp_Vbbp;
    double Isis_Vsis;
    double Ieie_Veie;
    double Ibpbi_Vbpbi, Ibpbi_Vciei, Ibpbi_Vbiei;
    double Ibpsi_Vbpci, Ibpsi_Vsici;
    double Icic_Vcic;

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

            Ibbp_Vbbp    = 1/here->HICUMrbx_t;
            Icic_Vcic    = 1/here->HICUMrcx_t;
            Ieie_Veie    = 1/here->HICUMre_t;
            Isis_Vsis    = 1/model->HICUMrsu;

            Ibiei_Vbiei = *(ckt->CKTstate0 + here->HICUMibiei_Vbiei);
            Ibpei_Vbpei = *(ckt->CKTstate0 + here->HICUMibpei_Vbpei);
            Iciei_Vbiei = *(ckt->CKTstate0 + here->HICUMiciei_Vbiei);
            Iciei_Vbici = *(ckt->CKTstate0 + here->HICUMiciei_Vbici);
            Ibici_Vbici = *(ckt->CKTstate0 + here->HICUMibici_Vbici);
            Ibpbi_Vbpbi = *(ckt->CKTstate0 + here->HICUMibpbi_Vbpbi);
            Ibpbi_Vbiei = *(ckt->CKTstate0 + here->HICUMibpbi_Vbiei);
            Ibpbi_Vciei = *(ckt->CKTstate0 + here->HICUMibpbi_Vbici);
            Isici_Vsici = *(ckt->CKTstate0 + here->HICUMisici_Vsici);
            Ibpsi_Vbpci = *(ckt->CKTstate0 + here->HICUMibpsi_Vbpci);
            Ibpsi_Vsici = *(ckt->CKTstate0 + here->HICUMibpsi_Vsici);
            Ibpci_Vbpci = *(ckt->CKTstate0 + here->HICUMibpci_Vbpci);

/*
c           The real part
*/
/*
c           Stamp element: Ibiei
*/
            *(here->HICUMbaseBIBaseBIPtr) +=  Ibiei_Vbiei;
            *(here->HICUMbaseBIEmitEIPtr) += -Ibiei_Vbiei;
            *(here->HICUMemitEIBaseBIPtr) += -Ibiei_Vbiei;
            *(here->HICUMemitEIEmitEIPtr) +=  Ibiei_Vbiei;
/*
c           Stamp element: Ibpei
*/
            *(here->HICUMbaseBPBaseBPPtr) +=  Ibpei_Vbpei;
            *(here->HICUMbaseBPEmitEIPtr) += -Ibpei_Vbpei;
            *(here->HICUMemitEIBaseBPPtr) += -Ibpei_Vbpei;
            *(here->HICUMemitEIEmitEIPtr) +=  Ibpei_Vbpei;

/*
c           Stamp element: Iciei
*/
            *(here->HICUMcollCIBaseBIPtr) +=  Iciei_Vbiei;
            *(here->HICUMcollCIEmitEIPtr) += -Iciei_Vbiei;
            *(here->HICUMemitEIBaseBIPtr) += -Iciei_Vbiei;
            *(here->HICUMemitEIEmitEIPtr) +=  Iciei_Vbiei;

            *(here->HICUMcollCIBaseBIPtr) +=  Iciei_Vbici;
            *(here->HICUMcollCICollCIPtr) += -Iciei_Vbici;
            *(here->HICUMemitEIBaseBIPtr) += -Iciei_Vbici;
            *(here->HICUMemitEICollCIPtr) +=  Iciei_Vbici;
/*
c           Stamp element: Ibici
*/
            *(here->HICUMbaseBIBaseBIPtr) +=  Ibici_Vbici;
            *(here->HICUMbaseBICollCIPtr) += -Ibici_Vbici;
            *(here->HICUMcollCIBaseBIPtr) += -Ibici_Vbici;
            *(here->HICUMcollCICollCIPtr) +=  Ibici_Vbici;
/*
c           Stamp element: Ibpci
*/
            *(here->HICUMbaseBPCollCIPtr) +=  Ibpci_Vbpci;
            *(here->HICUMbaseBPBaseBPPtr) += -Ibpci_Vbpci;
            *(here->HICUMcollCIBaseBPPtr) += -Ibpci_Vbpci;
            *(here->HICUMcollCICollCIPtr) +=  Ibpci_Vbpci;
/*
c           Stamp element: Rcx
*/
            *(here->HICUMcollCollPtr)     +=  Icic_Vcic;
            *(here->HICUMcollCICollPtr)   += -Icic_Vcic;
            *(here->HICUMcollCollCIPtr)   += -Icic_Vcic;
            *(here->HICUMcollCICollCIPtr) +=  Icic_Vcic;
/*
c           Stamp element: Rbx
*/
            *(here->HICUMbaseBasePtr)     +=  Ibbp_Vbbp;
            *(here->HICUMbaseBPBasePtr)   += -Ibbp_Vbbp;
            *(here->HICUMbaseBaseBPPtr)   += -Ibbp_Vbbp;
            *(here->HICUMbaseBPBaseBPPtr) +=  Ibbp_Vbbp;
/*
c           Stamp element: Ibpbi
*/
            *(here->HICUMbaseBPBaseBPPtr) +=  Ibpbi_Vbpbi;
            *(here->HICUMbaseBPBaseBIPtr) += -Ibpbi_Vbpbi;
            *(here->HICUMbaseBPBaseBIPtr) +=  Ibpbi_Vbiei;
            *(here->HICUMbaseBPEmitEIPtr) += -Ibpbi_Vbiei;
            *(here->HICUMbaseBPCollCIPtr) +=  Ibpbi_Vciei;
            *(here->HICUMbaseBPEmitEIPtr) += -Ibpbi_Vciei;
            *(here->HICUMbaseBIBaseBPPtr) += -Ibpbi_Vbpbi;
            *(here->HICUMbaseBIBaseBIPtr) +=  Ibpbi_Vbpbi;
            *(here->HICUMbaseBIBaseBIPtr) += -Ibpbi_Vbiei;
            *(here->HICUMbaseBIEmitEIPtr) +=  Ibpbi_Vbiei;
            *(here->HICUMbaseBICollCIPtr) += -Ibpbi_Vciei;
            *(here->HICUMbaseBIEmitEIPtr) +=  Ibpbi_Vciei;
/*
c           Stamp element: Re
*/
            *(here->HICUMemitEmitPtr)     +=  Ieie_Veie;
            *(here->HICUMemitEIEmitPtr)   += -Ieie_Veie;
            *(here->HICUMemitEmitEIPtr)   += -Ieie_Veie;
            *(here->HICUMemitEIEmitEIPtr) +=  Ieie_Veie;
/*
c           Stamp element: Isici
*/
            *(here->HICUMsubsSISubsSIPtr) +=  Isici_Vsici;
            *(here->HICUMsubsSICollCIPtr) += -Isici_Vsici;
            *(here->HICUMcollCISubsSIPtr) += -Isici_Vsici;
            *(here->HICUMcollCICollCIPtr) +=  Isici_Vsici;
/*
c           Stamp element: Ibpsi
*/
            *(here->HICUMbaseBPBaseBPPtr) +=  Ibpsi_Vbpci;
            *(here->HICUMbaseBPCollCIPtr) += -Ibpsi_Vbpci;
            *(here->HICUMbaseBPSubsSIPtr) +=  Ibpsi_Vsici;
            *(here->HICUMbaseBPCollCIPtr) += -Ibpsi_Vsici;
            *(here->HICUMsubsSIBaseBPPtr) += -Ibpsi_Vbpci;
            *(here->HICUMsubsSICollCIPtr) +=  Ibpsi_Vbpci;
            *(here->HICUMsubsSISubsSIPtr) += -Ibpsi_Vsici;
            *(here->HICUMsubsSICollCIPtr) +=  Ibpsi_Vsici;
/*
c           Stamp element: Rs
*/
            *(here->HICUMsubsSubsPtr)     +=  Isis_Vsis;
            *(here->HICUMsubsSISubsPtr)   += -Isis_Vsis;
            *(here->HICUMsubsSubsSIPtr)   += -Isis_Vsis;
            *(here->HICUMsubsSISubsSIPtr) +=  Isis_Vsis;
/*
c           The complex part
*/
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
            *(here->HICUMbaseBasePtr)         +=  XQbepar1_Vbe * (s->real);
            *(here->HICUMbaseEmitPtr)         += -XQbepar1_Vbe * (s->real);
            *(here->HICUMemitBasePtr)         += -XQbepar1_Vbe * (s->real);
            *(here->HICUMemitEmitPtr)         +=  XQbepar1_Vbe * (s->real);
            *(here->HICUMbaseBasePtr + 1)     +=  XQbepar1_Vbe * (s->imag);
            *(here->HICUMbaseEmitPtr + 1)     += -XQbepar1_Vbe * (s->imag);
            *(here->HICUMemitBasePtr + 1)     += -XQbepar1_Vbe * (s->imag);
            *(here->HICUMemitEmitPtr + 1)     +=  XQbepar1_Vbe * (s->imag);
/*
c           Stamp element: Qbepar2
*/
            *(here->HICUMbaseBPBaseBPPtr)     +=  XQbepar2_Vbpe * (s->real);
            *(here->HICUMemitBaseBPPtr)       += -XQbepar2_Vbpe * (s->real);
            *(here->HICUMemitEmitPtr)         += -XQbepar2_Vbpe * (s->real);
            *(here->HICUMbaseBPEmitPtr)       +=  XQbepar2_Vbpe * (s->real);
            *(here->HICUMbaseBPBaseBPPtr + 1) +=  XQbepar2_Vbpe * (s->imag);
            *(here->HICUMemitBaseBPPtr + 1)   += -XQbepar2_Vbpe * (s->imag);
            *(here->HICUMemitEmitPtr + 1)     += -XQbepar2_Vbpe * (s->imag);
            *(here->HICUMbaseBPEmitPtr + 1)   +=  XQbepar2_Vbpe * (s->imag);
/*
c           Stamp element: Qdeix, Qjei
*/
            *(here->HICUMbaseBIBaseBIPtr)     +=  XQdeix_Vbiei * (s->real);
            *(here->HICUMbaseBIEmitEIPtr)     += -XQdeix_Vbiei * (s->real);
            *(here->HICUMemitEIBaseBIPtr)     += -XQdeix_Vbiei * (s->real);
            *(here->HICUMemitEIEmitEIPtr)     +=  XQdeix_Vbiei * (s->real);
            *(here->HICUMbaseBIBaseBIPtr + 1) +=  XQdeix_Vbiei * (s->imag);
            *(here->HICUMbaseBIEmitEIPtr + 1) += -XQdeix_Vbiei * (s->imag);
            *(here->HICUMemitEIBaseBIPtr + 1) += -XQdeix_Vbiei * (s->imag);
            *(here->HICUMemitEIEmitEIPtr + 1) +=  XQdeix_Vbiei * (s->imag);
            *(here->HICUMbaseBIBaseBIPtr)     +=  XQjei_Vbiei * (s->real);
            *(here->HICUMbaseBIEmitEIPtr)     += -XQjei_Vbiei * (s->real);
            *(here->HICUMemitEIBaseBIPtr)     += -XQjei_Vbiei * (s->real);
            *(here->HICUMemitEIEmitEIPtr)     +=  XQjei_Vbiei * (s->real);
            *(here->HICUMbaseBIBaseBIPtr + 1) +=  XQjei_Vbiei * (s->imag);
            *(here->HICUMbaseBIEmitEIPtr + 1) += -XQjei_Vbiei * (s->imag);
            *(here->HICUMemitEIBaseBIPtr + 1) += -XQjei_Vbiei * (s->imag);
            *(here->HICUMemitEIEmitEIPtr + 1) +=  XQjei_Vbiei * (s->imag);
/*
c           Stamp element: Qjep
*/
            *(here->HICUMbaseBPBaseBPPtr)     +=  XQjep_Vbpei * (s->real);
            *(here->HICUMbaseBPEmitEIPtr)     += -XQjep_Vbpei * (s->real);
            *(here->HICUMemitEIBaseBPPtr)     += -XQjep_Vbpei * (s->real);
            *(here->HICUMemitEIEmitEIPtr)     +=  XQjep_Vbpei * (s->real);
            *(here->HICUMbaseBPBaseBPPtr + 1) +=  XQjep_Vbpei * (s->imag);
            *(here->HICUMbaseBPEmitEIPtr + 1) += -XQjep_Vbpei * (s->imag);
            *(here->HICUMemitEIBaseBPPtr + 1) += -XQjep_Vbpei * (s->imag);
            *(here->HICUMemitEIEmitEIPtr + 1) +=  XQjep_Vbpei * (s->imag);
/*
c           Stamp element: Qdci, Qjci
*/
            *(here->HICUMbaseBIBaseBIPtr)     +=  XQdci_Vbici * (s->real);
            *(here->HICUMbaseBICollCIPtr)     += -XQdci_Vbici * (s->real);
            *(here->HICUMcollCIBaseBIPtr)     += -XQdci_Vbici * (s->real);
            *(here->HICUMcollCICollCIPtr)     +=  XQdci_Vbici * (s->real);
            *(here->HICUMbaseBIBaseBIPtr + 1) +=  XQdci_Vbici * (s->imag);
            *(here->HICUMbaseBICollCIPtr + 1) += -XQdci_Vbici * (s->imag);
            *(here->HICUMcollCIBaseBIPtr + 1) += -XQdci_Vbici * (s->imag);
            *(here->HICUMcollCICollCIPtr + 1) +=  XQdci_Vbici * (s->imag);
            *(here->HICUMbaseBIBaseBIPtr)     +=  XQjci_Vbici * (s->real);
            *(here->HICUMbaseBICollCIPtr)     += -XQjci_Vbici * (s->real);
            *(here->HICUMcollCIBaseBIPtr)     += -XQjci_Vbici * (s->real);
            *(here->HICUMcollCICollCIPtr)     +=  XQjci_Vbici * (s->real);
            *(here->HICUMbaseBIBaseBIPtr + 1) +=  XQjci_Vbici * (s->imag);
            *(here->HICUMbaseBICollCIPtr + 1) += -XQjci_Vbici * (s->imag);
            *(here->HICUMcollCIBaseBIPtr + 1) += -XQjci_Vbici * (s->imag);
            *(here->HICUMcollCICollCIPtr + 1) +=  XQjci_Vbici * (s->imag);
/*
c           Stamp element: Qbcpar1, qjcx0_i
*/
            *(here->HICUMbaseBasePtr)         +=  XQbcpar1_Vbci * (s->real);
            *(here->HICUMbaseCollCIPtr)       += -XQbcpar1_Vbci * (s->real);
            *(here->HICUMcollCIBasePtr)       += -XQbcpar1_Vbci * (s->real);
            *(here->HICUMcollCICollCIPtr)     +=  XQbcpar1_Vbci * (s->real);
            *(here->HICUMbaseBasePtr + 1)     +=  XQbcpar1_Vbci * (s->imag);
            *(here->HICUMbaseCollCIPtr + 1)   += -XQbcpar1_Vbci * (s->imag);
            *(here->HICUMcollCIBasePtr + 1)   += -XQbcpar1_Vbci * (s->imag);
            *(here->HICUMcollCICollCIPtr + 1) +=  XQbcpar1_Vbci * (s->imag);
            *(here->HICUMbaseBasePtr)         +=  Xqjcx0_t_i_Vbci * (s->real);
            *(here->HICUMbaseCollCIPtr)       += -Xqjcx0_t_i_Vbci * (s->real);
            *(here->HICUMcollCIBasePtr)       += -Xqjcx0_t_i_Vbci * (s->real);
            *(here->HICUMcollCICollCIPtr)     +=  Xqjcx0_t_i_Vbci * (s->real);
            *(here->HICUMbaseBasePtr + 1)     +=  Xqjcx0_t_i_Vbci * (s->imag);
            *(here->HICUMbaseCollCIPtr + 1)   += -Xqjcx0_t_i_Vbci * (s->imag);
            *(here->HICUMcollCIBasePtr + 1)   += -Xqjcx0_t_i_Vbci * (s->imag);
            *(here->HICUMcollCICollCIPtr + 1) +=  Xqjcx0_t_i_Vbci * (s->imag);
/*
c           Stamp element: Qbcpar2, qjcx0_ii, Qdsu
*/
            *(here->HICUMbaseBPBaseBPPtr)     +=  XQbcpar2_Vbpci * (s->real);
            *(here->HICUMcollCICollCIPtr)     +=  XQbcpar2_Vbpci * (s->real);
            *(here->HICUMbaseBPCollCIPtr)     += -XQbcpar2_Vbpci * (s->real);
            *(here->HICUMcollCIBaseBPPtr)     += -XQbcpar2_Vbpci * (s->real);
            *(here->HICUMbaseBPBaseBPPtr + 1) +=  XQbcpar2_Vbpci * (s->imag);
            *(here->HICUMcollCICollCIPtr + 1) +=  XQbcpar2_Vbpci * (s->imag);
            *(here->HICUMbaseBPCollCIPtr + 1) += -XQbcpar2_Vbpci * (s->imag);
            *(here->HICUMcollCIBaseBPPtr + 1) += -XQbcpar2_Vbpci * (s->imag);
            *(here->HICUMbaseBPCollCIPtr)     +=  Xqjcx0_t_ii_Vbpci * (s->real);
            *(here->HICUMbaseBPBaseBPPtr)     += -Xqjcx0_t_ii_Vbpci * (s->real);
            *(here->HICUMcollCIBaseBPPtr)     += -Xqjcx0_t_ii_Vbpci * (s->real);
            *(here->HICUMcollCICollCIPtr)     +=  Xqjcx0_t_ii_Vbpci * (s->real);
            *(here->HICUMbaseBPCollCIPtr + 1) +=  Xqjcx0_t_ii_Vbpci * (s->imag);
            *(here->HICUMbaseBPBaseBPPtr + 1) += -Xqjcx0_t_ii_Vbpci * (s->imag);
            *(here->HICUMcollCIBaseBPPtr + 1) += -Xqjcx0_t_ii_Vbpci * (s->imag);
            *(here->HICUMcollCICollCIPtr + 1) +=  Xqjcx0_t_ii_Vbpci * (s->imag);
            *(here->HICUMbaseBPCollCIPtr)     +=  XQdsu_Vbpci * (s->real);
            *(here->HICUMbaseBPBaseBPPtr)     += -XQdsu_Vbpci * (s->real);
            *(here->HICUMcollCIBaseBPPtr)     += -XQdsu_Vbpci * (s->real);
            *(here->HICUMcollCICollCIPtr)     +=  XQdsu_Vbpci * (s->real);
            *(here->HICUMbaseBPCollCIPtr + 1) +=  XQdsu_Vbpci * (s->imag);
            *(here->HICUMbaseBPBaseBPPtr + 1) += -XQdsu_Vbpci * (s->imag);
            *(here->HICUMcollCIBaseBPPtr + 1) += -XQdsu_Vbpci * (s->imag);
            *(here->HICUMcollCICollCIPtr + 1) +=  XQdsu_Vbpci * (s->imag);
/*
c           Stamp element: Qrbi
*/
            *(here->HICUMbaseBPBaseBPPtr)     +=  XQrbi_Vbpbi * (s->real);
            *(here->HICUMbaseBPBaseBIPtr)     += -XQrbi_Vbpbi * (s->real);
            *(here->HICUMbaseBIBaseBPPtr)     += -XQrbi_Vbpbi * (s->real);
            *(here->HICUMbaseBIBaseBIPtr)     +=  XQrbi_Vbpbi * (s->real);
            *(here->HICUMbaseBPBaseBPPtr + 1) +=  XQrbi_Vbpbi * (s->imag);
            *(here->HICUMbaseBPBaseBIPtr + 1) += -XQrbi_Vbpbi * (s->imag);
            *(here->HICUMbaseBIBaseBPPtr + 1) += -XQrbi_Vbpbi * (s->imag);
            *(here->HICUMbaseBIBaseBIPtr + 1) +=  XQrbi_Vbpbi * (s->imag);
//todo:
//            *(here->HICUMbaseBPBaseBIPtr)     +=  XQrbi_Vbiei * (s->real);
//            *(here->HICUMbaseBPEmitEIPtr)     += -XQrbi_Vbiei * (s->real);
//            *(here->HICUMbaseBIBaseBIPtr)     += -XQrbi_Vbiei * (s->real);
//            *(here->HICUMbaseBIEmitEIPtr)     +=  XQrbi_Vbiei * (s->real);
//            *(here->HICUMbaseBPBaseBIPtr + 1) +=  XQrbi_Vbiei * (s->imag);
//            *(here->HICUMbaseBPEmitEIPtr + 1) += -XQrbi_Vbiei * (s->imag);
//            *(here->HICUMbaseBIBaseBIPtr + 1) += -XQrbi_Vbiei * (s->imag);
//            *(here->HICUMbaseBIEmitEIPtr + 1) +=  XQrbi_Vbiei * (s->imag);
//            *(here->HICUMbaseBPCollCIPtr)     +=  XQrbi_Vbici * (s->real);
//            *(here->HICUMbaseBPEmitEIPtr)     += -XQrbi_Vbici * (s->real);
//            *(here->HICUMbaseBICollCIPtr)     += -XQrbi_Vbici * (s->real);
//            *(here->HICUMbaseBIEmitEIPtr)     +=  XQrbi_Vbici * (s->real);
//            *(here->HICUMbaseBPCollCIPtr + 1) +=  XQrbi_Vbici * (s->imag);
//            *(here->HICUMbaseBPEmitEIPtr + 1) += -XQrbi_Vbici * (s->imag);
//            *(here->HICUMbaseBICollCIPtr + 1) += -XQrbi_Vbici * (s->imag);
//            *(here->HICUMbaseBIEmitEIPtr + 1) +=  XQrbi_Vbici * (s->imag);
/*
c           Stamp element: Cscp
*/
            *(here->HICUMsubsSubsPtr)         +=  XQscp_Vsc * (s->real);
            *(here->HICUMcollSubsPtr)         += -XQscp_Vsc * (s->real);
            *(here->HICUMcollCollPtr)         += -XQscp_Vsc * (s->real);
            *(here->HICUMsubsCollPtr)         +=  XQscp_Vsc * (s->real);
            *(here->HICUMsubsSubsPtr + 1)     +=  XQscp_Vsc * (s->imag);
            *(here->HICUMcollSubsPtr + 1)     += -XQscp_Vsc * (s->imag);
            *(here->HICUMcollCollPtr + 1)     += -XQscp_Vsc * (s->imag);
            *(here->HICUMsubsCollPtr + 1)     +=  XQscp_Vsc * (s->imag);
/*
c           Stamp element: Cjs
*/
            *(here->HICUMsubsSISubsSIPtr)     +=  XQjs_Vsici * (s->real);
            *(here->HICUMsubsSICollCIPtr)     += -XQjs_Vsici * (s->real);
            *(here->HICUMcollCISubsSIPtr)     += -XQjs_Vsici * (s->real);
            *(here->HICUMcollCICollCIPtr)     +=  XQjs_Vsici * (s->real);
            *(here->HICUMsubsSISubsSIPtr + 1) +=  XQjs_Vsici * (s->imag);
            *(here->HICUMsubsSICollCIPtr + 1) += -XQjs_Vsici * (s->imag);
            *(here->HICUMcollCISubsSIPtr + 1) += -XQjs_Vsici * (s->imag);
            *(here->HICUMcollCICollCIPtr + 1) +=  XQjs_Vsici * (s->imag);
/*
c           Stamp element: Csu
*/
            *(here->HICUMsubsSubsPtr)         +=  XQsu_Vsis * (s->real);
            *(here->HICUMsubsSISubsPtr)       += -XQsu_Vsis * (s->real);
            *(here->HICUMsubsSubsSIPtr)       += -XQsu_Vsis * (s->real);
            *(here->HICUMsubsSISubsSIPtr)     +=  XQsu_Vsis * (s->real);
            *(here->HICUMsubsSubsPtr + 1)     +=  XQsu_Vsis * (s->imag);
            *(here->HICUMsubsSISubsPtr + 1)   += -XQsu_Vsis * (s->imag);
            *(here->HICUMsubsSubsSIPtr + 1)   += -XQsu_Vsis * (s->imag);
            *(here->HICUMsubsSISubsSIPtr + 1) +=  XQsu_Vsis * (s->imag);

        }
    }
    return(OK);
}
