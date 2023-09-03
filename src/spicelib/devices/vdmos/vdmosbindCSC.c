/**********
Author: 2020 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

int
VDMOSbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    VDMOSmodel *model = (VDMOSmodel *)inModel ;
    VDMOSinstance *here ;
    BindElement i, *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->SMPkluMatrix->KLUmatrixBindStructCOO ;
    nz = (size_t)ckt->CKTmatrix->SMPkluMatrix->KLUmatrixLinkedListNZ ;

    /* loop through all the VDMOS models */
    for ( ; model != NULL ; model = VDMOSnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = VDMOSinstances(model); here != NULL ; here = VDMOSnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(VDMOSDdPtr,   VDMOSDdBinding,   VDMOSdNode,      VDMOSdNode);
            CREATE_KLU_BINDING_TABLE(VDMOSGgPtr,   VDMOSGgBinding,   VDMOSgNode,      VDMOSgNode);
            CREATE_KLU_BINDING_TABLE(VDMOSSsPtr,   VDMOSSsBinding,   VDMOSsNode,      VDMOSsNode);
            CREATE_KLU_BINDING_TABLE(VDMOSDPdpPtr, VDMOSDPdpBinding, VDMOSdNodePrime, VDMOSdNodePrime);
            CREATE_KLU_BINDING_TABLE(VDMOSSPspPtr, VDMOSSPspBinding, VDMOSsNodePrime, VDMOSsNodePrime);
            CREATE_KLU_BINDING_TABLE(VDMOSGPgpPtr, VDMOSGPgpBinding, VDMOSgNodePrime, VDMOSgNodePrime);
            CREATE_KLU_BINDING_TABLE(VDMOSDdpPtr,  VDMOSDdpBinding,  VDMOSdNode,      VDMOSdNodePrime);
            CREATE_KLU_BINDING_TABLE(VDMOSGPdpPtr, VDMOSGPdpBinding, VDMOSgNodePrime, VDMOSdNodePrime);
            CREATE_KLU_BINDING_TABLE(VDMOSGPspPtr, VDMOSGPspBinding, VDMOSgNodePrime, VDMOSsNodePrime);
            CREATE_KLU_BINDING_TABLE(VDMOSSspPtr,  VDMOSSspBinding,  VDMOSsNode,      VDMOSsNodePrime);
            CREATE_KLU_BINDING_TABLE(VDMOSDPspPtr, VDMOSDPspBinding, VDMOSdNodePrime, VDMOSsNodePrime);
            CREATE_KLU_BINDING_TABLE(VDMOSDPdPtr,  VDMOSDPdBinding,  VDMOSdNodePrime, VDMOSdNode);
            CREATE_KLU_BINDING_TABLE(VDMOSDPgpPtr, VDMOSDPgpBinding, VDMOSdNodePrime, VDMOSgNodePrime);
            CREATE_KLU_BINDING_TABLE(VDMOSSPgpPtr, VDMOSSPgpBinding, VDMOSsNodePrime, VDMOSgNodePrime);
            CREATE_KLU_BINDING_TABLE(VDMOSSPsPtr,  VDMOSSPsBinding,  VDMOSsNodePrime, VDMOSsNode);
            CREATE_KLU_BINDING_TABLE(VDMOSSPdpPtr, VDMOSSPdpBinding, VDMOSsNodePrime, VDMOSdNodePrime);

            CREATE_KLU_BINDING_TABLE(VDMOSGgpPtr, VDMOSGgpBinding, VDMOSgNode,      VDMOSgNodePrime);
            CREATE_KLU_BINDING_TABLE(VDMOSGPgPtr, VDMOSGPgBinding, VDMOSgNodePrime, VDMOSgNode);

            CREATE_KLU_BINDING_TABLE(VDMOSDsPtr, VDMOSDsBinding, VDMOSdNode, VDMOSsNode);
            CREATE_KLU_BINDING_TABLE(VDMOSSdPtr, VDMOSSdBinding, VDMOSsNode, VDMOSdNode);

            CREATE_KLU_BINDING_TABLE(VDIORPdPtr,  VDIORPdBinding,  VDIOposPrimeNode, VDMOSdNode);
            CREATE_KLU_BINDING_TABLE(VDIODrpPtr,  VDIODrpBinding,  VDMOSdNode,       VDIOposPrimeNode);
            CREATE_KLU_BINDING_TABLE(VDIOSrpPtr,  VDIOSrpBinding,  VDMOSsNode,       VDIOposPrimeNode);
            CREATE_KLU_BINDING_TABLE(VDIORPsPtr,  VDIORPsBinding,  VDIOposPrimeNode, VDMOSsNode);
            CREATE_KLU_BINDING_TABLE(VDIORPrpPtr, VDIORPrpBinding, VDIOposPrimeNode, VDIOposPrimeNode);

            if ((here->VDMOSthermal) && (model->VDMOSrthjcGiven)) {
                CREATE_KLU_BINDING_TABLE(VDMOSTemptempPtr, VDMOSTemptempBinding, VDMOStempNode,   VDMOStempNode);  /* Transistor thermal contribution */
                CREATE_KLU_BINDING_TABLE(VDMOSTempdpPtr,   VDMOSTempdpBinding,   VDMOStempNode,   VDMOSdNodePrime);
                CREATE_KLU_BINDING_TABLE(VDMOSTempspPtr,   VDMOSTempspBinding,   VDMOStempNode,   VDMOSsNodePrime);
                CREATE_KLU_BINDING_TABLE(VDMOSTempgpPtr,   VDMOSTempgpBinding,   VDMOStempNode,   VDMOSgNodePrime);
                CREATE_KLU_BINDING_TABLE(VDMOSGPtempPtr,   VDMOSGPtempBinding,   VDMOSgNodePrime, VDMOStempNode);
                CREATE_KLU_BINDING_TABLE(VDMOSDPtempPtr,   VDMOSDPtempBinding,   VDMOSdNodePrime, VDMOStempNode);
                CREATE_KLU_BINDING_TABLE(VDMOSSPtempPtr,   VDMOSSPtempBinding,   VDMOSsNodePrime, VDMOStempNode);

                CREATE_KLU_BINDING_TABLE(VDIOTempposPrimePtr, VDIOTempposPrimeBinding, VDMOStempNode,    VDIOposPrimeNode);/* Diode thermal contribution */
                CREATE_KLU_BINDING_TABLE(VDMOSTempdPtr,       VDMOSTempdBinding,       VDMOStempNode,    VDMOSdNode);
                CREATE_KLU_BINDING_TABLE(VDIOPosPrimetempPtr, VDIOPosPrimetempBinding, VDIOposPrimeNode, VDMOStempNode);
                CREATE_KLU_BINDING_TABLE(VDMOSDtempPtr,       VDMOSDtempBinding,       VDMOSdNode,       VDMOStempNode);
                CREATE_KLU_BINDING_TABLE(VDMOStempSPtr,       VDMOStempSBinding,       VDMOStempNode,    VDMOSsNode);
                CREATE_KLU_BINDING_TABLE(VDMOSSTempPtr,       VDMOSSTempBinding,       VDMOSsNode,       VDMOStempNode);

                CREATE_KLU_BINDING_TABLE(VDMOSTcasetcasePtr, VDMOSTcasetcaseBinding, VDMOStcaseNode,   VDMOStcaseNode);   /* Rthjc between tj and tcase*/
                CREATE_KLU_BINDING_TABLE(VDMOSTcasetempPtr,  VDMOSTcasetempBinding,  VDMOStcaseNode,   VDMOStempNode);
                CREATE_KLU_BINDING_TABLE(VDMOSTemptcasePtr,  VDMOSTemptcaseBinding,  VDMOStempNode,    VDMOStcaseNode);
                CREATE_KLU_BINDING_TABLE(VDMOSTptpPtr,       VDMOSTptpBinding,       VDMOStNodePrime,  VDMOStNodePrime);  /* Rthca between tcase and Vsrc */
                CREATE_KLU_BINDING_TABLE(VDMOSTptcasePtr,    VDMOSTptcaseBinding,    VDMOStNodePrime,  VDMOStempNode);
                CREATE_KLU_BINDING_TABLE(VDMOSTcasetpPtr,    VDMOSTcasetpBinding,    VDMOStempNode,    VDMOStNodePrime);
                CREATE_KLU_BINDING_TABLE(VDMOSCktTcktTPtr,   VDMOSCktTcktTBinding,   VDMOSvcktTbranch, VDMOSvcktTbranch); /* Vsrc=cktTemp to gnd */
                CREATE_KLU_BINDING_TABLE(VDMOSCktTtpPtr,     VDMOSCktTtpBinding,     VDMOSvcktTbranch, VDMOStNodePrime);
                CREATE_KLU_BINDING_TABLE(VDMOSTpcktTPtr,     VDMOSTpcktTBinding,     VDMOStNodePrime,  VDMOSvcktTbranch);
            }
        }
    }

    return (OK) ;
}

int
VDMOSbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    VDMOSmodel *model = (VDMOSmodel *)inModel ;
    VDMOSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VDMOS models */
    for ( ; model != NULL ; model = VDMOSnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = VDMOSinstances(model); here != NULL ; here = VDMOSnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSDdPtr,   VDMOSDdBinding,   VDMOSdNode,      VDMOSdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSGgPtr,   VDMOSGgBinding,   VDMOSgNode,      VDMOSgNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSSsPtr,   VDMOSSsBinding,   VDMOSsNode,      VDMOSsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSDPdpPtr, VDMOSDPdpBinding, VDMOSdNodePrime, VDMOSdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSSPspPtr, VDMOSSPspBinding, VDMOSsNodePrime, VDMOSsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSGPgpPtr, VDMOSGPgpBinding, VDMOSgNodePrime, VDMOSgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSDdpPtr,  VDMOSDdpBinding,  VDMOSdNode,      VDMOSdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSGPdpPtr, VDMOSGPdpBinding, VDMOSgNodePrime, VDMOSdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSGPspPtr, VDMOSGPspBinding, VDMOSgNodePrime, VDMOSsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSSspPtr,  VDMOSSspBinding,  VDMOSsNode,      VDMOSsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSDPspPtr, VDMOSDPspBinding, VDMOSdNodePrime, VDMOSsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSDPdPtr,  VDMOSDPdBinding,  VDMOSdNodePrime, VDMOSdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSDPgpPtr, VDMOSDPgpBinding, VDMOSdNodePrime, VDMOSgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSSPgpPtr, VDMOSSPgpBinding, VDMOSsNodePrime, VDMOSgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSSPsPtr,  VDMOSSPsBinding,  VDMOSsNodePrime, VDMOSsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSSPdpPtr, VDMOSSPdpBinding, VDMOSsNodePrime, VDMOSdNodePrime);

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSGgpPtr, VDMOSGgpBinding, VDMOSgNode,      VDMOSgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSGPgPtr, VDMOSGPgBinding, VDMOSgNodePrime, VDMOSgNode);

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSDsPtr, VDMOSDsBinding, VDMOSdNode, VDMOSsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSSdPtr, VDMOSSdBinding, VDMOSsNode, VDMOSdNode);

            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDIORPdPtr,  VDIORPdBinding,  VDIOposPrimeNode, VDMOSdNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDIODrpPtr,  VDIODrpBinding,  VDMOSdNode,       VDIOposPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDIOSrpPtr,  VDIOSrpBinding,  VDMOSsNode,       VDIOposPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDIORPsPtr,  VDIORPsBinding,  VDIOposPrimeNode, VDMOSsNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDIORPrpPtr, VDIORPrpBinding, VDIOposPrimeNode, VDIOposPrimeNode);

            if ((here->VDMOSthermal) && (model->VDMOSrthjcGiven)) {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSTemptempPtr, VDMOSTemptempBinding, VDMOStempNode,   VDMOStempNode);  /* Transistor thermal contribution */
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSTempdpPtr,   VDMOSTempdpBinding,   VDMOStempNode,   VDMOSdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSTempspPtr,   VDMOSTempspBinding,   VDMOStempNode,   VDMOSsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSTempgpPtr,   VDMOSTempgpBinding,   VDMOStempNode,   VDMOSgNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSGPtempPtr,   VDMOSGPtempBinding,   VDMOSgNodePrime, VDMOStempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSDPtempPtr,   VDMOSDPtempBinding,   VDMOSdNodePrime, VDMOStempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSSPtempPtr,   VDMOSSPtempBinding,   VDMOSsNodePrime, VDMOStempNode);

                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDIOTempposPrimePtr, VDIOTempposPrimeBinding, VDMOStempNode,    VDIOposPrimeNode);/* Diode thermal contribution */
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSTempdPtr,       VDMOSTempdBinding,       VDMOStempNode,    VDMOSdNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDIOPosPrimetempPtr, VDIOPosPrimetempBinding, VDIOposPrimeNode, VDMOStempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSDtempPtr,       VDMOSDtempBinding,       VDMOSdNode,       VDMOStempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOStempSPtr,       VDMOStempSBinding,       VDMOStempNode,    VDMOSsNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSSTempPtr,       VDMOSSTempBinding,       VDMOSsNode,       VDMOStempNode);

                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSTcasetcasePtr, VDMOSTcasetcaseBinding, VDMOStcaseNode,   VDMOStcaseNode);   /* Rthjc between tj and tcase*/
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSTcasetempPtr,  VDMOSTcasetempBinding,  VDMOStcaseNode,   VDMOStempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSTemptcasePtr,  VDMOSTemptcaseBinding,  VDMOStempNode,    VDMOStcaseNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSTptpPtr,       VDMOSTptpBinding,       VDMOStNodePrime,  VDMOStNodePrime);  /* Rthca between tcase and Vsrc */
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSTptcasePtr,    VDMOSTptcaseBinding,    VDMOStNodePrime,  VDMOStempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSTcasetpPtr,    VDMOSTcasetpBinding,    VDMOStempNode,    VDMOStNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSCktTcktTPtr,   VDMOSCktTcktTBinding,   VDMOSvcktTbranch, VDMOSvcktTbranch); /* Vsrc=cktTemp to gnd */
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSCktTtpPtr,     VDMOSCktTtpBinding,     VDMOSvcktTbranch, VDMOStNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VDMOSTpcktTPtr,     VDMOSTpcktTBinding,     VDMOStNodePrime,  VDMOSvcktTbranch);
            }
        }
    }

    return (OK) ;
}

int
VDMOSbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    VDMOSmodel *model = (VDMOSmodel *)inModel ;
    VDMOSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VDMOS models */
    for ( ; model != NULL ; model = VDMOSnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = VDMOSinstances(model); here != NULL ; here = VDMOSnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSDdPtr,   VDMOSDdBinding,   VDMOSdNode,      VDMOSdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSGgPtr,   VDMOSGgBinding,   VDMOSgNode,      VDMOSgNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSSsPtr,   VDMOSSsBinding,   VDMOSsNode,      VDMOSsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSDPdpPtr, VDMOSDPdpBinding, VDMOSdNodePrime, VDMOSdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSSPspPtr, VDMOSSPspBinding, VDMOSsNodePrime, VDMOSsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSGPgpPtr, VDMOSGPgpBinding, VDMOSgNodePrime, VDMOSgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSDdpPtr,  VDMOSDdpBinding,  VDMOSdNode,      VDMOSdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSGPdpPtr, VDMOSGPdpBinding, VDMOSgNodePrime, VDMOSdNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSGPspPtr, VDMOSGPspBinding, VDMOSgNodePrime, VDMOSsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSSspPtr,  VDMOSSspBinding,  VDMOSsNode,      VDMOSsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSDPspPtr, VDMOSDPspBinding, VDMOSdNodePrime, VDMOSsNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSDPdPtr,  VDMOSDPdBinding,  VDMOSdNodePrime, VDMOSdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSDPgpPtr, VDMOSDPgpBinding, VDMOSdNodePrime, VDMOSgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSSPgpPtr, VDMOSSPgpBinding, VDMOSsNodePrime, VDMOSgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSSPsPtr,  VDMOSSPsBinding,  VDMOSsNodePrime, VDMOSsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSSPdpPtr, VDMOSSPdpBinding, VDMOSsNodePrime, VDMOSdNodePrime);

            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSGgpPtr, VDMOSGgpBinding, VDMOSgNode,      VDMOSgNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSGPgPtr, VDMOSGPgBinding, VDMOSgNodePrime, VDMOSgNode);

            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSDsPtr, VDMOSDsBinding, VDMOSdNode, VDMOSsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSSdPtr, VDMOSSdBinding, VDMOSsNode, VDMOSdNode);

            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDIORPdPtr,  VDIORPdBinding,  VDIOposPrimeNode, VDMOSdNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDIODrpPtr,  VDIODrpBinding,  VDMOSdNode,       VDIOposPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDIOSrpPtr,  VDIOSrpBinding,  VDMOSsNode,       VDIOposPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDIORPsPtr,  VDIORPsBinding,  VDIOposPrimeNode, VDMOSsNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VDIORPrpPtr, VDIORPrpBinding, VDIOposPrimeNode, VDIOposPrimeNode);

            if ((here->VDMOSthermal) && (model->VDMOSrthjcGiven)) {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSTemptempPtr, VDMOSTemptempBinding, VDMOStempNode,   VDMOStempNode);  /* Transistor thermal contribution */
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSTempdpPtr,   VDMOSTempdpBinding,   VDMOStempNode,   VDMOSdNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSTempspPtr,   VDMOSTempspBinding,   VDMOStempNode,   VDMOSsNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSTempgpPtr,   VDMOSTempgpBinding,   VDMOStempNode,   VDMOSgNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSGPtempPtr,   VDMOSGPtempBinding,   VDMOSgNodePrime, VDMOStempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSDPtempPtr,   VDMOSDPtempBinding,   VDMOSdNodePrime, VDMOStempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSSPtempPtr,   VDMOSSPtempBinding,   VDMOSsNodePrime, VDMOStempNode);

                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDIOTempposPrimePtr, VDIOTempposPrimeBinding, VDMOStempNode,    VDIOposPrimeNode);/* Diode thermal contribution */
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSTempdPtr,       VDMOSTempdBinding,       VDMOStempNode,    VDMOSdNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDIOPosPrimetempPtr, VDIOPosPrimetempBinding, VDIOposPrimeNode, VDMOStempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSDtempPtr,       VDMOSDtempBinding,       VDMOSdNode,       VDMOStempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOStempSPtr,       VDMOStempSBinding,       VDMOStempNode,    VDMOSsNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSSTempPtr,       VDMOSSTempBinding,       VDMOSsNode,       VDMOStempNode);

                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSTcasetcasePtr, VDMOSTcasetcaseBinding, VDMOStcaseNode,   VDMOStcaseNode);   /* Rthjc between tj and tcase*/
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSTcasetempPtr,  VDMOSTcasetempBinding,  VDMOStcaseNode,   VDMOStempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSTemptcasePtr,  VDMOSTemptcaseBinding,  VDMOStempNode,    VDMOStcaseNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSTptpPtr,       VDMOSTptpBinding,       VDMOStNodePrime,  VDMOStNodePrime);  /* Rthca between tcase and Vsrc */
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSTptcasePtr,    VDMOSTptcaseBinding,    VDMOStNodePrime,  VDMOStempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSTcasetpPtr,    VDMOSTcasetpBinding,    VDMOStempNode,    VDMOStNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSCktTcktTPtr,   VDMOSCktTcktTBinding,   VDMOSvcktTbranch, VDMOSvcktTbranch); /* Vsrc=cktTemp to gnd */
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSCktTtpPtr,     VDMOSCktTtpBinding,     VDMOSvcktTbranch, VDMOStNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(VDMOSTpcktTPtr,     VDMOSTpcktTBinding,     VDMOStNodePrime,  VDMOSvcktTbranch);
            }
        }
    }

    return (OK) ;
}
