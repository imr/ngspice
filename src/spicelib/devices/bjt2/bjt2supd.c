/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* update the  charge sensitivities and their derivatives */

#include "ngspice.h"
#include "cktdefs.h"
#include "smpdefs.h"
#include "bjt2defs.h"
#include "const.h"
#include "sperror.h"
#include "ifsim.h"
#include "suffix.h"


int
BJT2sUpdate(GENmodel *inModel, CKTcircuit *ckt)
{
    BJT2model *model = (BJT2model*)inModel;
    BJT2instance *here;
    int    iparmno;
    double sb;
    double sbprm;
    double scprm;
    double seprm;
    double ss;
    double sxpbe;
    double sxpbc;
    double sxpcs;
    double sxpbx;
    double dummy1;
    double dummy2;
    SENstruct *info;

    info = ckt->CKTsenInfo;
    if(ckt->CKTtime == 0) return(OK);
#ifdef SENSDEBUG
    printf("BJT2senUpdate\n");
    printf("CKTtime = %.5e\n",ckt->CKTtime);
#endif /* SENSDEBUG */
    /*  loop through all the BJT2 models */
    for( ; model != NULL; model = model->BJT2nextModel ) {

        /* loop through all the instances of the model */
        for (here = model->BJT2instances; here != NULL ;
                here=here->BJT2nextInstance) {
            if (here->BJT2owner != ARCHme) continue;

            sxpbe = 0;
            sxpbc = 0;
            sxpcs = 0;
            sxpbx = 0;
#ifdef SENSDEBUG
            printf("senupdate Instance name: %s\n",here->BJT2name);
            printf("iparmno = %d,CKTag[0] = %.2e,CKTag[1] = %.2e\n",
                    iparmno,ckt->CKTag[0],ckt->CKTag[1]);

            printf("capbe = %.7e\n",here->BJT2capbe);
            printf("capbc = %.7e\n",here->BJT2capbc);
            printf("capcs = %.7e\n",here->BJT2capsub);
            printf("capbx = %.7e\n",here->BJT2capbx);
#endif /* SENSDEBUG */

            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){
                sb = *(info->SEN_Sap[here->BJT2baseNode] + iparmno);
                sbprm = *(info->SEN_Sap[here->BJT2basePrimeNode] + iparmno);
                scprm = *(info->SEN_Sap[here->BJT2colPrimeNode] + iparmno);
                seprm = *(info->SEN_Sap[here->BJT2emitPrimeNode] + iparmno);
                ss = *(info->SEN_Sap[here->BJT2substNode] + iparmno);
#ifdef SENSDEBUG
                printf("iparmno = %d \n",iparmno);
                printf("sb = %.7e,sbprm = %.7e,scprm=%.7e\n",sb,sbprm,scprm);
                printf("seprm = %.7e,ss = %.7e\n",seprm,ss);
#endif /* SENSDEBUG */

                sxpbe =  model ->BJT2type * (sbprm - seprm)*here->BJT2capbe;

                sxpbc = model ->BJT2type *  (sbprm - scprm)*here->BJT2capbc ;

                sxpcs = model ->BJT2type *  (ss - scprm)*here->BJT2capsub ;

                sxpbx = model ->BJT2type *  (sb - scprm)*here->BJT2capbx ;
                if(iparmno == here->BJT2senParmNo){
                    sxpbe += *(here->BJT2dphibedp);
                    sxpbc += *(here->BJT2dphibcdp);
                    sxpcs += *(here->BJT2dphisubdp);
                    sxpbx += *(here->BJT2dphibxdp);
                }


                *(ckt->CKTstate0 + here->BJT2sensxpbe + 8 * (iparmno - 1)) = 
                        sxpbe; 
                NIintegrate(ckt,&dummy1,&dummy2,here->BJT2capbe,
                        here->BJT2sensxpbe + 8*(iparmno -1));
                *(ckt->CKTstate0 + here->BJT2sensxpbc + 8 * (iparmno - 1)) = 
                        sxpbc; 
                NIintegrate(ckt,&dummy1,&dummy2,here->BJT2capbc,
                        here->BJT2sensxpbc + 8*(iparmno -1));
                *(ckt->CKTstate0 + here->BJT2sensxpsub + 8 * (iparmno - 1)) = 
                        sxpcs; 
                NIintegrate(ckt,&dummy1,&dummy2,here->BJT2capsub,
                        here->BJT2sensxpsub + 8*(iparmno -1));
                *(ckt->CKTstate0 + here->BJT2sensxpbx + 8 * (iparmno - 1)) = 
                        sxpbx; 
                NIintegrate(ckt,&dummy1,&dummy2,here->BJT2capbx,
                        here->BJT2sensxpbx + 8*(iparmno -1));

#ifdef SENSDEBUG
                printf("after loading\n");
                printf("sxpbe = %.7e,sdotxpbe = %.7e\n",
                        sxpbe,*(ckt->CKTstate0 + here->BJT2sensxpbe + 8 * 
                        (iparmno - 1) + 1));
                printf("sxpbc = %.7e,sdotxpbc = %.7e\n",
                        sxpbc,*(ckt->CKTstate0 + here->BJT2sensxpbc + 8 * 
                        (iparmno - 1) + 1));
                printf("sxpcs = %.7e,sdotxpsc = %.7e\n",
                        sxpcs,*(ckt->CKTstate0 + here->BJT2sensxpsub + 8 * 
                        (iparmno - 1) + 1));
                printf("sxpbx = %.7e,sdotxpbx = %.7e\n",
                        sxpbx,*(ckt->CKTstate0 + here->BJT2sensxpbx + 8 * 
                        (iparmno - 1) + 1));
                printf("\n");
#endif /* SENSDEBUG */
                if(ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate1 + here->BJT2sensxpbe + 8 * (iparmno - 1)) =
                            sxpbe;
                    *(ckt->CKTstate1 + here->BJT2sensxpbc + 8 * (iparmno - 1)) =
                            sxpbc;
                    *(ckt->CKTstate1 + here->BJT2sensxpsub + 8 * (iparmno - 1)) =
                            sxpcs;
                    *(ckt->CKTstate1 + here->BJT2sensxpbx + 8 * (iparmno - 1)) =
                            sxpbx;
                    *(ckt->CKTstate1 + here->BJT2sensxpbe + 8 * (iparmno - 1) +
                            1) = 0;
                    *(ckt->CKTstate1 + here->BJT2sensxpbc + 8 * (iparmno - 1) + 
                            1) = 0;
                    *(ckt->CKTstate1 + here->BJT2sensxpsub + 8 * (iparmno - 1) + 
                            1) = 0;
                    *(ckt->CKTstate1 + here->BJT2sensxpbx + 8 * (iparmno - 1) + 
                            1) = 0;
                }

            }
        }
    }
#ifdef SENSDEBUG
    printf("BJT2senUpdate end\n");
#endif /* SENSDEBUG */
    return(OK);
}

