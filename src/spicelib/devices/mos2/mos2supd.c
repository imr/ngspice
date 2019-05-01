/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* update the  charge sensitivities and their derivatives */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS2sUpdate(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;
    int    iparmno;
    double sb;
    double sg;
    double sdprm;
    double ssprm;
    double sxpgs;
    double sxpgd;
    double sxpbs;
    double sxpbd;
    double sxpgb;
    double dummy1;
    double dummy2;
    SENstruct *info;


    if(ckt->CKTtime == 0) return(OK);
    info = ckt->CKTsenInfo;

#ifdef SENSDEBUG
    printf("MOS2senupdate\n");
    printf("CKTtime = %.5e\n",ckt->CKTtime);
#endif /* SENSDEBUG */

    sxpgs = 0;
    sxpgd = 0;
    sxpbs = 0;
    sxpbd = 0;
    sxpgb = 0;
    dummy1 = 0;
    dummy2 = 0;

    /*  loop through all the MOS2 models */
    for( ; model != NULL; model = MOS2nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS2instances(model); here != NULL ;
                here=MOS2nextInstance(here)) {

#ifdef SENSDEBUG
            printf("senupdate instance name %s\n",here->MOS2name);
            printf("before loading\n");
            printf("CKTag[0] = %.2e,CKTag[1] = %.2e\n",
                    ckt->CKTag[0],ckt->CKTag[1]);
            printf("capgs = %.7e\n",here->MOS2cgs);
            printf("capgd = %.7e\n",here->MOS2cgd);
            printf("capgb = %.7e\n",here->MOS2cgb);
            printf("capbs = %.7e\n",here->MOS2capbs);
            printf("capbd = %.7e\n",here->MOS2capbd);
#endif /* SENSDEBUG */

            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){

                sb = *(info->SEN_Sap[here->MOS2bNode] + iparmno);
                sg = *(info->SEN_Sap[here->MOS2gNode] + iparmno);
                ssprm = *(info->SEN_Sap[here->MOS2sNodePrime] + iparmno);
                sdprm = *(info->SEN_Sap[here->MOS2dNodePrime] + iparmno);
#ifdef SENSDEBUG
                printf("iparmno = %d\n",iparmno);
                printf("sb = %.7e,sg = %.7e\n",sb,sg);
                printf("ssprm = %.7e,sdprm = %.7e\n",ssprm,sdprm);
#endif /* SENSDEBUG */

                sxpgs =  (sg - ssprm) * here->MOS2cgs ;
                sxpgd =  (sg - sdprm) * here->MOS2cgd ;
                sxpgb =  (sg - sb) * here->MOS2cgb ;
                sxpbs =  (sb - ssprm) * here->MOS2capbs ;
                sxpbd =  (sb - sdprm) * here->MOS2capbd ;

                if(here->MOS2sens_l && (iparmno == here->MOS2senParmNo)){
                    sxpgs += *(here->MOS2dphigs_dl);
                    sxpgd += *(here->MOS2dphigd_dl);
                    sxpbs += *(here->MOS2dphibs_dl);
                    sxpbd += *(here->MOS2dphibd_dl);
                    sxpgb += *(here->MOS2dphigb_dl);
                }
                if(here->MOS2sens_w && 
                        (iparmno == (here->MOS2senParmNo +
                            (int) here->MOS2sens_l))){
                    sxpgs += *(here->MOS2dphigs_dw);
                    sxpgd += *(here->MOS2dphigd_dw);
                    sxpbs += *(here->MOS2dphibs_dw);
                    sxpbd += *(here->MOS2dphibd_dw);
                    sxpgb += *(here->MOS2dphigb_dw);
                }
                if(ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate1 + here->MOS2sensxpgs + 
                            10 * (iparmno - 1)) = sxpgs;
                    *(ckt->CKTstate1 + here->MOS2sensxpgd + 
                            10 * (iparmno - 1)) = sxpgd;
                    *(ckt->CKTstate1 + here->MOS2sensxpbs + 
                            10 * (iparmno - 1)) = sxpbs;
                    *(ckt->CKTstate1 + here->MOS2sensxpbd + 
                            10 * (iparmno - 1)) = sxpbd;
                    *(ckt->CKTstate1 + here->MOS2sensxpgb + 
                            10 * (iparmno - 1)) = sxpgb;
                    *(ckt->CKTstate1 + here->MOS2sensxpgs + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS2sensxpgd + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS2sensxpbs + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS2sensxpbd + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS2sensxpgb + 
                            10 * (iparmno - 1) + 1) = 0;
                    goto next;
                }

                *(ckt->CKTstate0 + here->MOS2sensxpgs + 
                        10 * (iparmno - 1)) = sxpgs;
                *(ckt->CKTstate0 + here->MOS2sensxpgd + 
                        10 * (iparmno - 1)) = sxpgd;
                *(ckt->CKTstate0 + here->MOS2sensxpbs + 
                        10 * (iparmno - 1)) = sxpbs;
                *(ckt->CKTstate0 + here->MOS2sensxpbd + 
                        10 * (iparmno - 1)) = sxpbd;
                *(ckt->CKTstate0 + here->MOS2sensxpgb + 
                        10 * (iparmno - 1)) = sxpgb;

                NIintegrate(ckt,&dummy1,&dummy2,here->MOS2cgs,
                        here->MOS2sensxpgs + 10*(iparmno -1));
                NIintegrate(ckt,&dummy1,&dummy2,here->MOS2cgd,
                        here->MOS2sensxpgd + 10*(iparmno -1));
                NIintegrate(ckt,&dummy1,&dummy2,here->MOS2cgb,
                        here->MOS2sensxpgb + 10*(iparmno -1));
                NIintegrate(ckt,&dummy1,&dummy2,here->MOS2capbs,
                        here->MOS2sensxpbs + 10*(iparmno -1));
                NIintegrate(ckt,&dummy1,&dummy2,here->MOS2capbd,
                        here->MOS2sensxpbd + 10*(iparmno -1));
next:   
                ;
#ifdef SENSDEBUG
                printf("after loading\n");
                printf("sxpgs = %.7e,sdotxpgs = %.7e\n",
                        sxpgs,*(ckt->CKTstate0 + here->MOS2sensxpgs + 
                        10 * (iparmno - 1) + 1));
                printf("sxpgd = %.7e,sdotxpgd = %.7e\n",
                        sxpgd,*(ckt->CKTstate0 + here->MOS2sensxpgd + 
                        10 * (iparmno - 1) + 1));
                printf("sxpgb = %.7e,sdotxpgb = %.7e\n",
                        sxpgb,*(ckt->CKTstate0 + here->MOS2sensxpgb + 
                        10 * (iparmno - 1) + 1));
                printf("sxpbs = %.7e,sdotxpbs = %.7e\n",
                        sxpbs,*(ckt->CKTstate0 + here->MOS2sensxpbs + 
                        10 * (iparmno - 1) + 1));
                printf("sxpbd = %.7e,sdotxpbd = %.7e\n",
                        sxpbd,*(ckt->CKTstate0 + here->MOS2sensxpbd + 
                        10 * (iparmno - 1) + 1));
#endif /* SENSDEBUG */
            }
        }
    }
#ifdef SENSDEBUG
    printf("MOS2senupdate end\n");
#endif /* SENSDEBUG */
    return(OK);

}

