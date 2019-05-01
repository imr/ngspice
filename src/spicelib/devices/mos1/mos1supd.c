/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* update the  charge sensitivities and their derivatives */

int
MOS1sUpdate(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel;
    MOS1instance *here;
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
    printf("MOS1senupdate\n");
    printf("CKTtime = %.5e\n",ckt->CKTtime);
#endif /* SENSDEBUG */

    sxpgs = 0;
    sxpgd = 0;
    sxpbs = 0;
    sxpbd = 0;
    sxpgb = 0;
    dummy1 = 0;
    dummy2 = 0;

    /*  loop through all the MOS1 models */
    for( ; model != NULL; model = MOS1nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS1instances(model); here != NULL ;
                here=MOS1nextInstance(here)) {

#ifdef SENSDEBUG
            printf("senupdate instance name %s\n",here->MOS1name);
            printf("before loading\n");
            printf("CKTag[0] = %.2e,CKTag[1] = %.2e\n",
                    ckt->CKTag[0],ckt->CKTag[1]);
            printf("capgs = %.7e\n",here->MOS1cgs);
            printf("capgd = %.7e\n",here->MOS1cgd);
            printf("capgb = %.7e\n",here->MOS1cgb);
            printf("capbs = %.7e\n",here->MOS1capbs);
            printf("capbd = %.7e\n",here->MOS1capbd);
#endif /* SENSDEBUG */

            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){

                sb = *(info->SEN_Sap[here->MOS1bNode] + iparmno);
                sg = *(info->SEN_Sap[here->MOS1gNode] + iparmno);
                ssprm = *(info->SEN_Sap[here->MOS1sNodePrime] + iparmno);
                sdprm = *(info->SEN_Sap[here->MOS1dNodePrime] + iparmno);
#ifdef SENSDEBUG
                printf("iparmno = %d\n",iparmno);
                printf("sb = %.7e,sg = %.7e\n",sb,sg);
                printf("ssprm = %.7e,sdprm = %.7e\n",ssprm,sdprm);
#endif /* SENSDEBUG */

                sxpgs =  (sg - ssprm) * here->MOS1cgs ;
                sxpgd =  (sg - sdprm) * here->MOS1cgd ;
                sxpgb =  (sg - sb) * here->MOS1cgb ;
                sxpbs =  (sb - ssprm) * here->MOS1capbs ;
                sxpbd =  (sb - sdprm) * here->MOS1capbd ;

                if(here->MOS1sens_l && (iparmno == here->MOS1senParmNo)){
                    sxpgs += *(here->MOS1dphigs_dl);
                    sxpgd += *(here->MOS1dphigd_dl);
                    sxpbs += *(here->MOS1dphibs_dl);
                    sxpbd += *(here->MOS1dphibd_dl);
                    sxpgb += *(here->MOS1dphigb_dl);
                }
                if(here->MOS1sens_w &&
                        (iparmno == (here->MOS1senParmNo +
                            (int) here->MOS1sens_l))){
                    sxpgs += *(here->MOS1dphigs_dw);
                    sxpgd += *(here->MOS1dphigd_dw);
                    sxpbs += *(here->MOS1dphibs_dw);
                    sxpbd += *(here->MOS1dphibd_dw);
                    sxpgb += *(here->MOS1dphigb_dw);
                }
                if(ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate1 + here->MOS1sensxpgs + 
                            10 * (iparmno - 1)) = sxpgs;
                    *(ckt->CKTstate1 + here->MOS1sensxpgd + 
                            10 * (iparmno - 1)) = sxpgd;
                    *(ckt->CKTstate1 + here->MOS1sensxpbs + 
                            10 * (iparmno - 1)) = sxpbs;
                    *(ckt->CKTstate1 + here->MOS1sensxpbd + 
                            10 * (iparmno - 1)) = sxpbd;
                    *(ckt->CKTstate1 + here->MOS1sensxpgb + 
                            10 * (iparmno - 1)) = sxpgb;
                    *(ckt->CKTstate1 + here->MOS1sensxpgs + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS1sensxpgd + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS1sensxpbs + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS1sensxpbd + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS1sensxpgb + 
                            10 * (iparmno - 1) + 1) = 0;
                    goto next;
                }

                *(ckt->CKTstate0 + here->MOS1sensxpgs + 
                        10 * (iparmno - 1)) = sxpgs;
                *(ckt->CKTstate0 + here->MOS1sensxpgd + 
                        10 * (iparmno - 1)) = sxpgd;
                *(ckt->CKTstate0 + here->MOS1sensxpbs + 
                        10 * (iparmno - 1)) = sxpbs;
                *(ckt->CKTstate0 + here->MOS1sensxpbd + 
                        10 * (iparmno - 1)) = sxpbd;
                *(ckt->CKTstate0 + here->MOS1sensxpgb + 
                        10 * (iparmno - 1)) = sxpgb;

                NIintegrate(ckt,&dummy1,&dummy2,here->MOS1cgs,
                        here->MOS1sensxpgs + 10*(iparmno -1));

                NIintegrate(ckt,&dummy1,&dummy2,here->MOS1cgd,
                        here->MOS1sensxpgd + 10*(iparmno -1));

                NIintegrate(ckt,&dummy1,&dummy2,here->MOS1cgb,
                        here->MOS1sensxpgb + 10*(iparmno -1));

                NIintegrate(ckt,&dummy1,&dummy2,here->MOS1capbs,
                        here->MOS1sensxpbs + 10*(iparmno -1));

                NIintegrate(ckt,&dummy1,&dummy2,here->MOS1capbd,
                        here->MOS1sensxpbd + 10*(iparmno -1));
next:   
                ;
#ifdef SENSDEBUG
                printf("after loading\n");
                printf("sxpgs = %.7e,sdotxpgs = %.7e\n",
                        sxpgs,*(ckt->CKTstate0 + here->MOS1sensxpgs + 
                        10 * (iparmno - 1) + 1));
                printf("sxpgd = %.7e,sdotxpgd = %.7e\n",
                        sxpgd,*(ckt->CKTstate0 + here->MOS1sensxpgd + 
                        10 * (iparmno - 1) + 1));
                printf("sxpgb = %.7e,sdotxpgb = %.7e\n",
                        sxpgb,*(ckt->CKTstate0 + here->MOS1sensxpgb + 
                        10 * (iparmno - 1) + 1));
                printf("sxpbs = %.7e,sdotxpbs = %.7e\n",
                        sxpbs,*(ckt->CKTstate0 + here->MOS1sensxpbs + 
                        10 * (iparmno - 1) + 1));
                printf("sxpbd = %.7e,sdotxpbd = %.7e\n",
                        sxpbd,*(ckt->CKTstate0 + here->MOS1sensxpbd + 
                        10 * (iparmno - 1) + 1));
#endif /* SENSDEBUG */
            }
        }
    }
#ifdef SENSDEBUG
    printf("MOS1senupdate end\n");
#endif /* SENSDEBUG */
    return(OK);

}

