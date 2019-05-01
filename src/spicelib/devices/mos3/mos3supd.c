/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS3sUpdate(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel;
    MOS3instance *here;
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
    printf("MOS3senupdate\n");
    printf("CKTtime = %.5e\n",ckt->CKTtime);
#endif /* SENSDEBUG */

    sxpgs = 0;
    sxpgd = 0;
    sxpbs = 0;
    sxpbd = 0;
    sxpgb = 0;
    dummy1 = 0;
    dummy2 = 0;

    /*  loop through all the MOS3 models */
    for( ; model != NULL; model = MOS3nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS3instances(model); here != NULL ;
                here=MOS3nextInstance(here)) {

#ifdef SENSDEBUG
            printf("senupdate instance name %s\n",here->MOS3name);
            printf("before loading\n");
            printf("CKTag[0] = %.2e,CKTag[1] = %.2e\n",
                    ckt->CKTag[0],ckt->CKTag[1]);
            printf("capgs = %.7e\n",here->MOS3cgs);
            printf("capgd = %.7e\n",here->MOS3cgd);
            printf("capgb = %.7e\n",here->MOS3cgb);
            printf("capbs = %.7e\n",here->MOS3capbs);
            printf("capbd = %.7e\n",here->MOS3capbd);
#endif /* SENSDEBUG */

            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){

                sb = *(info->SEN_Sap[here->MOS3bNode] + iparmno);
                sg = *(info->SEN_Sap[here->MOS3gNode] + iparmno);
                ssprm = *(info->SEN_Sap[here->MOS3sNodePrime] + iparmno);
                sdprm = *(info->SEN_Sap[here->MOS3dNodePrime] + iparmno);
#ifdef SENSDEBUG
                printf("iparmno = %d\n",iparmno);
                printf("sb = %.7e,sg = %.7e\n",sb,sg);
                printf("ssprm = %.7e,sdprm = %.7e\n",ssprm,sdprm);
#endif /* SENSDEBUG */

                sxpgs =  (sg - ssprm) * here->MOS3cgs ;
                sxpgd =  (sg - sdprm) * here->MOS3cgd ;
                sxpgb =  (sg - sb) * here->MOS3cgb ;
                sxpbs =  (sb - ssprm) * here->MOS3capbs ;
                sxpbd =  (sb - sdprm) * here->MOS3capbd ;

                if(here->MOS3sens_l && (iparmno == here->MOS3senParmNo)){
                    sxpgs += *(here->MOS3dphigs_dl);
                    sxpgd += *(here->MOS3dphigd_dl);
                    sxpbs += *(here->MOS3dphibs_dl);
                    sxpbd += *(here->MOS3dphibd_dl);
                    sxpgb += *(here->MOS3dphigb_dl);
                }
                if(here->MOS3sens_w &&
                        (iparmno == (here->MOS3senParmNo +
                            (int) here->MOS3sens_l))){
                    sxpgs += *(here->MOS3dphigs_dw);
                    sxpgd += *(here->MOS3dphigd_dw);
                    sxpbs += *(here->MOS3dphibs_dw);
                    sxpbd += *(here->MOS3dphibd_dw);
                    sxpgb += *(here->MOS3dphigb_dw);
                }
                if(ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate1 + here->MOS3sensxpgs + 
                            10 * (iparmno - 1)) = sxpgs;
                    *(ckt->CKTstate1 + here->MOS3sensxpgd + 
                            10 * (iparmno - 1)) = sxpgd;
                    *(ckt->CKTstate1 + here->MOS3sensxpbs + 
                            10 * (iparmno - 1)) = sxpbs;
                    *(ckt->CKTstate1 + here->MOS3sensxpbd + 
                            10 * (iparmno - 1)) = sxpbd;
                    *(ckt->CKTstate1 + here->MOS3sensxpgb + 
                            10 * (iparmno - 1)) = sxpgb;
                    *(ckt->CKTstate1 + here->MOS3sensxpgs + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS3sensxpgd + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS3sensxpbs + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS3sensxpbd + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS3sensxpgb + 
                            10 * (iparmno - 1) + 1) = 0;
                    goto next;
                }

                *(ckt->CKTstate0 + here->MOS3sensxpgs + 
                        10 * (iparmno - 1)) = sxpgs;
                *(ckt->CKTstate0 + here->MOS3sensxpgd + 
                        10 * (iparmno - 1)) = sxpgd;
                *(ckt->CKTstate0 + here->MOS3sensxpbs + 
                        10 * (iparmno - 1)) = sxpbs;
                *(ckt->CKTstate0 + here->MOS3sensxpbd + 
                        10 * (iparmno - 1)) = sxpbd;
                *(ckt->CKTstate0 + here->MOS3sensxpgb + 
                        10 * (iparmno - 1)) = sxpgb;

                NIintegrate(ckt,&dummy1,&dummy2,here->MOS3cgs,
                        here->MOS3sensxpgs + 10*(iparmno -1));
                NIintegrate(ckt,&dummy1,&dummy2,here->MOS3cgd,
                        here->MOS3sensxpgd + 10*(iparmno -1));
                NIintegrate(ckt,&dummy1,&dummy2,here->MOS3cgb,
                        here->MOS3sensxpgb + 10*(iparmno -1));
                NIintegrate(ckt,&dummy1,&dummy2,here->MOS3capbs,
                        here->MOS3sensxpbs + 10*(iparmno -1));
                NIintegrate(ckt,&dummy1,&dummy2,here->MOS3capbd,
                        here->MOS3sensxpbd + 10*(iparmno -1));


next:   
                ;
#ifdef SENSDEBUG
                printf("after loading\n");
                printf("sxpgs = %.7e,sdotxpgs = %.7e\n",
                        sxpgs,*(ckt->CKTstate0 + here->MOS3sensxpgs + 
                        10 * (iparmno - 1) + 1));
                printf("sxpgd = %.7e,sdotxpgd = %.7e\n",
                        sxpgd,*(ckt->CKTstate0 + here->MOS3sensxpgd + 
                        10 * (iparmno - 1) + 1));
                printf("sxpgb = %.7e,sdotxpgb = %.7e\n",
                        sxpgb,*(ckt->CKTstate0 + here->MOS3sensxpgb + 
                        10 * (iparmno - 1) + 1));
                printf("sxpbs = %.7e,sdotxpbs = %.7e\n",
                        sxpbs,*(ckt->CKTstate0 + here->MOS3sensxpbs + 
                        10 * (iparmno - 1) + 1));
                printf("sxpbd = %.7e,sdotxpbd = %.7e\n",
                        sxpbd,*(ckt->CKTstate0 + here->MOS3sensxpbd + 
                        10 * (iparmno - 1) + 1));
#endif /* SENSDEBUG */
            }
        }
    }
#ifdef SENSDEBUG
    printf("MOS3senupdate end\n");
#endif /* SENSDEBUG */
    return(OK);

}

