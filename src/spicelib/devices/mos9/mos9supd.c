/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie

This function is obsolete (was used by an old sensitivity analysis)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS9sUpdate(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;
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
    printf("MOS9senupdate\n");
    printf("CKTtime = %.5e\n",ckt->CKTtime);
#endif /* SENSDEBUG */

    sxpgs = 0;
    sxpgd = 0;
    sxpbs = 0;
    sxpbd = 0;
    sxpgb = 0;
    dummy1 = 0;
    dummy2 = 0;

    /*  loop through all the MOS9 models */
    for( ; model != NULL; model = MOS9nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = MOS9instances(model); here != NULL ;
                here=MOS9nextInstance(here)) {

#ifdef SENSDEBUG
            printf("senupdate instance name %s\n",here->MOS9name);
            printf("before loading\n");
            printf("CKTag[0] = %.2e,CKTag[1] = %.2e\n",
                    ckt->CKTag[0],ckt->CKTag[1]);
            printf("capgs = %.7e\n",here->MOS9cgs);
            printf("capgd = %.7e\n",here->MOS9cgd);
            printf("capgb = %.7e\n",here->MOS9cgb);
            printf("capbs = %.7e\n",here->MOS9capbs);
            printf("capbd = %.7e\n",here->MOS9capbd);
#endif /* SENSDEBUG */

            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){

                sb = *(info->SEN_Sap[here->MOS9bNode] + iparmno);
                sg = *(info->SEN_Sap[here->MOS9gNode] + iparmno);
                ssprm = *(info->SEN_Sap[here->MOS9sNodePrime] + iparmno);
                sdprm = *(info->SEN_Sap[here->MOS9dNodePrime] + iparmno);
#ifdef SENSDEBUG
                printf("iparmno = %d\n",iparmno);
                printf("sb = %.7e,sg = %.7e\n",sb,sg);
                printf("ssprm = %.7e,sdprm = %.7e\n",ssprm,sdprm);
#endif /* SENSDEBUG */

                sxpgs =  (sg - ssprm) * here->MOS9cgs ;
                sxpgd =  (sg - sdprm) * here->MOS9cgd ;
                sxpgb =  (sg - sb) * here->MOS9cgb ;
                sxpbs =  (sb - ssprm) * here->MOS9capbs ;
                sxpbd =  (sb - sdprm) * here->MOS9capbd ;

                if(here->MOS9sens_l && (iparmno == here->MOS9senParmNo)){
                    sxpgs += *(here->MOS9dphigs_dl);
                    sxpgd += *(here->MOS9dphigd_dl);
                    sxpbs += *(here->MOS9dphibs_dl);
                    sxpbd += *(here->MOS9dphibd_dl);
                    sxpgb += *(here->MOS9dphigb_dl);
                }
                if(here->MOS9sens_w &&
                        (iparmno == (here->MOS9senParmNo +
                            (int) here->MOS9sens_l))){
                    sxpgs += *(here->MOS9dphigs_dw);
                    sxpgd += *(here->MOS9dphigd_dw);
                    sxpbs += *(here->MOS9dphibs_dw);
                    sxpbd += *(here->MOS9dphibd_dw);
                    sxpgb += *(here->MOS9dphigb_dw);
                }
                if(ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate1 + here->MOS9sensxpgs + 
                            10 * (iparmno - 1)) = sxpgs;
                    *(ckt->CKTstate1 + here->MOS9sensxpgd + 
                            10 * (iparmno - 1)) = sxpgd;
                    *(ckt->CKTstate1 + here->MOS9sensxpbs + 
                            10 * (iparmno - 1)) = sxpbs;
                    *(ckt->CKTstate1 + here->MOS9sensxpbd + 
                            10 * (iparmno - 1)) = sxpbd;
                    *(ckt->CKTstate1 + here->MOS9sensxpgb + 
                            10 * (iparmno - 1)) = sxpgb;
                    *(ckt->CKTstate1 + here->MOS9sensxpgs + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS9sensxpgd + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS9sensxpbs + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS9sensxpbd + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->MOS9sensxpgb + 
                            10 * (iparmno - 1) + 1) = 0;
                    goto next;
                }

                *(ckt->CKTstate0 + here->MOS9sensxpgs + 
                        10 * (iparmno - 1)) = sxpgs;
                *(ckt->CKTstate0 + here->MOS9sensxpgd + 
                        10 * (iparmno - 1)) = sxpgd;
                *(ckt->CKTstate0 + here->MOS9sensxpbs + 
                        10 * (iparmno - 1)) = sxpbs;
                *(ckt->CKTstate0 + here->MOS9sensxpbd + 
                        10 * (iparmno - 1)) = sxpbd;
                *(ckt->CKTstate0 + here->MOS9sensxpgb + 
                        10 * (iparmno - 1)) = sxpgb;

                NIintegrate(ckt,&dummy1,&dummy2,here->MOS9cgs,
                        here->MOS9sensxpgs + 10*(iparmno -1));
                NIintegrate(ckt,&dummy1,&dummy2,here->MOS9cgd,
                        here->MOS9sensxpgd + 10*(iparmno -1));
                NIintegrate(ckt,&dummy1,&dummy2,here->MOS9cgb,
                        here->MOS9sensxpgb + 10*(iparmno -1));
                NIintegrate(ckt,&dummy1,&dummy2,here->MOS9capbs,
                        here->MOS9sensxpbs + 10*(iparmno -1));
                NIintegrate(ckt,&dummy1,&dummy2,here->MOS9capbd,
                        here->MOS9sensxpbd + 10*(iparmno -1));


next:   
                ;
#ifdef SENSDEBUG
                printf("after loading\n");
                printf("sxpgs = %.7e,sdotxpgs = %.7e\n",
                        sxpgs,*(ckt->CKTstate0 + here->MOS9sensxpgs + 
                        10 * (iparmno - 1) + 1));
                printf("sxpgd = %.7e,sdotxpgd = %.7e\n",
                        sxpgd,*(ckt->CKTstate0 + here->MOS9sensxpgd + 
                        10 * (iparmno - 1) + 1));
                printf("sxpgb = %.7e,sdotxpgb = %.7e\n",
                        sxpgb,*(ckt->CKTstate0 + here->MOS9sensxpgb + 
                        10 * (iparmno - 1) + 1));
                printf("sxpbs = %.7e,sdotxpbs = %.7e\n",
                        sxpbs,*(ckt->CKTstate0 + here->MOS9sensxpbs + 
                        10 * (iparmno - 1) + 1));
                printf("sxpbd = %.7e,sdotxpbd = %.7e\n",
                        sxpbd,*(ckt->CKTstate0 + here->MOS9sensxpbd + 
                        10 * (iparmno - 1) + 1));
#endif /* SENSDEBUG */
            }
        }
    }
#ifdef SENSDEBUG
    printf("MOS9senupdate end\n");
#endif /* SENSDEBUG */
    return(OK);

}

