/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* update the  charge sensitivities and their derivatives */

int
VDMOSsUpdate(GENmodel *inModel, CKTcircuit *ckt)
{
    VDMOSmodel *model = (VDMOSmodel *)inModel;
    VDMOSinstance *here;
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
    printf("VDMOSsenupdate\n");
    printf("CKTtime = %.5e\n",ckt->CKTtime);
#endif /* SENSDEBUG */

    sxpgs = 0;
    sxpgd = 0;
    sxpbs = 0;
    sxpbd = 0;
    sxpgb = 0;
    dummy1 = 0;
    dummy2 = 0;

    /*  loop through all the VDMOS models */
    for( ; model != NULL; model = VDMOSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VDMOSinstances(model); here != NULL ;
                here=VDMOSnextInstance(here)) {

#ifdef SENSDEBUG
            printf("senupdate instance name %s\n",here->VDMOSname);
            printf("before loading\n");
            printf("CKTag[0] = %.2e,CKTag[1] = %.2e\n",
                    ckt->CKTag[0],ckt->CKTag[1]);
            printf("capgs = %.7e\n",here->VDMOScgs);
            printf("capgd = %.7e\n",here->VDMOScgd);
            printf("capgb = %.7e\n",here->VDMOScgb);
            printf("capbs = %.7e\n",here->VDMOScapbs);
            printf("capbd = %.7e\n",here->VDMOScapbd);
#endif /* SENSDEBUG */

            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){

                sb = *(info->SEN_Sap[here->VDMOSbNode] + iparmno);
                sg = *(info->SEN_Sap[here->VDMOSgNode] + iparmno);
                ssprm = *(info->SEN_Sap[here->VDMOSsNodePrime] + iparmno);
                sdprm = *(info->SEN_Sap[here->VDMOSdNodePrime] + iparmno);
#ifdef SENSDEBUG
                printf("iparmno = %d\n",iparmno);
                printf("sb = %.7e,sg = %.7e\n",sb,sg);
                printf("ssprm = %.7e,sdprm = %.7e\n",ssprm,sdprm);
#endif /* SENSDEBUG */

                sxpgs =  (sg - ssprm) * here->VDMOScgs ;
                sxpgd =  (sg - sdprm) * here->VDMOScgd ;
                sxpgb =  (sg - sb) * here->VDMOScgb ;
                sxpbs =  (sb - ssprm) * here->VDMOScapbs ;
                sxpbd =  (sb - sdprm) * here->VDMOScapbd ;

                if(here->VDMOSsens_l && (iparmno == here->VDMOSsenParmNo)){
                    sxpgs += *(here->VDMOSdphigs_dl);
                    sxpgd += *(here->VDMOSdphigd_dl);
                    sxpbs += *(here->VDMOSdphibs_dl);
                    sxpbd += *(here->VDMOSdphibd_dl);
                    sxpgb += *(here->VDMOSdphigb_dl);
                }
                if(here->VDMOSsens_w && 
                        (iparmno == (here->VDMOSsenParmNo+here->VDMOSsens_l))){
                    sxpgs += *(here->VDMOSdphigs_dw);
                    sxpgd += *(here->VDMOSdphigd_dw);
                    sxpbs += *(here->VDMOSdphibs_dw);
                    sxpbd += *(here->VDMOSdphibd_dw);
                    sxpgb += *(here->VDMOSdphigb_dw);
                }
                if(ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate1 + here->VDMOSsensxpgs + 
                            10 * (iparmno - 1)) = sxpgs;
                    *(ckt->CKTstate1 + here->VDMOSsensxpgd + 
                            10 * (iparmno - 1)) = sxpgd;
                    *(ckt->CKTstate1 + here->VDMOSsensxpbs + 
                            10 * (iparmno - 1)) = sxpbs;
                    *(ckt->CKTstate1 + here->VDMOSsensxpbd + 
                            10 * (iparmno - 1)) = sxpbd;
                    *(ckt->CKTstate1 + here->VDMOSsensxpgb + 
                            10 * (iparmno - 1)) = sxpgb;
                    *(ckt->CKTstate1 + here->VDMOSsensxpgs + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->VDMOSsensxpgd + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->VDMOSsensxpbs + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->VDMOSsensxpbd + 
                            10 * (iparmno - 1) + 1) = 0;
                    *(ckt->CKTstate1 + here->VDMOSsensxpgb + 
                            10 * (iparmno - 1) + 1) = 0;
                    goto next;
                }

                *(ckt->CKTstate0 + here->VDMOSsensxpgs + 
                        10 * (iparmno - 1)) = sxpgs;
                *(ckt->CKTstate0 + here->VDMOSsensxpgd + 
                        10 * (iparmno - 1)) = sxpgd;
                *(ckt->CKTstate0 + here->VDMOSsensxpbs + 
                        10 * (iparmno - 1)) = sxpbs;
                *(ckt->CKTstate0 + here->VDMOSsensxpbd + 
                        10 * (iparmno - 1)) = sxpbd;
                *(ckt->CKTstate0 + here->VDMOSsensxpgb + 
                        10 * (iparmno - 1)) = sxpgb;

                NIintegrate(ckt,&dummy1,&dummy2,here->VDMOScgs,
                        here->VDMOSsensxpgs + 10*(iparmno -1));

                NIintegrate(ckt,&dummy1,&dummy2,here->VDMOScgd,
                        here->VDMOSsensxpgd + 10*(iparmno -1));

                NIintegrate(ckt,&dummy1,&dummy2,here->VDMOScgb,
                        here->VDMOSsensxpgb + 10*(iparmno -1));

                NIintegrate(ckt,&dummy1,&dummy2,here->VDMOScapbs,
                        here->VDMOSsensxpbs + 10*(iparmno -1));

                NIintegrate(ckt,&dummy1,&dummy2,here->VDMOScapbd,
                        here->VDMOSsensxpbd + 10*(iparmno -1));
next:   
                ;
#ifdef SENSDEBUG
                printf("after loading\n");
                printf("sxpgs = %.7e,sdotxpgs = %.7e\n",
                        sxpgs,*(ckt->CKTstate0 + here->VDMOSsensxpgs + 
                        10 * (iparmno - 1) + 1));
                printf("sxpgd = %.7e,sdotxpgd = %.7e\n",
                        sxpgd,*(ckt->CKTstate0 + here->VDMOSsensxpgd + 
                        10 * (iparmno - 1) + 1));
                printf("sxpgb = %.7e,sdotxpgb = %.7e\n",
                        sxpgb,*(ckt->CKTstate0 + here->VDMOSsensxpgb + 
                        10 * (iparmno - 1) + 1));
                printf("sxpbs = %.7e,sdotxpbs = %.7e\n",
                        sxpbs,*(ckt->CKTstate0 + here->VDMOSsensxpbs + 
                        10 * (iparmno - 1) + 1));
                printf("sxpbd = %.7e,sdotxpbd = %.7e\n",
                        sxpbd,*(ckt->CKTstate0 + here->VDMOSsensxpbd + 
                        10 * (iparmno - 1) + 1));
#endif /* SENSDEBUG */
            }
        }
    }
#ifdef SENSDEBUG
    printf("VDMOSsenupdate end\n");
#endif /* SENSDEBUG */
    return(OK);

}

