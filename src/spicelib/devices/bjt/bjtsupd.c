/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* update the  charge sensitivities and their derivatives */

#include "ngspice.h"
#include "cktdefs.h"
#include "smpdefs.h"
#include "bjtdefs.h"
#include "const.h"
#include "sperror.h"
#include "ifsim.h"
#include "suffix.h"


int
BJTsUpdate(GENmodel *inModel, CKTcircuit *ckt)
{
    BJTmodel *model = (BJTmodel*)inModel;
    BJTinstance *here;
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
    printf("BJTsenUpdate\n");
    printf("CKTtime = %.5e\n",ckt->CKTtime);
#endif /* SENSDEBUG */
    /*  loop through all the BJT models */
    for( ; model != NULL; model = model->BJTnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->BJTinstances; here != NULL ;
                here=here->BJTnextInstance) {
	    if (here->BJTowner != ARCHme) continue;

            sxpbe = 0;
            sxpbc = 0;
            sxpcs = 0;
            sxpbx = 0;
#ifdef SENSDEBUG
            printf("senupdate Instance name: %s\n",here->BJTname);
            printf("iparmno = %d,CKTag[0] = %.2e,CKTag[1] = %.2e\n",
                    iparmno,ckt->CKTag[0],ckt->CKTag[1]);

            printf("capbe = %.7e\n",here->BJTcapbe);
            printf("capbc = %.7e\n",here->BJTcapbc);
            printf("capcs = %.7e\n",here->BJTcapcs);
            printf("capbx = %.7e\n",here->BJTcapbx);
#endif /* SENSDEBUG */

            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){
                sb = *(info->SEN_Sap[here->BJTbaseNode] + iparmno);
                sbprm = *(info->SEN_Sap[here->BJTbasePrimeNode] + iparmno);
                scprm = *(info->SEN_Sap[here->BJTcolPrimeNode] + iparmno);
                seprm = *(info->SEN_Sap[here->BJTemitPrimeNode] + iparmno);
                ss = *(info->SEN_Sap[here->BJTsubstNode] + iparmno);
#ifdef SENSDEBUG
                printf("iparmno = %d \n",iparmno);
                printf("sb = %.7e,sbprm = %.7e,scprm=%.7e\n",sb,sbprm,scprm);
                printf("seprm = %.7e,ss = %.7e\n",seprm,ss);
#endif /* SENSDEBUG */

                sxpbe =  model ->BJTtype * (sbprm - seprm)*here->BJTcapbe;

                sxpbc = model ->BJTtype *  (sbprm - scprm)*here->BJTcapbc ;

                sxpcs = model ->BJTtype *  (ss - scprm)*here->BJTcapcs ;

                sxpbx = model ->BJTtype *  (sb - scprm)*here->BJTcapbx ;
                if(iparmno == here->BJTsenParmNo){
                    sxpbe += *(here->BJTdphibedp);
                    sxpbc += *(here->BJTdphibcdp);
                    sxpcs += *(here->BJTdphicsdp);
                    sxpbx += *(here->BJTdphibxdp);
                }


                *(ckt->CKTstate0 + here->BJTsensxpbe + 8 * (iparmno - 1)) = 
                        sxpbe; 
                NIintegrate(ckt,&dummy1,&dummy2,here->BJTcapbe,
                        here->BJTsensxpbe + 8*(iparmno -1));
                *(ckt->CKTstate0 + here->BJTsensxpbc + 8 * (iparmno - 1)) = 
                        sxpbc; 
                NIintegrate(ckt,&dummy1,&dummy2,here->BJTcapbc,
                        here->BJTsensxpbc + 8*(iparmno -1));
                *(ckt->CKTstate0 + here->BJTsensxpcs + 8 * (iparmno - 1)) = 
                        sxpcs; 
                NIintegrate(ckt,&dummy1,&dummy2,here->BJTcapcs,
                        here->BJTsensxpcs + 8*(iparmno -1));
                *(ckt->CKTstate0 + here->BJTsensxpbx + 8 * (iparmno - 1)) = 
                        sxpbx; 
                NIintegrate(ckt,&dummy1,&dummy2,here->BJTcapbx,
                        here->BJTsensxpbx + 8*(iparmno -1));

#ifdef SENSDEBUG
                printf("after loading\n");
                printf("sxpbe = %.7e,sdotxpbe = %.7e\n",
                        sxpbe,*(ckt->CKTstate0 + here->BJTsensxpbe + 8 * 
                        (iparmno - 1) + 1));
                printf("sxpbc = %.7e,sdotxpbc = %.7e\n",
                        sxpbc,*(ckt->CKTstate0 + here->BJTsensxpbc + 8 * 
                        (iparmno - 1) + 1));
                printf("sxpcs = %.7e,sdotxpsc = %.7e\n",
                        sxpcs,*(ckt->CKTstate0 + here->BJTsensxpcs + 8 * 
                        (iparmno - 1) + 1));
                printf("sxpbx = %.7e,sdotxpbx = %.7e\n",
                        sxpbx,*(ckt->CKTstate0 + here->BJTsensxpbx + 8 * 
                        (iparmno - 1) + 1));
                printf("\n");
#endif /* SENSDEBUG */
                if(ckt->CKTmode & MODEINITTRAN) {
                    *(ckt->CKTstate1 + here->BJTsensxpbe + 8 * (iparmno - 1)) =
                            sxpbe;
                    *(ckt->CKTstate1 + here->BJTsensxpbc + 8 * (iparmno - 1)) =
                            sxpbc;
                    *(ckt->CKTstate1 + here->BJTsensxpcs + 8 * (iparmno - 1)) =
                            sxpcs;
                    *(ckt->CKTstate1 + here->BJTsensxpbx + 8 * (iparmno - 1)) =
                            sxpbx;
                    *(ckt->CKTstate1 + here->BJTsensxpbe + 8 * (iparmno - 1) +
                            1) = 0;
                    *(ckt->CKTstate1 + here->BJTsensxpbc + 8 * (iparmno - 1) + 
                            1) = 0;
                    *(ckt->CKTstate1 + here->BJTsensxpcs + 8 * (iparmno - 1) + 
                            1) = 0;
                    *(ckt->CKTstate1 + here->BJTsensxpbx + 8 * (iparmno - 1) + 
                            1) = 0;
                }

            }
        }
    }
#ifdef SENSDEBUG
    printf("BJTsenUpdate end\n");
#endif /* SENSDEBUG */
    return(OK);
}

