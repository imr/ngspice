/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "tradefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
TRAaccept(CKTcircuit *ckt, GENmodel *inModel)
{
    TRAmodel *model = (TRAmodel *)inModel;
    TRAinstance *here;
    int i=0,j;
    double v1,v2,v3,v4;
    double v5,v6,d1,d2,d3,d4;
    double *from,*to;
    int error;


    /*  loop through all the transmission line models */
    for( ; model != NULL; model = TRAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = TRAinstances(model); here != NULL ;
                here=TRAnextInstance(here)) {
            
            if( (ckt->CKTtime - here->TRAtd) > *(here->TRAdelays+6)) {
                /* shift! */
                for(i=2;i<here->TRAsizeDelay && 
                    (ckt->CKTtime - here->TRAtd > *(here->TRAdelays+3*i));i++)
                        { /* loop does it all */ ; }
                i -= 2;
                for(j=i;j<=here->TRAsizeDelay;j++) {
                    from = here->TRAdelays + 3*j;
                    to = here->TRAdelays + 3*(j-i);
                    *(to) = *(from);
                    *(to+1) = *(from+1);
                    *(to+2) = *(from+2);
                }
                here->TRAsizeDelay -= i;
            }
            if(ckt->CKTtime - *(here->TRAdelays+3*here->TRAsizeDelay) >
                    ckt->CKTminBreak) {
                if(here->TRAallocDelay <= here->TRAsizeDelay) {
                    /* need to grab some more space */
                    here->TRAallocDelay += 5;
                    here->TRAdelays = TREALLOC(double, here->TRAdelays, (here->TRAallocDelay + 1) * 3);
                }
                here->TRAsizeDelay ++;
                to = (here->TRAdelays  +3*here->TRAsizeDelay);
                *to = ckt->CKTtime;
                to = (here->TRAdelays+1+3*here->TRAsizeDelay);
                *to = ( *(ckt->CKTrhsOld + here->TRAposNode2)
                     -*(ckt->CKTrhsOld + here->TRAnegNode2))
                    + *(ckt->CKTrhsOld + here->TRAbrEq2)*
                        here->TRAimped;
                *(here->TRAdelays+2+3*here->TRAsizeDelay) = 
                    ( *(ckt->CKTrhsOld + here->TRAposNode1)
                     -*(ckt->CKTrhsOld + here->TRAnegNode1))
                    + *(ckt->CKTrhsOld + here->TRAbrEq1)*
                        here->TRAimped;
#ifdef NOTDEF
                v1 = *(here->TRAdelays+1+3*here->TRAsizeDelay);
                v2 = *(here->TRAdelays+1+3*(here->TRAsizeDelay-1));
                v3 = *(here->TRAdelays+2+3*here->TRAsizeDelay);
                v4 = *(here->TRAdelays+2+3*(here->TRAsizeDelay-1));
                if( (fabs(v1-v2) >= 50*ckt->CKTreltol*
                        MAX(fabs(v1),fabs(v2))+50*ckt->CKTvoltTol) ||
                    (fabs(v3-v4) >= 50*ckt->CKTreltol*
                        MAX(fabs(v3),fabs(v4))+50*ckt->CKTvoltTol)  ) {
                    /* changing - need to schedule after delay */
                    /*printf("%s:  at %g set for %g and %g\n",here->TRAname,
                        ckt->CKTtime,
                        ckt->CKTtime+here->TRAtd,
                        *(here->TRAdelays+3*here->TRAsizeDelay-3)+
                                here->TRAtd);*/
                    error = CKTsetBreak(ckt,ckt->CKTtime+here->TRAtd);
                    if(error) return(error);
                    /* also set for break after PREVIOUS point */
                    error = CKTsetBreak(ckt,
                            *(here->TRAdelays+3*here->TRAsizeDelay -3) +
                            here->TRAtd);
                    CKTbreakDump(ckt);
                    if(error) return(error);
                }
#else
                v1 = *(here->TRAdelays+1+3*here->TRAsizeDelay);
                v2 = *(here->TRAdelays+1+3*(here->TRAsizeDelay-1));
                v3 = *(here->TRAdelays+1+3*(here->TRAsizeDelay-2));
                v4 = *(here->TRAdelays+2+3*here->TRAsizeDelay);
                v5 = *(here->TRAdelays+2+3*(here->TRAsizeDelay-1));
                v6 = *(here->TRAdelays+2+3*(here->TRAsizeDelay-2));
                d1 = (v1-v2)/ckt->CKTdeltaOld[0];
                d2 = (v2-v3)/ckt->CKTdeltaOld[1];
                d3 = (v4-v5)/ckt->CKTdeltaOld[0];
                d4 = (v5-v6)/ckt->CKTdeltaOld[1];
                /*printf("%s: at %g derivs are %g, %g and %g, %g\n",
                        here->TRAname,ckt->CKTtime,d1,d2,d3,d4);*/
                if( (fabs(d1-d2) >= here->TRAreltol*MAX(fabs(d1),fabs(d2))+
                        here->TRAabstol) ||
                        (fabs(d3-d4) >= here->TRAreltol*MAX(fabs(d3),fabs(d4))+
                        here->TRAabstol) ) {
                    /* derivitive changing - need to schedule after delay */
                    /*printf("%s:  at %g set for %g\n",here->TRAname,
                        ckt->CKTtime,
                        *(here->TRAdelays+3*here->TRAsizeDelay-3)+here->TRAtd
                        );*/
                    /*printf("%g, %g, %g -> %g, %g \n",v1,v2,v3,d1,d2);*/
                    /*printf("%g, %g, %g -> %g, %g \n",v4,v5,v6,d3,d4);*/
                    /* also set for break after PREVIOUS point */
                    /*printf("setting break\n");*/
                    error = CKTsetBreak(ckt,
                            *(here->TRAdelays+3*here->TRAsizeDelay -3) +
                            here->TRAtd);
                    if(error) return(error);
                }
#endif /*NOTDEF*/
            }
        }
    }
    return(OK);
}
