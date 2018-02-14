/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "tradefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
TRAload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current values into the 
         * sparse matrix previously provided 
         */
{
    TRAmodel *model = (TRAmodel *)inModel;
    TRAinstance *here;
    double t1,t2,t3;
    double f1,f2,f3;
    int i;

    /*  loop through all the transmission line models */
    for( ; model != NULL; model = TRAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = TRAinstances(model); here != NULL ;
                here=TRAnextInstance(here)) {
            
            *(here->TRApos1Pos1Ptr) += here->TRAconduct;
            *(here->TRApos1Int1Ptr) -= here->TRAconduct;
            *(here->TRAneg1Ibr1Ptr) -= 1;
            *(here->TRApos2Pos2Ptr) += here->TRAconduct;
            *(here->TRAneg2Ibr2Ptr) -= 1;
            *(here->TRAint1Pos1Ptr) -= here->TRAconduct;
            *(here->TRAint1Int1Ptr) += here->TRAconduct;
            *(here->TRAint1Ibr1Ptr) += 1;
            *(here->TRAint2Int2Ptr) += here->TRAconduct;
            *(here->TRAint2Ibr2Ptr) += 1;
            *(here->TRAibr1Neg1Ptr) -= 1;
            *(here->TRAibr1Int1Ptr) += 1;
            *(here->TRAibr2Neg2Ptr) -= 1;
            *(here->TRAibr2Int2Ptr) += 1;
            *(here->TRApos2Int2Ptr) -= here->TRAconduct;
            *(here->TRAint2Pos2Ptr) -= here->TRAconduct;

            if(ckt->CKTmode & MODEDC) {
                *(here->TRAibr1Pos2Ptr) -= 1;
                *(here->TRAibr1Neg2Ptr) += 1;
                *(here->TRAibr1Ibr2Ptr) -= (1-ckt->CKTgmin)*here->TRAimped;
                *(here->TRAibr2Pos1Ptr) -= 1;
                *(here->TRAibr2Neg1Ptr) += 1;
                *(here->TRAibr2Ibr1Ptr) -= (1-ckt->CKTgmin)*here->TRAimped;
            } else {
                if (ckt->CKTmode & MODEINITTRAN) {
                    if(ckt->CKTmode & MODEUIC) {
                        here->TRAinput1 = here->TRAinitVolt2 + here->TRAinitCur2
                            * here->TRAimped;
                        here->TRAinput2 = here->TRAinitVolt1 + here->TRAinitCur1
                            * here->TRAimped;
                    } else {
                        here->TRAinput1 = 
                            ( *(ckt->CKTrhsOld+here->TRAposNode2)
                            - *(ckt->CKTrhsOld+here->TRAnegNode2) )
                            + ( *(ckt->CKTrhsOld+here->TRAbrEq2) 
                                *here->TRAimped);
                        here->TRAinput2 = 
                            ( *(ckt->CKTrhsOld+here->TRAposNode1)
                            - *(ckt->CKTrhsOld+here->TRAnegNode1) )
                            + ( *(ckt->CKTrhsOld+here->TRAbrEq1) 
                                *here->TRAimped);
                    }
                    *(here->TRAdelays  ) = -2*here->TRAtd;
                    *(here->TRAdelays+3) = -here->TRAtd;
                    *(here->TRAdelays+6) = 0;
                    *(here->TRAdelays+1) = *(here->TRAdelays +4) = 
                            *(here->TRAdelays+7) = here->TRAinput1;
                    *(here->TRAdelays+2) = *(here->TRAdelays +5) = 
                            *(here->TRAdelays+8) = here->TRAinput2;
                    here->TRAsizeDelay = 2;
                } else {
                    if(ckt->CKTmode & MODEINITPRED) {
                        for(i=2;(i<here->TRAsizeDelay) && 
                            (*(here->TRAdelays +3*i) <=
                            (ckt->CKTtime-here->TRAtd));i++) {;/*loop does it*/}
                        t1 = *(here->TRAdelays + (3*(i-2)));
                        t2 = *(here->TRAdelays + (3*(i-1)));
                        t3 = *(here->TRAdelays + (3*(i  )));
                        if( (t2-t1)==0  || (t3-t2) == 0) continue;
                        f1 = (ckt->CKTtime - here->TRAtd - t2) * 
                             (ckt->CKTtime - here->TRAtd - t3) ;
                        f2 = (ckt->CKTtime - here->TRAtd - t1) *
                             (ckt->CKTtime - here->TRAtd - t3) ;
                        f3 = (ckt->CKTtime - here->TRAtd - t1) *
                             (ckt->CKTtime - here->TRAtd - t2) ;
                        if((t2-t1)==0) { /* should never happen, but don't want
                                          * to divide by zero, EVER... */
                            f1=0;
                            f2=0;
                        } else {
                            f1 /= (t1-t2);
                            f2 /= (t2-t1);
                        }
                        if((t3-t2)==0) { /* should never happen, but don't want
                                          * to divide by zero, EVER... */
                            f2=0;
                            f3=0;
                        } else {
                            f2 /= (t2-t3);
                            f3 /= (t2-t3);
                        }
                        if((t3-t1)==0) { /* should never happen, but don't want
                                          * to divide by zero, EVER... */
                            f1=0;
                            f2=0;
                        } else {
                            f1 /= (t1-t3);
                            f3 /= (t1-t3);
                        }
                        /*printf("at time %g, using %g, %g, %g\n",ckt->CKTtime,
                                t1,t2,t3);
                        printf("values %g, %g, %g \n",
                            *(here->TRAdelays + (3*(i-2))+1),
                            *(here->TRAdelays + (3*(i-1))+1),
                            *(here->TRAdelays + (3*(i  ))+1) );
                        printf("and    %g, %g, %g \n",
                            *(here->TRAdelays + (3*(i-2))+2),
                            *(here->TRAdelays + (3*(i-1))+2),
                            *(here->TRAdelays + (3*(i  ))+2) );*/
                        here->TRAinput1 = f1 * *(here->TRAdelays + (3*(i-2))+1)
                                        + f2 * *(here->TRAdelays + (3*(i-1))+1)
                                        + f3 * *(here->TRAdelays + (3*(i  ))+1);
                        here->TRAinput2 = f1 * *(here->TRAdelays + (3*(i-2))+2)
                                        + f2 * *(here->TRAdelays + (3*(i-1))+2)
                                        + f3 * *(here->TRAdelays + (3*(i  ))+2);
                    }
                }
            *(ckt->CKTrhs + here->TRAbrEq1) += here->TRAinput1;
            *(ckt->CKTrhs + here->TRAbrEq2) += here->TRAinput2;
            }
        }
    }
    return(OK);
}
