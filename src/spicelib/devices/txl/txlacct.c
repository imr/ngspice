/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/


#include "ngspice.h"
#include "cktdefs.h"
#include "txldefs.h"
#include "sperror.h"
#include "suffix.h"


int
TXLaccept(CKTcircuit *ckt, GENmodel *inModel)
        /* set up the breakpoint table.
         */
{
     TXLmodel *model = (TXLmodel *)inModel;
     TXLinstance *here;
	int hint;
	double h, v, v1;
	NODE *nd;
	TXLine *tx;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->TXLnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->TXLinstances; here != NULL ;
                here=here->TXLnextInstance) {
            
			h = ckt->CKTdelta;
			hint = (int) (h * 1e12);
			if (hint != 0) {
				tx = here->txline;
				nd = tx->in_node;
				if (nd->dvtag == 0) {
					v = nd->V;
					v1 = nd->V = *(ckt->CKTrhs + here->TXLposNode);
					nd->dv = (v1 - v) / hint;
					nd->dvtag = 1;
				}
				nd = tx->out_node;
				if (nd->dvtag == 0) {
					v = nd->V;
					v1 = nd->V = *(ckt->CKTrhs + here->TXLnegNode);
					nd->dv = (v1 - v) / hint;
					nd->dvtag = 1;
				}
			}
			else {
				/* can't happen. */
				printf("zero h detected\n");
				exit(1);
			}
		}
    }
	model = (TXLmodel *)inModel;
	for( ; model != NULL; model = model->TXLnextModel ) {
		for (here = model->TXLinstances; here != NULL ;
			here=here->TXLnextInstance) {
			nd = here->txline->in_node;
			nd->dvtag = 0;
			nd = here->txline->out_node;
			nd->dvtag = 0;
		}
	}

    return(OK);
}
