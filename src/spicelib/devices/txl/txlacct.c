/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/


#include "ngspice/ngspice.h"
#include "ngspice/fteext.h"
#include "ngspice/cktdefs.h"
#include "txldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


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
    for( ; model != NULL; model = TXLnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = TXLinstances(model); here != NULL ;
                here=TXLnextInstance(here)) {
            
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
				controlled_exit(1);
			}
		}
    }
	model = (TXLmodel *)inModel;
	for( ; model != NULL; model = TXLnextModel(model)) {
		for (here = TXLinstances(model); here != NULL ;
			here=TXLnextInstance(here)) {
			nd = here->txline->in_node;
			nd->dvtag = 0;
			nd = here->txline->out_node;
			nd->dvtag = 0;
		}
	}

    return(OK);
}
