/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTinst2Node
     *  get the name and node pointer for a node given a device it is
     * bound to and the terminal of the device.
     */

#include "ngspice.h"
#include "ifsim.h"
#include "sperror.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "devdefs.h"



extern SPICEdev **DEVices;

int
CKTinst2Node(void *ckt, void *instPtr, int terminal, CKTnode **node, IFuid *nodeName)
{
    int nodenum;
    int type;
    CKTnode *here;

    type = ((GENinstance *)instPtr)->GENmodPtr->GENmodType;

    if(*((*DEVices[type]).DEVpublic.terms) >= terminal && terminal >0 ) {
        switch(terminal) {
            default: return(E_NOTERM);
            case 1:
                nodenum = ((GENinstance *)instPtr)->GENnode1;
                break;
            case 2:
                nodenum = ((GENinstance *)instPtr)->GENnode2;
                break;
            case 3:
                nodenum = ((GENinstance *)instPtr)->GENnode3;
                break;
            case 4:
                nodenum = ((GENinstance *)instPtr)->GENnode4;
                break;
            case 5:
                nodenum = ((GENinstance *)instPtr)->GENnode5;
                break;
	    case 6:
                nodenum = ((GENinstance *)instPtr)->GENnode6;
                break;
	    case 7:
                nodenum = ((GENinstance *)instPtr)->GENnode7;
                break;	
        }
        /* ok, now we know its number, so we just have to find it.*/
        for(here = ((CKTcircuit*)ckt)->CKTnodes;here;here = here->next) {
            if(here->number == nodenum) {
                /* found it */
                *node = (void*) here;
                *nodeName = here->name;
                return(OK);
            }
        }
        return(E_NOTFOUND);
    } else {
        return(E_NOTERM);
    }
}
