/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* CKTbindNode
 *  bind a node of the specified device of the given type to its place
 *  in the specified circuit.
 */

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "sperror.h"


extern SPICEdev *DEVices[];

/*ARGSUSED*/
int
CKTbindNode(void *ckt, void *fast, int term, void *node)
{
    int mappednode;
    register int type = ((GENinstance *)fast)->GENmodPtr->GENmodType;

    mappednode = ((CKTnode *)node)->number;

    if(*((*DEVices[type]).DEVpublic.terms) >= term && term >0 ) {
        switch(term) {
            default: return(E_NOTERM);
            case 1:
                ((GENinstance *)fast)->GENnode1 = mappednode;
                break;
            case 2:
                ((GENinstance *)fast)->GENnode2 = mappednode;
                break;
            case 3:
                ((GENinstance *)fast)->GENnode3 = mappednode;
                break;
            case 4:
                ((GENinstance *)fast)->GENnode4 = mappednode;
                break;
            case 5:
                ((GENinstance *)fast)->GENnode5 = mappednode;
                break;
        }
        return(OK);
    } else {
        return(E_NOTERM);
    }
}
