/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* CKTbindNode
 *
 * bind a node of the specified device of the given type to its place
 * in the specified circuit.  */

#include <config.h>
#include <stdio.h>
#include <devdefs.h>
#include <sperror.h>

#include "dev.h"

int
CKTbindNode(void *ckt, void *fast, int term, void *node)
{
    int mappednode;
    SPICEdev **devs;
    GENinstance *instance = (GENinstance *) fast;
    int type = instance->GENmodPtr->GENmodType;

    devs = devices();
    mappednode = ((CKTnode *)node)->number;

    if (*((*devs[type]).DEVpublic.terms) >= term && term >0 ) {
        switch(term) {
            default:
		return E_NOTERM;
            case 1:
                instance->GENnode1 = mappednode;
                break;
            case 2:
                instance->GENnode2 = mappednode;
                break;
            case 3:
                instance->GENnode3 = mappednode;
                break;
            case 4:
                instance->GENnode4 = mappednode;
                break;
            case 5:
                instance->GENnode5 = mappednode;
                break;
            case 6:/* added to consider the body node 01/06/99 */  
                instance->GENnode6 = mappednode;
                break;
            case 7:/* added to consider the temp node 02/03/99 */
                instance->GENnode7 = mappednode;
                break;
        }
        return OK;
    } else {
        return E_NOTERM;
    }
}
