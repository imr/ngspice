/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef GEN
#define GEN

#include "ngspice/typedefs.h"
#include "ngspice/ifsim.h"



        /* definitions used to describe generic devices */

/* information used to describe a single instance */

struct GENinstance {
    GENmodel *GENmodPtr;    /* backpointer to model */
    GENinstance *GENnextInstance;   /* pointer to next instance of
                                     * current model*/
    IFuid GENname;  /* pointer to character string naming this instance */
    int GENstate;   /* state index number */
    int GENnode1;   /* appropriate node numbers */
    int GENnode2;   /* appropriate node numbers */
    int GENnode3;   /* appropriate node numbers */
    int GENnode4;   /* appropriate node numbers */
    int GENnode5;   /* appropriate node numbers */
    int GENnode6;   /* added to create body node 01/06/99 */
    int GENnode7;   /* added to create temp node  2/03/99 */
};


/* Generic circuit data */

typedef void GENcircuit;


/* per model data */

struct GENmodel {       /* model structure for a resistor */
    int GENmodType;             /* type index of this device type */
    GENmodel *GENnextModel;     /* pointer to next possible model in
                                 * linked list */
    GENinstance *GENinstances;  /* pointer to list of instances that have this
                                 * model */
    IFuid GENmodName;           /* pointer to character string naming this model */
};

#endif /*GEN*/
