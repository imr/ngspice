/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef ngspice_GENDEFS_H
#define ngspice_GENDEFS_H

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

    /* The actual devices have to place their node elements
     *   right after the the end of struct GENinstance
     *   where there can be accessed by generic GENnode()[]
     * A notable exception is the XSPICE MIF device
     */
#if 0
    int GENnode[];  /* node numbers to which this instance is connected to */
                    /* carefull, thats overlayed into the actual device structs */
#endif
};

static inline int *GENnode(struct GENinstance *inst)
{ return (int*)(inst + 1); }


/* per model data */

struct GENmodel {       /* model structure for a resistor */
    int GENmodType;             /* type index of this device type */
    GENmodel *GENnextModel;     /* pointer to next possible model in
                                 * linked list */
    GENinstance *GENinstances;  /* pointer to list of instances that have this
                                 * model */
    IFuid GENmodName;           /* pointer to character string naming this model */
};


void GENinstanceFree(GENinstance *);
void GENmodelFree(GENmodel *);

#define GENmodelOf(p)    &((p)->gen)
#define GENinstanceOf(p) &((p)->gen)


#endif
