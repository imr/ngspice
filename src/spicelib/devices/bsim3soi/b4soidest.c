/***  B4SOI 12/16/2010 Released by Tanvir Morshed   ***/


/**********
 * Copyright 2010 Regents of the University of California.  All rights reserved.
 * Authors: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
 * Authors: 1999-2004 Pin Su, Hui Wan, Wei Jin, b3soidest.c
 * Authors: 2005- Hui Wan, Xuemei Xi, Ali Niknejad, Chenming Hu.
 * Authors: 2009- Wenwei Yang, Chung-Hsun Lin, Ali Niknejad, Chenming Hu.
 * File: b4soidest.c
 * Modified by Hui Wan, Xuemei Xi 11/30/2005
 * Modified by Wenwei Yang, Chung-Hsun Lin, Darsen Lu 03/06/2009
 * Modified by Tanvir Morshed 09/22/2009
 * Modified by Tanvir Morshed 12/31/2009
 **********/

#include "ngspice/ngspice.h"

#include "b4soidef.h"
#include "ngspice/suffix.h"

void
B4SOIdestroy(
GENmodel **inModel)
{
B4SOImodel **model = (B4SOImodel**)inModel;
B4SOIinstance *here;
B4SOIinstance *prev = NULL;
B4SOImodel *mod = *model;
B4SOImodel *oldmod = NULL;

    for (; mod ; mod = mod->B4SOInextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (B4SOIinstance *)NULL;
         for (here = mod->B4SOIinstances; here; here = here->B4SOInextInstance)
         {    
              if (here->B4SOIowner != ARCHme) continue;
              if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}



