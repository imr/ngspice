#include <config.h>

#include <devdefs.h>

#include "bsim3v1itf.h"
#include "bsim3v1ext.h"
#include "bsim3v1init.h"

SPICEdev B3v1info = {
    {   "BSIM3v1",
        "Berkeley Short Channel IGFET Model Version-3 (3.1)",

        &BSIM3v1nSize,
        &BSIM3v1nSize,
        BSIM3v1names,

        &BSIM3v1pTSize,
        BSIM3v1pTable,

        &BSIM3v1mPTSize,
        BSIM3v1mPTable,

#ifdef XSPICE
/*----  Fixed by SDB 5.2.2003 to enable XSPICE/tclspice integration  -----*/
        NULL,  /* This is a SPICE device, it has no MIF info data */

        0,     /* This is a SPICE device, it has no MIF info data */
        NULL,  /* This is a SPICE device, it has no MIF info data */

        0,     /* This is a SPICE device, it has no MIF info data */
        NULL,  /* This is a SPICE device, it has no MIF info data */

        0,     /* This is a SPICE device, it has no MIF info data */
        NULL,  /* This is a SPICE device, it has no MIF info data */
/*---------------------------  End of SDB fix   -------------------------*/
#endif
	
        DEV_DEFAULT
    },

    DEVparam      : BSIM3v1param,
    DEVmodParam   : BSIM3v1mParam,
    DEVload       : BSIM3v1load,
    DEVsetup      : BSIM3v1setup,  
    DEVunsetup    : BSIM3v1unsetup,
    DEVpzSetup    : BSIM3v1setup,
    DEVtemperature: BSIM3v1temp,
    DEVtrunc      : BSIM3v1trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : BSIM3v1acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : BSIM3v1destroy, 
    DEVmodDelete  : BSIM3v1mDelete,
    DEVdelete     : BSIM3v1delete,  
    DEVsetic      : BSIM3v1getic,
    DEVask        : BSIM3v1ask,
    DEVmodAsk     : BSIM3v1mAsk, 
    DEVpzLoad     : BSIM3v1pzLoad,    
    DEVconvTest   : BSIM3v1convTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,        
    DEVnoise      : BSIM3v1noise,
#ifdef CIDER    
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                    
    DEVinstSize   : &BSIM3v1iSize,
    DEVmodSize    : &BSIM3v1mSize

};


SPICEdev *
get_bsim3v1_info(void)
{
     return &B3v1info; 
}
