#include <config.h>

#include <devdefs.h>

#include "bsim3v0itf.h"
#include "bsim3v0ext.h"
#include "bsim3v0init.h"

SPICEdev B3v0info = {
    {   "BSIM3v0",
        "Berkeley Short Channel IGFET Model Version-3 (3.0)",

        &BSIM3v0nSize,
        &BSIM3v0nSize,
        BSIM3v0names,

        &BSIM3v0pTSize,
        BSIM3v0pTable,

        &BSIM3v0mPTSize,
        BSIM3v0mPTable,
	
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

    DEVparam      : BSIM3v0param,
    DEVmodParam   : BSIM3v0mParam,
    DEVload       : BSIM3v0load,
    DEVsetup      : BSIM3v0setup,  
    DEVunsetup    : BSIM3v0unsetup,
    DEVpzSetup    : BSIM3v0setup,
    DEVtemperature: BSIM3v0temp,
    DEVtrunc      : BSIM3v0trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : BSIM3v0acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : BSIM3v0destroy, 
    DEVmodDelete  : BSIM3v0mDelete,
    DEVdelete     : BSIM3v0delete,  
    DEVsetic      : BSIM3v0getic,
    DEVask        : BSIM3v0ask,
    DEVmodAsk     : BSIM3v0mAsk, 
    DEVpzLoad     : BSIM3v0pzLoad,    
    DEVconvTest   : BSIM3v0convTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,        
    DEVnoise      : BSIM3v0noise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                    
    DEVinstSize   : &BSIM3v0iSize,
    DEVmodSize    : &BSIM3v0mSize

};


SPICEdev *
get_bsim3v0_info(void)
{
     return &B3v0info; 
}
