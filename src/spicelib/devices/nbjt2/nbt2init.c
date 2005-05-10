#include <config.h>

#include <devdefs.h>

#include "nbjt2itf.h"
#include "nbjt2ext.h"
#include "nbt2init.h"


SPICEdev NBJT2info = {
    {
	"NBJT2",
        "2D Numerical Bipolar Junction Transistor model",

        &NBJT2nSize,
        &NBJT2nSize,
        NBJT2names,

        &NBJT2pTSize,
        NBJT2pTable,

        &NBJT2mPTSize,
        NBJT2mPTable,
	
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

    DEVparam      : NBJT2param,
    DEVmodParam   : NBJT2mParam,
    DEVload       : NBJT2load,
    DEVsetup      : NBJT2setup,
    DEVunsetup    : NULL,
    DEVpzSetup    : NBJT2setup,
    DEVtemperature: NBJT2temp,
    DEVtrunc      : NBJT2trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : NBJT2acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : NBJT2destroy,
    DEVmodDelete  : NBJT2mDelete,
    DEVdelete     : NBJT2delete,
    DEVsetic      : NULL,
    DEVask        : NBJT2ask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : NBJT2pzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : NULL,
#ifdef CIDER
    DEVdump       : NBJT2dump,
    DEVacct       : NBJT2acct,
#endif  
                    
    DEVinstSize   : &NBJT2iSize,
    DEVmodSize    : &NBJT2mSize

};


SPICEdev *
get_nbjt2_info(void)
{
    return &NBJT2info;
}
