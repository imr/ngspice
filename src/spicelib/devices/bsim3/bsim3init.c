#include "config.h"

#include "devdefs.h"

#include "bsim3itf.h"
#include "bsim3ext.h"
#include "bsim3init.h"


SPICEdev BSIM3info = {
    {   "BSIM3",
        "Berkeley Short Channel IGFET Model Version-3",

        &BSIM3nSize,
        &BSIM3nSize,
        BSIM3names,

        &BSIM3pTSize,
        BSIM3pTable,

        &BSIM3mPTSize,
        BSIM3mPTable,

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

    DEVparam      : BSIM3param,
    DEVmodParam   : BSIM3mParam,
    DEVload       : BSIM3load,
    DEVsetup      : BSIM3setup,
    DEVunsetup    : BSIM3unsetup,
    DEVpzSetup    : BSIM3setup,
    DEVtemperature: BSIM3temp,
    DEVtrunc      : BSIM3trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : BSIM3acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : BSIM3destroy,
    DEVmodDelete  : BSIM3mDelete,
    DEVdelete     : BSIM3delete, 
    DEVsetic      : BSIM3getic,
    DEVask        : BSIM3ask,
    DEVmodAsk     : BSIM3mAsk,
    DEVpzLoad     : BSIM3pzLoad,
    DEVconvTest   : BSIM3convTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : BSIM3noise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                       
    DEVinstSize   : &BSIM3iSize,
    DEVmodSize    : &BSIM3mSize

};


SPICEdev *
get_bsim3_info(void)
{
    return &BSIM3info;
}
