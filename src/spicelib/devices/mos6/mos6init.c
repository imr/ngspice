#include <config.h>

#include <devdefs.h>

#include "mos6itf.h"
#include "mos6ext.h"
#include "mos6init.h"


SPICEdev MOS6info = {
    {
        "Mos6",
        "Level 6 MOSfet model with Meyer capacitance model",

        &MOS6nSize,
        &MOS6nSize,
        MOS6names,

        &MOS6pTSize,
        MOS6pTable,

        &MOS6mPTSize,
        MOS6mPTable,

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

    DEVparam      : MOS6param,
    DEVmodParam   : MOS6mParam,
    DEVload       : MOS6load,
    DEVsetup      : MOS6setup,
    DEVunsetup    : MOS6unsetup,
    DEVpzSetup    : NULL, /* PZsetup routine */
    DEVtemperature: MOS6temp,
    DEVtrunc      : MOS6trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : NULL, /* MOS6acLoad, XXX */
    DEVaccept     : NULL,
    DEVdestroy    : MOS6destroy,
    DEVmodDelete  : NULL,
    DEVdelete     : NULL,
    DEVsetic      : MOS6getic,
    DEVask        : MOS6ask,
    DEVmodAsk     : MOS6mAsk,
    DEVpzLoad     : NULL, /*MOS6pzLoad, XXX */
    DEVconvTest   : MOS6convTest,
    DEVsenSetup   : NULL /* MOS6sSetup */,
    DEVsenLoad    : NULL /* MOS6sLoad */,
    DEVsenUpdate  : NULL /* MOS6sUpdate */,
    DEVsenAcLoad  : NULL /* MOS6sAcLoad */,
    DEVsenPrint   : NULL /* MOS6sPrint */,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL, /* Distortion routine */
    DEVnoise      : NULL, /* Noise routine */
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                        
    DEVinstSize   : &MOS6iSize,
    DEVmodSize    : &MOS6mSize
};


SPICEdev *
get_mos6_info(void)
{
    return &MOS6info;
}
