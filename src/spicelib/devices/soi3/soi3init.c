#include <config.h>

#include <devdefs.h>

#include "soi3itf.h"
#include "soi3ext.h"
#include "soi3init.h"


SPICEdev SOI3info = {
    {
        "SOI3",
        "Basic Thick Film SOI3 model v2.7",

        &SOI3nSize,
        &SOI3nSize,
        SOI3names,

        &SOI3pTSize,
        SOI3pTable,

        &SOI3mPTSize,
        SOI3mPTable,

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

    DEVparam      : SOI3param,
    DEVmodParam   : SOI3mParam,
    DEVload       : SOI3load,
    DEVsetup      : SOI3setup,
    DEVunsetup    : SOI3unsetup,
    DEVpzSetup    : SOI3setup,
    DEVtemperature: SOI3temp,
    DEVtrunc      : SOI3trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : SOI3acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : SOI3destroy,
    DEVmodDelete  : SOI3mDelete,
    DEVdelete     : SOI3delete,
    DEVsetic      : SOI3getic,
    DEVask        : SOI3ask,
    DEVmodAsk     : SOI3mAsk,
    DEVpzLoad     : NULL,
    DEVconvTest   : SOI3convTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : SOI3noise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                        
    DEVinstSize   : &SOI3iSize,
    DEVmodSize    : &SOI3mSize

};


SPICEdev *
get_soi3_info(void)
{
    return &SOI3info;
}
