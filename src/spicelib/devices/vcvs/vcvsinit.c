#include <config.h>

#include <devdefs.h>

#include "vcvsitf.h"
#include "vcvsext.h"
#include "vcvsinit.h"


SPICEdev VCVSinfo = {
    {
        "VCVS",
        "Voltage controlled voltage source",

        &VCVSnSize,
        &VCVSnSize,
        VCVSnames,

        &VCVSpTSize,
        VCVSpTable,

        0,
        NULL,

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

    DEVparam      : VCVSparam,
    DEVmodParam   : NULL,
    DEVload       : VCVSload,
    DEVsetup      : VCVSsetup,
    DEVunsetup    : VCVSunsetup,
    DEVpzSetup    : VCVSsetup,
    DEVtemperature: NULL,
    DEVtrunc      : NULL,
    DEVfindBranch : VCVSfindBr,
    DEVacLoad     : VCVSload,   /* AC and normal loads are identical */
    DEVaccept     : NULL,
    DEVdestroy    : VCVSdestroy,
    DEVmodDelete  : VCVSmDelete,
    DEVdelete     : VCVSdelete,
    DEVsetic      : NULL,
    DEVask        : VCVSask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : VCVSpzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : VCVSsSetup,
    DEVsenLoad    : VCVSsLoad,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : VCVSsAcLoad,
    DEVsenPrint   : VCVSsPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,	/* DISTO */
    DEVnoise      : NULL,	/* NOISE */
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                          
    DEVinstSize   : &VCVSiSize,
    DEVmodSize    : &VCVSmSize

};


SPICEdev *
get_vcvs_info(void)
{
    return &VCVSinfo;
}
