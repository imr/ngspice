#include <config.h>

#include <devdefs.h>

#include "inditf.h"
#include "indext.h"
#include "indinit.h"


SPICEdev INDinfo = {
    {
        "Inductor",
        "Inductors",

        &INDnSize,
        &INDnSize,
        INDnames,

        &INDpTSize,
        INDpTable,

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
	0
    },

    DEVparam      : INDparam,
    DEVmodParam   : NULL,
    DEVload       : INDload,
    DEVsetup      : INDsetup,
    DEVunsetup    : INDunsetup,
    DEVpzSetup    : INDsetup,
    DEVtemperature: NULL,
    DEVtrunc      : INDtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : INDacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : INDdestroy,
    DEVmodDelete  : INDmDelete,
    DEVdelete     : INDdelete,
    DEVsetic      : NULL,
    DEVask        : INDask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : INDpzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : INDsSetup,
    DEVsenLoad    : INDsLoad,
    DEVsenUpdate  : INDsUpdate,
    DEVsenAcLoad  : INDsAcLoad,
    DEVsenPrint   : INDsPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,	/* DISTO */
    DEVnoise      : NULL,	/* NOISE */
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                       
    DEVinstSize   : &INDiSize,
    DEVmodSize    : &INDmSize

};


SPICEdev MUTinfo = {
    {   
        "mutual",
        "Mutual inductors",
        0, /* term count */
        0, /* term count */
        NULL,

        &MUTpTSize,
        MUTpTable,

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

	0
    },

    MUTparam,
    NULL,
    NULL,/* load handled by INDload */
    MUTsetup,
    NULL,
    MUTsetup,
    NULL,
    NULL,
    NULL,
    MUTacLoad,
    NULL,
    MUTdestroy,
    MUTmDelete,
    MUTdelete,
    NULL,
    MUTask,
    NULL,
    MUTpzLoad,
    NULL,
    MUTsSetup,
    NULL,
    NULL,
    NULL,
    MUTsPrint,
    NULL,
    NULL,	/* DISTO */
    NULL,	/* NOISE */

    &MUTiSize,
    &MUTmSize

};


SPICEdev *
get_ind_info(void)
{
    return &INDinfo;
}


SPICEdev *
get_mut_info(void)
{
    return &MUTinfo;
}
