#include <config.h>

#include <devdefs.h>

#include "inditf.h"
#include "indext.h"
#include "indinit.h"


SPICEdev INDinfo = {
    {
        "Inductor",
        "Fixed inductor",

        &INDnSize,
        &INDnSize,
        INDnames,

        &INDpTSize,
        INDpTable,

        &INDmPTSize,
        INDmPTable,

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
    DEVmodParam   : INDmParam,
    DEVload       : INDload,
    DEVsetup      : INDsetup,
    DEVunsetup    : INDunsetup,
    DEVpzSetup    : INDsetup,
    DEVtemperature: INDtemp,
    DEVtrunc      : INDtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : INDacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : INDdestroy,
    DEVmodDelete  : INDmDelete,
    DEVdelete     : INDdelete,
    DEVsetic      : NULL,
    DEVask        : INDask,
    DEVmodAsk     : INDmAsk,
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

    DEVparam      : MUTparam,
    DEVmodParam   : NULL,
    DEVload       : NULL,/* load handled by INDload */
    DEVsetup      : MUTsetup,
    DEVunsetup    : NULL,
    DEVpzSetup    : MUTsetup,
    DEVtemperature: MUTtemp,
    DEVtrunc      : NULL,
    DEVfindBranch : NULL,
    DEVacLoad     : MUTacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : MUTdestroy,
    DEVmodDelete  : MUTmDelete,
    DEVdelete     : MUTdelete,
    DEVsetic      : NULL,
    DEVask        : MUTask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : MUTpzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : MUTsSetup,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : MUTsPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,	/* DISTO */
    DEVnoise      :NULL,	/* NOISE */
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif  
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
