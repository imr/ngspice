#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "b4soiitf.h"
#include "b4soiinit.h"

SPICEdev B4SOIinfo = {
    {   "B4SOI",
        "Berkeley SOI MOSFET model version 4.4.0",

        &B4SOInSize,
        &B4SOInSize,
        B4SOInames,

        &B4SOIpTSize,
        B4SOIpTable,

        &B4SOImPTSize,
        B4SOImPTable,
        
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

 /* DEVparam      */ B4SOIparam,
 /* DEVmodParam   */ B4SOImParam,
 /* DEVload       */ B4SOIload,
 /* DEVsetup      */ B4SOIsetup,
 /* DEVunsetup    */ B4SOIunsetup,
 /* DEVpzSetup    */ B4SOIsetup,
 /* DEVtemperature*/ B4SOItemp,
 /* DEVtrunc      */ B4SOItrunc,
 /* DEVfindBranch */ NULL,
 /* DEVacLoad     */ B4SOIacLoad,
 /* DEVaccept     */ NULL,
 /* DEVdestroy    */ B4SOIdestroy,
 /* DEVmodDelete  */ B4SOImDelete,
 /* DEVdelete     */ B4SOIdelete, 
 /* DEVsetic      */ B4SOIgetic,
 /* DEVask        */ B4SOIask,
 /* DEVmodAsk     */ B4SOImAsk,
 /* DEVpzLoad     */ B4SOIpzLoad,
 /* DEVconvTest   */ B4SOIconvTest,
 /* DEVsenSetup   */ NULL,
 /* DEVsenLoad    */ NULL,
 /* DEVsenUpdate  */ NULL,
 /* DEVsenAcLoad  */ NULL,
 /* DEVsenPrint   */ NULL,
 /* DEVsenTrunc   */ NULL,
 /* DEVdisto      */ NULL,
 /* DEVnoise      */ B4SOInoise,
#ifdef CIDER
 /* DEVdump       */ NULL,
 /* DEVacct       */ NULL,
#endif
 /* DEVinstSize   */ &B4SOIiSize,
 /* DEVmodSize    */ &B4SOImSize,
 /* DEVnodeIsNonLinear */ B4SOInodeIsNonLinear
};

SPICEdev *
get_b4soi_info (void)
{
  return &B4SOIinfo;
}


