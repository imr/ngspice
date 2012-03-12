/*
 * vbicinit.c
 */


#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "vbicitf.h"
#include "vbicext.h"
#include "vbicinit.h"


SPICEdev VBICinfo = {
    {
	"VBIC",
        "Vertical Bipolar Inter-Company Model",

        &VBICnSize,
        &VBICnSize,
        VBICnames,

        &VBICpTSize,
        VBICpTable,

        &VBICmPTSize,
        VBICmPTable,

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

    VBICparam,    /* DEVparam       */
    VBICmParam,   /* DEVmodParam    */
    VBICload,     /* DEVload        */
    VBICsetup,    /* DEVsetup       */
    VBICunsetup,  /* DEVunsetup     */
    VBICsetup,    /* DEVpzSetup     */
    VBICtemp,     /* DEVtemperature */
    VBICtrunc,    /* DEVtrunc       */
    NULL,         /* DEVfindBranch  */
    VBICacLoad,   /* DEVacLoad      */
    NULL,         /* DEVaccept      */
    VBICdestroy,  /* DEVdestroy     */
    VBICmDelete,  /* DEVmodDelete   */
    VBICdelete,   /* DEVdelete      */
    VBICgetic,    /* DEVsetic       */
    VBICask,      /* DEVask         */
    VBICmAsk,     /* DEVmodAsk      */
    VBICpzLoad,   /* DEVpzLoad      */
    VBICconvTest, /* DEVconvTest    */
    NULL,         /* DEVsenSetup    */
    NULL,         /* DEVsenLoad     */
    NULL,         /* DEVsenUpdate   */
    NULL,         /* DEVsenAcLoad   */
    NULL,         /* DEVsenPrint    */
    NULL,         /* DEVsenTrunc    */
    NULL,         /* DEVdisto       */
    VBICnoise,    /* DEVnoise       */
#ifdef CIDER
    NULL,         /* DEVdump       */
    NULL,         /* DEVacct       */
#endif                                                         
    &VBICiSize,   /* DEVinstSize    */
    &VBICmSize,   /* DEVmodSize     */
#if defined(KLU) || defined(SuperLU) || defined(UMFPACK)
 /* DEVbindCSC        */   VBICbindCSC,
 /* DEVbindCSCComplex */   VBICbindCSCComplex,
#endif

};


SPICEdev *
get_vbic_info(void)
{
    return &VBICinfo;
}
