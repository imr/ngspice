#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim4v4itf.h"
#include "bsim4v4ext.h"
#include "bsim4v4init.h"


SPICEdev BSIM4v4info = {
    {
        "BSIM4v4",
        "Berkeley Short Channel IGFET Model-4",

        &BSIM4v4nSize,
        &BSIM4v4nSize,
        BSIM4v4names,

        &BSIM4v4pTSize,
        BSIM4v4pTable,

        &BSIM4v4mPTSize,
        BSIM4v4mPTable,

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

    BSIM4v4param,    /* DEVparam       */
    BSIM4v4mParam,   /* DEVmodParam    */
    BSIM4v4load,     /* DEVload        */
    BSIM4v4setup,    /* DEVsetup       */
    BSIM4v4unsetup,  /* DEVunsetup     */
    BSIM4v4setup,    /* DEVpzSetup     */
    BSIM4v4temp,     /* DEVtemperature */
    BSIM4v4trunc,    /* DEVtrunc       */
    NULL,          /* DEVfindBranch  */
    BSIM4v4acLoad,   /* DEVacLoad      */
    NULL,          /* DEVaccept      */
    BSIM4v4destroy,  /* DEVdestroy     */
    BSIM4v4mDelete,  /* DEVmodDelete   */
    BSIM4v4delete,   /* DEVdelete      */
    BSIM4v4getic,    /* DEVsetic       */
    BSIM4v4ask,      /* DEVask         */
    BSIM4v4mAsk,     /* DEVmodAsk      */
    BSIM4v4pzLoad,   /* DEVpzLoad      */
    BSIM4v4convTest, /* DEVconvTest    */
    NULL,          /* DEVsenSetup    */
    NULL,          /* DEVsenLoad     */
    NULL,          /* DEVsenUpdate   */
    NULL,          /* DEVsenAcLoad   */
    NULL,          /* DEVsenPrint    */
    NULL,          /* DEVsenTrunc    */
    NULL,          /* DEVdisto       */
    BSIM4v4noise,    /* DEVnoise       */
#ifdef CIDER
    NULL,          /* DEVdump        */
    NULL,          /* DEVacct        */
#endif
    &BSIM4v4iSize,   /* DEVinstSize    */
    &BSIM4v4mSize,   /* DEVmodSize     */
#ifdef KLU
 /* DEVbindCSC        */   BSIM4v4bindCSC,
 /* DEVbindCSCComplex */   BSIM4v4bindCSCComplex,
#endif

};


SPICEdev *
get_bsim4v4_info(void)
{
    return &BSIM4v4info;
}
