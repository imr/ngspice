#include "config.h"

#include "devdefs.h"

#include "bsim4v4itf.h"
#include "bsim4v4ext.h"
#include "bsim4v4init.h"


SPICEdev BSIM4V4info = {
    {
        "BSIM4v4",
        "Berkeley Short Channel IGFET Model-4",

        &BSIM4V4nSize,
        &BSIM4V4nSize,
        BSIM4V4names,

        &BSIM4V4pTSize,
        BSIM4V4pTable,

        &BSIM4V4mPTSize,
        BSIM4V4mPTable,

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

    BSIM4V4param,    /* DEVparam       */
    BSIM4V4mParam,   /* DEVmodParam    */
    BSIM4V4load,     /* DEVload        */
    BSIM4V4setup,    /* DEVsetup       */
    BSIM4V4unsetup,  /* DEVunsetup     */
    BSIM4V4setup,    /* DEVpzSetup     */
    BSIM4V4temp,     /* DEVtemperature */
    BSIM4V4trunc,    /* DEVtrunc       */
    NULL,          /* DEVfindBranch  */
    BSIM4V4acLoad,   /* DEVacLoad      */
    NULL,          /* DEVaccept      */
    BSIM4V4destroy,  /* DEVdestroy     */
    BSIM4V4mDelete,  /* DEVmodDelete   */
    BSIM4V4delete,   /* DEVdelete      */
    BSIM4V4getic,    /* DEVsetic       */
    BSIM4V4ask,      /* DEVask         */
    BSIM4V4mAsk,     /* DEVmodAsk      */
    BSIM4V4pzLoad,   /* DEVpzLoad      */
    BSIM4V4convTest, /* DEVconvTest    */
    NULL,          /* DEVsenSetup    */
    NULL,          /* DEVsenLoad     */
    NULL,          /* DEVsenUpdate   */
    NULL,          /* DEVsenAcLoad   */
    NULL,          /* DEVsenPrint    */
    NULL,          /* DEVsenTrunc    */
    NULL,          /* DEVdisto       */
    BSIM4V4noise,    /* DEVnoise       */
#ifdef CIDER
    NULL,          /* DEVdump        */
    NULL,          /* DEVacct        */
#endif
    &BSIM4V4iSize,   /* DEVinstSize    */
    &BSIM4V4mSize    /* DEVmodSize     */
};


SPICEdev *
get_bsim4v4_info(void)
{
    return &BSIM4V4info;
}
