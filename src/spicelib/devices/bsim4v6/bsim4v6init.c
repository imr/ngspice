#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim4v6itf.h"
#include "bsim4v6ext.h"
#include "bsim4v6init.h"


SPICEdev BSIM4v6info = {
    {
        "BSIM4v6",
        "Berkeley Short Channel IGFET Model-4",

        &BSIM4v6nSize,
        &BSIM4v6nSize,
        BSIM4v6names,

        &BSIM4v6pTSize,
        BSIM4v6pTable,

        &BSIM4v6mPTSize,
        BSIM4v6mPTable,

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

    BSIM4v6param,    /* DEVparam       */
    BSIM4v6mParam,   /* DEVmodParam    */
    BSIM4v6load,     /* DEVload        */
    BSIM4v6setup,    /* DEVsetup       */
    BSIM4v6unsetup,  /* DEVunsetup     */
    BSIM4v6setup,    /* DEVpzSetup     */
    BSIM4v6temp,     /* DEVtemperature */
    BSIM4v6trunc,    /* DEVtrunc       */
    NULL,          /* DEVfindBranch  */
    BSIM4v6acLoad,   /* DEVacLoad      */
    NULL,          /* DEVaccept      */
    BSIM4v6destroy,  /* DEVdestroy     */
    BSIM4v6mDelete,  /* DEVmodDelete   */
    BSIM4v6delete,   /* DEVdelete      */
    BSIM4v6getic,    /* DEVsetic       */
    BSIM4v6ask,      /* DEVask         */
    BSIM4v6mAsk,     /* DEVmodAsk      */
    BSIM4v6pzLoad,   /* DEVpzLoad      */
    BSIM4v6convTest, /* DEVconvTest    */
    NULL,          /* DEVsenSetup    */
    NULL,          /* DEVsenLoad     */
    NULL,          /* DEVsenUpdate   */
    NULL,          /* DEVsenAcLoad   */
    NULL,          /* DEVsenPrint    */
    NULL,          /* DEVsenTrunc    */
    NULL,          /* DEVdisto       */
    BSIM4v6noise,    /* DEVnoise       */
#ifdef CIDER
    NULL,          /* DEVdump        */
    NULL,          /* DEVacct        */
#endif
    &BSIM4v6iSize,   /* DEVinstSize    */
    &BSIM4v6mSize,    /* DEVmodSize     */
    BSIM4v6nodeIsNonLinear /* DEVnodeIsNonLinear */
};


SPICEdev *
get_bsim4v6_info(void)
{
    return &BSIM4v6info;
}
