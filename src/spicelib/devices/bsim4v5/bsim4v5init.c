#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim4v5itf.h"
#include "bsim4v5ext.h"
#include "bsim4v5init.h"


SPICEdev BSIM4v5info = {
    {
        "BSIM4v5",
        "Berkeley Short Channel IGFET Model-4",

        &BSIM4v5nSize,
        &BSIM4v5nSize,
        BSIM4v5names,

        &BSIM4v5pTSize,
        BSIM4v5pTable,

        &BSIM4v5mPTSize,
        BSIM4v5mPTable,

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

    BSIM4v5param,    /* DEVparam       */
    BSIM4v5mParam,   /* DEVmodParam    */
    BSIM4v5load,     /* DEVload        */
    BSIM4v5setup,    /* DEVsetup       */
    BSIM4v5unsetup,  /* DEVunsetup     */
    BSIM4v5setup,    /* DEVpzSetup     */
    BSIM4v5temp,     /* DEVtemperature */
    BSIM4v5trunc,    /* DEVtrunc       */
    NULL,          /* DEVfindBranch  */
    BSIM4v5acLoad,   /* DEVacLoad      */
    NULL,          /* DEVaccept      */
    BSIM4v5destroy,  /* DEVdestroy     */
    BSIM4v5mDelete,  /* DEVmodDelete   */
    BSIM4v5delete,   /* DEVdelete      */
    BSIM4v5getic,    /* DEVsetic       */
    BSIM4v5ask,      /* DEVask         */
    BSIM4v5mAsk,     /* DEVmodAsk      */
    BSIM4v5pzLoad,   /* DEVpzLoad      */
    BSIM4v5convTest, /* DEVconvTest    */
    NULL,          /* DEVsenSetup    */
    NULL,          /* DEVsenLoad     */
    NULL,          /* DEVsenUpdate   */
    NULL,          /* DEVsenAcLoad   */
    NULL,          /* DEVsenPrint    */
    NULL,          /* DEVsenTrunc    */
    NULL,          /* DEVdisto       */
    BSIM4v5noise,    /* DEVnoise       */
#ifdef CIDER
    NULL,          /* DEVdump        */
    NULL,          /* DEVacct        */
#endif
    &BSIM4v5iSize,   /* DEVinstSize    */
    &BSIM4v5mSize,    /* DEVmodSize     */
    BSIM4v5nodeIsNonLinear /* DEVnodeIsNonLinear */
};


SPICEdev *
get_bsim4v5_info(void)
{
    return &BSIM4v5info;
}
