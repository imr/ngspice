/**********
Copyright ngspice team
Author: 2025 Holger Vogt
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/hbardefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/cpextern.h"

#include "analysis.h"

#ifdef WITH_HB

extern int hbnumfreqs[10];

struct variable {
    enum cp_types va_type;
    char* va_name;
    union {
        bool vV_bool;
        int vV_num;
        double vV_real;
        char* vV_string;
        struct variable* vV_list;
    } va_V;
    struct variable* va_next;      /* Link. */
};

#define va_bool   va_V.vV_bool
#define va_num    va_V.vV_num
#define va_real   va_V.vV_real
#define va_string va_V.vV_string
#define va_vlist  va_V.vV_list

/* Get the HB options.
   Currently supported:
   Number of frequencies for f1 
   Number of frequencies for f2
   ...
   Number of frequncies for f10 */

int
HBgetOptions(void)
{
    struct variable* var, *tv;

    if (hbnumfreqs[0] > 0)
        return 0;
    if (cp_getvar("hbnumfreq", CP_NUM, &hbnumfreqs[0], 0)) {
        for (int ii = 1; ii < 10; ii++)
            hbnumfreqs[ii] = 0;
    }
    else if (cp_getvar("hbnumfreq", CP_LIST, &var, 0)) {
        int ii = 0;
        for (tv = var; tv; tv = tv->va_next)
            if (tv->va_type == CP_NUM) {
                if (ii > 9) {
                    fprintf(stderr, "Warning: too many frequencies (> 10), ignored\n");
                    break;
                }
                hbnumfreqs[ii] = tv->va_num;
                ii++;
            }
            else
                fprintf(cp_err, "Error: bad syntax for hbnumfreq\n");
        for (int jj = ii; jj < 10; jj++)
            hbnumfreqs[jj] = 0;
    }

    return (0);

}

/* Set the .hb command parameters:
   Currently supported:
   fundamental frequency f1
   fundamental frequency f2 */
int
HBsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    HBAN *job = (HBAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case HB_F1:
        if (value->rValue <= 0.0) {
            errMsg = copy("A positive fundamental frequency is required for HB");
            job->HBFreq1 = 1.0;
            return(E_PARMVAL);
        }

        job->HBFreq1 = value->rValue;
        break;


    case HB_F2:
        if (value->rValue < 0.0) {
            errMsg = copy("A negative second fundamental frequency is invalid for multitone HB");
            job->HBFreq2 = 1.0;
            return(E_PARMVAL);
        }

        job->HBFreq2 = value->rValue;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm HBparms[] = {
    { "f1",    HB_F1,   IF_SET|IF_ASK|IF_REAL, "fundamental frequency" },
    { "f2",    HB_F2,   IF_SET|IF_ASK|IF_REAL, "optional second fundamental frequency" }
};

SPICEanalysis HBinfo  = {
    {
        "HB",
        "Harmonic Balance analysis",

        NUMELEMS(HBparms),
        HBparms
    },
    sizeof(HBAN),
    FREQUENCYDOMAIN,
    1,
    HBsetParm,
    HBaskQuest,
    NULL,
    HBan
};
#endif
