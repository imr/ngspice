#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/bool.h"
#include "circuits.h"
#include "ngspice/wordlist.h"
#include "variable.h"
#include "com_option.h"


/* The option command. Syntax is option [opt ...] [opt = val ...].
 * Val may be a string, an int, a float, or a list of the
 * form ( elt1 elt2 ... ).  */
void
com_option(wordlist *wl)
{
    struct variable *vars, *v;

    CKTcircuit *circuit = NULL;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: no circuit loaded\n");
        return;
    }

    circuit = (ft_curckt->ci_ckt);

    if (wl == NULL) {
        printf("******************************\n");
        printf("* Current simulation options *\n");
        printf("******************************\n\n");
        printf("Temperatures:\n");
        printf("temp = %f\n", circuit->CKTtemp);
        printf("tnom = %f\n", circuit->CKTnomTemp);

        printf("\nIntegration method summary:\n");
        switch (circuit->CKTintegrateMethod)
        {
        case TRAPEZOIDAL:
            printf("Integration Method = TRAPEZOIDAL\n");
            break;
        case GEAR:
            printf("Integration Method = GEAR\n");
            break;
        default:
            printf("Unknown integration method\n");
        }
        printf("MaxOrder = %d\n", circuit->CKTmaxOrder);
        printf("xmu = %g\n", circuit->CKTxmu);
        printf("indverbosity = %d\n", circuit->CKTindverbosity);
        printf("epsmin = %g\n", circuit->CKTepsmin);

        printf("\nTolerances (absolute):\n");
        printf("abstol      (current) = %g\n", circuit->CKTabstol);
        printf("chgtol      (charge)  = %g\n", circuit->CKTchgtol);
        printf("vntol       (voltage) = %g\n", circuit->CKTvoltTol);
        printf("pivtol      (pivot)   = %g\n", circuit->CKTpivotAbsTol);

        printf("\nTolerances (relative):\n");
        printf("reltol      (current) = %g\n", circuit->CKTreltol);
        printf("pivrel      (pivot)   = %g\n", circuit->CKTpivotRelTol);

        printf("\nIteration limits:\n");
        printf("itl1 (DC iterations) = %d\n", circuit->CKTdcMaxIter);
        printf("itl2 (DC transfer curve iterations) = %d\n", circuit->CKTdcTrcvMaxIter);
        printf("itl4 (transient iterations) = %d\n", circuit->CKTtranMaxIter);
        printf("gminsteps = %d\n", circuit->CKTnumGminSteps);
        printf("srcsteps = %d\n", circuit->CKTnumSrcSteps);

        printf("\nTruncation error correction:\n");
        printf("trtol = %f\n", circuit->CKTtrtol);
#ifdef NEWTRUNC
        printf("ltereltol = %g\n", circuit->CKTlteReltol);
        printf("lteabstol = %g\n", circuit->CKTlteAbstol);
#endif /* NEWTRUNC */

        printf("\nConductances:\n");
        printf("gmin     (devices)  = %g\n", circuit->CKTgmin);
        printf("diaggmin (stepping) = %g\n", circuit->CKTdiagGmin);
        printf("gshunt = %g\n", circuit->CKTgshunt);
        printf("cshunt = %g\n", circuit->CKTcshunt);

        printf("delmin = %g\n", circuit->CKTdelmin);

        printf("\nDefault parameters for MOS devices\n");
        printf("Default M: %f\n", circuit->CKTdefaultMosM);
        printf("Default L: %f\n", circuit->CKTdefaultMosL);
        printf("Default W: %f\n", circuit->CKTdefaultMosW);
        printf("Default AD: %f\n", circuit->CKTdefaultMosAD);
        printf("Default AS: %f\n", circuit->CKTdefaultMosAS);

        return;
    }

    vars = cp_setparse(wl);

    /* This is sort of a hassle... */
    for (v = vars; v; v = v->va_next) {
        void *s;
        switch (v->va_type) {
        case CP_BOOL:
            s = &v->va_bool;
            break;
        case CP_NUM:
            s = &v->va_num;
            break;
        case CP_REAL:
            s = &v->va_real;
            break;
        case CP_STRING:
            s = v->va_string;
            break;
        case CP_LIST:
            s = v->va_vlist;
            break;
        default:
            s = NULL;
        }

        /* qui deve settare le opzioni di simulazione */
        cp_vset(v->va_name, v->va_type, s);
    }

    free_struct_variable(vars);
}

