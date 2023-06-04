/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ltradefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
LTRAaccept(CKTcircuit* ckt, GENmodel* inModel)
{
    LTRAmodel* model = (LTRAmodel*)inModel;
    LTRAinstance* here;
    double v1, v2, v3, v4;
    double v5, v6, d1, d2, d3, d4;
    int tmp_test;
    int error;
    int compact = 1;


    /* loop through all the transmission line models */
    for (; model != NULL; model = LTRAnextModel(model)) {

        if (ckt->CKTmode & MODEINITTRAN) {

#define LTRAmemMANAGE(a,b) \
    if ( a != NULL) FREE(a);\
    a = TMALLOC(double, b);

            model->LTRAmodelListSize = 100;

            LTRAmemMANAGE(model->LTRAh1dashCoeffs, model->LTRAmodelListSize)
            LTRAmemMANAGE(model->LTRAh2Coeffs, model->LTRAmodelListSize)
            LTRAmemMANAGE(model->LTRAh3dashCoeffs, model->LTRAmodelListSize)
        }
        if (ckt->CKTtimeIndex >= model->LTRAmodelListSize) {	/* need more space */
            model->LTRAmodelListSize += ckt->CKTsizeIncr;

            model->LTRAh1dashCoeffs = TREALLOC(double, model->LTRAh1dashCoeffs, model->LTRAmodelListSize);
            model->LTRAh2Coeffs = TREALLOC(double, model->LTRAh2Coeffs, model->LTRAmodelListSize);
            model->LTRAh3dashCoeffs = TREALLOC(double, model->LTRAh3dashCoeffs, model->LTRAmodelListSize);
        }
        /* loop through all the instances of the model */
        for (here = LTRAinstances(model); here != NULL;
            here = LTRAnextInstance(here)) {

            if (ckt->CKTmode & MODEINITTRAN) {
                here->LTRAinstListSize = (int)MAX(10, ckt->CKTtimeListSize);

                LTRAmemMANAGE(here->LTRAv1, here->LTRAinstListSize)
                LTRAmemMANAGE(here->LTRAi1, here->LTRAinstListSize)
                LTRAmemMANAGE(here->LTRAv2, here->LTRAinstListSize)
                LTRAmemMANAGE(here->LTRAi2, here->LTRAinstListSize)
            }
            /*
             * why is this here? ask TQ
             *
             * if (ckt->CKTtimeIndex == 0? 1: (ckt->CKTtime-
             * (ckt->CKTtimePoints+ckt->CKTtimeIndex-1) > ckt->CKTminBreak)) {
             *
             */
            if (ckt->CKTtimeIndex >= here->LTRAinstListSize) {	/* need more space */
                here->LTRAinstListSize += ckt->CKTsizeIncr;

                here->LTRAv1 = TREALLOC(double, here->LTRAv1, here->LTRAinstListSize);
                here->LTRAi1 = TREALLOC(double, here->LTRAi1, here->LTRAinstListSize);
                here->LTRAi2 = TREALLOC(double, here->LTRAi2, here->LTRAinstListSize);
                here->LTRAv2 = TREALLOC(double, here->LTRAv2, here->LTRAinstListSize);
            }
            *(here->LTRAv1 + ckt->CKTtimeIndex) = *(ckt->CKTrhsOld +
                here->LTRAposNode1) - *(ckt->CKTrhsOld +
                    here->LTRAnegNode1);
            *(here->LTRAv2 + ckt->CKTtimeIndex) = *(ckt->CKTrhsOld +
                here->LTRAposNode2) - *(ckt->CKTrhsOld +
                    here->LTRAnegNode2);
            *(here->LTRAi1 + ckt->CKTtimeIndex) = *(ckt->CKTrhsOld +
                here->LTRAbrEq1);
            *(here->LTRAi2 + ckt->CKTtimeIndex) = *(ckt->CKTrhsOld +
                here->LTRAbrEq2);

            if (ckt->CKTtryToCompact && (ckt->CKTtimeIndex >= 2)) {

                /*
                 * figure out if the last 3 points lie on a st. line for all the
                 * terminal variables
                 */
                {
                    double t1, t2, t3;

                    t1 = *(ckt->CKTtimePoints + ckt->CKTtimeIndex - 2);
                    t2 = *(ckt->CKTtimePoints + ckt->CKTtimeIndex - 1);
                    t3 = *(ckt->CKTtimePoints + ckt->CKTtimeIndex);

                    if (compact) {
                        compact = LTRAstraightLineCheck(t1,
                            *(here->LTRAv1 + ckt->CKTtimeIndex - 2),
                            t2, *(here->LTRAv1 + ckt->CKTtimeIndex - 1),
                            t3, *(here->LTRAv1 + ckt->CKTtimeIndex),
                            model->LTRAstLineReltol, model->LTRAstLineAbstol);
                    }
                    if (compact) {
                        compact = LTRAstraightLineCheck(t1,
                            *(here->LTRAv2 + ckt->CKTtimeIndex - 2),
                            t2, *(here->LTRAv2 + ckt->CKTtimeIndex - 1),
                            t3, *(here->LTRAv2 + ckt->CKTtimeIndex),
                            model->LTRAstLineReltol, model->LTRAstLineAbstol);
                    }
                    if (compact) {
                        compact = LTRAstraightLineCheck(t1,
                            *(here->LTRAi1 + ckt->CKTtimeIndex - 2),
                            t2, *(here->LTRAi1 + ckt->CKTtimeIndex - 1),
                            t3, *(here->LTRAi1 + ckt->CKTtimeIndex),
                            model->LTRAstLineReltol, model->LTRAstLineAbstol);
                    }
                    if (compact) {
                        compact = LTRAstraightLineCheck(t1,
                            *(here->LTRAi2 + ckt->CKTtimeIndex - 2),
                            t2, *(here->LTRAi2 + ckt->CKTtimeIndex - 1),
                            t3, *(here->LTRAi2 + ckt->CKTtimeIndex),
                            model->LTRAstLineReltol, model->LTRAstLineAbstol);
                    }
                }
            }
            if (ckt->CKTtimeIndex > 0) {
#ifdef NOTDEF
                v1 = (*(here->LTRAv1 + ckt->CKTtimeIndex) +
                    *(here->LTRAi1 + ckt->CKTtimeIndex) *
                    model->LTRAimped) * model->LTRAattenuation;
                v2 = (*(here->LTRAv1 + ckt->CKTtimeIndex - 1) +
                    *(here->LTRAi1 + ckt->CKTtimeIndex - 1)
                    * model->LTRAimped) * model->LTRAattenuation;
                v3 = (*(here->LTRAv2 + ckt->CKTtimeIndex) +
                    *(here->LTRAi2 + ckt->CKTtimeIndex) *
                    model->LTRAimped) * model->LTRAattenuation;
                v4 = (*(here->LTRAv2 + ckt->CKTtimeIndex - 1) +
                    *(here->LTRAi2 + ckt->CKTtimeIndex - 1) *
                    model->LTRAimped) * model->LTRAattenuation;
                if ((fabs(v1 - v2) >= 50 * ckt->CKTreltol *
                    MAX(fabs(v1), fabs(v2)) + 50 * ckt->CKTvoltTol) ||
                    (fabs(v3 - v4) >= 50 * ckt->CKTreltol *
                        MAX(fabs(v3), fabs(v4)) + 50 * ckt->CKTvoltTol)) {
                    /* changing - need to schedule after delay */
                    /*
                     * don't really need this error =
                     * CKTsetBreak(ckt,ckt->CKTtime+model->LTRAtd); if(error)
                     * return(error);
                     */
                     /* the PREVIOUS point is the real breakpoint */
                    error = CKTsetBreak(ckt,
                        *(ckt->CKTtimePoints + ckt->CKTtimeIndex - 1) +
                        model->LTRAtd);
                    CKTbreakDump(ckt);
                    if (error)
                        return (error);
                }
#else
                /*
                 * remove the hack here - store the total inputs for the last 2 or 3
                 * timesteps
                 */

                v1 = (*(here->LTRAv1 + ckt->CKTtimeIndex) +
                    *(here->LTRAi1 + ckt->CKTtimeIndex) *
                    model->LTRAimped) * model->LTRAattenuation;
                v2 = (*(here->LTRAv1 + ckt->CKTtimeIndex - 1) +
                    *(here->LTRAi1 + ckt->CKTtimeIndex - 1) *
                    model->LTRAimped) * model->LTRAattenuation;
                v3 = ckt->CKTtimeIndex < 2 ? v2 : (*(here->LTRAv1 + ckt->CKTtimeIndex - 2) +
                    *(here->LTRAi1 + ckt->CKTtimeIndex - 2) *
                    model->LTRAimped) * model->LTRAattenuation;
                v4 = (*(here->LTRAv2 + ckt->CKTtimeIndex) +
                    *(here->LTRAi2 + ckt->CKTtimeIndex) *
                    model->LTRAimped) * model->LTRAattenuation;
                v5 = (*(here->LTRAv2 + ckt->CKTtimeIndex - 1) +
                    *(here->LTRAi2 + ckt->CKTtimeIndex - 1) *
                    model->LTRAimped) * model->LTRAattenuation;
                v6 = ckt->CKTtimeIndex < 2 ? v5 : (*(here->LTRAv2 + ckt->CKTtimeIndex - 2) +
                    *(here->LTRAi2 + ckt->CKTtimeIndex - 2) *
                    model->LTRAimped) * model->LTRAattenuation;

                d1 = (v1 - v2) / (*(ckt->CKTtimePoints + ckt->CKTtimeIndex) -
                    *(ckt->CKTtimePoints + ckt->CKTtimeIndex - 1));
                d2 = (ckt->CKTtimeIndex < 2)
                    ? 0
                    : (v2 - v3) / (*(ckt->CKTtimePoints + ckt->CKTtimeIndex - 1) -
                        *(ckt->CKTtimePoints + ckt->CKTtimeIndex - 2));
                d3 = (v4 - v5) / (*(ckt->CKTtimePoints + ckt->CKTtimeIndex) -
                    *(ckt->CKTtimePoints + ckt->CKTtimeIndex - 1));
                d4 = (ckt->CKTtimeIndex < 2)
                    ? 0
                    : (v5 - v6) / (*(ckt->CKTtimePoints + ckt->CKTtimeIndex - 1) -
                        *(ckt->CKTtimePoints + ckt->CKTtimeIndex - 2));

                /*
                 * here we have a big problem with the scheme boxed by the *s below.
                 * Note the following: if LTRAreltol == 1, (assuming LTRAabstol==0)
                 * then breakpoints are set if and only if d1 and d2 have opposite
                 * signs or one is zero. If LTRAreltol > 2, breakpoints are never
                 * set. The problem is that when the waveform is steady at a value,
                 * small random numerical inaccuracies may produce derivatives of
                 * opposite sign, and breakpoints get set. This can, in practice, get
                 * quite killing... To alleviate this, we try to determine if the
                 * waveform is actually steady using the following tests: 1. Check if
                 * the maximum difference between v1,v2 and v3 is less than
                 * 50*CKTreltol*(the average of v1,v2,and v3) + 50*ckt->CKTabstol
                 * (the 50 has been taken from the NOTDEF section above, reason
                 * unknown - hopefully there is a good reason for it - ask TQ)
                 *
                 * 2. Criterion 1 may be satisfied by a legitimate breakpoint. To
                 * further check, find one more derivative one timepoint ago and see
                 * if that is close to d2. If not, then the likelihood of numerical
                 * inaccuracies is greater...
                 */

                 /*********************************************************************
                                 if( (fabs(d1-d2) >= model->LTRAreltol*MAX(fabs(d1),fabs(d2))+
                                         model->LTRAabstol) ||
                                         (fabs(d3-d4) >= model->LTRAreltol*MAX(fabs(d3),fabs(d4))+
                                         model->LTRAabstol) ) {
                 *********************************************************************/
#define CHECK(a,b,c) (MAX(MAX(a,b),c)-MIN(MIN(a,b),c) >= \
    fabs(50.0*(ckt->CKTreltol/3.0*(a+b+c) +\
    ckt->CKTabstol)))

                tmp_test = (fabs(d1 - d2)
                    >= model->LTRAreltol * MAX(fabs(d1), fabs(d2)) +
                    model->LTRAabstol)
                    && CHECK(v1, v2, v3);
                if (tmp_test || ((fabs(d3 - d4)
                    >= model->LTRAreltol * MAX(fabs(d3), fabs(d4)) +
                    model->LTRAabstol)
                    && CHECK(v4, v5, v6))) {
                    /* criterion 2 not implemented yet... */
                    error = CKTsetBreak(ckt,
                        *(ckt->CKTtimePoints + ckt->CKTtimeIndex - 1) +
                        model->LTRAtd);
                    /*
                     * this is not necessary - the previous timepoint was the
                     * breakpoint error = CKTsetBreak(ckt, ckt->CKTtime +
                     * model->LTRAtd);
                     */
#ifdef LTRADEBUG
                    fprintf(stdout, "\nbreakpoints set at %14.14g at %14.14g at time %14.14g\n", ckt->CKTtime + model->LTRAtd, *(ckt->CKTtimePoints + ckt->CKTtimeIndex
                        - 1) + model->LTRAtd, ckt->CKTtime);
                    fprintf(stdout, "d1 through d4 are %14.14g %14.14g %14.14g %14.14g\n\n", d1, d2, d3, d4);
#endif
                    if (error)
                        return (error);
                }
                /* } */
#endif				/* NOTDEF */
            }
            /* ask TQ } */

        }				/* instance */
    }				/* model */


    if (ckt->CKTtryToCompact && compact && (ckt->CKTtimeIndex >= 2)) {

        /*
         * last three timepoints have variables lying on a straight line, do a
         * compaction
         */

        model = (LTRAmodel*)inModel;
        for (; model != NULL; model = LTRAnextModel(model)) {
            for (here = LTRAinstances(model); here != NULL;
                here = LTRAnextInstance(here)) {
                *(here->LTRAv1 + ckt->CKTtimeIndex - 1) = *(here->LTRAv1 +
                    ckt->CKTtimeIndex);
                *(here->LTRAv2 + ckt->CKTtimeIndex - 1) = *(here->LTRAv2 +
                    ckt->CKTtimeIndex);
                *(here->LTRAi1 + ckt->CKTtimeIndex - 1) = *(here->LTRAi1 +
                    ckt->CKTtimeIndex);
                *(here->LTRAi2 + ckt->CKTtimeIndex - 1) = *(here->LTRAi2 +
                    ckt->CKTtimeIndex);
            }
        }
        *(ckt->CKTtimePoints + ckt->CKTtimeIndex - 1) =
            *(ckt->CKTtimePoints + ckt->CKTtimeIndex);
        ckt->CKTtimeIndex--;
#ifdef LTRADEBUG
        fprintf(stdout, "compacted at time=%g\n", *(ckt->CKTtimePoints + ckt->CKTtimeIndex));
        fflush(stdout);
#endif
    }
    return (OK);
}
