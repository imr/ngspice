/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/1-f-code.h"


static void copy_coeffs(VSRCinstance *here, IFvalue *value)
{
    int n = value->v.numValue;

    if(here->VSRCcoeffs)
        tfree(here->VSRCcoeffs);

    here->VSRCcoeffs = TMALLOC(double, n);
    here->VSRCfunctionOrder = n;
    here->VSRCcoeffsGiven = TRUE;

    memcpy(here->VSRCcoeffs, value->v.vec.rVec, (size_t) n * sizeof(double));
}


/* ARGSUSED */
int
VSRCparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    int i;
    VSRCinstance *here = (VSRCinstance *) inst;

    NG_IGNORE(select);

    switch (param) {
        case VSRC_DC:
            here->VSRCdcValue = value->rValue;
            here->VSRCdcGiven = TRUE;
            break;

        case VSRC_AC_MAG:
            here->VSRCacMag = value->rValue;
            here->VSRCacMGiven = TRUE;
            here->VSRCacGiven = TRUE;
            break;

        case VSRC_AC_PHASE:
            here->VSRCacPhase = value->rValue;
            here->VSRCacPGiven = TRUE;
            here->VSRCacGiven = TRUE;
            break;

        case VSRC_AC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 2:
                    here->VSRCacPhase = *(value->v.vec.rVec+1);
                    here->VSRCacPGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->VSRCacMag = *(value->v.vec.rVec);
                    here->VSRCacMGiven = TRUE;
                    /* FALLTHROUGH */
                case 0:
                    here->VSRCacGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;

        case VSRC_PULSE:
            if(value->v.numValue < 2)
                return(E_BADPARM);
            here->VSRCfunctionType = PULSE;
            here->VSRCfuncTGiven = TRUE;
            copy_coeffs(here, value);
            break;

        case VSRC_SINE:
            if(value->v.numValue < 2)
                return(E_BADPARM);
            here->VSRCfunctionType = SINE;
            here->VSRCfuncTGiven = TRUE;
            copy_coeffs(here, value);
            break;

        case VSRC_EXP:
            if(value->v.numValue < 2)
                return(E_BADPARM);
            here->VSRCfunctionType = EXP;
            here->VSRCfuncTGiven = TRUE;
            copy_coeffs(here, value);
            break;

        case VSRC_PWL:
            if(value->v.numValue < 2)
                return(E_BADPARM);
            here->VSRCfunctionType = PWL;
            here->VSRCfuncTGiven = TRUE;
            copy_coeffs(here, value);

            for (i=0; i<(here->VSRCfunctionOrder/2)-1; i++) {
                  if (*(here->VSRCcoeffs+2*(i+1))<=*(here->VSRCcoeffs+2*i)) {
                     fprintf(stderr, "Warning : voltage source %s",
                                                               here->VSRCname);
                     fprintf(stderr, " has non-increasing PWL time points.\n");
                  }
            }

            break;

        case VSRC_TD:
            here->VSRCrdelay = value->rValue;
            break;

        case VSRC_R: {
            double end_time;
            /* Parameter r of pwl may now be parameterized:
               if r == -1, no repetition done.
               if r == 0, repeat forever.
               if r == xx, repeat from time xx to last time point given. */
            if (value->rValue < -0.5) {
                here->VSRCrGiven = FALSE;
                break;
            }

            /* buggy input? r is not a repetition coefficient */
            if (!here->VSRCcoeffs || here->VSRCfunctionOrder < 2) {
                here->VSRCrGiven = FALSE;
                break;
            }

            here->VSRCr = value->rValue;
            here->VSRCrGiven = TRUE;

            for ( i = 0; i < here->VSRCfunctionOrder; i += 2 ) {
              here->VSRCrBreakpt = i;
                  if ( here->VSRCr == *(here->VSRCcoeffs+i) ) break;
            }

            end_time     = *(here->VSRCcoeffs + here->VSRCfunctionOrder-2);
            if ( here->VSRCr >= end_time ) {
              fprintf(stderr, "ERROR: repeat start time value %g for pwl voltage source must be smaller than final time point given!\n", here->VSRCr );
              return ( E_PARMVAL );
            }

            if ( here->VSRCr != *(here->VSRCcoeffs+here->VSRCrBreakpt) ) {
              fprintf(stderr, "ERROR: repeat start time value %g for pwl voltage source does not match any time point given!\n", here->VSRCr );
              return ( E_PARMVAL );
            }

            break;
        }

        case VSRC_SFFM:
            if(value->v.numValue < 2)
                return(E_BADPARM);
            here->VSRCfunctionType = SFFM;
            here->VSRCfuncTGiven = TRUE;
            copy_coeffs(here, value);
            break;

        case VSRC_AM:
            if(value->v.numValue < 2)
                return(E_BADPARM);
            here->VSRCfunctionType = AM;
            here->VSRCfuncTGiven = TRUE;
            copy_coeffs(here, value);
            break;

        case VSRC_D_F1:
            here->VSRCdF1given = TRUE;
            here->VSRCdGiven = TRUE;
            switch (value->v.numValue) {
            case 2:
                here->VSRCdF1phase = *(value->v.vec.rVec+1);
                here->VSRCdF1mag = *(value->v.vec.rVec);
                break;
            case 1:
                here->VSRCdF1mag = *(value->v.vec.rVec);
                here->VSRCdF1phase = 0.0;
                break;
            case 0:
                here->VSRCdF1mag = 1.0;
                here->VSRCdF1phase = 0.0;
                break;
            default:
                return(E_BADPARM);
            }
            break;

        case VSRC_D_F2:
            here->VSRCdF2given = TRUE;
            here->VSRCdGiven = TRUE;
            switch (value->v.numValue) {
            case 2:
                here->VSRCdF2phase = *(value->v.vec.rVec+1);
                here->VSRCdF2mag = *(value->v.vec.rVec);
                break;
            case 1:
                here->VSRCdF2mag = *(value->v.vec.rVec);
                here->VSRCdF2phase = 0.0;
                break;
            case 0:
                here->VSRCdF2mag = 1.0;
                here->VSRCdF2phase = 0.0;
                break;
            default:
                return(E_BADPARM);
            }
            break;

        case VSRC_TRNOISE: {
            double NA, TS;
            double NALPHA = 0.0;
            double NAMP   = 0.0;
            double RTSAM   = 0.0;
            double RTSCAPT   = 0.0;
            double RTSEMT   = 0.0;

            here->VSRCfunctionType = TRNOISE;
            here->VSRCfuncTGiven = TRUE;
            copy_coeffs(here, value);

            NA = here->VSRCcoeffs[0]; // input is rms value
            TS = here->VSRCcoeffs[1]; // time step

            if (here->VSRCfunctionOrder > 2)
                NALPHA = here->VSRCcoeffs[2]; // 1/f exponent

            if (here->VSRCfunctionOrder > 3 && NALPHA != 0.0)
                NAMP = here->VSRCcoeffs[3]; // 1/f amplitude

            if (here->VSRCfunctionOrder > 4)
                RTSAM = here->VSRCcoeffs[4]; // RTS amplitude

            if (here->VSRCfunctionOrder > 5 && RTSAM != 0.0)
                RTSCAPT = here->VSRCcoeffs[5]; // RTS trap capture time

            if (here->VSRCfunctionOrder > 6 && RTSAM != 0.0)
                RTSEMT = here->VSRCcoeffs[6]; // RTS trap emission time

            /* after an 'alter' command to the TRNOISE voltage source the state gets re-written
               with the new parameters. So free the old state first. */
            trnoise_state_free(here->VSRCtrnoise_state);
            here->VSRCtrnoise_state =
                trnoise_state_init(NA, TS, NALPHA, NAMP, RTSAM, RTSCAPT, RTSEMT);
        }
        break;

        case VSRC_TRRANDOM: {
            double TD = 0.0, TS;
            int rndtype = 1;
            double PARAM1 = 1.0;
            double PARAM2 = 0.0;

            here->VSRCfunctionType = TRRANDOM;
            here->VSRCfuncTGiven = TRUE;
            copy_coeffs(here, value);

            rndtype = (int)here->VSRCcoeffs[0]; // type of random function
            TS = here->VSRCcoeffs[1]; // time step
            if (here->VSRCfunctionOrder > 2)
                TD = here->VSRCcoeffs[2]; // delay

            if (here->VSRCfunctionOrder > 3)
                PARAM1 = here->VSRCcoeffs[3]; // first parameter

            if (here->VSRCfunctionOrder > 4)
                PARAM2 = here->VSRCcoeffs[4]; // second parameter

            /* after an 'alter' command to the TRRANDOM voltage source the state gets re-written
               with the new parameters. So free the old state first. */
            tfree(here->VSRCtrrandom_state);
            here->VSRCtrrandom_state =
                trrandom_state_init(rndtype, TS, TD, PARAM1, PARAM2);
        }
        break;

#ifdef SHARED_MODULE
        case VSRC_EXTERNAL: {
            here->VSRCfunctionType = EXTERNAL;
            here->VSRCfuncTGiven = TRUE;
            /* no coefficients
            copy_coeffs(here, value);
            */
        }
        break;
#endif
#ifdef RFSPICE
        /*
        * NB If either Freq or Power are given, the Function type is overridden
        * If not, we have a passive port: can be used for AC/SP/Noise but the time
        * domain value is given by preceding Function definition (if present).
        */
        case VSRC_PORTNUM:
        {
            here->VSRCportNum = value->iValue;
            here->VSRCportNumGiven = TRUE;
            here->VSRCisPort = (here->VSRCportNum > 0);
            if (here->VSRCportZ0 <= 0.0) {
                here->VSRCportZ0 = 50;
                here->VSRCVAmplitude =
                    sqrt(here->VSRCportPower * 4.0 * here->VSRCportZ0);
            }
            break;
        }
        case VSRC_PORTZ0:
        {
            here->VSRCportZ0 = value->rValue;
            here->VSRCVAmplitude =
                sqrt(here->VSRCportPower * 4.0 * here->VSRCportZ0);
            here->VSRCportZ0Given = TRUE;
            break;
        }
        case VSRC_PORTPWR:
        {
            here->VSRCportPower = value->rValue;
            here->VSRCportPowerGiven = TRUE;

            here->VSRCfunctionType = PORT;

            break;
        }
        case VSRC_PORTFREQ:
        {
            here->VSRCportFreq = value->rValue;
            here->VSRCportFreqGiven = TRUE;

            here->VSRCfunctionType = PORT;

            break;
        }
        case VSRC_PORTPHASE:
        {
            here->VSRCportPhase = value->rValue;
            here->VSRCportPhaseGiven = TRUE;
        }
        break;
#endif
        default:
            return(E_BADPARM);
    }

    return(OK);
}
