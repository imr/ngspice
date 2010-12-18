/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice.h"
#include "vsrcdefs.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"
#include "1-f-code.h"


/* ARGSUSED */
int
VSRCparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    int i;
    VSRCinstance *here = (VSRCinstance *)inst;

    NG_IGNORE(select);

    switch(param) {
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
            switch(value->v.numValue) {
                case 2:
                    here->VSRCacPhase = *(value->v.vec.rVec+1);
                    here->VSRCacPGiven = TRUE;
                case 1:
                    here->VSRCacMag = *(value->v.vec.rVec);
                    here->VSRCacMGiven = TRUE;
                case 0:
                    here->VSRCacGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
        case VSRC_PULSE:
            here->VSRCfunctionType = PULSE;
            here->VSRCfuncTGiven = TRUE;
            here->VSRCcoeffs = value->v.vec.rVec;
            here->VSRCfunctionOrder = value->v.numValue;
            here->VSRCcoeffsGiven = TRUE;
            break;
        case VSRC_SINE:
            here->VSRCfunctionType = SINE;
            here->VSRCfuncTGiven = TRUE;
            here->VSRCcoeffs = value->v.vec.rVec;
            here->VSRCfunctionOrder = value->v.numValue;
            here->VSRCcoeffsGiven = TRUE;
            break;
        case VSRC_EXP:
            here->VSRCfunctionType = EXP;
            here->VSRCfuncTGiven = TRUE;
            here->VSRCcoeffs = value->v.vec.rVec;
            here->VSRCfunctionOrder = value->v.numValue;
            here->VSRCcoeffsGiven = TRUE;
            break;
        case VSRC_PWL:
            here->VSRCfunctionType = PWL;
            here->VSRCfuncTGiven = TRUE;
            here->VSRCcoeffs = value->v.vec.rVec;
            here->VSRCfunctionOrder = value->v.numValue;
            here->VSRCcoeffsGiven = TRUE;
            
            for(i=0;i<(here->VSRCfunctionOrder/2)-1;i++) {
                  if(*(here->VSRCcoeffs+2*(i+1))<=*(here->VSRCcoeffs+2*i)) {
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
            here->VSRCr = value->rValue;
            here->VSRCrGiven = TRUE;
	    
	    for ( i = 0; i < here->VSRCfunctionOrder; i += 2 ) {
	      here->VSRCrBreakpt = i;
              if ( here->VSRCr == *(here->VSRCcoeffs+i) ) break;
	    }
		
        end_time     = *(here->VSRCcoeffs + here->VSRCfunctionOrder-2);
	    if ( here->VSRCr > end_time ) {
	      fprintf(stderr, "ERROR: repeat start time value %g for pwl voltage source must be smaller than final time point given!\n", here->VSRCr );
	      return ( E_PARMVAL );
	    }

	    if ( here->VSRCr != *(here->VSRCcoeffs+here->VSRCrBreakpt) ) {
	      fprintf(stderr, "ERROR: repeat start time value %g for pwl voltage source does not match any time point given!\n", here->VSRCr );
	      return ( E_PARMVAL );
	    }

        break; }
        case VSRC_SFFM:
            here->VSRCfunctionType = SFFM;
            here->VSRCfuncTGiven = TRUE;
            here->VSRCcoeffs = value->v.vec.rVec;
            here->VSRCfunctionOrder = value->v.numValue;
            here->VSRCcoeffsGiven = TRUE;
            break;
	case VSRC_AM:
	    if(value->v.numValue <2) return(E_BADPARM);
            here->VSRCfunctionType = AM;
            here->VSRCfuncTGiven = TRUE;
            here->VSRCcoeffs = value->v.vec.rVec;
            here->VSRCfunctionOrder = value->v.numValue;
            here->VSRCcoeffsGiven = TRUE;	
	    break;    
	case VSRC_D_F1:
	    here->VSRCdF1given = TRUE;
	    here->VSRCdGiven = TRUE;
	    switch(value->v.numValue) {
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
		    break;
	    }
	case VSRC_D_F2:
	    here->VSRCdF2given = TRUE;
	    here->VSRCdGiven = TRUE;
	    switch(value->v.numValue) {
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
            here->VSRCcoeffs = value->v.vec.rVec;
            here->VSRCfunctionOrder = value->v.numValue;
            here->VSRCcoeffsGiven = TRUE;

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

            here->VSRCtrnoise_state =
                trnoise_state_init(NA, TS, NALPHA, NAMP, RTSAM, RTSCAPT, RTSEMT);
        }
            break;		
        default:
            return(E_BADPARM);
    }
    return(OK);
}
