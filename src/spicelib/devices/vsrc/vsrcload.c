/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
$Id$
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "vsrcdefs.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"
#undef WaGauss
#ifdef FastRand
#include "FastNorm3.h"
#elif defined (WaGauss)
#include "wallace.h"
#else
extern void rgauss(double* py1, double* py2);
#endif
#include "1-f-code.h"

#ifdef XSPICE_EXP
/* gtri - begin - wbk - modify for supply ramping option */
#include "cmproto.h"
/* gtri - end   - wbk - modify for supply ramping option */
#endif /* XSPICE_EXP */

int
VSRCload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    VSRCinstance *here;
    double time;
    double value = 0.0;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->VSRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->VSRCinstances; here != NULL ;
                here=here->VSRCnextInstance) {
            if (here->VSRCowner != ARCHme) continue;
            
            *(here->VSRCposIbrptr) += 1.0 ;
            *(here->VSRCnegIbrptr) -= 1.0 ;
            *(here->VSRCibrPosptr) += 1.0 ;
            *(here->VSRCibrNegptr) -= 1.0 ;
            if( (ckt->CKTmode & (MODEDCOP | MODEDCTRANCURVE)) &&
                    here->VSRCdcGiven ) {
                /* grab dc value */
#ifdef XSPICE_EXP
                value = here->VSRCdcValue;
#else
                value = ckt->CKTsrcFact * here->VSRCdcValue;
#endif
            } else {
                if(ckt->CKTmode & (MODEDC)) {
                    time = 0;
                } else {
                    time = ckt->CKTtime;
                }
                /* use the transient functions */
                switch(here->VSRCfunctionType) {
                default: { /* no function specified:   use the DC value */
                    value = here->VSRCdcValue;
                    break;
                }
                
                case PULSE: {
		            double V1, V2, TD, TR, TF, PW, PER;
                    double basetime = 0;
#ifdef XSPICE
                    double PHASE;
                    double phase;
                    double deltat;
#endif
                    V1 = here->VSRCcoeffs[0];
                    V2 = here->VSRCcoeffs[1];
                    TD = here->VSRCfunctionOrder > 2
                       ? here->VSRCcoeffs[2] : 0.0;
                    TR = here->VSRCfunctionOrder > 3
                       && here->VSRCcoeffs[3] != 0.0
                       ? here->VSRCcoeffs[3] : ckt->CKTstep;
                    TF = here->VSRCfunctionOrder > 4
                       && here->VSRCcoeffs[4] != 0.0
                       ? here->VSRCcoeffs[4] : ckt->CKTstep;
                    PW = here->VSRCfunctionOrder > 5
                       && here->VSRCcoeffs[5] != 0.0
                       ? here->VSRCcoeffs[5] : ckt->CKTfinalTime;
                    PER = here->VSRCfunctionOrder > 6
                       && here->VSRCcoeffs[6] != 0.0
                       ? here->VSRCcoeffs[6] : ckt->CKTfinalTime;

                    /* shift time by delay time TD */                   
                    time -=  TD;

#ifdef XSPICE
                    /* gtri - begin - wbk - add PHASE parameter */
                    PHASE = here->VSRCfunctionOrder > 7
                       ? here->VSRCcoeffs[7] : 0.0;

		            /* normalize phase to cycles */
                    phase = PHASE / 360.0;
                    phase = fmod(phase, 1.0);
                    deltat =  phase * PER;
                    while (deltat > 0) 
                        deltat -= PER;
                    /* shift time by pase (neg. for pos. phase value) */
                    time += deltat;
                    /* gtri - end - wbk - add PHASE parameter */
#endif		    
                    if(time > PER) {
                        /* repeating signal - figure out where we are */
                        /* in period */
                        basetime = PER * floor(time/PER);
                        time -= basetime;
                    }
                    if (time <= 0 || time >= TR + PW + TF) {
                        value = V1;
                    } else  if (time >= TR && time <= TR + PW) {
                        value = V2;
                    } else if (time > 0 && time < TR) {
                        value = V1 + (V2 - V1) * (time) / TR;
                    } else { /* time > TR + PW && < TR + PW + TF */
                        value = V2 + (V1 - V2) * (time - (TR + PW)) / TF;
                    }

                }
                break;

                case SINE: {
		
                    double VO, VA, FREQ, TD, THETA;
                    /* gtri - begin - wbk - add PHASE parameter */
#ifdef XSPICE
                    double PHASE;
                    double phase;

                    PHASE = here->VSRCfunctionOrder > 5
                       ? here->VSRCcoeffs[5] : 0.0;
		       		
	     	        /* compute phase in radians */ 
                    phase = PHASE * M_PI / 180.0;
#endif
                    VO = here->VSRCcoeffs[0];
                    VA = here->VSRCcoeffs[1];
                    FREQ =  here->VSRCfunctionOrder > 2 
	                   && here->VSRCcoeffs[2] != 0.0
		               ? here->VSRCcoeffs[2] : (1/ckt->CKTfinalTime);
	                TD = here->VSRCfunctionOrder > 3
	                   ? here->VSRCcoeffs[3] : 0.0;
                    THETA = here->VSRCfunctionOrder > 4
	                   ? here->VSRCcoeffs[4] : 0.0;
			   
                    time -= TD;
                    if (time <= 0) {
#ifdef XSPICE
                        value = VO + VA * sin(phase);
                    } else {

                        value = VO + VA * sin(FREQ*time * 2.0 * M_PI + phase) * 
                           exp(-time*THETA);
#else						    
                        value = VO;
                    } else {                        
                       value = VO + VA * sin(FREQ * time * 2.0 * M_PI) * 
                           exp(-(time*THETA));
#endif
/* gtri - end - wbk - add PHASE parameter */				
                    }
                }
                break;

                case EXP: {
                    double V1, V2, TD1, TD2, TAU1, TAU2;
                    
                    V1  = here->VSRCcoeffs[0];
                    V2  = here->VSRCcoeffs[1];
                    TD1 = here->VSRCfunctionOrder > 2 
                       && here->VSRCcoeffs[2] != 0.0
                       ? here->VSRCcoeffs[2] : ckt->CKTstep;
                    TAU1 = here->VSRCfunctionOrder > 3 
                       && here->VSRCcoeffs[3] != 0.0
                       ? here->VSRCcoeffs[3] : ckt->CKTstep;
                    TD2  = here->VSRCfunctionOrder > 4 
                       && here->VSRCcoeffs[4] != 0.0
                       ? here->VSRCcoeffs[4] : TD1 + ckt->CKTstep;
                    TAU2 = here->VSRCfunctionOrder > 5 
                       && here->VSRCcoeffs[5]
                       ? here->VSRCcoeffs[5] : ckt->CKTstep;
 		    
                    if(time <= TD1)  {
                        value = V1;
                    } else if (time <= TD2) {
                         value = V1 + (V2-V1)*(1-exp(-(time-TD1)/TAU1));
                    } else {
                        value = V1 + (V2-V1)*(1-exp(-(time-TD1)/TAU1)) +
                                     (V1-V2)*(1-exp(-(time-TD2)/TAU2)) ;
                    }
                }
                break;

                case SFFM:{
		
                    double VO, VA, FC, MDI, FS;
/* gtri - begin - wbk - add PHASE parameters */
#ifdef XSPICE

                    double PHASEC, PHASES;
                    double phasec;
                    double phases;
		    
                    PHASEC = here->VSRCfunctionOrder > 5
		               ? here->VSRCcoeffs[5] : 0.0;
                    PHASES = here->VSRCfunctionOrder > 6
		               ? here->VSRCcoeffs[6] : 0.0;
			
                    /* compute phases in radians */
                    phasec = PHASEC * M_PI / 180.0;
                    phases = PHASES * M_PI / 180.0;    
#endif				    
                    VO = here->VSRCcoeffs[0];
                    VA = here->VSRCcoeffs[1];
                    FC = here->VSRCfunctionOrder > 2 
                       && here->VSRCcoeffs[2]
                       ? here->VSRCcoeffs[2] : (1/ckt->CKTfinalTime);
                    MDI = here->VSRCfunctionOrder > 3
                       ? here->VSRCcoeffs[3] : 0.0;
                    FS  = here->VSRCfunctionOrder > 4 
                       && here->VSRCcoeffs[4]
                       ? here->VSRCcoeffs[4] : (1/ckt->CKTfinalTime);
#ifdef XSPICE
                    /* compute waveform value */
                    value = VO + VA * 
                        sin((2 * M_PI * FC * time + phasec) +
                        MDI * sin(2.0 * M_PI * FS * time + phases));
#else /* XSPICE */			
                    value = VO + VA * 
                            sin((2.0 * M_PI * FC * time) +
                            MDI * sin(2 * M_PI * FS * time));
#endif /* XSPICE */
/* gtri - end - wbk - add PHASE parameters */
                }
                break;
                case AM:{
		
                    double VA, FC, MF, VO, TD;
/* gtri - begin - wbk - add PHASE parameters */
#ifdef XSPICE
                    double PHASEC, PHASES;
                    double phasec;
                    double phases;
		    
                    PHASEC = here->VSRCfunctionOrder > 5
		            ? here->VSRCcoeffs[5] : 0.0;
                    PHASES = here->VSRCfunctionOrder > 6
		            ? here->VSRCcoeffs[6] : 0.0;
			
                    /* compute phases in radians */
                    phasec = PHASEC * M_PI / 180.0;
                    phases = PHASES * M_PI / 180.0;    

#endif			
                    VA = here->VSRCcoeffs[0];
                    VO = here->VSRCcoeffs[1];
                    MF = here->VSRCfunctionOrder > 2 
                       && here->VSRCcoeffs[2]
                       ? here->VSRCcoeffs[2] : (1/ckt->CKTfinalTime);
                    FC = here->VSRCfunctionOrder > 3
                       ? here->VSRCcoeffs[3] : 0.0;
                    TD  = here->VSRCfunctionOrder > 4 
                       && here->VSRCcoeffs[4]
                       ? here->VSRCcoeffs[4] : 0.0;

                    time -= TD;
                    if (time <= 0) {
                        value = 0;
                    } else {
#ifdef XSPICE
                        /* compute waveform value */
                        value = VA * (VO + sin(2.0 * M_PI * MF * time + phases )) *
                           sin(2 * M_PI * FC * time + phases);
                    
#else /* XSPICE */		    
                        value = VA * (VO + sin(2.0 * M_PI * MF * time)) *
                           sin(2 * M_PI * FC * time);
#endif			
                    }
		    
/* gtri - end - wbk - add PHASE parameters */
                }
                break;
                case PWL: {
                    int i = 0, num_repeat = 0, ii = 0;
                    double foo, repeat_time = 0, end_time, breakpt_time, itime;

                    time -= here->VSRCrdelay;

                    if(time < *(here->VSRCcoeffs)) {
                        foo = *(here->VSRCcoeffs + 1) ;
                        value = foo;
                        goto loadDone;
                    }

                    do {
                        for(i=ii ; i<(here->VSRCfunctionOrder/2)-1; i++ ) {
                            itime = *(here->VSRCcoeffs+2*i);
                            if (  AlmostEqualUlps(itime+repeat_time, time, 3 )) {
                                foo   = *(here->VSRCcoeffs+2*i+1);
                                value = foo;
                                goto loadDone;
                            } else if ( (*(here->VSRCcoeffs+2*i)+repeat_time < time) 
							   && (*(here->VSRCcoeffs+2*(i+1))+repeat_time > time) ) {
                                foo   = *(here->VSRCcoeffs+2*i+1) + (((time-(*(here->VSRCcoeffs+2*i)+repeat_time))/
								   (*(here->VSRCcoeffs+2*(i+1)) - *(here->VSRCcoeffs+2*i))) *
							       (*(here->VSRCcoeffs+2*i+3)    - *(here->VSRCcoeffs+2*i+1)));
                                value = foo;
                                goto loadDone;
                            }
                        }
                        foo = *(here->VSRCcoeffs+ here->VSRCfunctionOrder-1) ;
                        value = foo;

                        if ( !here->VSRCrGiven ) goto loadDone;
		      
                        end_time = *(here->VSRCcoeffs + here->VSRCfunctionOrder-2);
                        breakpt_time = *(here->VSRCcoeffs + here->VSRCrBreakpt);
                        repeat_time  = end_time + (end_time - breakpt_time)*num_repeat++ - breakpt_time;
                        ii            = here->VSRCrBreakpt/2;
                    } while ( here->VSRCrGiven );
                    break;
                }

/**** tansient noise routines: 
VNoi2 2 0  DC 0 TRNOISE(10n 0.5n 0 0n) : generate gaussian distributed noise
                        rms value, time step, 0 0
VNoi1 1 0  DC 0 TRNOISE(0n 0.5n 1 10n) : generate 1/f noise
                        0,  time step, exponent < 2, rms value
*/
                case TRNOISE: {
                /* Generate voltage point every TS with amplitude NA * ra,
                   where ra is drawn from a random number generator with
                   gaussian distribution with mean 0 and standard deviation 1 
				*/

//#define PRVAL
//                    typedef int bool;
   
                    double newval=0.0, lastval=0.0, lasttime=0.0;
                    double NA, NT, TS;                    
                    double V1, V2, basetime = 0.;
                    double scalef, ra1, ra2;
                    float NALPHA, NAMP;
   
                    long int nosteps, newsteps = 1, newexp = 0;
             
                    bool aof = FALSE;   
                       
                    NA = here->VSRCcoeffs[0]; // input is rms value
                    NT = here->VSRCcoeffs[1]; // time step

                    scalef = NA;
//                    scalef = NA*1.32;

                    NALPHA = here->VSRCfunctionOrder > 2
                       ? (float)here->VSRCcoeffs[2] : 0.0f;
                    NAMP = here->VSRCfunctionOrder > 3
                       && here->VSRCcoeffs[3] != 0.0
                       && here->VSRCcoeffs[2] != 0.0       
                       ? (float)here->VSRCcoeffs[3] : 0.0f;

                    if ((NT == 0.) || ((NA == 0.) && (NAMP == 0.))) {
                        value =  here->VSRCdcValue;
                        goto noiDone;
                    }
                    else
                        TS = NT; /* time step for noise */
        
                    if ((NALPHA > 0.0) && (NAMP > 0.0)) aof = TRUE;
   
                    lasttime = here->VSRCprevTime;
                    lastval = here->VSRCprevVal;
                    newval = here->VSRCnewVal;
                    /* set all data: DC, white, 1of */
                    if (time <= 0 /*ckt->CKTstep*/) {
                        /* data are already set */
                        if ((here->VSRCprevVal != 0) || (here->VSRCnewVal != 0)) {
                            value = here->VSRCprevVal;
                            goto noiDone;
                        }
                        lasttime = 0.0;
                        here->VSRCsecRand = 2.; /* > 1, invalid number out of the random number range */
                        /* get two random samples */
#ifdef FastRand
                        // use FastNorm3
                        here->VSRCprevVal = scalef * GaussWa;
                        here->VSRCnewVal = scalef * GaussWa; 
#elif defined (WaGauss)
                        // use WallaceHV
                        here->VSRCprevVal = scalef * GaussWa;
                        here->VSRCnewVal = scalef * GaussWa;
#else
                        // make use of two random variables per call to rgauss()
                        rgauss(&ra1, &ra2);
                        here->VSRCprevVal = scalef * ra1;
                        // choose to set start value to 0
                        here->VSRCprevVal = 0;
                        here->VSRCnewVal = scalef * ra2;
#endif
                        /* generate 1 over f noise at time 0 */
                        if (aof) {
                            if (here->VSRCncount==0) {
                                // add 10 steps for start up sequence
                                nosteps = (long)((ckt->CKTfinalTime)/TS) + 10;
                                // generate number of steps as power of 2
                                while(newsteps < nosteps) {
                                    newsteps <<= 1;
                                    newexp++;
                                }
                                here->VSRConeof = TMALLOC(float, newsteps); //(float *)tmalloc(sizeof(float) * newsteps);
#ifdef PRVAL
                                printf("ALPHA: %f, GAIN: %e\n", NALPHA, NAMP);
#endif
                                f_alpha(newsteps, newexp, here->VSRConeof, NAMP, NALPHA);
#ifdef PRVAL
                                printf("Noi1: %e, Noi2: %e\n", here->VSRConeof[10], here->VSRConeof[100]);                            
#endif
                                here->VSRCprevVal += here->VSRConeof[here->VSRCncount];                         
                                here->VSRCncount++;
                                here->VSRCnewVal += here->VSRConeof[here->VSRCncount];
                                here->VSRCncount++;
                                value = newval;
                                // add DC
                                here->VSRCprevVal += here->VSRCdcValue;
                                here->VSRCnewVal += here->VSRCdcValue;
                                value = here->VSRCprevVal;
#ifdef PRVAL
                                printf("start1, time: %e, outp: %e, rnd: %e\n", time, newval, testval);
#endif
                            } else { // here->VSRCncount > 0
                                // add DC
                                here->VSRCprevVal += here->VSRCdcValue;
                                here->VSRCnewVal += here->VSRCdcValue;
                                value = here->VSRCprevVal;
#ifdef PRVAL
						        printf("start2, time: %e, outp: %e, rnd: %e\n", time, here->VSRCprevVal, testval);
#endif                        
                            }         
#ifdef PRVAL
                            printf("time 0 value: %e for %s\n", here->VSRCprevVal, here->VSRCname);
#endif   
                            goto loadDone;                         
                        }  //aof
                        // add DC
                        here->VSRCprevVal += here->VSRCdcValue;
                        here->VSRCnewVal += here->VSRCdcValue;
                        value = here->VSRCprevVal;
                        here->VSRCprevTime = 0.;
                        goto loadDone;
                    }  // time < 0                   

                    V1 = here->VSRCprevVal;
                    V2 = here->VSRCnewVal;
                    if (here->VSRCprevTime == ckt->CKTtime) {
                        value = here->VSRCprevVal;
                        goto noiDone;
                    }

                    if (time > 0 && time < TS) {
                        value = V1 + (V2 - V1) * (time) / TS;
                    }
                    else if (time >= TS) {
                     /* repeating signal - figure out where we are in period */
                     /* numerical correction to avoid basetime less than 
                     next step, e.g. 4.99999999999999995 delivers a floor
                     of 4 instead of 5 */
                        basetime = TS * floor(time*1.000000000001/TS);
                        time -= basetime;

#define NSAMETIME(a,b) (fabs((a)-(b))<= NTIMETOL * TS)
#define NTIMETOL 1e-7

                        if NSAMETIME(time,0.) {

                        /* get new random number */
#ifdef FastRand
                            // use FastNorm3
                            newval = scalef * FastNorm;
#elif defined (WaGauss)
                            // use WallaceHV
                            newval = scalef * GaussWa;
#else
                            // make use of two random variables per call to rgauss()
                            if (here->VSRCsecRand == 2.0) {  
                                rgauss(&ra1, &ra2);
                                newval = scalef * ra1;
                                here->VSRCsecRand = scalef * ra2;
                            }
                            else {
                                newval = here->VSRCsecRand;
                                here->VSRCsecRand = 2.0;
                            }
#endif
                            V1 = here->VSRCprevVal = here->VSRCnewVal;
                            V2 = newval; // scale factor t.b.d.
                            if(here->VSRCdcGiven) V2 += here->VSRCdcValue;
                            if (aof) {                        
                                V2 += here->VSRConeof[here->VSRCncount];
#ifdef PRVAL
                                printf("aof: %d\n", here->VSRCncount);
#endif
                            }
                            here->VSRCncount++;
                            value = V1;
                            here->VSRCnewVal = V2;
                        } else if (time > 0 && time < TS) {
                            value = V1 + (V2 - V1) * (time) / TS;
#ifdef PRVAL
                            printf("if1, time: %e, outp: %e, rnd: %e\n", ckt->CKTtime, 
						        V1 + (V2 - V1) * (time) / TS, V2);
#endif
                        } else { /* time > TS should be never reached */
                            value = V1 + (V2 - V1) * (time-TS) / TS;
#ifdef PRVAL
                            printf("if2, time: %e, outp: %e, rnd: %e\n", ckt->CKTtime, 
						        V1 + (V2 - V1) * (time-TS) / TS, V2);
#endif                  
                        }
                        here->VSRCprevTime = ckt->CKTtime;
                    }
noiDone:                   
                    if (time >=ckt->CKTfinalTime) {
                        /* free the 1of memory */
                        if (here->VSRConeof) tfree(here->VSRConeof);
                        /* reset the 1of counter */
                        here->VSRCncount = 0;
                    }
                    goto loadDone;     
                } // case
                break; 				
                } // switch
            }
loadDone:
/* gtri - begin - wbk - modify for supply ramping option */
#ifdef XSPICE_EXP
            value *= ckt->CKTsrcFact;
            value *= cm_analog_ramp_factor();
#else
            if (ckt->CKTmode & MODETRANOP) value *= ckt->CKTsrcFact;
            /* load the new voltage value into the matrix */
			*(ckt->CKTrhs + (here->VSRCbranch)) += value;
#endif
/* gtri - end - wbk - modify to process srcFact, etc. for all sources */
        } // for loop instances
    } // for loop models
    return(OK);
}
