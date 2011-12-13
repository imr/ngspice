/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE filesource/cfunc.mod

Copyright 2011
Thomas Sailer

              

AUTHORS                      

    20 May 2011     Thomas Sailer


MODIFICATIONS   


SUMMARY

    This file contains the model-specific routines used to
    functionally describe the gain code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    N/A                  N/A                     


REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/

#if defined(__MINGW32__) || defined(_MSC_VER)
#define DIR_PATHSEP	"\\"
#else
#define DIR_PATHSEP	"/"
#endif
  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         

struct filesource_state {
	FILE *fp;
	long pos;
	unsigned char atend;
};
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*==============================================================================

FUNCTION void cm_gain()

AUTHORS                      

     2 Oct 1991     Jeffrey P. Murray

MODIFICATIONS   

    NONE

SUMMARY

    This function implements the gain code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    N/A                  N/A


RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/


/*=== CM_FILESOURCE ROUTINE ===*/
                                                   

void cm_filesource(ARGS)   /* structure holding parms, inputs, outputs, etc.     */
{
	int size;
	int amplscalesize;
	int amploffssize;
	double *timeinterval;
	double *amplinterval;
	struct filesource_state *state;

	if(ANALYSIS == MIF_AC) {
		return;
	}
	size = PORT_SIZE(out);
	if (INIT == 1) {
		/* Allocate storage for internal state */
		cm_analog_alloc(0, 2 * sizeof(double));
		cm_analog_alloc(1, size * (int) (2 * sizeof(double)));
		cm_analog_alloc(2, sizeof(struct filesource_state));
	}
	timeinterval = (double *)cm_analog_get_ptr(0, 0);
	amplinterval = (double *)cm_analog_get_ptr(1, 0);
	state = (struct filesource_state *)cm_analog_get_ptr(2, 0);
	if (INIT == 1) {
		int i;
		timeinterval[0] = timeinterval[1] = PARAM_NULL(timeoffset) ? 0.0 : PARAM(timeoffset);
		for (i = 0; i < size; ++i)
			amplinterval[2 * i] = amplinterval[2 * i + 1] = PARAM_NULL(amploffset) ? 0.0 : PARAM(amploffset[i]);
		state->fp = fopen(PARAM(file), "r");
		state->pos = 0;
		state->atend = 0;
		if (!state->fp) {
			char *lbuffer, *p;
            lbuffer = getenv("NGSPICE_INPUT_DIR");
            if (lbuffer && *lbuffer) {
                p = (char*) malloc(strlen(lbuffer) + strlen(DIR_PATHSEP) + strlen(PARAM(file)) + 1);
                sprintf(p, "%s%s%s", lbuffer, DIR_PATHSEP, PARAM(file));
                state->fp = fopen(p, "r");
                free(p);
            } 
			if (!state->fp) {			
				char msg[512];
				snprintf(msg, sizeof(msg), "cannot open file %s", PARAM(file));
				cm_message_send(msg);
				state->atend = 1;
			}
		}
	}
       	amplscalesize = PARAM_NULL(amplscale) ? 0 : PARAM_SIZE(amplscale);
	amploffssize = PARAM_NULL(amploffset) ? 0 : PARAM_SIZE(amploffset);
	while (TIME >= timeinterval[1] && !state->atend) {
		char line[512];
		char *cp, *cpdel;
		char *cp2;
		double t;
		int i;
		if (ftell(state->fp) != state->pos) {
			clearerr(state->fp);
			fseek(state->fp, state->pos, SEEK_SET);
		}
		if (!fgets(line, sizeof(line), state->fp)) {
			state->atend = 1;
			break;
		}
		state->pos = ftell(state->fp);
		cpdel = cp = strdup(line);
		while (*cp && isspace(*cp))
			++cp;
		if (*cp == '#' || *cp == ';') {
			free(cpdel);
			continue;
		}
		t = strtod(cp, &cp2);
		if (cp2 == cp) {
			free(cpdel);
			continue;
		}
		cp = cp2;
		if (!PARAM_NULL(timescale))
			t *= PARAM(timescale);
		if (!PARAM_NULL(timerelative) && PARAM(timerelative) == MIF_TRUE)
			t += timeinterval[1];
		else if (!PARAM_NULL(timeoffset))
			t += PARAM(timeoffset);
		timeinterval[0] = timeinterval[1];
		timeinterval[1] = t;
		for (i = 0; i < size; ++i)
			amplinterval[2 * i] = amplinterval[2 * i + 1];
		for (i = 0; i < size; ++i) {
			while (*cp && (isspace(*cp) || *cp == ','))
				++cp;
			t = strtod(cp, &cp2);
			if (cp2 == cp)
				break;
			cp = cp2;
			if (i < amplscalesize)
				t *= PARAM(amplscale[i]);
			if (i < amploffssize)
				t += PARAM(amploffset[i]);
			amplinterval[2 * i + 1] = t;
		}
		free(cpdel);
	}
	if (TIME < timeinterval[1] && timeinterval[0] < timeinterval[1] && 0.0 <= timeinterval[0]) {
		if (!PARAM_NULL(amplstep) && PARAM(amplstep) == MIF_TRUE) {
			int i;
			for (i = 0; i < size; ++i)
				OUTPUT(out[i]) = amplinterval[2 * i];
		} else {
			double mul0 = (timeinterval[1] - TIME) / (timeinterval[1] - timeinterval[0]);
			double mul1 = 1.0 - mul0;
			int i;
			for (i = 0; i < size; ++i)
				OUTPUT(out[i]) = mul0 * amplinterval[2 * i] + mul1 * amplinterval[2 * i + 1];
		}
	} else {
		int i;
		for (i = 0; i < size; ++i)
			OUTPUT(out[i]) = amplinterval[2 * i + 1];
	}
}
