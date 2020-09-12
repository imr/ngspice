/*******************************************************************************
 * Copyright 2020 Florian Ballenegger, Anamosic Ballenegger Design
 *******************************************************************************
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cktdefs.h"
#include "ngspice/loopdefs.h"

#include "analysis.h"

int 
LOOPsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    LOOPAN *job = (LOOPAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {
    
    case LOOP_PROBESRC:
        job->LOOPprobeSrc = value->uValue;
        break;
    
    case LOOP_REFNODE:
        job->LOOPrefNode = value->nValue;
	job->LOOPrefNodeGiven = 1;
	break;

    case LOOP_START:
	if (value->rValue <= 0.0) {
	    errMsg = copy("Frequency of 0 is invalid");
            job->LOOPstartFreq = 1.0;
	    return(E_PARMVAL);
	}

        job->LOOPstartFreq = value->rValue;
        break;

    case LOOP_STOP:
	if (value->rValue <= 0.0) {
	    errMsg = copy("Frequency of 0 is invalid");
            job->LOOPstartFreq = 1.0;
	    return(E_PARMVAL);
	}

        job->LOOPstopFreq = value->rValue;
        break;

    case LOOP_STEPS:
        job->LOOPnumSteps = value->iValue;
        break;

    case LOOP_DEC:
        job->LOOPstepType = DECADE;
        break;

    case LOOP_OCT:
        job->LOOPstepType = OCTAVE;
        break;

    case LOOP_LIN:
        job->LOOPstepType = LINEAR;
        break;
    
    case LOOP_NAME:
        job->LOOPname = value->sValue;
	break;
	
    case LOOP_DIR:
        job->LOOPdirection = value->iValue;
	break;
	
    case LOOP_PORTNUM:
        job->LOOPportnum = value->iValue;
	job->LOOPportnumGiven = 1;
	break;
	
    case LOOP_PORTNAME:
        job->LOOPportname = value->sValue;
	job->LOOPportnameGiven = 1;
	break;
	
    case LOOP_INSRC:
        job->LOOPinSrc = value->uValue;
	job->LOOPinSrcGiven = 1;
	break;
    
    case LOOP_OUTSRC:
        job->LOOPoutSrc = value->uValue;
	job->LOOPoutSrcGiven = 1;
	job->LOOPoutPosGiven = 0;  /* exclusive */
	job->LOOPoutNegGiven = 0;  /* exclusive */
	break;
	
    case LOOP_OUTPOS:
        job->LOOPoutPos = value->nValue;
	job->LOOPoutPosGiven = 1;
	job->LOOPoutSrcGiven = 0; /* exclusive */
	break;
	
    case LOOP_OUTNEG:
        job->LOOPoutNeg = value->nValue;
	job->LOOPoutNegGiven = 1;
	job->LOOPoutSrcGiven = 0; /* exclusive */
	break;
	
	
    default:
        printf("loopsetp Badparam %d\n", which);
        return(E_BADPARM);
    }
    return(OK);
}

static IFparm LOOPparms[] = {
    { "start",      LOOP_START,   IF_SET|IF_REAL, "starting frequency" },
    { "stop",       LOOP_STOP,    IF_SET|IF_REAL, "ending frequency" },
    { "numsteps",   LOOP_STEPS,   IF_SET|IF_INTEGER,  "number of frequencies" },
    { "dec",        LOOP_DEC,     IF_SET|IF_FLAG, "step by decades" },
    { "oct",        LOOP_OCT,     IF_SET|IF_FLAG, "step by octaves" },
    { "lin",        LOOP_LIN,     IF_SET|IF_FLAG, "step linearly" },
    { "name",       LOOP_NAME,    IF_SET|IF_STRING, "analysis name" },
    { "probe",      LOOP_PROBESRC,IF_SET|IF_INSTANCE, "probe inloop" },
    { "refnode",    LOOP_REFNODE, IF_SET|IF_NODE, "reference node" },
    { "portnum",    LOOP_PORTNUM,    IF_SET|IF_INTEGER, "terminal number for breaking the loop" },
    { "portname",       LOOP_PORTNAME,    IF_SET|IF_STRING, "terminal name for breaking the loop" },
    { "dir",       LOOP_DIR,    IF_SET|IF_INTEGER, "loop direction" },
    { "probe-",      LOOP_PROBESRC,IF_SET|IF_INSTANCE, "probe inloop (diff)" },
    { "portnum-",    LOOP_PORTNUM,    IF_SET|IF_INTEGER, "terminal number for breaking the loop (diff)" },
    { "portname-",       LOOP_PORTNAME,    IF_SET|IF_STRING, "terminal name for breaking the loop (diff)" },
    { "dir-",       LOOP_DIR,    IF_SET|IF_INTEGER, "loop direction (diff)" },
    { "insrc",       LOOP_INSRC,    IF_SET|IF_INSTANCE, "input source" },
    { "outsrc",       LOOP_OUTSRC,    IF_SET|IF_INSTANCE, "output source branch" },
    { "outpos",       LOOP_OUTPOS,    IF_SET|IF_NODE, "Positive output node" },
    { "outneg",       LOOP_OUTNEG,    IF_SET|IF_NODE, "Negative output node" },
};

SPICEanalysis LOOPinfo  = {
    { 
        "LOOP",
        "Small signal loop stability analysis",

        NUMELEMS(LOOPparms),
        LOOPparms
    },
    sizeof(LOOPAN),
    FREQUENCYDOMAIN,
    1,
    LOOPsetParm,
    LOOPaskQuest,
    NULL,
    LOOPan,
    LOOPpreset
};
