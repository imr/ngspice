/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

/*
 * This is a simple routine to dump the internal device states. It produces
 * states for .OP, .DC, & .TRAN simulations.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "numosdef.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"


/* Forward Declarations */
static void NUMOSputHeader(FILE *, CKTcircuit *, NUMOSinstance *);

/* State Counter */
static int state_numOP = 0;
static int state_numDC = 0;
static int state_numTR = 0;

void
NUMOSdump(GENmodel *inModel, CKTcircuit *ckt)
{
  register NUMOSmodel *model = (NUMOSmodel *) inModel;
  register NUMOSinstance *inst;
  OUTPcard *output;
  FILE *fpState;
  char fileName[BSIZE_SP];
  char description[BSIZE_SP];
  char *prefix;
  int *state_num;
  int anyOutput = 0;
  BOOLEAN writeAscii = TRUE;

  if (ckt->CKTmode & MODEDCOP) {
    prefix = "OP";
    state_num = &state_numOP;
    sprintf(description, "...");
  } else if (ckt->CKTmode & MODEDCTRANCURVE) {
    prefix = "DC";
    state_num = &state_numDC;
    sprintf(description, "sweep = % e", ckt->CKTtime);
  } else if (ckt->CKTmode & MODETRAN) {
    prefix = "TR";
    state_num = &state_numTR;
    sprintf(description, "time = % e", ckt->CKTtime);
  } else {
    /* Not a recognized CKT mode. */
    return;
  }

  for (; model != NULL; model = NUMOSnextModel(model)) {
    output = model->NUMOSoutputs;
    for (inst = NUMOSinstances(model); inst != NULL;
         inst = NUMOSnextInstance(inst)) {

      if (inst->NUMOSprintGiven) {
	if ((ckt->CKTmode & MODETRAN) &&
	    ((ckt->CKTstat->STATaccepted - 1) % inst->NUMOSprint != 0)) {
	  continue;
	}
	anyOutput = 1;
	sprintf(fileName, "%s%s.%d.%s", output->OUTProotFile, prefix,
	    *state_num, inst->NUMOSname);

	writeAscii = compareFiletypeVar("ascii");

	fpState = fopen(fileName, (writeAscii ? "w" : "wb"));
	if (!fpState) {
	  perror(fileName);
	} else {
	  NUMOSputHeader(fpState, ckt, inst);
	  TWOprnSolution(fpState, inst->NUMOSpDevice,
	      model->NUMOSoutputs, writeAscii, "numos");
	  fclose(fpState);
	  LOGmakeEntry(fileName, description);
	}
      }
    }
  }
  if (anyOutput) {
    (*state_num)++;
  }
}

#define NUMOSnumOutputs 10

static
void
NUMOSputHeader(FILE *file, CKTcircuit *ckt, NUMOSinstance *inst)
{
  char *reference;
  double refVal = 0.0;
  int numVars = NUMOSnumOutputs;

  if (ckt->CKTmode & MODEDCOP) {
    reference = NULL;
  } else if (ckt->CKTmode & MODEDCTRANCURVE) {
    reference = "sweep";
    refVal = ckt->CKTtime;
    numVars++;
  } else if (ckt->CKTmode & MODETRAN) {
    reference = "time";
    refVal = ckt->CKTtime;
    numVars++;
  } else {
    reference = NULL;
  }
  fprintf(file, "Title: Device %s external state\n", inst->NUMOSname);
  fprintf(file, "Plotname: Device Operating Point\n");
  fprintf(file, "Command: deftype v conductance S\n");
  fprintf(file, "Flags: real\n");
  fprintf(file, "No. Variables: %d\n", numVars);
  fprintf(file, "No. Points: 1\n");
  numVars = 0;
  fprintf(file, "Variables:\n");
  if (reference) {
    fprintf(file, "\t%d	%s	unknown\n", numVars++, reference);
  }
  fprintf(file, "\t%d	v14	voltage\n", numVars++);
  fprintf(file, "\t%d	v24	voltage\n", numVars++);
  fprintf(file, "\t%d	v34	voltage\n", numVars++);
  fprintf(file, "\t%d	i1	current\n", numVars++);
  fprintf(file, "\t%d	i2	current\n", numVars++);
  fprintf(file, "\t%d	i3	current\n", numVars++);
  fprintf(file, "\t%d	i4	current\n", numVars++);
  fprintf(file, "\t%d	g11	conductance\n", numVars++);
  fprintf(file, "\t%d	g12	conductance\n", numVars++);
  fprintf(file, "\t%d	g13	conductance\n", numVars++);
  fprintf(file, "Values:\n0");
  if (reference) {
    fprintf(file, "\t% e\n", refVal);
  }
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NUMOSvdb));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NUMOSvgb));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NUMOSvsb));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NUMOSid));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NUMOSig));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NUMOSis));
  fprintf(file, "\t% e\n", -*(ckt->CKTstate0 + inst->NUMOSid)
      - *(ckt->CKTstate0 + inst->NUMOSig)
      - *(ckt->CKTstate0 + inst->NUMOSis));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NUMOSdIdDVdb));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NUMOSdIdDVgb));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NUMOSdIdDVsb));
}

void
NUMOSacct(GENmodel *inModel, CKTcircuit *ckt, FILE *file)
{
  register NUMOSmodel *model = (NUMOSmodel *) inModel;
  register NUMOSinstance *inst;
  OUTPcard *output;

  NG_IGNORE(ckt);

  for (; model != NULL; model = NUMOSnextModel(model)) {
    output = model->NUMOSoutputs;
    for (inst = NUMOSinstances(model); inst != NULL;
         inst = NUMOSnextInstance(inst)) {

      if (output->OUTPstats) {
	TWOmemStats(file, inst->NUMOSpDevice);
	TWOcpuStats(file, inst->NUMOSpDevice);
      }
    }
  }
}
