/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/onedev.h"
#include "ngspice/onemesh.h"
#include "ngspice/spmatrix.h"
#include "onedext.h"
#include "oneddefs.h"

extern void CiderLoaded(int);

void
ONEdestroy(ONEdevice *pDevice)
{
  int index, eIndex;
  ONEelem *pElem;
  ONEnode *pNode;
  ONEedge *pEdge;


  if (!pDevice)
    return;

  switch (pDevice->solverType) {
  case SLV_SMSIG:
  case SLV_BIAS:
    /* free up memory allocated for the bias solution */
    FREE(pDevice->dcSolution);
    FREE(pDevice->dcDeltaSolution);
    FREE(pDevice->copiedSolution);
    FREE(pDevice->rhs);
    FREE(pDevice->rhsImag);
    spDestroy(pDevice->matrix);
    break;
  case SLV_EQUIL:
    /* free up the vectors allocated in the equilibrium solution */
    FREE(pDevice->dcSolution);
    FREE(pDevice->dcDeltaSolution);
    FREE(pDevice->copiedSolution);
    FREE(pDevice->rhs);
    spDestroy(pDevice->matrix);
    break;
  case SLV_NONE:
    break;
  default:
    fprintf(stderr, "Panic: Unknown solver type in ONEdestroy.\n");
    exit(-1);
    break;
  }

  /* destroy the mesh */
  if (pDevice->elemArray) {
    for (eIndex = 1; eIndex < pDevice->numNodes; eIndex++) {
      pElem = pDevice->elemArray[eIndex];
      pEdge = pElem->pEdge;
      FREE(pEdge);
      for (index = 0; index <= 1; index++) {
	if (pElem->evalNodes[index]) {
	  pNode = pElem->pNodes[index];
	  FREE(pNode);
	}
      }
      FREE(pElem);
    }
    FREE(pDevice->elemArray);
  }

  if (pDevice->pMaterials) {
      ONEmaterial* pMtmp = pDevice->pMaterials;
      while (pMtmp) {
          ONEmaterial* pMtmpnext = pMtmp->next;
          FREE(pMtmp);
          pMtmp = pMtmpnext;
      }
  }

  if (pDevice->pStats) {
    FREE(pDevice->pStats);
  }

  /* destroy any other lists */
  /* NOT IMPLEMENTED */

  FREE(pDevice);
  {
    CiderLoaded(-1);
  }
}
