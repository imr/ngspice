/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

/*
 * Functions needed to read solutions for 1D devices.
 */
 
#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/onedev.h"
#include "ngspice/onemesh.h"
#include "ngspice/plot.h"
#include "onedext.h"
#include "oneddefs.h"
#include "ngspice/cidersupt.h"

int
ONEreadState(ONEdevice *pDevice, char *fileName, int numVolts, 
             double *pV1, double *pV2 )
/* fileName          | File containing raw data */
/* int numVolts      | Number of voltage differences */
/* double *pV1, *pV2 | Pointer to return them in */
{
  int dataLength;
  int index, i;
  ONEnode **nodeArray=NULL;
  ONEnode *pNode;
  ONEelem *pElem;
  double refPsi = 0.0;
  double *psiData, *nData, *pData;
  double *vData[2];
  struct plot *stateDB;
  struct plot *voltsDB;
  char voltName[80];


  stateDB = DBread( fileName );
  if (stateDB == NULL) return (-1);
  voltsDB = stateDB->pl_next;
  if (voltsDB == NULL) return (-1);

  for (i=0; i < numVolts; i++ ) {
    sprintf( voltName, "v%d%d", i+1, numVolts+1 );
    vData[i] = DBgetData( voltsDB, voltName, 1 );
    if (vData[i] == NULL) return (-1);
  }
  dataLength = pDevice->numNodes;
  psiData = DBgetData( stateDB, "psi", dataLength );
  nData = DBgetData( stateDB, "n", dataLength );
  pData = DBgetData( stateDB, "p", dataLength );
  if (psiData == NULL || nData == NULL || pData == NULL) return (-1);

  if (pV1 != NULL) {
    *pV1 = vData[0][0];
    FREE( vData[0] );
  }
  if (pV2 != NULL) {
    *pV2 = vData[1][0];
    FREE( vData[1] );
  }

  /* generate the work array for copying node info */
  XCALLOC(nodeArray, ONEnode *, 1 + pDevice->numNodes);

  /* store the nodes in this work array and print out later */
  for (index = 1; index < pDevice->numNodes; index++) {
    pElem = pDevice->elemArray[index];
    if (refPsi == 0.0 && pElem->matlInfo->type == SEMICON) {
      refPsi = pElem->matlInfo->refPsi;
    }
    for (i = 0; i <= 1; i++) {
      if (pElem->evalNodes[i]) {
	pNode = pElem->pNodes[i];
	nodeArray[pNode->nodeI] = pNode;
      }
    }
  }
  for (index = 1; index <= pDevice->numNodes; index++) {
    pNode = nodeArray[index];
    pNode->psi = psiData[index-1]/VNorm + refPsi;
    pNode->nConc = nData[index-1]/NNorm;
    pNode->pConc = pData[index-1]/NNorm;
  }
  FREE(nodeArray);
  FREE(psiData);
  FREE(nData);
  FREE(pData);

  return (0);
}
