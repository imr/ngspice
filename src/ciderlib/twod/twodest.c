/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/twodev.h"
#include "ngspice/twomesh.h"
#include "ngspice/spmatrix.h"
#include "twoddefs.h"
#include "twodext.h"

extern void CiderLoaded(int);

void
TWOdestroy(TWOdevice *pDevice)
{
  int index, eIndex; 
  TWOelem *pElem;
  TWOnode *pNode;
  TWOedge *pEdge;

  if ( !pDevice ) return;

  switch ( pDevice->solverType ) {
  case SLV_SMSIG:
  case SLV_BIAS:
    /* free up memory allocated for the bias solution */
    FREE( pDevice->dcSolution );
    FREE( pDevice->dcDeltaSolution );
    FREE( pDevice->copiedSolution );
    FREE( pDevice->rhs );
    FREE( pDevice->rhsImag );
    spDestroy( pDevice->matrix );
    break;
  case SLV_EQUIL:
    /* free up the vectors allocated in the equilibrium solution */
    FREE( pDevice->dcSolution );
    FREE( pDevice->dcDeltaSolution );
    FREE( pDevice->copiedSolution );
    FREE( pDevice->rhs );
    spDestroy( pDevice->matrix );
    break;
  case SLV_NONE:
    break;
  default:
    fprintf( stderr, "Panic: Unknown solver type in TWOdestroy.\n" );
    exit( -1 );
    break;
  }
  
  /* destroy the mesh */
  if ( pDevice->elements ) {
    for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
      pElem = pDevice->elements[ eIndex ];
      for ( index = 0; index <= 3; index++ ) {
	if ( pElem->evalNodes[ index ] ) {
	  pNode = pElem->pNodes[ index ];
	  FREE( pNode );
	}
	if ( pElem->evalEdges[ index ] ) {
	  pEdge = pElem->pEdges[ index ];
	  FREE( pEdge );
	}
      }
      FREE( pElem );
    }
    FREE( pDevice->elements );
    for (int xIndex = 1; xIndex < pDevice->numXNodes; xIndex++) {
      FREE(pDevice->elemArray[xIndex]);
    }
    FREE( pDevice->elemArray );
  }

  if (pDevice->pChannel) {
      TWOchannel* pCtmp = pDevice->pChannel;
      while (pCtmp) {
          TWOchannel* pCtmpnext = pCtmp->next;
          FREE(pCtmp);
          pCtmp = pCtmpnext;
      }
  }

  if (pDevice->pMaterials) {
      TWOmaterial* pMtmp = pDevice->pMaterials;
      while (pMtmp) {
          TWOmaterial* pMtmpnext = pMtmp->next;
          FREE(pMtmp);
          pMtmp = pMtmpnext;
      }
  }

  if (pDevice->pFirstContact) {
      struct sTWOcontact* pFCtmp = pDevice->pFirstContact;
      while (pFCtmp) {
          struct sTWOcontact* pFCtmpnext = pFCtmp->next;
          if (pFCtmp->pNodes) {
              FREE(pFCtmp->pNodes);
          }
          FREE(pFCtmp);
          pFCtmp = pFCtmpnext;
      }
  }

  if (pDevice->pStats) {
    FREE(pDevice->pStats);
  }
  if (pDevice->xScale) {
    FREE(pDevice->xScale);
  }
  if (pDevice->yScale) {
    FREE(pDevice->yScale);
  }

  FREE( pDevice );
  {
    CiderLoaded(-1);
  }
}
