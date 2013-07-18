/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/twomesh.h"
#include "ngspice/twodev.h"
#include "twoddefs.h"
#include "twodext.h"

/* Forward declarations */
static void setDirichlet(TWOcontact *, double);


void NUMD2setBCs(TWOdevice *pDevice, double vd)
{
    TWOcontact *pContact = pDevice->pLastContact;

    setDirichlet( pContact, - vd );
}

void NBJT2setBCs(TWOdevice *pDevice, double vce, double vbe)
{
    TWOcontact *pCollContact = pDevice->pFirstContact;
    TWOcontact *pBaseContact = pDevice->pFirstContact->next;

    setDirichlet( pCollContact, vce );
    setDirichlet( pBaseContact, vbe );
}

void NUMOSsetBCs(TWOdevice *pDevice, double vdb, double vsb, double vgb)
{
    TWOcontact *pDContact = pDevice->pFirstContact;
    TWOcontact *pGContact = pDevice->pFirstContact->next;
    TWOcontact *pSContact = pDevice->pFirstContact->next->next;

    setDirichlet( pDContact, vdb );
    setDirichlet( pSContact, vsb );
    setDirichlet( pGContact, vgb );
}

static void
  setDirichlet(TWOcontact *pContact, double voltage)
{
  int index, numContactNodes, i;
  TWOelem *pElem = NULL;
  TWOnode *pNode;
  double psi, ni, pi, nie;
  double conc, sign, absConc;

  voltage /= VNorm;
  
  numContactNodes = pContact->numNodes;
  for ( index = 0; index < numContactNodes; index++ ) {
    pNode = pContact->pNodes[ index ];

    /* Find this node's owner element. */
    for ( i = 0; i <= 3; i++ ) {
      pElem = pNode->pElems[ i ];
      if ( pElem != NULL && pElem->evalNodes[ (i+2)%4 ] ) {
	break; /* got it */
      }
    }

    if (pElem->elemType == INSULATOR) {
      pNode->psi = RefPsi - pNode->eaff;
      pNode->nConc = 0.0;
      pNode->pConc = 0.0;
    }
    else if (pElem->elemType == SEMICON) {
      nie = pNode->nie;
      conc = pNode->netConc / nie;
      sign = SGN( conc );
      absConc = ABS( conc );
      if ( conc != 0.0 ) {
	psi = sign * log( 0.5 * absConc + sqrt( 1.0 + 0.25*absConc*absConc ));
	ni = nie * exp( psi );
	pi = nie * exp( - psi );
      }
      else {
	psi = 0.0;
	ni = nie;
	pi = nie;
      }
      pNode->psi = pElem->matlInfo->refPsi + psi;
      pNode->nConc = ni;
      pNode->pConc = pi;
    }
    pNode->psi += voltage;
  }
}
