/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* CKTsenComp(ckt)
 * this is a program to solve the sensitivity equation 
 * of the given circuit 
 */

#include "spice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "sperror.h"
#include "trandefs.h"
#include "suffix.h"


int
CKTsenComp(ckt)
register CKTcircuit *ckt;
{
    register int size;
    int row;
    int col;
    SENstruct *info;
#ifdef SENSDEBUG
    char *rowe;
    SMPelement *elt;
#endif /* SENSDEBUG */

#ifdef SENSDEBUG
    printf("CKTsenComp\n");
#endif /* SENSDEBUG */

    size = SMPmatSize(ckt->CKTmatrix);
    info = ckt->CKTsenInfo;

    if((info->SENmode == DCSEN)||
        (info->SENmode == TRANSEN))
    {

        /*  loop throgh all the columns of RHS
            matrix - each column corresponding to a design
            parameter */

        for (col=1;col<=info->SENparms;col++) {
            for(row=1;row<=size;row++){
                *(ckt->CKTsenRhs + row ) = *(info->SEN_RHS[row] + col);
            }
            /* solve for the sensitivity values */
            SMPsolve(ckt->CKTmatrix,ckt->CKTsenRhs,ckt->CKTrhsSpare); 

            /* store the sensitivity values */
            for(row=1;row<=size;row++){
                *(info->SEN_Sap[row] + col) = *(ckt->CKTsenRhs+row);
                *(info->SEN_RHS[row] + col) = *(ckt->CKTsenRhs+row);
            }
        }
#ifdef SENSDEBUG
        printf("\n");
        printf("Sensitivity matrix :\n");

        for(row=1;row<=size;row++){
            rowe=CKTnodName(ckt,row) ;
            if(strcmp("4",rowe)==0){
                for (col=1;col<=info->SENparms;col++) {
                    printf("\t");
                    printf("Sap(%s,%d) = %.5e\t",rowe,col,
                            *(info->SEN_Sap[row] + col));
                }
                printf("\n\n");
            }
        }

        printf("  RHS matrix   :\n");
        for(row=1;row<=size;row++){
            for (col=1;col<=info->SENparms;col++) {
                printf("  ");
                printf("RHS(%d,%d) = %.7e ",row,col,
                        *(info->SEN_RHS[row] + col));
            }
            printf("\n");
        }

        printf("      Jacobian  matrix :\n");
        for(row=1; row<=size; row++){
            for(col=1; col<=size; col++){
                if(elt = SMPfindElt(ckt->CKTmatrix, row , col , 0)){
                    printf("%.7e ",elt->SMPvalue);
                }
                else{
                    printf("0.0000000e+00 ");
                }
            }
            printf("\n");
        }
#endif 

    }

    if(info->SENmode == ACSEN){

        /*  loop throgh all the columns of RHS
            matrix - each column corresponding to a design
            parameter */

        for (col=1;col<=info->SENparms;col++) {

            for(row=1;row<=size;row++){
                *(ckt->CKTsenRhs+row) = *(info->SEN_RHS[row] + col);
                *(ckt->CKTseniRhs+row) = *(info->SEN_iRHS[row] + col);
            }

            /* solve for the sensitivity values ( both real and imag parts)*/
            SMPcSolve(ckt->CKTmatrix,ckt->CKTsenRhs,ckt->CKTseniRhs,
                    ckt->CKTrhsSpare,ckt->CKTirhsSpare);

            /* store the sensitivity values ( both real and imag parts)*/
            for(row=1;row<=size;row++){
                *(info->SEN_RHS[row] + col) = *(ckt->CKTsenRhs+row);
                *(info->SEN_iRHS[row] + col) = *(ckt->CKTseniRhs+row);
            }   
        }
#ifdef SENSDEBUG
        printf("\n");
        printf("CKTomega = %.7e rad/sec\t\n",ckt->CKTomega);
        printf("Sensitivity matrix :\n");
        for(row=1;row<=size;row++){
            rowe=CKTnodName(ckt,row);
            for (col=1;col<=info->SENparms;col++) {
                printf("\t");
                printf("RHS(%s,%d) = %.5e",rowe,col,
                        *(info->SEN_RHS[row] + col));
                printf(" + j %.5e\t",*(info->SEN_iRHS[row] + col));
                printf("\n\n");

            }
            printf("\n");
        }


        printf("CKTomega = %.7e rad/sec\t\n",ckt->CKTomega);
        printf("  RHS matrix   :\n");
        for(row=1;row<=size;row++){
            for (col=1;col<=info->SENparms;col++) {
                printf("  ");
                printf("RHS(%d,%d) = %.7e ",row,col,
                        *(info->SEN_RHS[row] + col));
                printf("+j %.7e ",*(info->SEN_iRHS[row] + col));
            }
            printf("\n");
        }

        printf("      Jacobian  matrix for AC :\n");
        for(row=1; row<=size; row++){
            for(col=1; col<=size; col++){
                if(elt = SMPfindElt(ckt->CKTmatrix, row , col , 0)){
                    printf("%.7e ",elt->SMPvalue);
                    printf("+j%.7e\t",elt->SMPiValue);
                }
                else{
                    printf("0.0000000e+00 ");
                    printf("+j0.0000000e+00\t");
                }
            }
            printf("\n\n");
        }
#endif

    }
    return(OK);
}

