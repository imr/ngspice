/* This entire routine written by PW. */

#include <stdio.h>
#include <string.h>
#include "xgraph.h"

void
Der1()
{
  PointList *theList, *D1List, *D2List;
  double m,b;
  int i;

  
  for (i=0;i<MAXSETS;i++) {
    theList = PlotData[i].list;
    if (theList) {
      DataD1[i].list = (PointList *)malloc(sizeof(PointList));
      DataD2[i].list = (PointList *)malloc(sizeof(PointList));
      D1List = DataD1[i].list;
      D2List = DataD2[i].list;
    } else 
      DataD1[i].list = DataD2[i].list = NULL;
    DataD1[i].setName = 
      (char *)malloc(sizeof(char)*strlen(PlotData[i].setName));
    DataD2[i].setName = 
      (char *)malloc(sizeof(char)*strlen(PlotData[i].setName));
    strcpy(DataD1[i].setName,PlotData[i].setName);
    strcpy(DataD2[i].setName,PlotData[i].setName);
    while (theList) {
      int j;
 
      D1List->numPoints = D2List->numPoints = theList->numPoints;
      D1List->xvec = (double *)malloc(sizeof(double)*theList->numPoints);
      D1List->yvec = (double *)malloc(sizeof(double)*theList->numPoints);
      D1List->next = NULL;
      D2List->xvec = (double *)malloc(sizeof(double)*theList->numPoints);
      D2List->yvec = (double *)malloc(sizeof(double)*theList->numPoints);
      D2List->next = NULL;

      for (j=1;j<theList->numPoints-1;j++) {
        D1List->xvec[j] = D2List->xvec[j] = theList->xvec[j];
        D1List->yvec[j] = (theList->yvec[j+1] - theList->yvec[j-1]) /
                (theList->xvec[j+1] - theList->xvec[j-1]);
        D2List->yvec[j] = (theList->yvec[j+1] + theList->yvec[j-1] -
                 2.0*theList->yvec[j]) /
                ((theList->xvec[j+1] - theList->xvec[j]) * 
                 (theList->xvec[j]-theList->xvec[j-1]));
 
      }

      /* Extrapolate to get the endpoints ... */
      /* end near 0 */
      D1List->xvec[0] = theList->xvec[0];
      D1List->xvec[theList->numPoints-1] = 
       theList->xvec[theList->numPoints-1];
      m = (D1List->yvec[2]-D1List->yvec[1]) / 
          (theList->xvec[2]-theList->xvec[1]);
      b = D1List->yvec[1] - m*theList->xvec[1];
      D1List->yvec[0] = m*theList->xvec[0] + b;
      /* end near numPoints-1 */
      m = (D1List->yvec[theList->numPoints-2]-
           D1List->yvec[theList->numPoints-3]) / 
        (theList->xvec[theList->numPoints-2]-
         theList->xvec[theList->numPoints-3]);
      b = D1List->yvec [theList->numPoints-2] - 
          m*theList->xvec[theList->numPoints-2];
      D1List->yvec[theList->numPoints-1] = 
          m*theList->xvec[theList->numPoints-1] + b;

      /* Extrapolate to get the endpoints ... */
      /* end near 0 */
      D2List->xvec[0] = theList->xvec[0];
      D2List->xvec[theList->numPoints-1] = 
       theList->xvec[theList->numPoints-1];
      m = (D2List->yvec[2]-D2List->yvec[1]) / 
          (theList->xvec[2]-theList->xvec[1]);
      b = D2List->yvec[1] - m*theList->xvec[1];
      D2List->yvec[0] = m*theList->xvec[0] + b;
      /* end near numPoints-1 */
      m = (D2List->yvec[theList->numPoints-2]-
           D2List->yvec[theList->numPoints-3]) / 
        (theList->xvec[theList->numPoints-2]-
         theList->xvec[theList->numPoints-3]);
      b = D2List->yvec[theList->numPoints-2] - 
          m*theList->xvec[theList->numPoints-2];
      D2List->yvec[theList->numPoints-1] = 
           m*theList->xvec[theList->numPoints-1] + b;

      theList = theList->next;
      if (theList) {
        D1List->next = (PointList *)malloc(sizeof(PointList)); 
        D2List->next = (PointList *)malloc(sizeof(PointList)); 
        D1List = D1List->next;
        D2List = D2List->next;
      }
    }
  }
} 

void
Bounds(loX, loY, hiX, hiY, Ord)
  double *loX, *loY, *hiX, *hiY;
  int Ord;
{
  int i;
  PointList *theList;
  if ((Ord<1) || (Ord>2)) {
    printf ("Internal Error - Cannot eval deriv > 2 in Bounds.\n");
    exit(1);
  }
  *loX = *loY = *hiX = *hiY = 0.0;
  for (i=0;i<MAXSETS;i++) {
    if (Ord == 1)
      theList = DataD1[i].list;
    else
      theList = DataD2[i].list;
    while (theList) {
      int j;
 
      for (j=0;j<theList->numPoints;j++) {
        *loX = (theList->xvec[j]<*loX?theList->xvec[j]:*loX);
        *loY = (theList->yvec[j]<*loY?theList->yvec[j]:*loY);
        *hiX = (theList->xvec[j]>*hiX?theList->xvec[j]:*hiX);
        *hiY = (theList->yvec[j]>*hiY?theList->yvec[j]:*hiY);
      }
      theList = theList->next;
    }
  }
}
