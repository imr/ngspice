/**********
Permit to use it as your wish.
Author:	2007 Gong Ding, gdiso@ustc.edu 
University of Science and Technology of China 
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ndevdefs.h"
#include "ngspice/suffix.h"


IFparm NDEVpTable[] = {		/* parameters */
  /* numerical-device models no longer have parameters */
  /* one is left behind to keep the table from being empty */
  IP("ndev", NDEV_MOD_NDEV, IF_FLAG, "Numerical Device"),
};

IFparm NDEVmPTable[] = {	/* model parameters */
  IP("ndev",       NDEV_MOD_NDEV,   IF_FLAG,    "Numerical Device"),
  IP("remote",     NDEV_REMOTE,     IF_STRING,  "Remote computer run device simulation"),
  IP("port",       NDEV_PORT,       IF_INTEGER, "Remote port")
};

char *NDEVnames[] = {
  "pin1",
  "pin2",
  "pin3",
  "pin4",
  "pin5",
  "pin6",
  "pin7"
};

int NDEVnSize = NUMELEMS(NDEVnames);
int NDEVpTSize = NUMELEMS(NDEVpTable);
int NDEVmPTSize = NUMELEMS(NDEVmPTable);
int NDEViSize = sizeof(NDEVinstance);
int NDEVmSize = sizeof(NDEVmodel);
