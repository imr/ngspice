/**********
Permit to use it as your wish.
Author:	2007 Gong Ding, gdiso@ustc.edu 
University of Science and Technology of China 
**********/

#ifndef NDEV_H
#define NDEV_H


/* circuit level includes */
#include "ngspice/ifsim.h"
#include "ngspice/inpmacs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ndevexch.h"

/* network function */
#include <errno.h>
#include <netinet/in.h> /* IPv4 socket address structres. */
#include <netdb.h> /* Access to DNS lookups. */
#include <arpa/inet.h> /* inet_ntop function. */
#include <sys/socket.h> /* Socket functions. */

/* information needed per instance */
typedef struct sNDEVinstance {
  struct sNDEVmodel *NDEVmodPtr;/* back pointer to model */
  struct sNDEVinstance *NDEVnextInstance;	/* pointer to next instance
						 * of current model */
  IFuid NDEVname;		/* pointer to character string naming this
				 * instance */
  int NDEVstate;		/* pointer to start of state vector for diode */
  int pin[7];                   /* max 7 terminals are allowed */
  int  term;                    /* the real number of terminals */
  CKTnode *node[7];		/* the array of CKT node's node pointer */
  char *bname[7];               /* the electrode boundary label for numerical solver */
  sCKTinfo    CKTInfo;
  sDeviceinfo Ndevinfo;
  sPINinfo    PINinfos[7];           
  double  * mat_pointer[49];    /* the pointer array to matrix */ 
} NDEVinstance;


/* per model data */

typedef struct sNDEVmodel {	/* model structure for a diode */
  /* the following 4 entries should always exist */
  int NDEVmodType;		/* type index of this device type */
  struct sNDEVmodel *NDEVnextModel;	/* pointer to next possible model in linked list */
  NDEVinstance *NDEVinstances;	/* pointer to list of instances that have this model */
  IFuid NDEVmodName;		/* pointer to character string naming this model */
  /* here can be freely defined as your wish*/
  
  char * NDEVmodelfile;
  char * host;
  int    port;              /* Port number. */
  int    sock;              /* Our connection socket. */
  
} NDEVmodel;




/* device parameters */
#define NDEV_MODEL_FILE 1
/* model parameters */
#define NDEV_MOD_NDEV 101
#define NDEV_REMOTE   102
#define NDEV_PORT     103 

#include "ndevext.h"


#endif				/* NDEV_H */
