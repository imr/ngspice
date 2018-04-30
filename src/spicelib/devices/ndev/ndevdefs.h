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

  struct GENinstance gen;

#define NDEVmodPtr(inst) ((struct sNDEVmodel *)((inst)->gen.GENmodPtr))
#define NDEVnextInstance(inst) ((struct sNDEVinstance *)((inst)->gen.GENnextInstance))
#define NDEVstate gen.GENstate

  const int pin[7];                   /* max 7 terminals are allowed */
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

  struct GENmodel gen;

#define NDEVmodType gen.GENmodType
#define NDEVnextModel(inst) ((struct sNDEVmodel *)((inst)->gen.GENnextModel))
#define NDEVinstances(inst) ((NDEVinstance *)((inst)->gen.GENinstances))
#define NDEVmodName gen.GENmodName

  char * NDEVmodelfile;
  char * host;
  int    port;              /* Port number. */
  int    sock;              /* Our connection socket. */
  
} NDEVmodel;




/* device parameters */
#define NDEV_MODEL_FILE 1
/* model parameters */
enum {
    NDEV_MOD_NDEV = 101,
    NDEV_REMOTE,
    NDEV_PORT,
};

#include "ndevext.h"


#endif				/* NDEV_H */
