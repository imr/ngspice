/**********
Permit to use it as your wish.
Author:	2007 Gong Ding, gdiso@ustc.edu 
University of Science and Technology of China 
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "ndevdefs.h"
#include "ngspice/numconst.h"
#include "ngspice/numenum.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define TSCALLOC(var, size, type)\
if (size && (var =(type *)calloc(1, (unsigned)(size)*sizeof(type))) == NULL) {\
   return(E_NOMEM);\
}

int NDEVmodelConnect(NDEVmodel *inModel);


int NDEVsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
/*
 * load the structure with those pointers needed later for fast matrix
 * loading
 */
{
  NDEVmodel *model = (NDEVmodel *)inModel;
  NDEVinstance *here;
  int i,j;
  CKTnode *node;        
  
  NG_IGNORE(ckt);
  NG_IGNORE(states);

    /*  loop through all the ndev models */
    for( ; model != NULL; model = NDEVnextModel(model)) {

        /* connect to remote device simulator */ 
	if(NDEVmodelConnect(model)) return E_PRIVATE;
		
        /* loop through all the instances of the model */
        for (here = NDEVinstances(model); here != NULL ;
                here=NDEVnextInstance(here)) {
            
	    here->Ndevinfo.term = here->term;
	    strncpy(here->Ndevinfo.NDEVname, here->gen.GENname, 32);
	    send(model->sock,&(here->Ndevinfo),sizeof(here->Ndevinfo),0);
/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

	    for(i=0;i<here->term;i++)
	      for(j=0;j<here->term;j++)
	      {
                TSTALLOC(mat_pointer[i*here->term+j], pin[i], pin[j]);
	      }

	    for(i=0;i<here->term;i++)   
	    {
	         node = here->node[i];
		 here->PINinfos[i].pin=node->number;
	         strncpy(here->PINinfos[i].name,here->bname[i],32);
		 here->PINinfos[i].V  = 0.0;
		 send(model->sock,&here->PINinfos[i],sizeof(here->PINinfos[i]),0);
	    } 
	     	 
        }
    }
  return(OK);

}

int NDEVmodelConnect(NDEVmodel *inModel)
{
  NDEVmodel *model = inModel;
  struct hostent *hostlist; /* List of hosts returned by gethostbyname. */
  char dotted_ip[15];       /* Buffer for converting
                               the resolved address to
                               a readable format. */
  struct sockaddr_in sa;    /* Connection address. */
  char *buf = TMALLOC(char, 128);

  /* Look up the hostname with DNS. gethostbyname
  (at least most UNIX versions of it) properly
  handles dotted IP addresses as well as hostnames. */
  hostlist = gethostbyname(model->host);
  if (hostlist == NULL)
  {
    fprintf(stderr,"NDEV: Unable to resolve host %s.\n", model->host);
    return E_PRIVATE;
  }
  /* Good, we have an address. However, some sites
  are moving over to IPv6 (the newer version of
  IP), and we're not ready for it (since it uses
  a new address format). It's a good idea to check
  for this. */
  if (hostlist->h_addrtype != AF_INET)
  {
    fprintf(stderr,"NDEV: Host %s doesn't seem to be an IPv4 address.\n",model->host);
    return E_PRIVATE;
  }

  /* inet_ntop converts a 32-bit IP address to
  the dotted string notation (suitable for printing).
  hostlist->h_addr_list is an array of possible addresses
  (in case a name resolves to more than on IP). In most
  cases we just want the first. */
  inet_ntop(AF_INET, hostlist->h_addr_list[0], dotted_ip, 15);
  /* Create a socket for the connection. */
  model->sock = socket(PF_INET, SOCK_STREAM, IPPROTO_IP);

  if (model->sock < 0)
  {
    fprintf(stderr, "NDEV: Unable to create a socket %s.\n", strerror(errno));
    return E_PRIVATE;
  }
  /* Fill in the sockaddr_in structure. The address is
  already in network byte order (from the gethostbyname
  call). We need to convert the port number with the htons
  macro. Before we do anything else, we'll zero out the
  entire structure. */
  
  memset(&sa, 0, sizeof(struct sockaddr_in));
  sa.sin_port = htons(model->port);
  /* The IP address was returned as a char * for
  various reasons.
  Just memcpy it into the sockaddr_in structure. */
  memcpy(&sa.sin_addr, hostlist->h_addr_list[0],
         (size_t) hostlist->h_length);
  /* This is an Internet socket. */
  sa.sin_family = AF_INET;
  /* Connect! */
  if (connect(model->sock, (struct sockaddr *)&sa, sizeof(sa)) < 0)
  {
    fprintf(stderr, "NDEV: Unable to connect %s\n",strerror(errno));
    return  E_PRIVATE;
  }
  
  sprintf(buf, NG_QUERY);
  send(model->sock, buf, 128, 0);
  if(recv(model->sock, buf, 128, MSG_WAITALL) < 128)
  {
    fprintf(stderr, "NDEV: Remote answer error. %s\n",strerror(errno));
    return  E_PRIVATE;
  }
  
  if(strncmp(buf, NDEV_REPLY, sizeof(NDEV_REPLY)))
  {
    fprintf(stderr, "NDEV: Remote answer error. %s\n", buf);
    return  E_PRIVATE;
  }  
  
  
  free(buf);
  return (OK);
}


