/* Spice hooks */
#include "ngspice/ngspice.h"
#ifdef CLUSTER
#include "ngspice/inpdefs.h"
#include "ngspice/cluster.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"

/* Misc stuff */
#include <pthread.h>
#include <string.h>

/*Network stuff*/
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>


struct input_pipe {
  /*the names of the local and remote nodes*/
  char remote[32];
  char local[32];
  int fd;
  FILE *stream;
  /* the data recieved */
  double time;
  double data;
  /*resistance of this link*/
  double res;
  /* The value controled */
  double *currentPtr;
  /*The output it is linked to*/
  struct output_pipe *link;
  struct input_pipe *next;
};

struct output_pipe {
  int fd;
  FILE *stream;
  /*the names of the local and remote nodes*/
  char local[32];
  char remote[32];
  /* The index of the local node value in the ckt->CKTrhsOld array */
  int outIndex;
  /*Last values sent*/
  double time,data;
  struct input_pipe *link;
  struct output_pipe *next;
};

static double lastTimeSent=0;

static int time_sock=0;
static FILE *time_outfile=NULL;
static FILE *time_infile=NULL;

static struct input_pipe* input_pipes=NULL;
static struct output_pipe* output_pipes=NULL;

/* sets up deamon which waits for connections 
 *and sets up input pipes as it recieves them */
static void *start_listener(void *);

/* Setup the output pipes*/
static int setup_output(CKTcircuit *ckt);
static int setup_input(CKTcircuit *ckt);
static int setup_time();

int CLUsetup(CKTcircuit *ckt){
  pthread_t tid;
  struct input_pipe *curr;
  int i, connections=0;
  GENmodel *mod;
  GENinstance *inst;

  /* count the number of connections expected */
  i = INPtypelook("Isource");
  for(mod = ckt->CKThead[i];
      mod != NULL;
      mod = mod->GENnextModel)
    for (inst = mod->GENinstances;
	 inst != NULL; 
	 inst = inst->GENnextInstance) 
      if(strncmp("ipcx",inst->GENname,4) == 0)
	connections++;
  
  /* allocate the input connections */
  for(i=0;i<connections;i++) {
    curr = TMALLOC(struct input_pipe, 1);
    if(input_pipes)
      curr->next = input_pipes;
    else
      curr->next = NULL;
    input_pipes = curr;
  }

  pthread_create(&tid,NULL,start_listener,(void *)&connections);
  setup_output(ckt);
  pthread_join(tid,NULL);
  setup_input(ckt);
  setup_time();
  return 0;
}

#include "../devices/isrc/isrcdefs.h"
/*Connect to remote machine and find the data*/
static int setup_output(CKTcircuit *ckt){
  int type;
  GENmodel *mod;
  GENinstance *inst;
  char hostname[64];
  
  lastTimeSent = 0;
  type = INPtypelook("Isource");

  for(mod = ckt->CKThead[type];
      mod != NULL;
      mod = mod->GENnextModel)

    for (inst = mod->GENinstances;
	 inst != NULL; 
	 inst = inst->GENnextInstance) 

      if(strncmp("ipcx",inst->GENname,4) == 0){
	ISRCinstance *isrc = (ISRCinstance *)inst;
	CKTnode *node;
	struct output_pipe *curr;
	struct sockaddr_in address;
	struct hostent *host=NULL;
	int sock,nodeNum,i;

	/*Create the struct*/
	curr = TMALLOC(struct output_pipe, 1);
	if(output_pipes)
	  curr->next = output_pipes;
	else
	  curr->next = NULL;
	output_pipes = curr;

	/* The node names */
	strcpy(curr->local,CKTnodName(ckt,isrc->ISRCnegNode));/*weird*/
	strcpy(curr->remote,isrc->ISRCname);
	
	/* extract remote node number */
	nodeNum = /*Xcoord*/(curr->remote[4] - '0') * CLUSTER_WIDTH 
	  + /*Ycoord*/(curr->remote[9] - '0');
	sprintf(hostname,"n%d."DOMAIN_NAME,nodeNum);

	/* network stuff */
	host = gethostbyname(hostname);
	if(!host){
	  printf("Host not found in setup_output\n");
	  exit(0);
	}
	
	if((sock = socket(PF_INET,SOCK_STREAM,0)) < 0){
	  printf("Socket open in setup_output\n");
	  exit(0);
	}
	
	address.sin_family = AF_INET;
	address.sin_port = htons(PORT);
	memcpy(&address.sin_addr,host->h_addr_list[0],
	       sizeof(address.sin_addr));

	printf("connecting to %s ...... ",hostname);
	fflush(stdout);

	while(connect(sock,(struct sockaddr *)&address,sizeof(address))){
	  usleep(500);/*wait for the sever to start*/
	}

	printf("connected\n");
	
	curr->fd = sock;

	/* send stuff */
	/* buffer */
	i = (strlen(curr->remote) + strlen(curr->local) + 2)*sizeof(char);
	setsockopt(sock,SOL_SOCKET,SO_SNDBUF,&i,sizeof(i));

	curr->stream = fdopen(curr->fd,"w");

	fwrite(curr->remote,sizeof(char),strlen(curr->remote),curr->stream);
	fputc('\0',curr->stream);
	fwrite(curr->local,sizeof(char),strlen(curr->local),curr->stream);
	fputc('\0',curr->stream);
	fflush(curr->stream);

	/* buffer, what is done per time point */
	i = sizeof(double)*2;
	setsockopt(sock,SOL_SOCKET,SO_SNDBUF,&i,sizeof(i));
	
	/* find the index in ckt->rhsOld which contains the local node */
	i = 0;
	for(node = ckt->CKTnodes->next;node;node = node->next){
	  i++;
	  if(strcmp(node->name,curr->local)==0){
	    curr->outIndex = i;
	    goto next;
	  }
	}
	printf("Local node %s not found\n",curr->local);
	exit(0);
      next:
	
      }
  return 0;
}

/*Processes the connections recieved by start_listener*/
static int setup_input(CKTcircuit *ckt){
  int type;
  GENmodel *mod;
  GENinstance *inst;
  struct input_pipe *input;
  type = INPtypelook("Isource");
  
  for(input = input_pipes;input;input = input->next){
    int i;

    input->stream = fdopen(input->fd,"r");

    /*Get the local and remote node names*/
    i=0;
    do {
      while(fread(&input->local[i],sizeof(char),1,input->stream) != 1);
    }while(input->local[i++] != '\0');
    
    i=0;
    do {
      while(fread(&input->remote[i],sizeof(char),1,input->stream) != 1);
    }while(input->remote[i++] != '\0');

    /* initilise */
    input->time = -1;

    /*Find the Isource to control*/
    for(mod = ckt->CKThead[type];
	mod != NULL;
	mod = mod->GENnextModel)
      
      for (inst = mod->GENinstances;
	   inst != NULL; 
	   inst = inst->GENnextInstance) 

	if(strcmp(input->remote,&inst->GENname[11]) == 0){

	  ISRCinstance *isrc = (ISRCinstance *)inst;
	  input->res = isrc->ISRCdcValue;
	  isrc->ISRCdcValue = 0;
	  input->currentPtr = &isrc->ISRCdcValue;
	  goto next;
      }
    /* We get here if no Isource matches */
    printf("Current source %s not found\n",input->remote);
    exit(0);
    
  next:
    
    /* Now find the corresponding output */
    {
      struct output_pipe *output;
      for(output = output_pipes;output;output = output->next)
	if(strcmp(&input->local[11],output->local)==0){
	  input->link = output;
	  output->link = input;
	  goto next2;
	}
      printf("Parent to %s not found\n",&input->local[11]);
      exit(0);
    next2:  
    } 
  }
  return 0;
}

/* This starts a server and waits for connections, number given by argument*/
static void *start_listener(void *v){
  int *connections = (int *)v;
  int count=0;
  struct sockaddr_in address;
  int sock, conn,i;
  size_t addrLength = sizeof(struct sockaddr_in);
  struct input_pipe *curr;

  if((sock = socket(PF_INET,SOCK_STREAM,0)) < 0){
    printf("socket in start_listener\n");
    exit(0); 
  }

  /* Allow reuse of the socket */
  i = 1;
  setsockopt(sock,SOL_SOCKET,SO_REUSEADDR,&i,sizeof(i));

  /* port, inferface ..*/
  address.sin_family = AF_INET;
  address.sin_port = htons(PORT);
  memset(&address.sin_addr,0,sizeof(address.sin_addr));

  if(bind(sock, (struct sockaddr *)&address,sizeof(address))){
    printf("bind in start_listener\n");
    exit(0); 
  }
  if(listen(sock,5)){
    printf("listen in start_listener\n");
    exit(0);
  }
  
  /* Loop till recieved all connections */
  curr = input_pipes;
  while (count < *connections){
    if((conn = accept(sock, (struct sockaddr *)&address,&addrLength)) < 0){
      printf("accept in start_listener\n");
      exit(0);
    }
    
    curr->fd = conn;
    /* will fill rest of structure later in setup_input*/
    count ++;
    curr = curr->next;
  }

  close(sock);

  return NULL;
}

/*Writes data to remote computer*/
int CLUoutput(CKTcircuit *ckt){
  struct output_pipe *output;
  lastTimeSent = ckt->CKTtime;
  for(output = output_pipes;
      output;
      output = output->next){
    output->time = ckt->CKTtime;
    output->data = ckt->CKTrhsOld[output->outIndex];
    fwrite(&output->time,sizeof(double),1,output->stream);
    fwrite(&output->data,
	   sizeof(double),1,output->stream);
    fflush(output->stream);
  }
  return 0;
}

/*Maniputates the local circuit based on the links*/
int CLUinput(CKTcircuit *ckt){
  struct input_pipe *input;
  double tmp;
  for(input= input_pipes;input;input = input->next){
    /*recieve data till we get a good time point*/
    while (input->time < lastTimeSent){
      while(fread(&input->time, sizeof(double), 1, input->stream) != 1){}
      while(fread(&input->data, sizeof(double), 1, input->stream) != 1){}
    }
    tmp = (input->link->data - input->data) / input->res;

    /*dampen out large currents*/
    if(tmp > 0)
	*input->currentPtr = 0.2 * (1 - exp(-tmp/0.2));
    else
	*input->currentPtr = -0.2 * (1 - exp(tmp/0.2));

    /*GND is the posNode, local node is the negNode*/
  }
  return 0;
}

static int setup_time(){
  struct sockaddr_in address;
  struct hostent *host=NULL;
  char *hostname = TIME_HOST;
  int sock,i;

  /* network stuff */
  host = gethostbyname(hostname);
  if(!host){
    printf("Host not found in setup_time\n");
    exit(0);
  }
  if((sock = socket(PF_INET,SOCK_STREAM,0)) < 0){
    printf("Socket open in setup_time\n");
    exit(0);
  }

  i = sizeof(double)*2;
  setsockopt(sock,SOL_SOCKET,SO_SNDBUF ,&i,sizeof(i));
 
  address.sin_family = AF_INET;
  address.sin_port = htons(TIME_PORT);
  memcpy(&address.sin_addr,host->h_addr_list[0],
	 sizeof(address.sin_addr));


  while(connect(sock,(struct sockaddr *)&address,sizeof(address))){
    usleep(500);/*wait for the sever to start*/
  }
  time_sock=sock;
  time_outfile=fdopen(sock,"w");
  time_infile=fdopen(sock,"r");

  return 0;
}


int CLUsync(double time,double *delta, int error){
  double tmp;
  if(error)
    tmp = *delta * (-1);
  else
    tmp = *delta;
  fwrite(&time,sizeof(double),1,time_outfile);
  fwrite(&tmp,sizeof(double),1,time_outfile);
  fflush(time_outfile);
  while(fread(&tmp,sizeof(double),1,time_infile) != 1);
  if(tmp < 0){
    *delta = -tmp;
    return 0;
  } else {
    *delta = tmp;
    return 1;
  }
}
#endif
