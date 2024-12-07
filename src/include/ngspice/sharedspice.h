/* header file for shared ngspice */
/* Copyright 2021-2024 Holger Vogt */
/* Modified BSD license */

/*
Interface between a calling program (caller) and ngspice.dll (libngspice.so)

**
ngSpice_nospinit(void)
Set variable no_spinit, if reading the initialization file 'spinit' is not wanted.
To be called before ngSpice_Init()

**
ngSpice_nospiceinit(void)
Set variable no_spiceinit, if reading the user defined initialization file 
'.spiceinit' is not wanted.
To be called before ngSpice_Init().
Use with care, as this removes the last chance to send preparative commands
before the netlist is loaded. Then use the the caller to send such commands.

**
ngSpice_Init(SendChar*, SendStat*, ControlledExit*,
             SendData*, SendInitData*, BGThreadRunning*, void*)
After caller has loaded ngspice.dll, the simulator has to be initialized
by calling ngSpice_Init(). Address pointers of several callback functions
defined in the caller are sent to ngspice.dll.

Callback funtion typedefs
SendChar       typedef of callback function for reading printf, fprintf, fputs
SendStat       typedef of callback function for reading status string and percent value
ControlledExit typedef of callback function for tranferring a signal upon
               ngspice controlled_exit to caller. May be used by caller
               to detach ngspice.dll.
SendData       typedef of callback function for sending an array of structs containing
               data values of all vectors in the current plot (simulation output)
SendInitData   typedef of callback function for sending an array of structs containing info on
               all vectors in the current plot (immediately before simulation starts)
BGThreadRunning typedef of callback function for sending a boolean signal (true if thread
                is running)

The void pointer may contain the object address of the calling
function ('self' or 'this' pointer), so that the answer may be directed
to a calling object. Callback functions are defined in the global section.

**
ngSpice_Command(char*)
Send a valid command (see the control or interactive commands) from caller
to ngspice.dll. Will be executed immediately (as if in interactive mode).
Some commands are rejected (e.g. 'plot', because there is no graphics interface).
Command 'quit' will remove internal data, and then send a notice to caller via
ngexit().

**
ngGet_Vec_Info(char*)
receives the name of a vector (may be in the form 'vectorname' or
<plotname>.vectorname) and returns a pointer to a vector_info struct.
The caller may then directly assess the vector data (but probably should
not modify them).

***************** If XSPICE is enabled *************************************
**
ngCM_Input_Path(const char*)
Set the input path for files loaded by code models
like d_state, file_source, d_source.
Useful when netlist is sent by ngSpice_Circ and therefore
Infile_Path cannot be retrieved automatically.
If NULL is sent, return the current Infile_Path.

**
ngGet_Evt_NodeInfo(char*)
receives the name of a event node vector (may be in the form 'vectorname' or
<plotname>.vectorname) and returns a pointer to a evt_node_info struct.
The caller may then directly assess the vector data.

**
char** ngSpice_AllEvtNodes(void);
returns to the caller a pointer to an array of all event node names.
****************************************************************************

**
ngSpice_Circ(char**)
sends an array of null-terminated char* to ngspice.dll. Each char* contains a
single line of a circuit (each line like in an input file **.sp). The last
entry to char** has to be NULL. Upon receiving the arry, ngspice.dll will
immediately parse the input and set up the circuit structure (as if received
the circuit from a file by the 'source' command.

**
char* ngSpice_CurPlot();
returns to the caller a pointer to the name of the current plot

**
char** ngSpice_AllPlots()
returns to the caller a pointer to an array of all plots (by their typename)

**
char** ngSpice_AllVecs(char*);
returns to the caller a pointer to an array of vector names in the plot
named by the string in the argument.

**
int ngSpice_LockRealloc(void)
int ngSpice_UnlockRealloc(void)
Locking and unlocking the realloc of output vectors during simulation. May be set
during reading output vectors in the primary thread, while the simulation in the
background thread is moving on.

**
int ngSpice_Reset(void)
Reset ngspice as far as possible

**
Additional basics:
No memory mallocing and freeing across the interface:
Memory allocated in ngspice.dll has to be freed in ngspice.dll.
Memory allocated in the calling program has to be freed only there.

ngspice.dll should never call exit() directly, but handle either the 'quit'
request to the caller or an request for exiting upon error,
done by callback function ngexit().

All boolean signals (NG_BOOL) are of type _Bool, if ngspice is compiled. They
are of type bool if sharedspice.h is used externally.
*/

#ifndef NGSPICE_PACKAGE_VERSION
#define NGSPICE_PACKAGE_VERSION "43+"
#endif
/* we have NG_BOOL instead of BOOL */
#ifndef HAS_NG_BOOL
#define HAS_NG_BOOL 1
#endif

#ifndef NGSPICE_DLL_H
#define NGSPICE_DLL_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__MINGW32__) || defined(_MSC_VER) || defined(__CYGWIN__)
  #ifdef SHARED_MODULE
    #define IMPEXP __declspec(dllexport)
  #else
    #define IMPEXP __declspec(dllimport)
  #endif
#else
  /* use with gcc flag -fvisibility=hidden */
  #if __GNUC__ >= 4
    #define IMPEXP __attribute__ ((visibility ("default")))
    #define IMPEXPLOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define IMPEXP
    #define IMPEXP_LOCAL
  #endif
#endif

/* required only if header is used by the caller,
   is already defined in ngspice.dll */
#ifndef ngspice_NGSPICE_H
/* Complex numbers. */
struct ngcomplex {
    double cx_real;
    double cx_imag;
} ;

typedef struct ngcomplex ngcomplex_t;
#endif

/* NG_BOOL is the boolean variable at the ngspice interface.*/
typedef bool NG_BOOL;

/* vector info obtained from any vector in ngspice.dll.
   Allows direct access to the ngspice internal vector structure,
   as defined in include/ngspice/devc.h . */
typedef struct vector_info {
    char *v_name;		/* Same as so_vname. */
    int v_type;			/* Same as so_vtype. */
    short v_flags;		/* Flags (a combination of VF_*). */
    double *v_realdata;		/* Real data. */
    ngcomplex_t *v_compdata;	/* Complex data. */
    int v_length;		/* Length of the vector. */
} vector_info, *pvector_info;

typedef struct vecvalues {
    char* name;        /* name of a specific vector */
    double creal;      /* actual data value */
    double cimag;      /* actual data value */
    NG_BOOL is_scale;     /* if 'name' is the scale vector */
    NG_BOOL is_complex;   /* if the data are complex numbers */
} vecvalues, *pvecvalues;

typedef struct vecvaluesall {
    int veccount;      /* number of vectors in plot */
    int vecindex;      /* index of actual set of vectors. i.e. the number of accepted data points */
    pvecvalues *vecsa; /* values of actual set of vectors, indexed from 0 to veccount - 1 */
} vecvaluesall, *pvecvaluesall;

/* info for a specific vector */
typedef struct vecinfo
{
    int number;     /* number of vector, as postion in the linked list of vectors, starts with 0 */
    char *vecname;  /* name of the actual vector */
    NG_BOOL is_real;   /* TRUE if the actual vector has real data */
    void *pdvec;    /* a void pointer to struct dvec *d, the actual vector */
    void *pdvecscale; /* a void pointer to struct dvec *ds, the scale vector */
} vecinfo, *pvecinfo;

/* info for the current plot */
typedef struct vecinfoall
{
    /* the plot */
    char *name;
    char *title;
    char *date;
    char *type;
    int veccount;

    /* the data as an array of vecinfo with length equal to the number of vectors in the plot */
    pvecinfo *vecs;

} vecinfoall, *pvecinfoall;

/* to be used by ngGet_Evt_NodeInfo, returns all data of a specific node after simulation */
#ifdef XSPICE
/* a single data point */
typedef struct evt_data
{
    int           dcop;        /* t.b.d. */
    double        step;        /* simulation time */
    char          *node_value; /* one of 0s, 1s, Us, 0r, 1r, Ur, 0z, 1z, Uz, 0u, 1u, Uu */
} evt_data, *pevt_data;

/* a list of all data points of the node selected by the char* argument to ngGet_Evt_NodeInfo */
typedef struct evt_shared_data
{
    pevt_data *evt_dect; /* array of data */
    int num_steps;       /* length of the array */
} evt_shared_data, *pevt_shared_data;
#endif


/* callback functions
addresses received from caller with ngSpice_Init() function
*/
/* sending output from stdout, stderr to caller */
typedef int (SendChar)(char*, int, void*);
/*
   char* string to be sent to caller output
   int   identification number of calling ngspice shared lib
   void* return pointer received from caller, e.g. pointer to object having sent the request
*/
/* sending simulation status to caller */
typedef int (SendStat)(char*, int, void*);
/*
   char* simulation status and value (in percent) to be sent to caller
   int   identification number of calling ngspice shared lib
   void* return pointer received from caller
*/
/* asking for controlled exit */
typedef int (ControlledExit)(int, NG_BOOL, NG_BOOL, int, void*);
/*
   int   exit status
   NG_BOOL  if true: immediate unloading dll, if false: just set flag, unload is done when function has returned
   NG_BOOL  if true: exit upon 'quit', if false: exit due to ngspice.dll error
   int   identification number of calling ngspice shared lib
   void* return pointer received from caller
*/
/* send back actual vector data */
typedef int (SendData)(pvecvaluesall, int, int, void*);
/*
   vecvaluesall* pointer to array of structs containing actual values from all vectors
   int           number of structs (one per vector)
   int           identification number of calling ngspice shared lib
   void*         return pointer received from caller
*/

/* send back initailization vector data */
typedef int (SendInitData)(pvecinfoall, int, void*);
/*
   vecinfoall* pointer to array of structs containing data from all vectors right after initialization
   int         identification number of calling ngspice shared lib
   void*       return pointer received from caller
*/

/* indicate if background thread is running */
typedef int (BGThreadRunning)(NG_BOOL, int, void*);
/*
   NG_BOOL        true if background thread is running
   int         identification number of calling ngspice shared lib
   void*       return pointer received from caller
*/

/* callback functions
   addresses received from caller with ngSpice_Init_Sync() function
*/

/* ask for VSRC EXTERNAL value */
typedef int (GetVSRCData)(double*, double, char*, int, void*);
/*
   double*     return voltage value
   double      actual time
   char*       node name
   int         identification number of calling ngspice shared lib
   void*       return pointer received from caller
*/

/* ask for ISRC EXTERNAL value */
typedef int (GetISRCData)(double*, double, char*, int, void*);
/*
   double*     return current value
   double      actual time
   char*       node name
   int         identification number of calling ngspice shared lib
   void*       return pointer received from caller
*/

/* ask for new delta time depending on synchronization requirements */
typedef int (GetSyncData)(double, double*, double, int, int, int, void*);
/*
   double      actual time (ckt->CKTtime)
   double*     delta time (ckt->CKTdelta)
   double      old delta time (olddelta)
   int         redostep (as set by ngspice)
   int         identification number of calling ngspice shared lib
   int         location of call for synchronization in dctran.c
   void*       return pointer received from caller
*/

#ifdef XSPICE
/* callback functions
addresses received from caller with ngSpice_Init_Evt() function
*/

/* Upon time step finished, called per node */
typedef int (SendEvtData)(int, double, double, char *, void *, int, int, int, void*);
/*
   int         node index
   double      step, actual simulation time
   double      dvalue, a real value for specified structure component for plotting purposes
   char        *svalue, a string value for specified structure component for printing
   void        *pvalue, a binary data structure
   int         plen, size of the *pvalue structure
   int         mode, the mode (op, dc, tran) we are in
   int         ident, identification number of calling ngspice shared lib
   void*       return pointer received from caller
*/

/* Upon initialization, called once per event node
   To build up a dictionary of nodes */
typedef int (SendInitEvtData)(int, int, char*, char*, int, void*);
/*
   int         node index
   int         maximum node index, number of nodes
   char*       node name
   char*       udn-name, node type
   int         identification number of calling ngspice shared lib
   void*       return pointer received from caller
*/
#endif

/* ngspice initialization,
printfcn: pointer to callback function for reading printf, fprintf
statfcn: pointer to callback function for the status string and percent value
ControlledExit: pointer to callback function for setting a 'quit' signal in caller
SendData: pointer to callback function for returning data values of all current output vectors
SendInitData: pointer to callback function for returning information of all output vectors just initialized
BGThreadRunning: pointer to callback function indicating if workrt thread is running
userData: pointer to user-defined data, will not be modified, but
          handed over back to caller during Callback, e.g. address of calling object */
IMPEXP
int  ngSpice_Init(SendChar* printfcn, SendStat* statfcn, ControlledExit* ngexit,
                  SendData* sdata, SendInitData* sinitdata, BGThreadRunning* bgtrun, void* userData);

/* initialization of synchronizing functions
vsrcdat: pointer to callback function for retrieving a voltage source value from caller
isrcdat: pointer to callback function for retrieving a current source value from caller
syncdat: pointer to callback function for synchronization
ident: pointer to integer unique to this shared library (defaults to 0)
userData: pointer to user-defined data, will not be modified, but
          handed over back to caller during Callback, e.g. address of calling object.
          If NULL is sent here, userdata info from ngSpice_Init() will be kept, otherwise
          userdata will be overridden by new value from here.
*/
IMPEXP
int  ngSpice_Init_Sync(GetVSRCData *vsrcdat, GetISRCData *isrcdat, GetSyncData *syncdat, int *ident, void *userData);

/* Caller may send ngspice commands to ngspice.dll.
Commands are executed immediately */
IMPEXP
int  ngSpice_Command(char* command);

/* get info about a vector */
IMPEXP
pvector_info ngGet_Vec_Info(char* vecname);

#ifdef XSPICE
/* Set the input path for files loaded by code models.
   If NULL is sent, return the current Infile_Path. */
IMPEXP
char* ngCM_Input_Path(const char* path);

/* get info about the event node vector */
IMPEXP
pevt_shared_data ngGet_Evt_NodeInfo(char* nodename);

/* get a list of all event nodes */
IMPEXP
char** ngSpice_AllEvtNodes(void);

/* initialization of XSPICE callback functions 
sevtdata: data for a specific event node at time 'step'
sinitevtdata: single line entry of event node dictionary (list)
userData: pointer to user-defined data, will not be modified, but
handed over back to caller during Callback, e.g. address of calling object */
IMPEXP
int  ngSpice_Init_Evt(SendEvtData* sevtdata, SendInitEvtData* sinitevtdata, void* userData);
#endif


/* send a circuit to ngspice.dll
   The circuit description is a dynamic array
   of char*. Each char* corresponds to a single circuit
   line. The last-but-one entry of the array has to be a .end card,
   followed by the last entry NULL. */
IMPEXP
int ngSpice_Circ(char** circarray);


/* return to the caller a pointer to the name of the current plot */
IMPEXP
char* ngSpice_CurPlot(void);


/* return to the caller a pointer to an array of all plots created
so far by ngspice.dll */
IMPEXP
char** ngSpice_AllPlots(void);


/* return to the caller a pointer to an array of vector names in the plot
named by plotname */
IMPEXP
char** ngSpice_AllVecs(char* plotname);

/* returns TRUE if ngspice is running in a second (background) thread */
IMPEXP
NG_BOOL ngSpice_running(void);

/* set a breakpoint in ngspice */
IMPEXP
NG_BOOL ngSpice_SetBkpt(double time);

/* Set variable no_spinit, if reading 'spinit' is not wanted. */
IMPEXP
int ngSpice_nospinit(void);

/* Set variable no_spiceinit, if reading '.spiceinit' is not wanted. */
IMPEXP
int ngSpice_nospiceinit(void);

#ifdef __cplusplus
}
#endif

#endif
