/*
Test file for shared ngspice with event nodes
Copyright Holger Vogt 2018

ngspice library loaded dynamically

Test 1
Load and initialize ngspice
Source an input file adder_mos.cir
Run the simulation for 0.5 seconds in a background thread
Stop the simulation for 3 seconds
Resume the simulation in the background thread
Write rawfile
Unload ngspice

*/



#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#ifndef _MSC_VER
#include <stdbool.h>
#include <pthread.h>
#else
#define bool int
#define true 1
#define false 0
#define strdup _strdup
#endif

#ifndef _MSC_VER
#include "../../../ngspice/src/include/ngspice/sharedspice.h"
#else
#include "../../../src/include/ngspice/sharedspice.h"
#endif

#if defined(__MINGW32__) ||  defined(_MSC_VER)
#undef BOOLEAN
#include <windows.h>
typedef FARPROC funptr_t;
void *dlopen (const char *, int);
funptr_t dlsym (void *, const char *);
int dlclose (void *);
char *dlerror (void);
#define RTLD_LAZY	1	/* lazy function call binding */
#define RTLD_NOW	2	/* immediate function call binding */
#define RTLD_GLOBAL	4	/* symbols in this dlopen'ed obj are visible to other dlopen'ed objs */
static char errstr[128];
#else
#include <dlfcn.h> /* to load libraries*/
#include <unistd.h>
#include <ctype.h>
typedef void *  funptr_t;
#endif

bool no_bg = true;
bool not_yet = true;
bool will_unload = false;
bool error_ngspice = false;

static bool firsttime = true;

int cieq(register char *p, register char *s);
int ciprefix(const char *p, const char *s);

/* callback functions used by ngspice */
int
ng_getchar(char* outputreturn, int ident, void* userdata);

int
ng_getstat(char* outputreturn, int ident, void* userdata);

int
ng_thread_runs(bool noruns, int ident, void* userdata);

/* callback functions used by XSPICE for event data  */
int
ng_getevtdata(int index, double step, double dvalue, char *svalue,
    void *pvalue, int plen, int mode, int ident, void *userdata);

int
ng_getinitevtdata(int index, int max_index, char * name, char *type, int ident, void *userdata);

ControlledExit ng_exit;
SendData ng_data;
SendInitData ng_initdata;

int vecgetnumber = 0;
double v2dat;
static bool has_break = false;
int testnumber = 0;
void alterp(int sig);

/* functions exported by ngspice */
funptr_t ngSpice_Init_handle = NULL;
funptr_t ngSpice_Command_handle = NULL;
funptr_t ngSpice_Circ_handle = NULL;
funptr_t ngSpice_CurPlot_handle = NULL;
funptr_t ngSpice_AllVecs_handle = NULL;
funptr_t ngSpice_GVI_handle = NULL;
funptr_t ngSpice_AllEvtNodes_handle = NULL;
funptr_t ngSpice_EVT_handle = NULL;
funptr_t ngSpice_Init_Evt_handle = NULL;

void * ngdllhandle = NULL;

#ifndef _MSC_VER
pthread_t mainthread;
#endif // _MSC_VER

int main(int argc, char* argv[])
{
    char *errmsg = NULL, *loadstring = NULL, *curplot, *vecname;
    int *ret, i;
    char **vecarray;
    printf("\n");
    printf("****************************\n");
    printf("**  ngspice shared start  **\n");
    printf("****************************\n");

#ifndef _MSC_VER
    mainthread = pthread_self();
#endif // _MSC_VER
    printf("ng_shared_test_v.exe found in %s\n", argv[0]);

    printf("Load ngspice.dll\n");

    /* path to shared ngspice via command line */
    if (argc > 1) {
        loadstring = argv[1];
        ngdllhandle = dlopen(loadstring, RTLD_NOW);
    }
    /* try some standard paths */
    if (!ngdllhandle) {
#ifdef __CYGWIN__
        loadstring = "/cygdrive/c/cygwin/usr/local/bin/cygngspice-0.dll";
#elif _MSC_VER
        loadstring = "..\\..\\sharedspice\\Debug.x64\\ngspice.dll";
#elif __MINGW32__
        loadstring = "D:\\Spice_general\\ngspice\\visualc-shared\\Debug\\bin\\ngspice.dll";
#else
        loadstring = "libngspice.so";
#endif
        ngdllhandle = dlopen(loadstring, RTLD_NOW);
    }
    errmsg = dlerror();
    if (errmsg)
        printf("%s\n", errmsg);
    if (ngdllhandle)
        printf("ngspice.dll loaded\n");
    else {
        printf("ngspice.dll not loaded !\n");
/* Delay closing the Windows console */
#if defined(__MINGW32__) || defined(_MSC_VER)
        Sleep(3000);
#endif
        exit(1);
    }

    ngSpice_Init_handle = dlsym(ngdllhandle, "ngSpice_Init");
    errmsg = dlerror();
    if (errmsg)
        printf(errmsg);
    ngSpice_Command_handle = dlsym(ngdllhandle, "ngSpice_Command");
    errmsg = dlerror();
    if (errmsg)
        printf(errmsg);
    ngSpice_CurPlot_handle = dlsym(ngdllhandle, "ngSpice_CurPlot");
    errmsg = dlerror();
    if (errmsg)
        printf(errmsg);
    ngSpice_AllVecs_handle = dlsym(ngdllhandle, "ngSpice_AllVecs");
    errmsg = dlerror();
    if (errmsg)
        printf(errmsg);
    ngSpice_GVI_handle = dlsym(ngdllhandle, "ngGet_Vec_Info");
    errmsg = dlerror();
    if (errmsg)
        printf(errmsg);
    ngSpice_AllEvtNodes_handle = dlsym(ngdllhandle, "ngSpice_AllEvtNodes");
    errmsg = dlerror();
    if (errmsg)
        printf(errmsg);
    ngSpice_EVT_handle = dlsym(ngdllhandle, "ngGet_Evt_NodeInfo");
    errmsg = dlerror();
    if (errmsg)
        printf(errmsg);
    ngSpice_Init_Evt_handle = dlsym(ngdllhandle, "ngSpice_Init_Evt");
    errmsg = dlerror();
    if (errmsg)
        printf(errmsg);

    /* general callback initialization */
    ret = ((int * (*)(SendChar*, SendStat*, ControlledExit*, SendData*, SendInitData*,
        BGThreadRunning*, void*)) ngSpice_Init_handle)(ng_getchar, ng_getstat,
            ng_exit, ng_data, ng_initdata, ng_thread_runs, NULL);

    /* event data initialization */
    ret = ((int * (*)(SendEvtData*, SendInitEvtData*, void*)) ngSpice_Init_Evt_handle)(ng_getevtdata,
            ng_getinitevtdata, NULL);

 //   goto test2;

    testnumber = 1;
    printf("\n**  Test no. %d with sourcing digital input file **\n\n", testnumber);
    error_ngspice = false;
    will_unload = false;

#if defined(__CYGWIN__)
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("source /cygdrive/d/Spice_general/ngspice_sh/examples/shared-ngspice/counter-test.cir");
#elif __MINGW32__
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("source D:\\Spice_general\\ngspice_sh\\examples\\shared-ngspice\\counter-test.cir");
#elif _MSC_VER
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("source ../examples/counter-test.cir");
#else
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("source ../examples/counter-test.cir");
//    ret = ((int * (*)(char*)) ngSpice_Command_handle)("source ../examples/adder_mos.cir");
#endif
    /* reset firsttime for ng_data() */
    firsttime = true;
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("bg_run");
    printf("Background thread started\n");

    /* wait to catch error signal, if available */
#if defined(__MINGW32__) || defined(_MSC_VER)
    Sleep(100);
#else
    usleep(100000);
#endif
    /* Upon error: unload ngspice and skip rest of test code */
    if (error_ngspice) {
        printf("Error detected, let's unload\n");
        ret = ((int * (*)(char*)) ngSpice_Command_handle)("quit");
        dlclose(ngdllhandle);
        printf("ngspice.dll unloaded\n\n");
        ngdllhandle = NULL;
        return 1;
    }

    /* simulate for 500 milli seconds or until simulation has finished */
    for (i = 0; i < 5; i++) {
#if defined(__MINGW32__) || defined(_MSC_VER)
        Sleep(100);
#else
        usleep(100000);
#endif
        /* we are faster than anticipated */
        if (no_bg) {
            printf("Faster than 500ms!\n");
            goto endsim;
        }
    }

    ret = ((int * (*)(char*)) ngSpice_Command_handle)("bg_halt");
    for (i = 3; i > 0; i--) {
        printf("Pause for %d seconds\n", i);
#if defined(__MINGW32__) || defined(_MSC_VER)
        Sleep(1000);
#else
        usleep(1000000);
#endif
    }
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("bg_resume");

    /* wait for 1s while simulation continues */
#if defined(__MINGW32__) || defined(_MSC_VER)
    Sleep(1000);
#else
    usleep(1000000);
#endif
    /* read current plot while simulation continues */
    curplot = ((char * (*)()) ngSpice_CurPlot_handle)();
    printf("\nCurrent plot is %s\n\n", curplot);
    vecarray = ((char ** (*)(char*)) ngSpice_AllVecs_handle)(curplot);
    /* get length of first vector */
    if (vecarray) {
        char plotvec[256];
        pvector_info myvec;
        int veclength;
        vecname = vecarray[0];
        sprintf(plotvec, "%s.%s", curplot, vecname);
        myvec = ((pvector_info(*)(char*)) ngSpice_GVI_handle)(plotvec);
        if (myvec) {
            veclength = myvec->v_length;
            printf("\nActual length of vector %s is %d\n\n", plotvec, veclength);
        }
        else
            printf("\nCould not read vector %s\n\n", plotvec);
    }
    /* wait until simulation finishes */
    for (;;) {
#if defined(__MINGW32__) || defined(_MSC_VER)
        Sleep(100);
#else
        usleep(100000);
#endif
        if (no_bg) {
            printf("Faster than 500ms!\n");
            break;
        }
    }
endsim:
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("edisplay");

    /* Print all event nodes to stdout */
    vecarray = ((char ** (*)()) ngSpice_AllEvtNodes_handle)();
    i = 0;
    if (vecarray) {
        char* node;
        printf("\nWe print all event node names:\n");
        for (i = 0; ; i++) {
            node = vecarray[i];
            if (!vecarray[i])
                break;
            fprintf(stdout, "%s\n", vecarray[i]);
        }
    }
    printf("We have %d event nodes\n\n", i);

    /* just to select some nodes */
    int j = (int)(i / 2);
    char * nodename;
    if (j > 0) {
        nodename = vecarray[j];
        printf("We analyse event node %s\n", nodename);
        pevt_shared_data evtnode;
        /* Get data from ngspice.dll */
        evtnode = ((pevt_shared_data(*)(char*)) ngSpice_EVT_handle)(nodename);
        pevt_data *nodevals = evtnode->evt_dect;
        int count = evtnode->num_steps;
        for (i = 0; i < count; i++) {
            char *nodeval = nodevals[i]->node_value;
            double step = nodevals[i]->step;
            fprintf(stdout, "%e    %s\n", step, nodeval);
        }
        /* Delete data */
        ((pevt_shared_data(*)(char*)) ngSpice_EVT_handle)(NULL);
        /* just watch another node */
        nodename = vecarray[j - 1];
        printf("We analyse event node %s\n", nodename);
        evtnode = ((pevt_shared_data(*)(char*)) ngSpice_EVT_handle)(nodename);
        nodevals = evtnode->evt_dect;
        count = evtnode->num_steps;
        for (i = 0; i < count; i++) {
            char *nodeval = nodevals[i]->node_value;
            double step = nodevals[i]->step;
            fprintf(stdout, "%e    %s\n", step, nodeval);
        }
        /* Delete data */
        ((pevt_shared_data(*)(char*)) ngSpice_EVT_handle)(NULL);
    }
    else
        printf("Not enough nodes for selection\n\n");
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("remcirc");

test2:
    testnumber = 2;
    printf("\n**  Test no. %d with sourcing analog input file **\n\n", testnumber);
    error_ngspice = false;
    will_unload = false;

#if defined(__CYGWIN__)
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("source /cygdrive/d/Spice_general/ngspice_sh/examples/shared-ngspice/counter-test.cir");
#elif __MINGW32__
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("source D:\\Spice_general\\ngspice_sh\\examples\\shared-ngspice\\counter-test.cir");
#elif _MSC_VER
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("source ../examples/adder_mos.cir");
#else
//    ret = ((int * (*)(char*)) ngSpice_Command_handle)("source ../examples/counter-test.cir");
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("source ../examples/adder_mos.cir");
#endif
    /* test the new feature */
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("save none");
    /* reset firsttime for ng_data() */
    firsttime = true;
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("bg_run");
    printf("Background thread started\n");

    /* wait to catch error signal, if available */
#if defined(__MINGW32__) || defined(_MSC_VER)
    Sleep(100);
#else
    usleep(100000);
#endif
    /* Upon error: unload ngspice and skip rest of test code */
    if (error_ngspice) {
        printf("Error detected, let's unload\n");
        ret = ((int * (*)(char*)) ngSpice_Command_handle)("quit");
        dlclose(ngdllhandle);
        printf("ngspice.dll unloaded\n\n");
        ngdllhandle = NULL;
        return 0;
    }

    /* simulate for 500 milli seconds or until simulation has finished */
    for (i = 0; i < 5; i++) {
#if defined(__MINGW32__) || defined(_MSC_VER)
        Sleep(100);
#else
        usleep(100000);
#endif
        /* we are faster than anticipated */
        if (no_bg) {
            printf("Faster than 500ms!\n");
            goto endsim2;
        }
    }

    ret = ((int * (*)(char*)) ngSpice_Command_handle)("bg_halt");
    for (i = 3; i > 0; i--) {
        printf("Pause for %d seconds\n", i);
#if defined(__MINGW32__) || defined(_MSC_VER)
        Sleep(1000);
#else
        usleep(1000000);
#endif
    }
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("alter VCC = 5");

    ret = ((int * (*)(char*)) ngSpice_Command_handle)("bg_resume");

    /* wait for 1s while simulation continues */
#if defined(__MINGW32__) || defined(_MSC_VER)
    Sleep(1000);
#else
    usleep(1000000);
#endif

endsim2:
    /* read current plot while simulation continues */
    curplot = ((char * (*)()) ngSpice_CurPlot_handle)();
    printf("\nCurrent plot is %s\n\n", curplot);
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("bg_halt");
    vecarray = ((char ** (*)(char*)) ngSpice_AllVecs_handle)(curplot);
    /* get length of first vector */
    if (vecarray) {
        char plotvec[256];
        pvector_info myvec;
        int veclength;
        vecname = vecarray[0];
        sprintf(plotvec, "%s.%s", curplot, vecname);
        myvec = ((pvector_info(*)(char*)) ngSpice_GVI_handle)(plotvec);
        if (myvec) {
            veclength = myvec->v_length;
            printf("\nActual length of vector %s is %d\n\n", plotvec, veclength);
        }
        else
            printf("\nCould not read vector %s\n\n", plotvec);
    }
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("display");
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("bg_resume");
    /* wait until simulation finishes */
    for (;;) {
#if defined(__MINGW32__) || defined(_MSC_VER)
        Sleep(100);
#else
        usleep(100000);
#endif
        if (no_bg) {
            printf("Faster than 100ms!\n");
            break;
        }
    }
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("display");
    ret = ((int * (*)(char*)) ngSpice_Command_handle)("quit");

    /* unload now */
    dlclose(ngdllhandle);
    ngdllhandle = NULL;
    printf("Unloaded\n\n");
#if 0
    if (will_unload) {
        printf("Unload now\n");
        dlclose(ngdllhandle);
        ngdllhandle = NULL;
        printf("Unloaded\n\n");
    }
#endif
/* wait before closing the command window */
    puts("\nPress <enter> to quit:");
    getchar();

    return 0;
}


/* Callback function called from bg thread in ngspice to transfer
   any string created by printf or puts. Output to stdout in ngspice is
   preceded by token stdout, same with stderr.*/
int
ng_getchar(char* outputreturn, int ident, void* userdata)
{
    printf("%s\n", outputreturn);
    /* setting a flag if an error message occurred */
    if (ciprefix("stderr Error:", outputreturn))
        error_ngspice = true;
    return 0;
}

/* Callback function called from bg thread in ngspice to transfer
   simulation status (type and progress in percent. */
int
ng_getstat(char* outputreturn, int ident, void* userdata)
{
    printf("%s\n", outputreturn);
    return 0;
}

/* Callback function called from ngspice upon starting (returns true) or
  leaving (returns false) the bg thread. */
int
ng_thread_runs(bool noruns, int ident, void* userdata)
{
    no_bg = noruns;
    if (noruns)
        printf("bg not running\n");
    else
        printf("bg running\n");

    return 0;
}

/* Callback function called from bg thread in ngspice if fcn controlled_exit()
   is hit. Do not exit, but unload ngspice. */
int
ng_exit(int exitstatus, bool immediate, bool quitexit, int ident, void* userdata)
{

    if(quitexit) {
        printf("DNote: Returned from quit with exit status %d\n", exitstatus);
    }
    if(immediate) {
        printf("DNote: Unload ngspice\n");
        ((int * (*)(char*)) ngSpice_Command_handle)("quit");
        dlclose(ngdllhandle);
    }

    else {
        printf("DNote: Prepare unloading ngspice\n");
        will_unload = true;
    }

    /* Delay closing the Windows console */
#if defined(__MINGW32__) || defined(_MSC_VER)
    Sleep(3000);
#endif

    return exitstatus;

}

/* Callback function called from bg thread in ngspice once per accepted data point */
int
ng_data(pvecvaluesall vdata, int numvecs, int ident, void* userdata)
{
    double tscale;
    static double olddat = 0.0;
    static int i, j;
    char *vecname = "V(12)";

    if (firsttime) {
        for (i = 0; i < numvecs; i++) {
            /* We have a look at vector V(12) from adder_mos.cir.
               Get the index i */
            if (cieq(vdata->vecsa[i]->name, vecname))
                break;
        }
        firsttime = false;
        /* get the scale vector index j */
        for (j = 0; j < numvecs; j++) {
            if (vdata->vecsa[j]->is_scale)
                break;
        }
        if (i == numvecs) {
            fprintf(stderr, "Error: Vector %s not found!\n", vecname);
        }
    }
    if ((testnumber == 2)  && (i < numvecs)){
        v2dat = vdata->vecsa[i]->creal;
        /* print output value only if it has changed by more than 10 mV */
        if ((olddat + 0.01 < v2dat) || (olddat - 0.01 > v2dat)) {
            tscale = vdata->vecsa[j]->creal;
            printf("real value of vector %s is %e at %e\n", vdata->vecsa[i]->name, v2dat, tscale);
        }
        olddat = v2dat;
    }
    return 0;
}


/* Callback function called from bg thread in ngspice once upon intialization
   of the analog simulation vectors)*/
int
ng_initdata(pvecinfoall intdata, int ident, void* userdata)
{
    int i;
    /* suppress second printing (after bg_resume) */
    static bool printonce = true;
    int vn = intdata->veccount;
    if (printonce) {
        for (i = 0; i < vn; i++) {
            printf("Vector: %s\n", intdata->vecs[i]->vecname);
            /* find the location of a vector */
            char *myvec = "adacout";
            if (cieq(intdata->vecs[i]->vecname, myvec)) {
                vecgetnumber = i;
                printf("Vector %s has index %d\n", myvec, i);
            }
        }
        printonce = false;
    }
    return 0;
}

int mindex = -1;
char* mynode = "dout8";
/* callback functions used by XSPICE for event data. */

/* Function is executed each time EVTdump() in ngspice is called and output
   data for the node indexed by index have changed. */
int
ng_getevtdata(int index, double step, double dvalue, char *svalue,
    void *pvalue, int plen, int mode, int ident, void *userdata)
{
    static bool once = true;
    if (mindex == -1) {
        if (once) {
            fprintf(stderr, "Error: Cannot watch node %s, not found\n", mynode);
            once = false;
        }
        return 1;
    }
    /* Select an action only for a specific node.
       The value of mindex for node mynode has been determined in function
       ng_getinitevtdata() given below. */
    if (index == mindex)
        fprintf(stdout, "Node %s, Index %d, Step %e, Value %s\n", mynode, index, step, svalue);
    return 0;
}

char *evt_nodes[2][1000];
/* Immediately after initialization, get the list of event nodes as a string array:
   evt_nodes[0][x]: node name
   evt_nodes[1][x]: node type
   index x: node number, may be used to access node value by using
   function ng_getevtdata(index, ...) */
int
ng_getinitevtdata(int index, int max_index, char * name, char *type, int ident, void *userdata)
{
    evt_nodes[0][index] = strdup(name);
    evt_nodes[1][index] = strdup(type);
    /* for a given node name, find the corresponding index */
    if (cieq(name, mynode))
       mindex = index;
    return 0;
}

/* Funcion called from main thread upon receiving signal SIGTERM */
void
alterp(int sig) {
    ((int * (*)(char*)) ngSpice_Command_handle)("bg_halt");
}

/* Unify LINUX and Windows dynamic library handling:
   Add functions dlopen, dlsym, dlerror, dlclose to Windows by
   tranlating to Windows API functions.
*/
#if defined(__MINGW32__) ||  defined(_MSC_VER)

void *dlopen(const char *name,int type)
{
	return LoadLibrary((LPCSTR)name);
}

funptr_t dlsym(void *hDll, const char *funcname)
{
	return GetProcAddress(hDll, funcname);
}

char *dlerror(void)
{
	LPVOID lpMsgBuf;
    char * testerr;
    DWORD dw = GetLastError();

	FormatMessage(
		FORMAT_MESSAGE_ALLOCATE_BUFFER |
		FORMAT_MESSAGE_FROM_SYSTEM |
		FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		dw,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		(LPTSTR) &lpMsgBuf,
		0,
		NULL
	);
	testerr = (char*)lpMsgBuf;
    strcpy(errstr,lpMsgBuf);
	LocalFree(lpMsgBuf);
    if (ciprefix("Der Vorgang wurde erfolgreich beendet.", errstr))
        return NULL;
    else
        return errstr;
}

int dlclose (void *lhandle)
{
    return (int)FreeLibrary(lhandle);
}
#endif

/* Case insensitive str eq.
   Like strcasecmp( ) XXX */
int
cieq(register char *p, register char *s)
{
    while (*p) {
        if ((isupper(*p) ? tolower(*p) : *p) !=
            (isupper(*s) ? tolower(*s) : *s))
            return(false);
        p++;
        s++;
    }
    return (*s ? false : true);
}

/* Case insensitive prefix. */
int
ciprefix(const char *p, const char *s)
{
    while (*p) {
        if ((isupper(*p) ? tolower(*p) : *p) !=
            (isupper(*s) ? tolower(*s) : *s))
            return(false);
        p++;
        s++;
    }
    return (true);
}
