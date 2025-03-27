/*
Test file for shared ngspice
Copyright Holger Vogt 2017-2024
Local commands: Giles Atkinson 2025
New BSD license

ngspice library loaded dynamically
simple manual input
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

#define XSPICE
#include "sharedspice.h"

typedef void *funptr_t;

#if defined(__MINGW32__) || defined(__CYGWIN__) || defined(_MSC_VER)

#undef BOOLEAN
#define LOAD_STRING "libngspice.DLL"

#include <windows.h>

void *dlopen (const char *, int);
funptr_t dlsym (void *, const char *);
int dlclose (void *);
char *dlerror (void);
#define RTLD_LAZY	1	/* lazy function call binding */
#define RTLD_NOW	2	/* immediate function call binding */
#define RTLD_GLOBAL	4	/* Symbols are externally visible. */

static char errstr[128];

#else

#if defined(__APPLE__)
#define LOAD_STRING "./libngspice.dylib"
#else
#define LOAD_STRING "./libngspice.so"
#endif

#include <dlfcn.h> /* to load libraries*/
#include <unistd.h>

#endif /* not Windows */

#include <ctype.h>

/* pointers to functions exported by ngspice */

char      ** (*ngSpice_AllEvtNodes_handle)(void);
char      ** (*ngSpice_AllPlots_handle)(void);
char      ** (*ngSpice_AllVecs_handle)(char*);
int          (*ngSpice_Command_handle)(char*);
int          (*ngSpice_Circ_handle)(char**);
char       * (*ngSpice_CurPlot_handle)(void);
int          (*ngSpice_Decode_Evt_handle)(void *, int,
                                          double *, const char **);
pvector_info (*ngSpice_GVI_handle)(char*);
int          (*ngSpice_Init_handle)(SendChar*, SendStat*, ControlledExit*,
                                    SendData*, SendInitData*, BGThreadRunning*,
                                    void*);
int          (*ngSpice_Init_Evt_handle)(SendEvtData*, SendInitEvtData*, void*);
int          (*ngSpice_Init_Sync_handle)(GetVSRCData*, GetISRCData*,
                                         GetSyncData*, int*, void*);
int          (*ngSpice_Raw_Evt_handle)(char *, SendRawEvtData *, void *);
int          (*ngSpice_Reset_handle)(void);
bool         (*ngSpice_running_handle)(void);

bool no_bg = true;
bool not_yet = true;

/* Limit output from SendData, SendEvtData and VSRCData callbacks. */

static unsigned int sd_limit = 10, sd_count, sd_list;
static unsigned int se_limit = 10, se_count;
static unsigned int sq_limit = 10, sq_count, sq_ask;

/* Automatic ramp of source query replies. */

double sr_last_time = -1, sr_target_time = -1;
double sr_last_val, sr_target_val, sim_time;

static int cieq(register char *p, register char *s);
static int ciprefix(const char *p, const char *s);
static int getLine(char *prmpt, char *buff, size_t sz);

/* Callback functions used by ngspice and defined below. */

static int
ng_getchar(char* outputreturn, int ident, void* userdata);

static int
ng_getstat(char* outputreturn, int ident, void* userdata);

static int
ng_thread_runs(bool noruns, int ident, void* userdata);

static int ng_rawevt(double, void *, void *, int);

static ControlledExit  ng_exit;
static SendData        ng_data;
static SendInitData    ng_initdata;
static SendInitEvtData ng_initevtdata;
static SendEvtData     ng_evtdata;
static GetVSRCData     ng_srcdata;
static GetSyncData     ng_syncdata;

char comd[1024];
void * ngdllhandle = NULL;

/* Register call-back functions. */

static void register_cbs(void) {
    static int  z =  0;
    char       *bad;
    int         ret;

    ret = ngSpice_Init_handle(ng_getchar, ng_getstat,
                              ng_exit, ng_data, ng_initdata, ng_thread_runs,
                              NULL);
    if (ret) {
        bad = "ngSpice_Init";
        goto fail;
    }
    ret = ngSpice_Init_Evt_handle(ng_evtdata, ng_initevtdata, NULL);
    if (!ret) {
        bad = "ngSpice_Init_Evt";
        goto fail;
    }
    ret = ngSpice_Init_Sync_handle(ng_srcdata, ng_srcdata, ng_syncdata,
                                   &z, NULL);
    if (ret == 0)
        return;
    bad = "ngSpice_Init_Sync";
 fail:
    fprintf(stderr, "Init call %s() failed: %d\n", bad, ret);
    exit(2);
}

/* Non-interactive test: run a selection of circuits. */

static int auto_test(void)
{
    #define SRC_FMT "source ../%s"          /* Assume cd is examples/shared. */
    static const char * const tests[] = {
        "transient-noise/shot_ng.cir",
        "soi/inv_tr.sp",
        "p-to-n-examples/op-test-adi.cir",
        "digital/compare/adder_Xspice.cir",
        NULL
    };
    int ret, i;
    char msgbuf[128];

    for (i = 0; tests[i]; ++i) {
        register_cbs();
        snprintf(msgbuf, sizeof msgbuf, "echo run no. %d", i + 1);
        ret = ngSpice_Command_handle(msgbuf);
        if (ret)
            return ret;
        snprintf(msgbuf, sizeof msgbuf, SRC_FMT, tests[i]);
        ret = ngSpice_Command_handle(msgbuf);
        if (ret)
            return ret;
        ret = ngSpice_Reset_handle();
        if (ret)
            return ret;
    }
    return 0;
}

static funptr_t getsym(const char *sym)
{
    funptr_t  value;

    value = dlsym(ngdllhandle, sym);
    if (!value) {
        fprintf(stderr, "Ngspice symbol %s was not found: %s\n",
                sym, dlerror());
        exit(1);
    }
    return value;
}

/* String utilities. */

char *skip(char *cp)
{
    while (isspace(*cp))
        ++cp;
    return cp;
}

char *tail(char *cp)
{
    while (*cp && !isspace(*cp))
        ++cp;
    return cp;
}

/* Execute a local command, identified by a leading '/'. */

static void help(char *cmd) {
    puts("Local commands are:\n"
         "  /aevt\t\t\tList event nodes.\n"
         "  /aplot\t\tList plot names.\n"
         "  /avec <plot_name>\tList vectors in plot.\n"
         "  /bgr\t\t\tQuery background thread.\n"
         "  /cplot\t\tShow name of current plot.\n"
         "  /dlim <limit>\t\tSet output limit for SendData CB.\n"
         "  /elim <limit>\t\tSet output limit for SendEvtData CB.\n"
         "  /help\t\t\tPrint this message.\n"
         "  /lvals\t\tList node values on data callback (toggle).\n"
         "  /reset\t\tReset Ngspice.\n"
         "  /sask\t\t\tAsk for V/ISRC values (toggles).\n"
         "  /slim <limit>\t\tSet output limit for SendxSRCData CB.\n"
         "  /sramp [new_val] [interval end_val]\n"
         "\t\t\tAuto-ramp sources\n"
         "  /vec <vector>\t\tQuery vector.\n"
         "  /xnode <node ...>\tRequest raw event callbacks for event node.\n"
         "All other input is passed to Ngspice.\n");
}

static void aevt(char *cmd) {
    char **cpp;

    cpp = ngSpice_AllEvtNodes_handle();
    if (!cpp)
        return;
    printf("Event nodes:\n");
    while (*cpp)
        printf("  %s\n", *cpp++);
}

static void aplot(char *cmd) {
    char **cpp;

    cpp = ngSpice_AllPlots_handle();
    if (!cpp)
        return;
    printf("Plots:\n");
    while (*cpp)
        printf("  %s\n", *cpp++);
}

static void avec(char *cmd) {
    char **cpp;

    cpp = ngSpice_AllVecs_handle(cmd);
    if (!cpp)
        return;
    printf("Vectors in plot %s:\n", cmd);
    while (*cpp)
        printf("  %s\n", *cpp++);
}

static void bgr(char *cmd) {
    printf("Background thread is %srunning\n",
           ngSpice_running_handle() ? "": "not ");
}

static void cplot(char *cmd) {
    printf("Current plot: %s\n", ngSpice_CurPlot_handle());
}

static void dlim(char *cmd) {
    sd_limit = atoi(cmd);
    sd_count = 0;
}

static void elim(char *cmd) {
    se_limit = atoi(cmd);
    se_count = 0;
}

static void lvals(char *cmd) {
    sd_list ^= 1;
    printf("Listing node values is now %s.\n",
           sd_list ? "on" : "off");
}

static void reset(char *cmd) {
    int ret;

    ret = ngSpice_Reset_handle();
    if (ret) {
        fprintf(stderr, "Reset error %d\n", ret);
        return;
    }
    register_cbs();
    sd_count = se_count = sq_count = 10;
    sr_last_time = -1;
    sr_target_time = -1;
    sim_time = 0;
}

static void sask(char *cmd) {
    pvector_info vp;

    sq_ask ^= 1;
    printf("Prompting for V/ISRC values is now %s.\n",
           sq_ask ? "on" : "off");
}

static void slim(char *cmd) {
    sq_limit = atoi(cmd);
    sq_count = 0;
}

static void sramp(char *cmd) {
    double v[3];
    int i;

    for (i = 0; i < 3; ++i) {
        if (!*cmd)
            break;
        v[i] = strtod(cmd, NULL);
        cmd = skip(tail(cmd));
    }
    if (sr_last_time < 0.0)
        sr_last_time = sim_time;
    switch (i) {
    case 0:
        break;
    case 1:
        sr_last_val = sr_target_val = v[0];
        break;
    case 2:
        sr_target_time = sr_last_time + v[0];
        sr_target_val = v[1];
        break;
    default:
        sr_last_val = sr_target_val = v[0];
        sr_target_time = sr_last_time + v[1];
        sr_target_val = v[2];
        break;
    }
    if (sr_target_time < 0.0)
        sr_target_time = sim_time + 1.0;
}

static void vec(char *cmd) {
    pvector_info vp;

    vp = ngSpice_GVI_handle(cmd);
    if (!vp)
        return;
    printf("Vector %s: length %d type %d flags %0hx\n",
           vp->v_name, vp->v_length, vp->v_type, vp->v_flags);
}

struct raw_cb_ctx {
    int   type;
    char  name[1];
};
        
static void xnode(char *cmd) {
    struct raw_cb_ctx *ctx;
    char              *end, c;
    int                ret;
    
    while (*cmd) {
        /* Memory leak here. */

        end = tail(cmd);
        ctx = (struct raw_cb_ctx *)malloc(sizeof *ctx + (end - cmd));
        c = *end;
        *end = '\0';
        strcpy(ctx->name, cmd);
        ret = ngSpice_Raw_Evt_handle(cmd, ng_rawevt, ctx);
        if (ret >= 0) {
            ctx->type = ret;
        } else {
            free(ctx);
            fprintf(stderr, "Node name not recognised\n");
        }
        *end = c;
        cmd = skip(end);
    }
}

#define E(name) { #name, name }

static void local(char *cmd)
{
    static const struct {
        const char *cmd;
        void       (*fn)(char *);
    }     table[] = { E(help),          // First, so that just "/" works.
                      E(aevt), E(avec), E(aplot), E(bgr), E(cplot),
                      E(dlim), E(elim), E(lvals), E(reset), E(sask), E(slim),
                      E(sramp), E(vec), E(xnode),
                     { NULL, NULL }};
    char *end;
    int   i, len;

    end = tail(cmd);
    len = end - cmd;
    for (i = 0; table[i].cmd; ++i) {
        if (!strncmp(cmd, table[i].cmd, len)) {
            table[i].fn(skip(end));
            return;
        }
    }
    fprintf(stderr, "No such local command\n");
}

int main(int argc, char **argv)
{
    char *libpath = NULL;
    int ret, i, do_auto = 0;;

    for (i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
            case 'a':
                do_auto = 1;
                break;
            case 'l':
                libpath = argv[i + 1];
                break;
            default:
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                return 2;
            }
        }
    }

    if (!libpath)
        libpath = LOAD_STRING;
    ngdllhandle = dlopen(libpath, RTLD_GLOBAL | RTLD_NOW);
    if (ngdllhandle) {
        printf("Ngspice library loaded.\n");
    } else {
        fprintf(stderr, "Ngspice library not loaded!\n  %s\n", dlerror());
        return 1;
    }

    ngSpice_AllEvtNodes_handle = getsym("ngSpice_AllEvtNodes");
    ngSpice_Command_handle = getsym("ngSpice_Command");
    ngSpice_CurPlot_handle = getsym("ngSpice_CurPlot");
    ngSpice_AllPlots_handle = getsym("ngSpice_AllPlots");
    ngSpice_AllVecs_handle = getsym("ngSpice_AllVecs");
    ngSpice_Decode_Evt_handle = getsym("ngSpice_Decode_Evt");
    ngSpice_GVI_handle = getsym("ngGet_Vec_Info");
    ngSpice_Init_handle = getsym("ngSpice_Init");
    ngSpice_Init_Evt_handle = getsym("ngSpice_Init_Evt");
    ngSpice_Init_Sync_handle = getsym("ngSpice_Init_Sync");
    ngSpice_Raw_Evt_handle = getsym("ngSpice_Raw_Evt");
    ngSpice_Reset_handle = getsym("ngSpice_Reset");
    ngSpice_running_handle = getsym("ngSpice_running");
    register_cbs();

    if (do_auto)
        return auto_test();

    /* Interactive. */

    printf("Enter \"/h\" for local command descriptions.\n\n");
    
    for (;;) {
        /* get command from stdin */

        if (getLine("Command: ", comd, sizeof(comd)))
            return 0; // EOF

        /* Check for a locally-executed command. */

        if (comd[0] == '/') {
            local(comd + 1);
            continue;
        }

        /* return upon 'exit' */

        if (cieq("exit", comd))
            break;

        /* If command 'bg_run' is given, ngSpice_Command_handle() will return immediately.
           To guarantee that the primary thread here waits until the run is finished, we
           may set no_bg to 0 already here. Risk: if starting the simulation fails, we never
           may leave the waiting loop. As an alternative callback function ng_thread_runs()
           will set no_bg to 0. This has to happen within the first 200ms waiting time. */
        if (cieq("bg_run", comd))
            no_bg = false;

        ret = ngSpice_Command_handle(comd);
        if (ret)
            fprintf(stderr, "Ngspice command execution error %d\n", ret);

        /* wait until simulation finishes */

        for (;;) {
#if defined(__MINGW32__) || defined(__CYGWIN__) || defined(_MSC_VER)
            Sleep(200);
#else
            usleep(200000);
#endif
            /* after 200ms the callback function ng_thread_runs() should have
               set no_bg to 0, otherwise we would not wait for the end of the
               background thread.*/
            if (no_bg)
                break;
        }
    }
    ret = ngSpice_Reset_handle();   
    return 0;
}


/* Callback function called from bg thread in ngspice to transfer
   any string created by printf or puts. Output to stdout in ngspice is
   preceded by token stdout, same with stderr.*/
static int
ng_getchar(char* outputreturn, int ident, void* userdata)
{
    printf("%s\n", outputreturn);
    return 0;
}

/* Callback function called from bg thread in ngspice to transfer
   simulation status (type and progress in percent). */
static int
ng_getstat(char* outputreturn, int ident, void* userdata)
{
    printf("Getstat callback: %s\n", outputreturn);
    return 0;
}

/* Callback function called from ngspice upon starting (returns true) or
  leaving (returns false) the bg thread. */
static int
ng_thread_runs(bool noruns, int ident, void* userdata)
{
    no_bg = noruns;
    if (noruns)
        printf("\nbg not running\n");
    else
        printf("bg running\n\n");

    return 0;
}

/* Callback function called from bg thread in ngspice if fcn controlled_exit()
   is hit. Do not exit, but unload ngspice. */
static int
ng_exit(int exitstatus, bool immediate, bool quitexit, int ident, void* userdata)
{
    printf("Exit callback status %d immediate %d quit %d\n",
            exitstatus, immediate, quitexit);
    return exitstatus;
}

/* Callback function called from bg thread in ngspice once per
 * accepted data point: Sendata callbick.
 */

static int
ng_data(pvecvaluesall vdata, int numvecs, int ident, void* userdata)
{
    if (sd_limit > sd_count) {
        ++sd_count;
        if (numvecs > 0) {
            printf("New data %d (%d) vectors, first is %s\n",
                   numvecs, vdata->veccount, vdata->vecsa[0]->name);
            if (sd_list) {
                int i;

                for (i = 0; i <  vdata->veccount; ++i) {
                    if (vdata->vecsa[i]->is_complex) {
                        printf("%s: (%g, %g)\n", vdata->vecsa[i]->name,
                               vdata->vecsa[i]->creal, vdata->vecsa[i]->cimag);
                    } else {
                        printf("%s: %g\n",
                               vdata->vecsa[i]->name, vdata->vecsa[i]->creal);
                    }
                }
            }
        } else {
            printf("New data callback, no data!\n");
        }
    }
    return 0;
}

/* Callback function called from bg thread in ngspice once upon intialization
 * of the simulation vectors.
 */

static int
ng_initdata(pvecinfoall intdata, int ident, void* userdata)
{
    int i;
    int vn = intdata->veccount;

    printf("Init data callback:\n");
    for (i = 0; i < vn; i++)
        printf("  Vector: %s\n", intdata->vecs[i]->vecname);
    return 0;
}

static int ng_initevtdata(int idx, int max_idx, char *node, char *type,
                          int ident, void* userdata)
{
    printf("Evt init: node %s type %s\n", node, type);
    return 0;
}

static int ng_evtdata(int idx, double time, double pval, char *sval,
                          void *sp, int sz, int sim_node,
                          int ident, void* userdata)
{
    if (se_limit > se_count) {
        ++se_count;
        printf("Evt val: node %d val %s/%g at time %g\n",
               idx, sval, pval, time);
    }
    return 0;
}

static int ng_rawevt(double time, void *valp, void *userData, int last)
{
    struct raw_cb_ctx *ctx = (struct raw_cb_ctx *)userData;
    const char        *printstr, *typestr;
    double             plotval;

    if (se_limit > se_count) {
        ++se_count;
        ngSpice_Decode_Evt_handle(valp, ctx->type, &plotval, &printstr);
        ngSpice_Decode_Evt_handle(NULL, ctx->type, NULL, &typestr);
        printf("Raw event: node %s type %s val %s/%g at time %g%s\n",
               ctx->name, typestr, printstr, plotval, time,
               last ? " is last." : "");
        return 0;
    } else {
        free(ctx);
        return 1;
    }
}

/* EXTERNAL source callback. */

static int ng_srcdata(double *vp, double time, char *source, int id, void *udp)
{
    if (sq_limit > sq_count) {
        ++sq_count;
        printf("V or ISRC request: source %s at time %g\n", source, time);
        if (sq_ask) {
            getLine("Value: ", comd, sizeof(comd));
            if (!strncmp("/s", comd, 2)) {
                /* Allow "/sask" as respone. */

                sq_ask = 0;
            } else {
                sr_last_val = *vp = strtod(comd, NULL);
                sr_last_time = time;
            }
            return 0;
        }
    }

    if (sr_last_time >= 0.0) {
        /* Provide a value. */

        if (sr_target_time >= 0.0) {
            if (time < sr_target_time) {
                sr_last_val += (sr_target_val - sr_last_val) *
                    (time - sr_last_time) / (sr_target_time - sr_last_time);
            } else {
                sr_last_val = sr_target_val;
            }
        }
        *vp = sr_last_val;
        sr_last_time = time;
    }
    return 0;
}

static int ng_syncdata(double time, double *deltap, double old_delta,
                       int redo, int loc, int id, void *udp)
{
    if (sd_limit > sd_count) {
        ++sd_count;
        printf("Sync data redo %d delta %g (old %g) location %d at %g\n",
               redo, *deltap, old_delta, loc, time);
    }
    sim_time = time;
    return 0;
}

/* Unify LINUX and Windows dynamic library handling:
   Add functions dlopen, dlsym, dlerror, dlclose to Windows by
   tranlating to Windows API functions.
*/
#if defined(__MINGW32__) || defined(__CYGWIN__) || defined(_MSC_VER)

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
static int
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
static int
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

/* read a line from console input
   source:
   https://stackoverflow.com/questions/4023895/how-to-read-string-entered-by-user-in-c
   */
#define OK       0
#define NO_INPUT 1

static int
getLine(char *prmpt, char *buff, size_t sz)
{
    int ch, len, extra;

    // Get line with buffer overrun protection.

    for (;;) {
        if (prmpt != NULL) {
            printf("%s", prmpt);
            fflush(stdout);
        }
        if (fgets(buff, sz, stdin) == NULL)
            return NO_INPUT;

        // If it was too long, there'll be no newline. In that case, we flush
        // to end of line so that excess doesn't affect the next call.

        len = strlen(buff);
        if (buff[len - 1] != '\n') {
            extra = 0;
            while (((ch = getchar()) != '\n') && (ch != EOF))
                extra = 1;
            if (extra) {
                fprintf(stderr,
                        "Line longer than %zd characters was ignored.\n",
                        sz - 1);
                continue;
            }
        }

        // Otherwise remove newline and give string back to caller.

        buff[len - 1] = '\0';
        return OK;
    }
}
