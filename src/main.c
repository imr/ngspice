/* Copyright 1990
   Regents of the University of California.
   All rights reserved.

   Author: 1985 Wayne A. Christopher

   The main routine for ngspice
*/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/dstring.h"

#include <setjmp.h>
#include <signal.h>

/* Added GNU Readline Support 11/3/97 -- Andrew Veliath <veliaa@rpi.edu> */
/* from spice3f4 patch to ng-spice. jmr */
#ifdef HAVE_GNUREADLINE
# include <readline/readline.h>
# include <readline/history.h>
# include "../misc/tilde.h"
#endif

/* editline development has added the following typdef to readline.h in 06/2018.
   It is not vailable with older libedit versions (pre-1.42.2) , thus we have to set it ourselves */
#ifdef HAVE_BSDEDITLINE
#include <editline/readline.h>
#ifndef rl_hook_func_t
    typedef int rl_hook_func_t(void);
#endif
extern rl_hook_func_t *rl_event_hook;    /* missing from editline/readline.h */
extern int rl_catch_signals;        /* missing from editline/readline.h */
#endif

#include "ngspice/iferrmsg.h"
#include "ngspice/ftedefs.h"
#include "ngspice/devdefs.h"
#include "spicelib/devices/dev.h"
#include "spicelib/analysis/analysis.h"
#include "misc/ivars.h"
#include "misc/misc_time.h"
#include "misc/util.h"

#if defined(HAS_WINGUI) || defined(_MSC_VER) || defined(__MINGW32__)
# include "misc/mktemp.h"
#endif

#if defined(HAVE_GETOPT_LONG) && defined(HAVE_GETOPT_H)
# include <getopt.h>
#else
# include "misc/getopt_bsd.h"
#endif

#include "frontend/spiceif.h"
#include "frontend/resource.h"
#include "frontend/variable.h"
#include "frontend/display.h"  /* added by SDB to pick up Input() fcn */
#include "frontend/signal_handler.h"
#include "frontend/misccoms.h"
#include "ngspice/compatmode.h"
#include "ngspice/randnumb.h"

/* saj xspice headers */
#ifdef XSPICE
# include "ngspice/ipctiein.h"
# include "ngspice/mif.h"
# include "ngspice/enh.h"
# include "ngspice/mifproto.h"
# include "ngspice/evtproto.h"
#endif

#ifdef CIDER
# include "ngspice/numenum.h"
# include "maths/misc/accuracy.h"
#endif

#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
char history_file[512] = {'\0'};
static char *application_name;
#endif

/* Undefine this next line for debug tracing */
/* #define TRACE */

/* Main options */
static bool ft_servermode = FALSE;
bool ft_batchmode = FALSE;
bool ft_pipemode = FALSE;
bool rflag = FALSE; /* has rawfile */

/* Frontend options */
bool ft_intrpt = FALSE;     /* Set by the (void) signal handlers. TRUE = we've been interrupted. */
bool ft_setflag = FALSE;    /* TRUE = Don't abort simulation after an interrupt. */
char *ft_rawfile = "rawspice.raw";

#ifdef XSPICE
bool wantevtdata = FALSE;
#endif

#ifdef HAS_WINGUI
extern void winmessage(char *new_msg); /* display a message box (defined in winmain.c)*/
extern void SetSource(char *Name);    /* display the source file name in the source window */
extern int  xmain(int argc, char **argv);
FILE *flogp = NULL;         /* log file ('-o logfile' command line option) */
#else
bool orflag = FALSE; /* global for -o option */
#endif

extern int IsCiderLoaded(void);

FILE *slogp = NULL;          /* soa log file ('--soa-log file' command line option) */

/* Frontend and circuit options */
IFsimulator *ft_sim = NULL;

char *errRtn;     /* name of the routine declaring error */
char *errMsg = NULL;     /* descriptive message about what went wrong */
char *cp_program; /* program name 'ngspice' */

char *Infile_Path = NULL; /* Path to netlist input file */


/* Globals definitions for Machine Accuracy Limits
 * (needed by CIDER)
 */
double BMin;                /* lower limit for B(x) */
double BMax;                /* upper limit for B(x) */
double ExpLim;              /* limit for exponential */
double Accuracy;            /* accuracy of the machine */
double MuLim, MutLim;

#ifdef CIDER
/* Global debug flags from CIDER, soon they will become
 * spice variables :)
 */
BOOLEAN ONEacDebug   = FALSE;
BOOLEAN ONEdcDebug   = TRUE;
BOOLEAN ONEtranDebug = TRUE;
BOOLEAN ONEjacDebug  = FALSE;

BOOLEAN TWOacDebug   = FALSE;
BOOLEAN TWOdcDebug   = TRUE;
BOOLEAN TWOtranDebug = TRUE;
BOOLEAN TWOjacDebug  = FALSE;

/* CIDER Global Variable Declarations */

int BandGapNarrowing;
int TempDepMobility, ConcDepMobility, FieldDepMobility, TransDepMobility;
int SurfaceMobility, MatchingMobility, MobDeriv;
int CCScattering;
int Srh, Auger, ConcDepLifetime, AvalancheGen;
int FreezeOut = FALSE;
int OneCarrier;

int MaxIterations = 100;
int AcAnalysisMethod = DIRECT;

double Temp, RelTemp, Vt;
double RefPsi;/* potential at Infinity */
double EpsNorm, VNorm, NNorm, LNorm, TNorm, JNorm, GNorm, ENorm;

/* end cider globals */
#endif /* CIDER */

struct variable *(*if_getparam)(CKTcircuit *ckt, char **name, char *param, int ind, int do_model);

/* static functions */
int SIMinit(IFfrontEnd *frontEnd, IFsimulator **simulator);

static ATTRIBUTE_NORETURN void sp_shutdown(int exitval);
static void app_rl_readlines(void);

#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
static char *prompt(void);
#ifndef X_DISPLAY_MISSING
# include "frontend/plotting/x11.h"
static int app_event_func(void);
#endif
#endif

static void show_help(void);
static bool read_initialisation_file(const char *dir, const char *name);

#ifdef SIMULATOR
static void append_to_stream(FILE *dest, FILE *source);
#endif

extern IFsimulator SIMinfo;

#ifdef SIMULATOR

const bool ft_nutmeg = FALSE;
extern struct comm spcp_coms[];
struct comm *cp_coms = spcp_coms;

extern int OUTpBeginPlot(CKTcircuit *, JOB *, IFuid, IFuid, int, int, IFuid *, int, runDesc **);
extern int OUTpData(runDesc *, IFvalue *, IFvalue *);
extern int OUTwBeginPlot(CKTcircuit *, JOB *, IFuid, IFuid, int, int, IFuid *, int, runDesc **);
extern int OUTwReference(runDesc *, IFvalue *, void **);
extern int OUTwData(runDesc *, int, IFvalue *, void *), OUTwEnd(runDesc *), OUTendPlot(runDesc *);
extern int OUTbeginDomain(runDesc *, IFuid, int, IFvalue *);
extern int OUTendDomain(runDesc *), OUTstopnow(void);
extern void OUTerror(int, char *, IFuid *);

#ifdef __GNUC__
extern void OUTerrorf(int, const char *fmt, ...)  __attribute__ ((format (__printf__, 2, 3)));
#else
extern void OUTerrorf(int, const char *fmt, ...);
#endif

extern int OUTattributes(runDesc *, IFuid, int, IFvalue *);

IFfrontEnd nutmeginfo = {
    IFnewUid,
    IFdelUid,
    OUTstopnow,
    seconds,
    OUTerror,
    OUTerrorf,
    OUTpBeginPlot,
    OUTpData,
    OUTwBeginPlot,
    OUTwReference,
    OUTwData,
    OUTwEnd,
    OUTendPlot,
    OUTbeginDomain,
    OUTendDomain,
    OUTattributes
};

#else /* SIMULATOR */

const bool ft_nutmeg = TRUE;
extern struct comm nutcp_coms[];
struct comm *cp_coms = nutcp_coms;
IFfrontEnd nutmeginfo;

/* -------------------------------------------------------------------------- */
int
if_run(CKTcircuit *t, char *w, wordlist *s, INPtables *b)
{
    NG_IGNORE(b);
    NG_IGNORE(s);
    NG_IGNORE(w);
    NG_IGNORE(t);

    return 0;
}

/* -------------------------------------------------------------------------- */
int
if_sens_run(CKTcircuit *t, wordlist *args, INPtables *tab)
{
    NG_IGNORE(tab);
    NG_IGNORE(args);
    NG_IGNORE(t);

    return 0;
}

/* -------------------------------------------------------------------------- */
void
if_dump(CKTcircuit *ckt, FILE *fp)
{
    NG_IGNORE(fp);
    NG_IGNORE(ckt);
}

/* -------------------------------------------------------------------------- */
CKTcircuit *
if_inpdeck(struct card *deck, INPtables **tab)
{
    NG_IGNORE(tab);
    NG_IGNORE(deck);

    return NULL;
}

/* -------------------------------------------------------------------------- */
int
if_option(CKTcircuit *ckt, char *name, enum cp_types type, void *value)
{
    NG_IGNORE(value);
    NG_IGNORE(type);
    NG_IGNORE(name);
    NG_IGNORE(ckt);

    return 0;
}

/* -------------------------------------------------------------------------- */
void
if_cktfree(CKTcircuit *ckt, INPtables *tab)
{
    NG_IGNORE(tab);
    NG_IGNORE(ckt);
}

/* -------------------------------------------------------------------------- */
void
if_setndnames(char *line)
{
    NG_IGNORE(line);
}

/* -------------------------------------------------------------------------- */
char *
if_errstring(int code)
{
    NG_IGNORE(code);

    return copy("spice error");
}

/* -------------------------------------------------------------------------- */
void
if_setparam_model(CKTcircuit *ckt, char **name, char *val)
{
    NG_IGNORE(val);
    NG_IGNORE(name);
    NG_IGNORE(ckt);
}

void
if_setparam(CKTcircuit *ckt, char **name, char *param, struct dvec *val, int do_model)
{
    NG_IGNORE(do_model);
    NG_IGNORE(val);
    NG_IGNORE(param);
    NG_IGNORE(name);
    NG_IGNORE(ckt);
}

/* -------------------------------------------------------------------------- */
bool
if_tranparams(struct circ *ci, double *start, double *stop, double *step)
{
    NG_IGNORE(step);
    NG_IGNORE(stop);
    NG_IGNORE(start);
    NG_IGNORE(ci);

    return FALSE;
}

/* -------------------------------------------------------------------------- */
struct variable *
if_getstat(CKTcircuit *ckt, char *name)
{
    NG_IGNORE(name);
    NG_IGNORE(ckt);

    return NULL;
}

void
com_snload(wordlist *wl)
{
    NG_IGNORE(wl);
}

void
com_snsave(wordlist *wl)
{
    NG_IGNORE(wl);
}

void
com_optran(wordlist *wl)
{
    NG_IGNORE(wl);
}

void
SMPprint(SMPmatrix *n1, char *n2)
{
    NG_IGNORE(n1);
    NG_IGNORE(n2);
}

void
SMPprintRHS(SMPmatrix *n1, char *n2, double *n3, double *n4)
{
    NG_IGNORE(n1);
    NG_IGNORE(n2);
    NG_IGNORE(n3);
    NG_IGNORE(n4);
}

void
spice_destroy_devices(void)
{
}

#endif /* SIMULATOR */

#ifndef SIMULATOR

#ifdef XSPICE
/* saj to get nutmeg to compile, not nice but necessary */
Ipc_Tiein_t  g_ipc;

Ipc_Status_t
ipc_send_errchk(void)
{
    Ipc_Status_t x = IPC_STATUS_OK;
    return x;
}

Ipc_Status_t
ipc_get_line(char *str, int *len, Ipc_Wait_t wait)
{
    Ipc_Status_t x = IPC_STATUS_OK;

    NG_IGNORE(str);
    NG_IGNORE(len);
    NG_IGNORE(wait);

    return x;
}

struct card *
ENHtranslate_poly(struct card *deck)
{
    NG_IGNORE(deck);
    return NULL;
}

int
EVTswitch_plot(CKTcircuit* ckt, const char* plottypename)
{
    NG_IGNORE(ckt);
    NG_IGNORE(plottypename);
    return 1;
};

void
EVTsave(wordlist* wl)
{
    NG_IGNORE(wl);
}

int
load_opus(const char *name)
{
    NG_IGNORE(name);
    return 1;
}

char *
MIFgettok(char **s)
{
    NG_IGNORE(s);
    return NULL;
}

void
EVTprint(wordlist *wl)
{
    NG_IGNORE(wl);
}

void
EVTprintvcd(wordlist *wl)
{
    NG_IGNORE(wl);
}

void
EVTdisplay(wordlist *wl)
{
    NG_IGNORE(wl);
}

struct dvec *
EVTfindvec(char *node) {
    NG_IGNORE(node);
    return NULL;
}

void NIresetwarnmsg(void)
{}

#endif /* XSPICE */

#endif /* SIMULATOR */

char *hlp_filelist[] = { "ngspice", NULL };


/* Allocate space for global constants declared in const.h
 * and set their values */
double CONSTroot2 = CONSTsqrt2;
double CONSTvt0 = CONSTboltz * REFTEMP / CHARGE;
double CONSTKoverQ = CONSTboltz / CHARGE;
double CONSTe = CONSTnap;

IFfrontEnd *SPfrontEnd = NULL;
int DEVmaxnum = 0;


/* -------------------------------------------------------------------------- */
int
SIMinit(IFfrontEnd *frontEnd, IFsimulator **simulator)
{
#ifdef SIMULATOR
    spice_init_devices();
    SIMinfo.numDevices = DEVmaxnum = num_devices();
    SIMinfo.devices = devices_ptr();
    SIMinfo.numAnalyses = spice_num_analysis();

    /* va: we recast, because we use only the public part */
    SIMinfo.analyses = (IFanalysis **)spice_analysis_ptr();


#ifdef CIDER
/* Evaluates limits of machine accuracy for CIDER */
    evalAccLimits();
#endif /* CIDER */

#endif /* SIMULATOR */

    SPfrontEnd = frontEnd;
    *simulator = &SIMinfo;

    return OK;
} /* end of function SIMinit */


/* -------------------------------------------------------------------------- */
/* Shutdown gracefully. */
static ATTRIBUTE_NORETURN void
sp_shutdown(int exitval)
{
#ifdef CIDER
    {

        char *cider_quit_s = NULL;
        bool cider_com_quit = TRUE;
        if (exitval != EXIT_BAD) {
            cider_quit_s = getenv("CIDER_COM_QUIT");
            if (cider_quit_s && *cider_quit_s && eq(cider_quit_s, "OFF")) {
                cider_com_quit = FALSE;
            }
            if (cider_com_quit && IsCiderLoaded() > 0) {
                com_quit(NULL);
            }
        }
    }
#endif
    destroy_ivars();
#ifdef HAS_WINGUI
    if (exitval == EXIT_BAD)
        winmessage("Fatal error in SPICE");
    else if (exitval == EXIT_INFO)
        winmessage("Information during setup, see main window!");
#endif
    if  (exitval == EXIT_INFO)
        exitval = EXIT_NORMAL;
    exit(exitval);
}

/* -------------------------------------------------------------------------- */

#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
/* Adapted ../lib/cp/lexical.c:prompt() for GNU Readline -- Andrew Veliath <veliaa@rpi.edu> */
static char *
prompt(void)
{
    static char pbuf[128];
    char *p = pbuf, *s;

    if (cp_interactive == FALSE)
        return NULL;    /* NULL means no prompt */

    s = get_alt_prompt();

    if (!s)
        s = cp_promptstring;

    if (!s)
        s = "->";

    while (*s) {
        char c = (*s++);

        /* FALLTHROUGH added to suppress GCC warning due to
         * -Wimplicit-fallthrough flag */
        switch (c) {
        case '!':
            p += sprintf(p, "%d", where_history() + 1);
            break;
        case '\\': /* skip an escape char */
            if (*s)
                c = (char) (*s++);
            /* FALLTHROUGH */
        default:
            *p++ = c;
            break;
        }
    }

    *p = '\0';

    return pbuf;
}
#endif /* HAVE_GNUREADLINE || HAVE_BSDEDITLINE */

#ifndef X_DISPLAY_MISSING
#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
/* -------------------------------------------------------------------------- */
/* Process device events in Readline's hook since there is no where
   else to do it now - AV */
static int
app_event_func(void)
/* called by GNU readline periodically to know what to do about keypresses */
{
    static REQUEST reqst = { char_option, 0 };
    reqst.fp = rl_instream;
    X11_Input(&reqst, NULL);
    return 0;
}
#endif

#endif

/* Initialisation for readline. */

static void
app_rl_init(void)
{
#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
    char *home;

    /* ---  set up readline params --- */

    if (get_local_home(0, &home) < 0)
        return;
    snprintf(history_file, sizeof history_file, "%s/.%s_history",
             home, application_name);
    tfree(home);

    using_history();
    read_history(history_file);

    rl_readline_name = application_name;
    rl_instream = cp_in;
    rl_outstream = cp_out;
    rl_catch_signals = 0;   /* disable signal handling  */

    /* sjb - what to do for editline?
       This variable is not supported by editline. */
#if defined(HAVE_GNUREADLINE)
    rl_catch_sigwinch = 1;  /* allow readline to respond to resized windows  */
#endif
#endif
}

/* -------------------------------------------------------------------------- */
/* This is the command processing loop for spice and nutmeg.
   The function is called even when GNU readline is unavailable, in which
   case it falls back to repeatable calling cp_evloop()
   SJB 26th April 2005 */
static void
app_rl_readlines(void)
{
#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
    /* GNU Readline Support -- Andrew Veliath <veliaa@rpi.edu> */
    char *line, *expanded_line;

#ifndef X_DISPLAY_MISSING
    if (dispdev->Input == X11_Input)
        rl_event_hook = app_event_func;
#endif

    /* note that we want some mechanism to detect ctrl-D and expand it to exit */
    for (;;) {
        history_set_pos(history_length);

        if (SETJMP(jbuf, 1)) { /* Set location to jump to after handling SIGINT (ctrl-C)  */
            ft_sigintr_cleanup();
        }

        line = readline(prompt());

        if (!line) {
            cp_evloop("quit");
            continue;
        }

        if (*line) {
            int s = history_expand(line, &expanded_line);

            if (s == 2) {
                fprintf(stderr, "-> %s\n", expanded_line);
            }
            else if (s == -1) {
                fprintf(stderr, "readline: %s\n", expanded_line);
            }
            else {
                cp_evloop(expanded_line);
                add_history(expanded_line);
            }
            tfree(expanded_line);
        }

        tfree(line);
    }
    /* History gets written in ../fte/misccoms.c com_quit */

#else
    while (cp_evloop(NULL) == 1) {
        ;
    }
#endif /* defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE) */
} /* end of function app_rl_readlines */


/* -------------------------------------------------------------------------- */
static void
show_help(void)
{
    printf("Usage: %s [OPTION]... [FILE]...\n"
           "Simulate the electical circuits in FILE.\n"
           "\n"
           "  -a  --autorun             run the loaded netlist\n"
           "  -b, --batch               process FILE in batch mode\n"
           "  -c, --circuitfile=FILE    set the circuitfile\n"
           "  -D, --define=variable[=value] define variable to true/[value]\n"
           "  -i, --interactive         run in interactive mode\n"
           "  -n, --no-spiceinit        don't load the local or user's config file\n"
           "  -o, --output=FILE         set the outputfile\n"
           "  -p, --pipe                run in I/O pipe mode\n"
           "  -q, --completion          activate command completion\n"
           "  -r, --rawfile=FILE        set the rawfile output\n"
           "      --soa-log=FILE        set the outputfile for SOA warnings\n"
           "  -s, --server              run spice as a server process\n"
           "  -t, --term=TERM           set the terminal type\n"
           "  -h, --help                display this help and exit\n"
           "  -v, --version             output version information and exit\n"
           "\n"
           "Report bugs to %s.\n", cp_program, Bug_Addr);
}

#ifdef SIMULATOR
/* -------------------------------------------------------------------------- */
static void
append_to_stream(FILE *dest, FILE *source)
{
    char buf[BSIZE_SP];
    size_t n;

    while ((n = fread(buf, 1, BSIZE_SP, source)) > 0)
        fwrite(buf, n, 1, dest);
}
#endif /* SIMULATOR */

/* -------------------------------------------------------------------------- */
/* Read an initialisation file.
   dir    is the directory (use NULL or "" for current directory)
   name   is the initialisation file's name
   Return true on success
   SJB 25th April 2005 */
static bool read_initialisation_file(const char *dir, const char *name)
{
    const char *path;
    bool result = FALSE;

    /* check name */
    if (!name || *name == '\0') {
        return FALSE;   /* Fail; name needed */
    }

    /* contruct the full path */
    if (!dir || *dir == '\0') {
        path = name;
    }
    else {
        path = tprintf("%s" DIR_PATHSEP "%s", dir, name);
        if (!path) {
            return FALSE;    /* memory allocation error */
        }
    }

    /* now access the file */
#ifdef HAVE_UNISTD_H
    if (access(path, R_OK) == 0) {
        result = TRUE;
    }
#else
    {
        FILE *fp = fopen(path, "r");
        if (fp) {
            fclose(fp);
            result = TRUE;
        }
    }
#endif

    if (result) {
        inp_source(path);
#ifdef TRACE
        printf("Init file: '%s'\n", path);
#endif
    }

    if (path != name) { /* Allocated by tprintf() */
        txfree(path);
    }

    return result;
} /* end of function read_initialisation_file */

/* -------------------------------------------------------------------------- */
static void print_news(void)
{
    if (News_File && *News_File) {
        const char * const fname = cp_tildexpand(News_File); /*DG  Memory leak */
        FILE * const fp = fopen(fname, "r");
        txfree(fname);
        if (fp) {
            char buf[BSIZE_SP];
            while (fgets(buf, BSIZE_SP, fp)) {
                fputs(buf, stdout);
            }
            fclose(fp);
        }
    }
} /* end of function print_news */

#ifdef HAS_WINGUI
#define main xmain
#endif

int main(int argc, char **argv)
{
    char log_file[BSIZE_SP];
    char soa_log_file[BSIZE_SP];

    /* volatile added to resolve GCC -Wclobbered */
    volatile bool readinit = TRUE; /* read initialization file */
    volatile bool istty = TRUE;

    bool iflag = FALSE; /* flag for interactive mode */
    bool qflag = FALSE; /* flag for command completion */

    bool t = TRUE;

    FILE * volatile circuit_file;
    bool volatile oflag = FALSE;
    bool srflag = FALSE;

#ifdef TRACE
    /* this is used to detect memory leaks during debugging */
    /* added by SDB during debug . . . . */
    /* mtrace(); */
#endif

#ifdef TRACE
    /* this is also used for memory leak plugging . . . */
    /* added by SDB during debug . . . . */
    /* mwDoFlush(1); */
#endif
#ifdef TRACE1
    int jj;
    for (jj = 0; jj < argc; jj++)
    {
        fprintf(stdout, "%s\n", argv[jj]);
    }
#endif

    /* MFB tends to jump to 0 on errors.  This tends to catch it. */
    {
        static int started = 0;
        if (started++) {
            fprintf(cp_err, "main: Internal Error: jump to zero\n");
            sp_shutdown(EXIT_BAD);
        }
    }

#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
    application_name = strrchr(argv[0], '/');
    if (application_name) {
        ++application_name;
    }
    else {
        application_name = argv[0];
    }
#endif


    ivars(argv[0]); /* Create internal variables */

    /* Set default data sources */
    cp_in  = stdin;
    cp_out = stdout;
    cp_err = stderr;
    circuit_file = stdin;

#if defined(HAVE_ISATTY) && !defined(HAS_WINGUI)
    istty = (bool) isatty(fileno(stdin));
#endif

    init_time();

    {
        int rv = SIMinit(&nutmeginfo, &ft_sim);
        if (rv != OK) {
            ft_sperror(rv, "SIMinit");
            sp_shutdown(EXIT_BAD);
        }
    }

    cp_program = ft_sim->simulator;

    /* initialze random number generator with seed = 1 */
    int ii = 1;
    cp_vset("rndseed", CP_NUM, &ii);
    com_sseed(NULL);

    /* set a boolean variable when XSPICE and/or OSDI is enabled,
       to be used in spinit etc. */
#if defined(SIMULATOR) && defined(XSPICE)
    cp_vset("xspice_enabled", CP_BOOL, &t);
#endif
#if defined(SIMULATOR) && defined(OSDI)
    cp_vset("osdi_enabled", CP_BOOL, &t);
#endif

    /* --- Process command line options --- */
    for (;;) {
        enum { soa_log = 1001, };

        static struct option long_options[] = {
            {"define",       required_argument, NULL, 'D'},
            {"help",         no_argument,       NULL, 'h'},
            {"version",      no_argument,       NULL, 'v'},
            {"batch",        no_argument,       NULL, 'b'},
            {"autorun",      no_argument,       NULL, 'a'},
            {"circuitfile",  required_argument, NULL, 'c'},
            {"interactive",  no_argument,       NULL, 'i'},
            {"no-spiceinit", no_argument,       NULL, 'n'},
            {"output",       required_argument, NULL, 'o'},
            {"pipe",         no_argument,       NULL, 'p'},
            {"completion",   no_argument,       NULL, 'q'},
            {"rawfile",      required_argument, NULL, 'r'},
            {"server",       no_argument,       NULL, 's'},
            {"terminal",     required_argument, NULL, 't'},
            {"soa-log",      required_argument, NULL, soa_log},
            {NULL,           0,                 NULL, 0}
        };

        int option_index = 0;

        int c = getopt_long(argc, argv, "D:hvbac:ino:pqr:st:",
                            long_options, &option_index);

        if (c == -1) {
            break;
        }

        switch (c) {
        case 'D':       /* Definition of variable */
            if (optarg) {
                const char *eq = strchr(optarg, '=');
                if (eq == (char *) NULL) { /* no assignment */
                    bool true_val = TRUE;
                    cp_vset(optarg, CP_BOOL, &true_val);
                }
                else {
                    DS_CREATE(ds, 100);
                    if (ds_cat_mem(&ds, optarg, (size_t) (eq - optarg)) == 0) {
                        cp_vset(ds_get_buf(&ds), CP_STRING, eq + 1);
                    }
                    ds_free(&ds);
                }
            }
            break;

        case 'h':       /* Help */
            show_help();
            sp_shutdown(EXIT_INFO);
            break;

        case 'v':       /* Version info */
            com_version(NULL);
            sp_shutdown(EXIT_INFO);
            break;

        case 'b':       /* Batch mode */
        {
            bool x_false = FALSE;
            bool x_true = TRUE;
            cp_vset("addcontrol", CP_BOOL, &x_false);
            cp_vset("batchmode", CP_BOOL, &x_true);
            ft_batchmode = TRUE;
        }
        break;

        case 'a':       /* Add control section for autorun */
            if (!ft_batchmode) {
                bool x_true = TRUE;
                cp_vset("addcontrol", CP_BOOL, &x_true);
            }
            break;

        case 'c':       /* Circuit file */
            if (optarg) {
                if ((circuit_file = fopen(optarg, "r")) == NULL) {
                    perror(optarg);
                    sp_shutdown(EXIT_BAD);
                }
                istty = FALSE;
            }
            break;

        case 'i':       /* Interactive mode */
            iflag = TRUE;
            break;

        case 'n':       /* Don't read initialisation file */
            readinit = FALSE;
            break;

        case 'o':       /* Output file */
            if (optarg) {
                /* switch to line buffering for stdout */
                setvbuf(stdout, NULL, _IOLBF, BUFSIZ);
                sprintf(log_file, "%s", optarg);
                oflag = TRUE;
            }
            break;

        case 'p':       /* Run in pipe mode */
            iflag = TRUE;
            istty = TRUE;
            ft_pipemode = TRUE;
            /* switch to line buffering for stdout */
            setvbuf(stdout, NULL, _IOLBF, BUFSIZ);
            break;

        case 'q':       /* Command completion */
            qflag = TRUE;
            break;

        case 'r':       /* The raw file */
            if (optarg) {
                cp_vset("rawfile", CP_STRING, optarg);
            }
            rflag = TRUE;
            break;

        case 's':       /* Server mode */
            ft_servermode = TRUE;
            break;

        case 't':
            if (optarg) {
                cp_vset("term", CP_STRING, optarg);
            }
            break;

        case soa_log:
            if (optarg) {
                sprintf(soa_log_file, "%s", optarg);
                srflag = TRUE;
            }
            break;

        case '?':
            break;

        default:
            printf ("?? getopt returned character code 0%o ??\n", c);
        }
    }  /* --- End of command line option processing (While(1)-loop) --- */


    if (oflag) {   /* -o option has been set */

        com_version(NULL);

        if (ft_servermode) {
            fprintf(stdout, "\nServer mode\n\n");
        }
        else if (ft_batchmode) {
            fprintf(stdout, "\nBatch mode\n\n");
        }
        else {
            fprintf(stdout, "\nInteractive mode, better used without -o option\n\n");
        }

        if (rflag) {
            fprintf(stdout, "Simulation output goes to rawfile: %s\n", ft_rawfile);
        }

        fprintf(stdout, "Comments and warnings go to log-file: %s\n\n", log_file);

        /* Open the log file */
#ifdef HAS_WINGUI
        /* flogp used by winmain's putc which writes to file 'log_file' */
        flogp = fopen(log_file, "w");
        if (!flogp) {
            perror(log_file);
            sp_shutdown(EXIT_BAD);
        }
#else
        orflag = TRUE;
        /* Connect stdout and stderr to file log_file and log stdout */
        if (!freopen(log_file, "w", stdout)) {
            perror(log_file);
            sp_shutdown(EXIT_BAD);
        }
        dup2(fileno(stdout), fileno(stderr));
        setvbuf(stdout, NULL, _IOLBF, BUFSIZ); /* enable line buffering */
#endif
    } /* orflag */

    if (srflag) {   /* --soa-log option has been set */

        fprintf(stdout, "\nSOA warnings go to log-file: %s\n", soa_log_file);

        /* Open the soa log file */
        slogp = fopen(soa_log_file, "w");
        if (!slogp) {
            perror(soa_log_file);
            sp_shutdown(EXIT_BAD);
        }
    } /* srflag */

#ifdef SIMULATOR
    if_getparam = spif_getparam_special;
#else
    if_getparam = nutif_getparam;
#endif

    if ((!iflag && !istty) || ft_servermode) { /* (batch and file) or
                                                * server operation */
        ft_batchmode = TRUE;
    }

    if ((iflag && !istty) || qflag)  { /* (interactive and file) or
                                        * command completion */
        cp_nocc = TRUE;              /* set command completion */
    }
    else {
        cp_nocc = FALSE;
    }

#ifndef HAS_WINGUI
    if ((iflag || istty) && !ft_batchmode) {
        /* Enable interactive prompting. */
#else
    if (iflag && !ft_batchmode) {
#endif
        bool x_true = TRUE;
        cp_vset("interactive", CP_BOOL, &x_true);
    }


    if (ft_servermode) {             /* in server no init file */
        readinit = FALSE;
    }

    if (!istty || ft_batchmode) {    /* file or batch - no more output */
        out_moremode = FALSE;
    }

    /* Get information on memory status upon startup.
       Would like to do this later, but cpinit evals commands.
       fcn is in resource.c */
    init_rlimits();

    /* Have to initialize cp now.
       fcn is in cpitf.c */
    ft_cpinit();


    /* Set up signal handling */
    if (!ft_batchmode) {
        /*  Set up interrupt handler  */
        signal(SIGINT, (SIGNAL_FUNCTION) ft_sigintr);

        /* floating point exception  */
        signal(SIGFPE, (SIGNAL_FUNCTION) sigfloat);

#ifdef SIGTSTP
        signal(SIGTSTP, (SIGNAL_FUNCTION) sigstop);
#endif
    }

    /* Set up signal handling for fatal errors. */
    signal(SIGILL, (SIGNAL_FUNCTION) sigill);

#ifdef SIGBUS
    signal(SIGBUS, (SIGNAL_FUNCTION) sigbus);
#endif
#if defined(SIGSEGV) && !defined(NGDEBUG) && defined(HAS_WINGUI)
/* Allow a comment and graceful shutdown after seg fault */
    signal(SIGSEGV, (SIGNAL_FUNCTION) sigsegv);
#endif
#ifdef SIGSYS
    signal(SIGSYS, (SIGNAL_FUNCTION) sig_sys);
#endif

#ifdef TRACE1
    fprintf(stdout, "We are ready to read initialization files.\n");
#endif

    /* To catch interrupts during .spiceinit... Readline must be initialised
     * so that it is safe to call ft_sigintr_cleanup();
     */

    app_rl_init();
    if (SETJMP(jbuf, 1)) {
        ft_sigintr_cleanup();
        fprintf(cp_err, "Warning: error executing .spiceinit.\n");
    }
    else {
        if (readinit) {
            /* load user's initialisation file
              try accessing the initialisation file .spiceinit in a user provided
              path read from environmental variable SPICE_USERINIT_DIR,
              if that fails try the alternate name spice.rc, then look into
              the current directory, then the HOME directory, then into USERPROFILE */
            do {
                {
                    const char* const userinit = getenv("SPICE_USERINIT_DIR");
                    if (userinit) {
                        if (read_initialisation_file(userinit, INITSTR) != FALSE) {
                            break;
                        }
                        if (read_initialisation_file(userinit, ALT_INITSTR) != FALSE) {
                            break;
                        }
                    }
                }

                if (read_initialisation_file("", INITSTR) != FALSE) {
                    break;
                }
                if (read_initialisation_file("", ALT_INITSTR) != FALSE) {
                    break;
                }

                {
                    const char * const home = getenv("HOME");
                    if (home) {
                        if (read_initialisation_file(home, INITSTR) != FALSE) {
                            break;
                        }
                        if (read_initialisation_file(home, ALT_INITSTR) != FALSE) {
                            break;
                        }
                    }
                }

                {
                    const char * const usr = getenv("USERPROFILE");
                    if (usr) {
                        if (read_initialisation_file(usr, INITSTR) != FALSE) {
                            break;
                        }
                        if (read_initialisation_file(usr, ALT_INITSTR) != FALSE) {
                            break;
                        }
                    }
                }
            } while (0);
        } /* end of case that init file is read */

        if (!ft_batchmode) {
            com_version(NULL);
            DevInit();
            print_news();
        }
    } /* end of normal execution for setjmp() */


#ifdef SIMULATOR

    /* Pass 2 -- get the filenames. If we are spice, then this means
     * build a circuit for this file. If this is in server mode, don't
     * process any of these args.  */

    if (SETJMP(jbuf, 1)) {
        ft_sigintr_cleanup();
        fprintf(cp_err, "Warning: error executing during ngspice startup.\n");
    }
    else {
        bool gotone = FALSE;

        cp_interactive = FALSE;

#ifdef FastRand
// initialization and seed for FastNorm Gaussian random generator
        {
            unsigned int rseed = 66;
            initnorm(0, 0);
            if (!cp_getvar("rndseed", CP_NUM, &rseed, 0)) {
                time_t acttime = time(NULL);
                rseed = (unsigned int) acttime;
            }
            initnorm(rseed, 2);
            fprintf(cp_out, "SoS %f, seed value: %ld\n", renormalize(), rseed);
        }
#elif defined(WaGauss)
        initw();
#endif

        if (!ft_servermode) {

            int err = 0;

            /* Concatenate all non-option arguments into a temporary file
               and load that file into the spice core.

               The original routine took a special path if there was only
               one non-option argument.  In that case, it didn't create
               the temporary file but used the original file instead.  The
               current algorithm is uniform at the expense of a little
               startup time.  */

            FILE *tempfile = tmpfile();
            char *dname = NULL;   /* input file */

#if defined(HAS_WINGUI) || defined(_MSC_VER) || defined(__MINGW32__)
            char *tpf = NULL;     /* temporary file */

            /* tmpfile() returns NULL, if in MS Windows as non admin user
               in directory C:\something (no write permission to root C:).
               Then we add a tempfile in the user's home directory.
               File will be removed automatically due to TD option in fopen */
            if (tempfile == NULL) {
                tpf = smktemp("sp");
                tempfile = fopen(tpf, "w+bTD");
                if (tempfile == NULL) {
                    fprintf(stderr, "Could not open a temporary file "
                            "to save and use optional arguments.\n");
                    sp_shutdown(EXIT_BAD);
                }
            }
#endif

            if (!tempfile) {
                perror("tmpfile()");
                sp_shutdown(EXIT_BAD);
            }

            if (optind == argc && !istty) {
                append_to_stream(tempfile, stdin);
            }

            while (optind < argc) {
                char *arg = argv[optind++];
                FILE *tp;
                /* Copy the the path of the first filename only */
                if (!Infile_Path) {
                    Infile_Path = ngdirname(arg);
                }

             /* unquote the input string, needed if it results from double clicking the filename */
#if defined(HAS_WINGUI)
                arg = cp_unquote(arg);
#endif
                /* Copy all the arguments into the temporary file */
                tp = fopen(arg, "r");
                if (!tp) {
                    char *lbuffer = getenv("NGSPICE_INPUT_DIR");
                    if (lbuffer && *lbuffer) {
                        char *p = tprintf("%s" DIR_PATHSEP "%s",
                                lbuffer, arg);
                        tp = fopen(p, "r");
                        tfree(p);
                    }
                    if (!tp) {
                        perror(arg);
                        err = 1;
                        break;
                    }
                }

                /* Copy the input file name which otherwise will be lost due to the
                   temporary file */
                dname = copy(arg);
#if defined(HAS_WINGUI)
                /* write source file name into source window */
                SetSource(dname);
                /* free arg that has been unquoted above */
                tfree(arg);
#endif

                append_to_stream(tempfile, tp);
                fclose(tp);
            }

            fseek(tempfile, 0L, SEEK_SET);

            if (tempfile && (!err || !ft_batchmode)) {
                /* Copy the input file name for becoming another file search path */
                if (inp_spsource(tempfile, FALSE, dname, FALSE) != 0) {
                    fprintf(stderr, "    Simulation interrupted due to error!\n\n");
                    if (oflag && !cp_getvar("interactive", CP_BOOL, NULL, 0))
                        exit(EXIT_BAD);
                }
                tfree(dname);
                gotone = TRUE;
            }

            if (ft_batchmode && err) {
                sp_shutdown(EXIT_BAD);
            }

        }   /* ---  if (!ft_servermode) --- */

        if (!gotone && ft_batchmode) {
            if (inp_spsource(circuit_file, FALSE, NULL, FALSE) != 0) {
                    fprintf(stderr, "    Simulation interrupted due to error!\n\n");
            }
        }

    }

    if (ft_batchmode) {

        int error3 = 1;

        /* If we get back here in batch mode then something is wrong,
         * so exit.  */

        if (SETJMP(jbuf, 1)) {
            ft_sigintr_cleanup();
            sp_shutdown(EXIT_BAD);
        }


        if (ft_servermode) {
            if (ft_curckt == NULL) {
                fprintf(cp_err, "Error: no circuit loaded!\n");
                sp_shutdown(EXIT_BAD);
            }
            if (ft_dorun(""))
                sp_shutdown(EXIT_BAD);
            sp_shutdown(EXIT_NORMAL);
        }


        cp_interactive = FALSE;

        /* Check if a simulation has run from a .control section */
        cp_getvar("sim_status", CP_NUM, &error3, 0);

        if (rflag) {
            /* If -r is specified, then dot cards (.width, .plot, .print, .op, .meas, .tf)
               are ignored, except .save, which has been handled by ft_dotsaves()
               from within inp_spsource (), data are put into linked list dbs.
            */
            int error2 = ft_dorun(ft_rawfile);
            /* Execute the .whatever lines found in the deck, after we are done running. */
            if (ft_cktcoms(TRUE) || error2) {
                sp_shutdown(EXIT_BAD);
            }
        }
        else if (ft_savedotargs()) {
            /* all dot card data to be put into dbs */
            int error2 = ft_dorun(NULL);
            /* Execute the .whatever lines found in the deck, after we are done running. */
            if (ft_cktcoms(FALSE) || error2)
                sp_shutdown(EXIT_BAD);
        }
        else if (error3 == 0) {
            fprintf(stdout, "Note: Simulation executed from .control section \n");
            sp_shutdown(EXIT_NORMAL);
        }
        else {
            fprintf(stderr,
                    "Note: No \".plot\", \".print\", or \".fourier\" lines; "
                    "no simulations run\n");
            sp_shutdown(EXIT_BAD);
        }

        sp_shutdown(EXIT_NORMAL);
    }  /* ---  if (ft_batchmode) ---  */


#else  /* ~ SIMULATOR */

    if (SETJMP(jbuf, 1)) {

        ft_sigintr_cleanup();
        fprintf(cp_err, "Warning: error executing during ft_loadfile().\n");

    }
    else {
        cp_interactive = FALSE;

        while (optind < argc) {
            ft_loadfile(argv[optind++]);
        }
    }

#endif /* ~ SIMULATOR */

    for (;;) {
        if (!SETJMP(jbuf, 1)) {
            /*  enter the command processing loop  */
            cp_interactive = TRUE;
#ifdef HAS_WINGUI
            int i;
            if (argv) {
                for (i = 0; i < argc; i++) {
                    txfree(argv[i]);
                }
                tfree(argv);
            }
#endif
            app_rl_readlines();
        }
        else {
            ft_sigintr_cleanup();
        }
    }
} /* end of function main */
