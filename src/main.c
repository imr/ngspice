/* Copyright 1990
   Regents of the University of California.
   All rights reserved.

   Author: 1985 Wayne A. Christopher

   The main routine for ngspice
   $Id$
*/

#include <ngspice.h>

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif /* HAVE_STRING_H */

#ifdef __MINGW32__
#define  srandom(a) srand(a) /* srandom */
#endif /* __MINGW32__ */

#include <setjmp.h>
#include <signal.h>
#include <sys/types.h>

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif
#ifdef HAVE_GNUREADLINE
/* Added GNU Readline Support 11/3/97 -- Andrew Veliath <veliaa@rpi.edu> */
/* from spice3f4 patch to ng-spice. jmr */
#include <readline/readline.h>
#include <readline/history.h>
#endif  /* HAVE_GNUREADLINE */

#ifdef HAVE_BSDEDITLINE
/* SJB added editline support 2005-05-05 */
#include <editline/readline.h>
extern VFunction *rl_event_hook;    /* missing from editline/readline.h */
extern int rl_catch_signals;	    /* missing from editline/readline.h */
#endif /* HAVE_BSDEDITLINE */

#ifndef HAVE_GETRUSAGE
#ifdef HAVE_FTIME
#include <sys/timeb.h>
#endif
#endif

#include "iferrmsg.h"
#include "ftedefs.h"
#include "devdefs.h"
#include "spicelib/devices/dev.h"
#include "spicelib/analysis/analysis.h"
#include "misc/ivars.h"
#include "misc/getopt.h"
#include "frontend/resource.h"
#include "frontend/variable.h"
#include "frontend/display.h"  /*  added by SDB to pick up Input() fcn  */
#include "frontend/signal_handler.h"

/* saj xspice headers */
#ifdef XSPICE
#include "ipctiein.h"
#include "mif.h"
#include "enh.h"
#endif

#ifdef CIDER
#include "numenum.h"
#include "maths/misc/accuracy.h"
#endif 


#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
char history_file[512];
static char *application_name;
#endif  /* HAVE_GNUREADLINE || HAVE_BSDEDITLINE */

/* Undefine this next line for dubug tracing */
/* #define TRACE */

/* Main options */
static bool ft_servermode = FALSE;
static bool ft_batchmode = FALSE;

/* Frontend options */
bool ft_intrpt = FALSE;     /* Set by the (void) signal handlers. TRUE = we've been interrupted. */
bool ft_setflag = FALSE;    /* TRUE = Don't abort simulation after an interrupt. */
char *ft_rawfile = "rawspice.raw";

#ifdef HAS_WINDOWS
bool oflag = FALSE;         /* Output over redefined I/O functions */
FILE *flogp;  /* hvogt 15.12.2001 */
#endif /* HAS_WINDOWS */

/* Frontend and circuit options */
IFsimulator *ft_sim = NULL;

/* (Virtual) Machine architecture parameters */
int ARCHme;
int ARCHsize;

char *errRtn;
char *errMsg;
char *cp_program;

#ifdef CIDER
/* Globals definitions for Machine Accuracy Limits
 * (needed by CIDER)
 */   
double BMin;                /* lower limit for B(x) */
double BMax;                /* upper limit for B(x) */
double ExpLim;              /* limit for exponential */
double Accuracy;            /* accuracy of the machine */
double Acc, MuLim, MutLim;


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

struct variable *(*if_getparam)( );

static int started = FALSE;

/* static functions */
static int SIMinit(IFfrontEnd *frontEnd, IFsimulator **simulator);
static int shutdown(int exitval);
static void app_rl_readlines(void);

#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
static char * prompt(void);
#endif /* HAVE_GNUREADLINE || HAVE_BSDEDITLINE */

#ifdef HAVE_GNUREADLINE
static int rl_event_func(void) ;
#endif /* HAVE_GNUREADLINE || HAVE_BSDEDITLINE */
#ifdef HAVE_BSDEDITLINE
static void rl_event_func(void) ;
#endif /* HAVE_BSDEDITLINE */

static void show_help(void);
static void show_version(void);
static bool read_initialisation_file(char * dir, char * name);
#ifdef SIMULATOR
static void append_to_stream(FILE *dest, FILE *source);
#endif /* SIMULATOR */


#ifndef HAVE_GETRUSAGE
#ifdef HAVE_FTIME
extern struct timeb timebegin;		/* for use w/ ftime */
#endif
#endif

extern IFsimulator SIMinfo;

#ifdef SIMULATOR

bool ft_nutmeg = FALSE;
extern struct comm spcp_coms[ ];
struct comm *cp_coms = spcp_coms;

#else /* SIMULATOR */

bool ft_nutmeg = TRUE;
extern struct comm nutcp_coms[ ];
struct comm *cp_coms = nutcp_coms;
static IFfrontEnd nutmeginfo;

/* -------------------------------------------------------------------------- */
int
if_run(char *t, char *w, wordlist *s, char *b)
{
    return (0);
}

/* -------------------------------------------------------------------------- */
int
if_sens_run(char *t, char *w, wordlist *s, char *b)
{
    return (0);
}

/* -------------------------------------------------------------------------- */
void
if_dump(char *ckt, FILE *fp)
{}

/* -------------------------------------------------------------------------- */
char *
if_inpdeck(struct line *deck, char **tab)
{
    return ((char *) 0);
}

/* -------------------------------------------------------------------------- */
int
if_option(char *ckt, char *name, int type, char *value)
{
    return 0;
}

/* -------------------------------------------------------------------------- */
void if_cktfree(char *ckt, char *tab)
{}

/* -------------------------------------------------------------------------- */
void if_setndnames(char *line)
{}

/* -------------------------------------------------------------------------- */
char *
if_errstring(int code)
{
    return ("spice error");
}

/* -------------------------------------------------------------------------- */
void
if_setparam_model(char *ckt, char *name, struct variable *val)
{}

void
if_setparam(char *ckt, char *name, char *param, struct variable *val)
{}

/* -------------------------------------------------------------------------- */
bool
if_tranparams(struct circ *ckt, double *start, double *stop, double *step)
{
    return (FALSE); 
}

/* -------------------------------------------------------------------------- */
struct variable *
if_getstat(char *n, char *c)
{
    return (NULL);
}

#ifdef EXPERIMENTAL_CODE
void com_loadsnap(wordlist *wl) { return; }
void com_savesnap(wordlist *wl) { return; }
#endif

#endif /* SIMULATOR */

#ifndef SIMULATOR

#ifdef XSPICE
/* saj to get nutmeg to compile, not nice but necessary */
Ipc_Tiein_t  g_ipc;
Ipc_Status_t ipc_send_errchk(void ) {
  Ipc_Status_t x=0;
  return(x);
}
Ipc_Status_t ipc_get_line(char *str , int *len , Ipc_Wait_t wait ){
  Ipc_Status_t x=0;
  return(x);
}
struct line *ENHtranslate_poly(struct line *deck){
  return(NULL);
}
int load_opus(char *name){
  return(1);
}
char  *MIFgettok(char **s){
  return(NULL);
}
void EVTprint(wordlist *wl){
  return;
}
struct dvec *EVTfindvec(char *node){
  return NULL;
}
#endif /* XSPICE */

#endif /* SIMULATOR */

char *hlp_filelist[] = { "ngspice", 0 };


/* allocate space for global constants in 'CONST.h' */

double CONSTroot2;
double CONSTvt0;
double CONSTKoverQ;
double CONSTe;
IFfrontEnd *SPfrontEnd = NULL;
int DEVmaxnum = 0;

/* -------------------------------------------------------------------------- */
static int
SIMinit(IFfrontEnd *frontEnd, IFsimulator **simulator)
{
#ifdef SIMULATOR
    spice_init_devices();
    SIMinfo.numDevices = DEVmaxnum = num_devices();
    SIMinfo.devices = devices_ptr();
    SIMinfo.numAnalyses = spice_num_analysis();
    SIMinfo.analyses = (IFanalysis **)spice_analysis_ptr(); /* va: we recast, because we use 
                                                             * only the public part 
							     */
							
#ifdef CIDER
/* Evaluates limits of machine accuracy for CIDER */
    evalAccLimits();
#endif /* CIDER */  
   						     
#endif /* SIMULATOR */

    SPfrontEnd = frontEnd;
    *simulator = &SIMinfo;
    CONSTroot2 = sqrt(2.);
    CONSTvt0 = CONSTboltz * (27 /* deg c */ + CONSTCtoK ) / CHARGE;
    CONSTKoverQ = CONSTboltz / CHARGE;
    CONSTe = exp((double)1.0);
    return(OK);
}


/* -------------------------------------------------------------------------- */
/* Shutdown gracefully. */
static int
shutdown(int exitval)
{
    cleanvars();
#ifdef PARALLEL_ARCH
    if (exitval == EXIT_BAD) {
	Error("Fatal error in SPICE", -1);
    } else {
	PEND_();
    }
#endif /* PARALLEL_ARCH */
    exit (exitval);
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
        return NULL;	/* NULL means no prompt */
    
    s = get_alt_prompt();
    if(s==NULL)
	s = cp_promptstring;
    if(s==NULL)
	s = "->";
    
    while (*s) {
	switch (strip(*s)) {
	    case '!':
#ifdef HAVE_BSDEDITLINE
		{
		    /* SJB In the present version of editline (v2.9)
		      it seems that where_history() is broken.
		      This is a hack that works round this problem.
		      WARNING: It may fail to work in the future
		      as it relies on undocumented structure */
		    int where = 0;
		    HIST_ENTRY * he = current_history();
		    if(he!=NULL) where = *(int*)(he->data);
		    p += sprintf(p, "%d", where + 1);
		}
#else
		p += sprintf(p, "%d", where_history() + 1);
#endif	/* HAVE_BSDEDITLINE*/
		break;
	    case '\\':
		if (*(s + 1))
		    p += sprintf(p, "%c", strip(*++s));
	    default:
		*p = strip(*s); ++p;
		break;
	}
        s++;
    }
    *p = 0;
    return pbuf;
}
#endif /* HAVE_GNUREADLINE || HAVE_BSDEDITLINE */

#ifdef HAVE_GNUREADLINE
/* -------------------------------------------------------------------------- */
/* Process device events in Readline's hook since there is no where
   else to do it now - AV */
static int
rl_event_func()  
/* called by GNU readline periodically to know what to do about keypresses */
{
    static REQUEST reqst = { checkup_option, 0 };
    Input(&reqst, NULL);
    return 0;
}
#endif /* HAVE_GNUREADLINE */

#ifdef HAVE_BSDEDITLINE
/* -------------------------------------------------------------------------- */
/* Process device events in Editline's hook.
   similar to the readline function above but returns void */
static void
rl_event_func()  
/* called by GNU readline periodically to know what to do about keypresses */
{
    static REQUEST reqst = { checkup_option, 0 };
    Input(&reqst, NULL);
}
#endif /* HAVE_BSDEDITLINE */

/* -------------------------------------------------------------------------- */
/* This is the command processing loop for spice and nutmeg.
   The function is called even when GNU readline is unavailable, in which
   case it falls back to repeatable calling cp_evloop()
   SJB 26th April 2005 */
static void
app_rl_readlines()
{
#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
    /* GNU Readline Support -- Andrew Veliath <veliaa@rpi.edu> */
    char *line, *expanded_line;
    
    /* ---  set up readline params --- */
    strcpy(history_file, getenv("HOME"));
    strcat(history_file, "/.");
    strcat(history_file, application_name);
    strcat(history_file, "_history");
    
    using_history();
    read_history(history_file);
    
    rl_readline_name = application_name;
    rl_instream = cp_in;
    rl_outstream = cp_out;
    rl_event_hook = rl_event_func;
    rl_catch_signals = 0;   /* disable signal handling  */
    
    /* sjb - what to do for editline?
       This variable is not supported by editline. */   
#if defined(HAVE_GNUREADLINE) 
    rl_catch_sigwinch = 1;  /* allow readline to respond to resized windows  */
#endif  
    
    /* note that we want some mechanism to detect ctrl-D and expand it to exit */
    while (1) {
       history_set_pos(history_length);

       SETJMP(jbuf, 1);    /* Set location to jump to after handling SIGINT (ctrl-C)  */

       line = readline(prompt());
       if (line && *line) {
           int s = history_expand(line, &expanded_line);

           if (s == 2) {
               fprintf(stderr, "-> %s\n", expanded_line);
           } else if (s == -1) {
               fprintf(stderr, "readline: %s\n", expanded_line);
           } else {
               cp_evloop(expanded_line);
               add_history(expanded_line);
           }
           free(expanded_line);
       }
       if (line) free(line);
    }
    /* History gets written in ../fte/misccoms.c com_quit */
    
#else
    while (cp_evloop((char *) NULL) == 1) ;
#endif /* defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE) */
}


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
	   "  -i, --interactive         run in interactive mode\n"
	   "  -n, --no-spiceinit        don't load the local or user's config file\n"
	   "  -o, --output=FILE         set the outputfile\n"
	   "  -q, --completion          activate command completion\n"
	   "  -r, --rawfile=FILE        set the rawfile output\n"             
	   "  -s, --server              run spice as a server process\n"
	   "  -t, --term=TERM           set the terminal type\n"
 	   "  -h, --help                display this help and exit\n"
	   "  -v, --version             output version information and exit\n"
	   "\n"
	   "Report bugs to %s.\n", cp_program, Bug_Addr);
}

/* -------------------------------------------------------------------------- */
static void
show_version(void)
{
    printf("%s compiled from %s revision %s\n"
	   "Written originally by Berkeley University\n"
	   "Currently maintained by the NGSpice Project\n\n"
	   "Copyright (C) 1985-1996,"
	   "  The Regents of the University of California\n"
	   "Copyright (C) 1999-2005,"
	   "  The NGSpice Project\n", cp_program, PACKAGE, VERSION);
}

#ifdef SIMULATOR
/* -------------------------------------------------------------------------- */
static void
append_to_stream(FILE *dest, FILE *source)
{
    char *buf[BSIZE_SP];
    int i;

    while ((i = fread(buf, 1, BSIZE_SP, source)) > 0)
	fwrite(buf, i, 1, dest);
}
#endif /* SIMULATOR */

/* -------------------------------------------------------------------------- */
/* Read an initialisation file.
   dir    is the directory (use NULL or "" for current directory)
   name   is the initialisation file's name
   Return true on success
   SJB 25th April 2005 */
static bool
read_initialisation_file(char * dir, char * name)
{
#ifndef HAVE_ASPRINTF
    FILE * fp = NULL;
#endif /* not HAVE_ASPRINTF */
    char * path;
    bool result = FALSE;
    
    /* check name */
    if(name==NULL || name[0]=='\0')
    	return FALSE;	/* Fail; name needed */
    
    /* contruct the full path */
    if(dir == NULL || dir[0]=='\0') {
	path = name;
    } else {
#ifdef HAVE_ASPRINTF
	asprintf(&path, "%s" DIR_PATHSEP "%s", dir,name);
	if(path==NULL) return FALSE;	/* memory allocation error */
#else /* ~ HAVE_ASPRINTF */
	path=(char*)tmalloc(1 + strlen(dir)+strlen(name));
	if(path==NULL) return FALSE;	/* memory allocation error */
	sprintf(path,"%s" DIR_PATHSEP "%s",dir,name);
#endif /* HAVE_ASPRINTF */
    }

    /* now access the file */
#ifdef HAVE_UNISTD_H
    if (access(path, R_OK) == 0) {		
#else
    if ((fp = fopen(path, "r")) != NULL) {
	(void) fclose(fp);
#endif /* HAVE_UNISTD_H */
	inp_source(path);
#ifdef TRACE
	printf("Init file: '%s'\n",path);
#endif /* TRACE */	
	result = TRUE;	/* loaded okay */
    }
    
    /* if dir was not NULL and not empty then we allocated memory above */
    if(dir!=NULL && dir[0] !='\0')
#ifdef HAVE_ASPRINTF
    	free(path);
#else
	tfree(path);
#endif /* HAVE_ASPRINTF */
    
    return result;
}

/* -------------------------------------------------------------------------- */

#ifdef SIMULATOR
extern int OUTpBeginPlot(), OUTpData(), OUTwBeginPlot(), OUTwReference();
extern int OUTwData(), OUTwEnd(), OUTendPlot(), OUTbeginDomain();
extern int OUTendDomain(), OUTstopnow(), OUTerror(), OUTattributes();
#endif /* SIMULATOR */    

int
#ifdef HAS_WINDOWS
xmain(int argc, char **argv)
#else
main(int argc, char **argv)
#endif /* HAS_WINDOWS */
{
    int c;
    int		err;
    bool	gotone = FALSE;
    char*       copystring;/*DG*/
    char        addctrlsect = TRUE; /* PN: for autorun */


#ifdef SIMULATOR
    int error2;
    
    static IFfrontEnd nutmeginfo = {
	IFnewUid,
	IFdelUid,
	OUTstopnow,
	seconds,
	OUTerror,
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
#else  /* ~ SIMULATOR */
    bool gdata = TRUE;
#endif /* ~ SIMULATOR */

    char buf[BSIZE_SP];
    bool readinit = TRUE;
    bool rflag = FALSE;
    bool istty = TRUE;
    bool iflag = FALSE;
    bool qflag = FALSE;
    FILE *fp;
    FILE *circuit_file;

#ifdef TRACE
    /* this is used to detect memory leaks during debugging */
    /* added by SDB during debug . . . . */
    /* mtrace();  */
#endif

#ifdef TRACE
    /* this is also used for memory leak plugging . . . */
    /* added by SDB during debug . . . . */
    /*     mwDoFlush(1);  */
#endif

    /* MFB tends to jump to 0 on errors.  This tends to catch it. */
    if (started) {
        fprintf(cp_err, "main: Internal Error: jump to zero\n");
        shutdown(EXIT_BAD);
    }
    started = TRUE;

#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
    if (!(application_name = strrchr(argv[0],'/')))
        application_name = argv[0];
    else
        ++application_name;
#endif  /* defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)  */

#ifdef PARALLEL_ARCH
    PBEGIN_(argc, argv);
    ARCHme = NODEID_();
    ARCHsize = NNODES_();
    SETDBG_(&debug_flag);
    fprintf( stderr, "On-line: process %d of %d total.\n", ARCHme, ARCHsize );
    evlog(EVKEY_ENABLE, EVKEY_EVENT, "On-line", EVKEY_DUMP, EVKEY_DISABLE,
	  EVKEY_LAST_ARG);
#else
    ARCHme = 0;
    ARCHsize = 1;
#endif /* PARALLEL_ARCH */

    ivars( );

    cp_in = stdin;
    cp_out = stdout;
    cp_err = stderr;

    circuit_file = stdin;

#ifdef MALLOCTRACE
    mallocTraceInit("malloc.out");
#endif
#if defined(HAVE_ISATTY) && !defined(HAS_WINDOWS)
    istty = (bool) isatty(fileno(stdin));
#endif

    init_time( );

    err = SIMinit(&nutmeginfo,&ft_sim);
    if(err != OK) {
        ft_sperror(err,"SIMinit");
        shutdown(EXIT_BAD);
    }
    cp_program = ft_sim->simulator;

    srandom(getpid());

    /* --- Process command line options --- */
    while (1) {
	int option_index = 0;
	static struct option long_options[] = {
	    {"help", 0, 0, 'h'},
	    {"version", 0, 0, 'v'},
	    {"batch", 0, 0, 'b'},
            {"autorun", 0, 0, 'a'},
	    {"circuitfile", 0, 0, 'c'},
	    {"interactive", 0, 0, 'i'},
	    {"no-spiceinit", 0, 0, 'n'},
	    {"output", 0, 0, 'o'},
	    {"completion", 0, 0, 'q'},
	    {"rawfile", 1, 0, 'r'},
	    {"server", 0, 0, 's'},
	    {"terminal", 1, 0, 't'},
	    {0, 0, 0, 0}
	};

	c = getopt_long (argc, argv, "hvbac:ihno:qr:st:",
			 long_options, &option_index);
	if (c == -1)
	    break;

	switch (c) {
	    case 'h':		/* Help */
		show_help();
		shutdown (EXIT_NORMAL);
		break;

	    case 'v':		/* Version info */
		show_version();
		shutdown (EXIT_NORMAL);
		break;

	    case 'b':		/* Batch mode */
		ft_batchmode = TRUE;
                addctrlsect = FALSE;
                cp_vset("addcontrol",VT_BOOL,&addctrlsect);
		break;

            case 'a':           /* Add control section for autorun */
                if (!ft_batchmode) {
                    addctrlsect = TRUE;
                    cp_vset("addcontrol",VT_BOOL, &addctrlsect);
                    }
                break;

	    case 'c':		/* Circuit file */
		if (optarg) {
		    if (!(circuit_file = fopen(optarg, "r"))) {
			perror("circuit file not available");
			shutdown(EXIT_BAD);
		    }
		    istty = FALSE;
		}
		break;

	    case 'i':		/* Interactive mode */
		iflag = TRUE;
		break;

	    case 'n':		/* Don't read initialisation file */
		readinit = FALSE;
		break;

	    case 'o':		/* Output file */
		if (optarg) {
		    /* turn off buffering for stdout */
		    setbuf(stdout, NULL);
#ifdef PARALLEL_ARCH
		    sprintf (buf, "%s%03d", optarg, ARCHme);
#else
		    sprintf (buf, "%s", optarg);
#endif
		    /* Open the log file */
#ifdef HAS_WINDOWS
		    /* flogp goes to winmain's putc and writes to file buf */
		    if (!(flogp = fopen(buf, "w"))) {	
#else
		    /* Connect stdout to file buf and log stdout */
		    if (!(freopen (buf, "w", stdout))) {    
#endif
			perror (buf);
			shutdown (EXIT_BAD);
		    }

#ifdef HAS_WINDOWS
		    com_version(NULL);  /* hvogt 11.11.2001 */
#endif
		    fprintf(stdout, "\nBatch mode\n\n");
		    fprintf(stdout, "Simulation output goes to rawfile: %s\n\n", ft_rawfile);
		    fprintf(stdout, "Comments and warnings go to log-file: %s\n", buf);
#ifdef HAS_WINDOWS
		    oflag = TRUE;
#endif
		}
		break;

	    case 'q':		/* Command completion */
		qflag = TRUE;
		break;

	    case 'r':		/* The raw file */
		if (optarg) {
		    cp_vset("rawfile", VT_STRING, optarg);
		}
		//rflag = TRUE;
		break;

	    case 's':		/* Server mode */
		ft_servermode = TRUE;
		break;

	    case 't':
		if (optarg) {
		    cp_vset("term", VT_STRING, optarg);
		}
		break;

	    case '?':
		break;

	    default:
		printf ("?? getopt returned character code 0%o ??\n", c);
	}
    }  /* --- End of command line option processing  --- */


#ifdef SIMULATOR
    if_getparam = spif_getparam;
#else
    if_getparam = nutif_getparam;

    if (optind == argc) {
	/* No raw file */
	gdata = FALSE;
    }
#endif


    if ((!iflag && !istty) || ft_servermode)
        ft_batchmode = TRUE;
    if ((iflag && !istty) || qflag)
        cp_nocc = TRUE;
    if (ft_servermode)
        readinit = FALSE;
    if (!istty || ft_batchmode)
        out_moremode = FALSE;

    /* Would like to do this later, but cpinit evals commands */
    init_rlimits( );

    /* Have to initialize cp now. */
    ft_cpinit();

    /* To catch interrupts during .spiceinit... */
    if (SETJMP(jbuf, 1) == 1) {
        fprintf(cp_err, "Warning: error executing .spiceinit.\n");
        if (!ft_batchmode)
            goto bot;
    }
 
    /* Set up signal handling */
    if (!ft_batchmode) {
        /*  Set up interrupt handler  */
        (void) signal(SIGINT, (SIGNAL_FUNCTION) ft_sigintr);

        /* floating point exception  */
        (void) signal(SIGFPE, (SIGNAL_FUNCTION) sigfloat);

#ifdef SIGTSTP
        signal(SIGTSTP, (SIGNAL_FUNCTION) sigstop);
#endif
    }

    /* Set up signal handling for fatal errors. */
    signal(SIGILL, (SIGNAL_FUNCTION) sigill);

#ifdef SIGBUS
    signal(SIGBUS, (SIGNAL_FUNCTION) sigbus);
#endif
#ifdef SIGSEGV
/* Want core files!
 *   signal(SIGSEGV, sigsegv);
 */
#endif
#ifdef SIGSYS
    signal(SIGSYS, (SIGNAL_FUNCTION) sig_sys);
#endif

    /* load user's initialisation file */
    if (readinit) {
	bool good;
	
	/* Try accessing the initialisation file in the current directory */
	good = read_initialisation_file("",INITSTR);
	
	/* if that fail try the alternate name */
	if(good == FALSE)
	    good = read_initialisation_file("",ALT_INITSTR);
	
	/* if that failed try in the user's home directory
	   if their HOME environment variable is set */
	if(good == FALSE) {
	    char * homedir;
	    homedir = getenv("HOME");
	    if(homedir !=NULL) {
		good = read_initialisation_file(homedir,INITSTR);
		if(good == FALSE) {
		    good = read_initialisation_file(homedir,ALT_INITSTR);
		}
    	    }
	}
    }

    if (!ft_batchmode) {
	com_version(NULL);
        DevInit( );
	if (News_File && *News_File) {
            copystring=cp_tildexpand(News_File);/*DG  Memory leak */
	    fp = fopen(copystring, "r");
            tfree(copystring);
	    if (fp) {
		while (fgets(buf, BSIZE_SP, fp))
		    fputs(buf, stdout);
		(void) fclose(fp);
	    }
	}
    }


bot:

    /* Pass 2 -- get the filenames. If we are spice, then this means
     * build a circuit for this file. If this is in server mode, don't
     * process any of these args.  */

    if (SETJMP(jbuf, 1) == 1)
        goto evl;


    cp_interactive = FALSE;
    err = 0;

#ifdef SIMULATOR
    if (!ft_servermode && !ft_nutmeg) {
	/* Concatenate all non-option arguments into a temporary file
	   and load that file into the spice core.
	   
	   The original routine took a special path if there was only
	   one non-option argument.  In that case, it didn't create
	   the temporary file but used the original file instead.  The
	   current algorithm is uniform at the expense of a little
	   startup time.  */
	FILE *tempfile;

	tempfile = tmpfile();
	if (optind == argc && !istty) {
	    append_to_stream(tempfile, stdin);
	}
	while (optind < argc) {
	    char *arg;
	    FILE *tp;

	    /* Copy all the arguments into the temporary file */
	    arg = argv[optind++];
	    tp = fopen(arg, "r");
	    if (!tp) {
		perror(arg);
		err = 1;
		break;
	    }
	    append_to_stream(tempfile, tp);
	    fclose(tp);
	}
	fseek(tempfile, (long) 0, 0);

        if (tempfile && (!err || !ft_batchmode)) {
            inp_spsource(tempfile, FALSE, NULL);
            gotone = TRUE;
        }
	if (ft_batchmode && err)
	    shutdown(EXIT_BAD);
    }   /* ---  if (!ft_servermode && !ft_nutmeg) --- */

    if (!gotone && ft_batchmode && !ft_nutmeg)
        inp_spsource(circuit_file, FALSE, (char *) NULL);

evl:
    if (ft_batchmode) {
        /* If we get back here in batch mode then something is wrong,
         * so exit.  */
        bool st = FALSE;

        (void) SETJMP(jbuf, 1);


        if (st == TRUE) {
            shutdown(EXIT_BAD);
	}
        st = TRUE;
        if (ft_servermode) {
            if (ft_curckt == NULL) {
                fprintf(cp_err, "Error: no circuit loaded!\n");
                shutdown(EXIT_BAD);
            }
            if (ft_dorun(""))
		shutdown(EXIT_BAD);
            shutdown(EXIT_NORMAL);
        }

        /* If -r is specified, then we don't bother with the dot
         * cards. Otherwise, we use wrd_run, but we are careful not to
         * save too much.  */
        cp_interactive = FALSE;
        if (rflag) {
	  /* saj done already in inp_spsource ft_dotsaves();*/
	    error2 = ft_dorun(ft_rawfile);
	    if (ft_cktcoms(TRUE) || error2)
		shutdown(EXIT_BAD);
        } else if (ft_savedotargs()) {
	    error2 = ft_dorun(NULL);
	    if (ft_cktcoms(FALSE) || error2)
		shutdown(EXIT_BAD);
	} else {
	    fprintf(stderr,
		    "Note: No \".plot\", \".print\", or \".fourier\" lines; "
		    "no simulations run\n");
	    shutdown(EXIT_BAD);
        }
    }  /* ---  if (ft_batchmode) ---  */ 
    else {
        cp_interactive = TRUE;
        app_rl_readlines();  /*  enter the command processing loop  */
    }  /* --- else (if (ft_batchmode)) --- */

#else  /* ~ SIMULATOR */

    if (ft_nutmeg && gdata) {
	while (optind < argc) {
	  ft_loadfile(argv[optind++]);
	  gotone = TRUE;
	}
        if (!gotone)
            ft_loadfile(ft_rawfile);
    }

evl:
    /* Nutmeg "main" */
    (void) SETJMP(jbuf, 1);
    cp_interactive = TRUE;
    app_rl_readlines();  /*  enter the command processing loop  */

#endif /* ~ SIMULATOR */

    return shutdown(EXIT_NORMAL);
}
