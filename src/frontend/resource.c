/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Resource-related routines.
 *
 * Time information is acquired here.
 * Memory information is obtained in functions get_... for
 * a large variety of current operating systems.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"

#include "circuits.h"
#include "resource.h"
#include "variable.h"
#include "ngspice/cktdefs.h"

#include <inttypes.h>

#include "../misc/misc_time.h" /* timediff */

#ifdef XSPICE
/* gtri - add - 12/12/90 - wbk - include ipc stuff */
#include "ngspice/ipctiein.h"
/* gtri - end - 12/12/90 */
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

/* We might compile for Windows, but only as a console application (e.g. tcl) */
#if defined(HAS_WINGUI) || defined(__MINGW32__) || defined(_MSC_VER)
#define PSAPI_VERSION 1
#define HAVE_WIN32

#define WIN32_LEAN_AND_MEAN

/*
 * The ngspice.h file included above defines BOOLEAN (via bool.h) and this
 * clashes with the definition obtained from windows.h (via winnt.h).
 * However, BOOLEAN is not used by this file so we can work round this problem
 * by undefining BOOLEAN before including windows.h
 * SJB - April 2005
 */
#undef BOOLEAN
#include <windows.h>
#include <psapi.h>

#else
#include <unistd.h>
#endif /* HAVE_WIN32 */

static void printres(char *name);
static void fprintmem(FILE *stream, unsigned long long memory);

#if defined(HAVE_WIN32) || defined(HAVE__PROC_MEMINFO)
static int get_procm(struct proc_mem *memall);

struct sys_mem mem_t, mem_t_act;
struct proc_mem mem_ng, mem_ng_act;

#endif

#if defined(HAVE_WIN32) &&  defined(SHARED_MODULE) && defined(__MINGW32__)
static int get_sysmem(struct sys_mem *memall);
#endif

void
init_rlimits(void)
{
    ft_ckspace();
}


void
init_time(void)
{
#ifdef HAVE_GETRUSAGE
#else
#  ifdef HAVE_TIMES
#  else
#    ifdef HAVE_FTIME
    ftime(&timebegin);
#    endif
#  endif
#endif
}


void
com_rusage(wordlist *wl)
{
    char *copyword;
    /* Fill in the SPICE accounting structure... */

    if (wl && (eq(wl->wl_word, "everything") || eq(wl->wl_word, "all"))) {
        printres(NULL);
    } else if (wl) {
        for (; wl; wl = wl->wl_next) {
            /*   printres(cp_unquote(wl->wl_word)); DG: bad, memory leak*/
            copyword = cp_unquote(wl->wl_word);/*DG*/
            printres(copyword);
            tfree(copyword);
            if (wl->wl_next)
                (void) putc('\n', cp_out);
        }
    } else {
        printf("\n");
        printres("time");
        (void) putc('\n', cp_out);
        printres("totalcputime");
        (void) putc('\n', cp_out);
        printres("space");
    }
}


/* Find out if the user is approaching his maximum data size.
   If usage is withing 95% of total available then a warning message is sent
   to the error stream (cp_err) */
void ft_ckspace(void)
{
#ifdef SHARED_MODULE
    /* False warning on some OSs, especially on Linux when loaded during runtime.
       The caller then has to take care of memory available */
    return;
#else
    const unsigned long long freemem = getAvailableMemorySize();
    const unsigned long long usage = getCurrentRSS();

    if (freemem == 0 || usage == 0) { /* error obtaining data */
        return;
    }

    const unsigned long long avail = usage + freemem;
    if ((double) usage > (double) avail * 0.95) {
        (void) fprintf(cp_err,
                "Warning - approaching max data size: "
                "current size = ");
        fprintmem(cp_err, usage);
        (void) fprintf(cp_err, ", limit = ");
        fprintmem(cp_err, avail);
        (void) fprintf(cp_err, "\n");
    }
#endif
} /* end of function ft_chkspace */



/* Print out one piece of resource usage information. */
static void
printres(char *name)
{
#ifdef CIDER
    char *paramname = NULL;
#endif
    bool yy = FALSE;
    static bool called = FALSE;
    static long last_sec = 0, last_msec = 0;
    struct variable *v, *vfree = NULL;
    char *cpu_elapsed;

    if (!name || eq(name, "totalcputime") || eq(name, "cputime")) {
        int total_sec, total_msec;

#  ifdef HAVE_GETRUSAGE
        int ret;
        struct rusage ruse;
        memset(&ruse, 0, sizeof(ruse));
        ret = getrusage(RUSAGE_SELF, &ruse);
        if (ret == -1)
            perror("getrusage(): ");

        total_sec = (int) (ruse.ru_utime.tv_sec + ruse.ru_stime.tv_sec);
        total_msec = (int) (ruse.ru_utime.tv_usec + ruse.ru_stime.tv_usec) / 1000;
        cpu_elapsed = "CPU";
#  else
#    ifdef HAVE_TIMES
        struct tms ruse;
        times(&ruse);
        clock_t x = ruse.tms_utime + ruse.tms_stime;
        clock_t hz = (clock_t) sysconf(_SC_CLK_TCK);
        total_sec = x / hz;
        total_msec = ((x % hz) * 1000) / hz;
        cpu_elapsed = "CPU";
#    else
#      ifdef HAVE_FTIME
        struct timeb timenow;
        ftime(&timenow);
        timediff(&timenow, &timebegin, &total_sec, &total_msec);
        cpu_elapsed = "elapsed";
#      else
#        define NO_RUDATA
#      endif
#    endif
#  endif


#ifndef NO_RUDATA

        if (total_msec >= 1000) {
            total_msec -= 1000;
            total_sec += 1;
        }

        if (!name || eq(name, "totalcputime")) {
            fprintf(cp_out, "Total %s time (seconds) = %u.%03u \n",
                    cpu_elapsed, total_sec, total_msec);
        }

        if (!name || eq(name, "cputime")) {
            last_msec = 1000 + total_msec - last_msec;
            last_sec = total_sec - last_sec - 1;
            if (last_msec >= 1000) {
                last_msec -= 1000;
                last_sec += 1;
            }
            /* do not print it the first time, doubling totalcputime */
            if (called)
                fprintf(cp_out, "%s time since last call (seconds) = %lu.%03lu \n",
                        cpu_elapsed, last_sec, last_msec);

            last_sec = total_sec;
            last_msec = total_msec;
            called = TRUE;
        }

#ifdef XSPICE
        /* gtri - add - 12/12/90 - wbk - record cpu time used for ipc */
        g_ipc.cpu_time = (double) last_msec;
        g_ipc.cpu_time /= 1000.0;
        g_ipc.cpu_time += (double) last_sec;
        /* gtri - end - 12/12/90 */
#endif

        yy = TRUE;
#else
        if (!name || eq(name, "totalcputime"))
            fprintf(cp_out, "Total CPU time: ??.??? seconds.\n");
        if (!name || eq(name, "cputime"))
            fprintf(cp_out, "CPU time since last call: ??.??? seconds.\n");
        yy = TRUE;
#endif

    }

    if (!name || eq(name, "space")) {
        unsigned long long mem = getMemorySize();
        fprintf(cp_out, "Total DRAM available = ");
        fprintmem(cp_out, mem);
        fprintf(cp_out, ".\n");
        mem = getAvailableMemorySize();
        fprintf(cp_out, "DRAM currently available = ");
        fprintmem(cp_out, mem);
        fprintf(cp_out, ".\n");
        mem = getPeakRSS();
        fprintf(cp_out, "Maximum ngspice program size = ");
        fprintmem(cp_out, mem);
        fprintf(cp_out, ".\n");
        mem = getCurrentRSS();
        fprintf(cp_out, "Current ngspice program size = ");
        fprintmem(cp_out, mem);
        fprintf(cp_out, ".\n");

#if defined(HAVE__PROC_MEMINFO)
        get_procm(&mem_ng_act);
//        fprintf(cp_out, "Resident set size = ");
//        fprintmem(cp_out, mem_ng_act.resident);
//        fprintf(cp_out, ".\n");
        fprintf(cp_out, "\n");  
        fprintf(cp_out, "Shared ngspice pages = ");
        fprintmem(cp_out, mem_ng_act.shared);
        fprintf(cp_out, ".\n");

        fprintf(cp_out, "Text (code) pages = ");
        fprintmem(cp_out, mem_ng_act.trs);
        fprintf(cp_out, ".\n");

        fprintf(cp_out, "Stack = ");
        fprintmem(cp_out, mem_ng_act.drs);
        fprintf(cp_out, ".\n");

        fprintf(cp_out, "Library pages = ");
        fprintmem(cp_out, mem_ng_act.lrs);
        fprintf(cp_out, ".\n");
        /* not used
           fprintf(cp_out, "Dirty pages = ");
           fprintmem(cp_out, all_memory.dt);
           fprintf(cp_out, ".\n"); */
#endif  /* HAVE__PROC_MEMINFO */
        yy = TRUE;
    }

    if (!name || eq(name, "faults")) {
#ifdef HAVE_GETRUSAGE
        int ret;
        struct rusage ruse;
        memset(&ruse, 0, sizeof(ruse));
        ret = getrusage(RUSAGE_SELF, &ruse);
        if (ret == -1)
            perror("getrusage(): ");
        fprintf(cp_out,
                "%lu page faults, %lu vol + %lu invol = %lu context switches.\n",
                ruse.ru_majflt, ruse.ru_nvcsw, ruse.ru_nivcsw,
                ruse.ru_nvcsw + ruse.ru_nivcsw);
        yy = TRUE;
#endif
    }

    /* PN Now get all the frontend resource stuff */
    if (ft_curckt) {
        if (name && eq(name, "task"))
            vfree = v = ft_getstat(ft_curckt, NULL);
        else
            vfree = v = ft_getstat(ft_curckt, name);

        if (name && v) {
            fprintf(cp_out, "%s= ", v->va_name);
            wl_print(cp_varwl(v), cp_out);
            (void)putc('\n', cp_out);
            yy = TRUE;
        } else if (v) {
            (void) putc('\n', cp_out);
            while (v) {
                wordlist *wlpr = cp_varwl(v);
                fprintf(cp_out, "%s = ", v->va_name);
                wl_print(wlpr, cp_out);
                wl_free(wlpr);
                (void) putc('\n', cp_out);
                v = v->va_next;
            }
            yy = TRUE;
        }
    }

    if (vfree)
        free_struct_variable(vfree);

    /* Now get all the spice resource stuff. */
    if (ft_curckt && ft_curckt->ci_ckt) {

#ifdef CIDER
/* begin cider integration */
        if (!name || eq(name, "circuit") || eq(name, "task"))
            paramname = NULL;
        else
            paramname = name;

        vfree = v = if_getstat(ft_curckt->ci_ckt, paramname);
        if (paramname && v) {
/* end cider integration */
#else /* ~CIDER */
        if (name && eq(name, "task"))
            vfree = v = if_getstat(ft_curckt->ci_ckt, NULL);
        else
            vfree = v = if_getstat(ft_curckt->ci_ckt, name);

        if (name && v) {
#endif
            fprintf(cp_out, "%s = ", v->va_name);
            wordlist *wltmp = cp_varwl(v);
            wl_print(wltmp, cp_out);
            wl_free(wltmp);
            (void) putc('\n', cp_out);
            yy = TRUE;
        } else if (v) {
            (void) putc('\n', cp_out);
            while (v) {
                wordlist *wlpr = cp_varwl(v);
                fprintf(cp_out, "%s = ", v->va_name);
                wl_print(wlpr, cp_out);
                wl_free(wlpr);
                (void) putc('\n', cp_out);
                v = v->va_next;
            }
            yy = TRUE;
        }

#ifdef CIDER
        /* begin cider integration */
        /* Now print out interesting stuff about numerical devices. */
        if (!name || eq(name, "devices")) {
            (void) NDEVacct(ft_curckt->ci_ckt, cp_out);
            yy = TRUE;
        }
        /* end cider integration */
#endif
    }

    if (!yy) {
        fprintf(cp_err, "Note: no resource usage information for '%s',\n", name);
        fprintf(cp_err, "\tor no active circuit available\n");
    }

    if (vfree)
        free_struct_variable(vfree);
}


/* Print to stream the given memory size in a human friendly format */
static void
fprintmem(FILE *stream, unsigned long long memory) {
    if (memory > 1048576)
        fprintf(stream, "%8.3f MB", (double)memory / 1048576.);
    else if (memory > 1024)
        fprintf(stream, "%5.3f kB", (double)memory / 1024.);
    else
        fprintf(stream, "%llu bytes", memory);
}


#if defined(HAVE_WIN32) || defined(HAVE__PROC_MEMINFO)

static int get_procm(struct proc_mem *memall) {

#ifdef HAVE_WIN32
    /* FIXME: shared module should be allowed, but currently does not link to psapi within MINGW/MSYS2 */
#if !defined(SHARED_MODULE) || !defined(__MINGW32__)
/* Use Windows API function to obtain size of memory - more accurate */
    PROCESS_MEMORY_COUNTERS pmc;

    /* psapi library required */
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        memall->size = pmc.WorkingSetSize;
        memall->resident = pmc.QuotaNonPagedPoolUsage;
        memall->trs = pmc.QuotaPagedPoolUsage;
    } else
        return 0;
#else
   /* Use Windows GlobalMemoryStatus or /proc/memory to obtain size of memory -
    * not accurate */
    get_sysmem(&mem_t_act); /* size is the difference between free memory at
                             * start time and now */
    if (mem_t.free > mem_t_act.free) /* it can happen that that ngspice is */
        memall->size = (mem_t.free - mem_t_act.free); /* too small compared to
                                                       * os memory usage */
    else
        memall->size = 0;       /* sure, it is more */
    memall->resident = 0;
    memall->trs = 0;
#endif
#else
/* Use Linux/UNIX /proc/<pid>/statm file information */
    FILE *fp;
    char buffer[1024];
    size_t bytes_read;
    long sz;
    /* page size */
    if ((sz = sysconf(_SC_PAGESIZE)) == -1) {
        perror("sysconf() error");
        return 0;
    }
    if ((fp = fopen("/proc/self/statm", "r")) == NULL) {
        perror("fopen(\"/proc/%d/statm\")");
        return 0;
    }
    bytes_read = fread(buffer, 1, sizeof(buffer), fp);
    fclose(fp);
    if (bytes_read == 0 || bytes_read == sizeof(buffer))
        return 0;
    buffer[bytes_read] = '\0';

    sscanf(buffer, "%llu %llu %llu %llu %llu %llu %llu", &memall->size, &memall->resident, &memall->shared, &memall->trs, &memall->drs, &memall->lrs, &memall->dt);
    /* scale by page size */
    memall->size *= (long long unsigned)sz;
    memall->resident *= (long long unsigned)sz;
    memall->shared *= (long long unsigned)sz;
    memall->trs *= (long long unsigned)sz;
    memall->drs *= (long long unsigned)sz;
    memall->lrs *= (long long unsigned)sz;
    memall->dt *= (long long unsigned)sz;

#endif /* HAVE_WIN32 */
    return 1;
}


#if defined(HAVE_WIN32) &&  defined(SHARED_MODULE) && defined(__MINGW32__)
static int get_sysmem(struct sys_mem *memall)
{
#ifdef HAVE_WIN32
#if (_WIN32_WINNT >= 0x0500)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&ms);
    memall->size = ms.ullTotalPhys;
    memall->free = ms.ullAvailPhys;
    memall->swap_t = ms.ullTotalPageFile;
    memall->swap_f = ms.ullAvailPageFile;
#else
    MEMORYSTATUS ms;
    ms.dwLength = sizeof(MEMORYSTATUS);
    GlobalMemoryStatus(&ms);
    memall->size = ms.dwTotalPhys;
    memall->free = ms.dwAvailPhys;
    memall->swap_t = ms.dwTotalPageFile;
    memall->swap_f = ms.dwAvailPageFile;
#endif /*_WIN32_WINNT 0x0500*/
#else
    FILE *fp;
    char buffer[2048];
    size_t bytes_read;
    char *match;
    unsigned long long mem_got;

    if ((fp = fopen("/proc/meminfo", "r")) == NULL) {
        perror("fopen(\"/proc/meminfo\")");
        return 0;
    }

    bytes_read = fread(buffer, 1, sizeof(buffer), fp);
    fclose(fp);
    if (bytes_read == 0 || bytes_read == sizeof(buffer))
        return 0;
    buffer[bytes_read] = '\0';

    /* Search for string "MemTotal" */
    match = strstr(buffer, "MemTotal");
    if (match == NULL) /* not found */
        return 0;
    sscanf(match, "MemTotal: %llu", &mem_got);
    memall->size = mem_got*1024; /* 1MB = 1024KB */
    /* Search for string "MemFree" */
    match = strstr(buffer, "MemFree");
    if (match == NULL) /* not found */
        return 0;
    sscanf(match, "MemFree: %llu", &mem_got);
    memall->free = mem_got*1024; /* 1MB = 1024KB */
    /* Search for string "SwapTotal" */
    match = strstr(buffer, "SwapTotal");
    if (match == NULL) /* not found */
        return 0;
    sscanf(match, "SwapTotal: %llu", &mem_got);
    memall->swap_t = mem_got*1024; /* 1MB = 1024KB */
    /* Search for string "SwapFree" */
    match = strstr(buffer, "SwapFree");
    if (match == NULL) /* not found */
        return 0;
    sscanf(match, "SwapFree: %llu", &mem_got);
    memall->swap_f = mem_got*1024; /* 1MB = 1024KB */
#endif
    return 1;
}
#endif


#else


#include <signal.h>
#include <setjmp.h>

/*
 * baseaddr() returns the base address of the data segment on most Unix
 * systems.  It's an ugly hack for info that should be provided by the OS.
 */

/* Does anyone use a pagesize < 256 bytes??  I'll bet not;
 * too small doesn't hurt
 */

#define LOG2_PAGESIZE   8

static JMP_BUF env;


static void
fault(void)
{
    signal(SIGSEGV, (SIGNAL_FUNCTION) fault);   /* SysV style */
    LONGJMP(env, 1);
}


static void *
baseaddr(void)
{
#if defined(__CYGWIN__) || defined(__MINGW32__) || defined(HAVE_WIN32) || defined(__APPLE__) || defined(__SUNPRO_C)
    return 0;
#else
    char *low, *high, *at;
    long x;
    void  (*orig_signal)();

    if (getenv("SPICE_NO_DATASEG_CHECK"))
        return 0;

    low = 0;
    high = (char *) ((unsigned long) sbrk(0) & ~((1 << LOG2_PAGESIZE) - 1));

    orig_signal = signal(SIGSEGV, (SIGNAL_FUNCTION) fault);

    for (;;) {

        at = (char *) ((((long)low >> LOG2_PAGESIZE) +
                        ((long)high >> LOG2_PAGESIZE))
                       << (LOG2_PAGESIZE - 1));

        if (at == low || at == high)
            break;

        if (SETJMP(env, 1)) {
            low = at;
            continue;
        } else {
            x = *at;
        }

        if (SETJMP(env, 1)) {
            low = at;
            continue;
        } else {
            *at = x;
        }

        high = at;

    }

    (void) signal(SIGSEGV, (SIGNAL_FUNCTION) orig_signal);
    return (void *) high;

#endif
}


#endif


#  ifdef notdef
main()
{
    printf("testing\n");
    printf("baseaddr: %#8x  topaddr: %#8x\n", baseaddr(), sbrk(0));
}
#  endif
