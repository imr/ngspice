/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Resource-related routines.
 */

#include <config.h>
#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"

#include "circuits.h"
#include "quote.h"
#include "resource.h"
#include "variable.h"

#include <sys/types.h>

#ifdef HAS_BSDRLIMIT
#  include <sys/time.h>
#  include <sys/resource.h>
#endif
#ifdef HAS_BSDRUSAGE
#  ifndef HAS_BSDRLIMIT
#    include <sys/time.h>
#    include <sys/resource.h>
#  endif
#else
#  ifdef HAS_SYSVRUSAGE
#    include <sys/times.h>
#    include <sys/param.h>
#  else
#    ifdef HAS_FTIME
#      include <sys/timeb.h>
struct timeb timebegin;
#    endif
#  endif
#endif

/* static declarations */
static void printres(char *name);
static RETSIGTYPE fault(void);
static void * baseaddr(void);

#ifdef HAS_RLIMIT_
#  ifdef HAS_MEMAVL
size_t mem_avail;
#  else
char *startdata;
char *enddata;
#  endif
#endif


void
init_rlimits(void)
{
#ifdef HAS_RLIMIT_
#  ifdef HAS_MEMAVL
    mem_avail = _memavl( );
#  else
    startdata = (char *) baseaddr( );
    enddata = sbrk(0);
#  endif
#endif
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
char* copyword;
    /* Fill in the SPICE accounting structure... */

    if (wl && (eq(wl->wl_word, "everything") || eq(wl->wl_word, "all"))) {
        printres((char *) NULL);
    } else if (wl) {
        for (; wl; wl = wl->wl_next) {
         /*   printres(cp_unquote(wl->wl_word)); DG: bad, memory leak*/
              copyword=cp_unquote(wl->wl_word);/*DG*/
              printres(copyword);
              tfree(copyword);                         
            if (wl->wl_next)
                (void) putc('\n', cp_out);
        }
    } else {
        printres("cputime");
        (void) putc('\n', cp_out);
        printres("totalcputime");
        (void) putc('\n', cp_out);
        printres("space");
    }
    return;
}

/* Find out if the user is approaching his maximum data size. */

void
ft_ckspace(void)
{
#ifdef HAS_RLIMIT_
    long usage, limit;
    static long old_usage = 0;

#  ifdef HAS_MEMAVL

    size_t mem_avail_now;

    mem_avail_now = _memavl( );
    usage = mem_avail - mem_avail_now;
    limit = mem_avail;

#  else

    char *hi;

#    ifdef HAS_BSDRLIMIT

    struct rlimit rld;
    getrlimit(RLIMIT_DATA, &rld);
    if (rld.rlim_cur == RLIM_INFINITY)
        return;
    limit = rld.rlim_cur - (enddata - startdata); /* rlim_max not used */

#    else

    /* SYSVRLIMIT */
    limit = ulimit(3, 0L) - (enddata - startdata);

#    endif

    usage = sbrk(0) - enddata;

#  endif

    if (limit < 0)
	return;	/* what else do you do? */

    if (usage <= old_usage)
	return;

    old_usage = usage;

    if (usage > limit * 0.9) {
        fprintf(cp_err, "Warning - approaching max data size: ");
        fprintf(cp_err, "current size = %ld, limit = %ld.\n", usage, limit);
    }

#endif
    return;
}

/* Print out one piece of resource usage information. */


static void
printres(name)
    char *name;
{
    bool yy = false;
    static long lastsec = 0, lastusec = 0;
    struct variable *v;
    long   realt;
    char   *cpu_elapsed;

    if (!name || eq(name, "totalcputime") || eq(name, "cputime")) {
	int	total, totalu;

#ifdef ipsc
#        define NO_RUDATA
#else

#  ifdef HAS_BSDRUSAGE
        struct rusage ruse;
        (void) getrusage(RUSAGE_SELF, &ruse);
	total = ruse.ru_utime.tv_sec + ruse.ru_stime.tv_sec;
	totalu = (ruse.ru_utime.tv_usec + ruse.ru_stime.tv_usec) / 1000;
	cpu_elapsed = "CPU";
#  else
#    ifdef HAS_SYSVRUSAGE
	struct tms ruse;
	realt = times(&ruse);
	total = (ruse.tms_utime + ruse.tms_stime)/ HZ;
	totalu = (ruse.tms_utime + ruse.tms_utime) * 1000 / HZ;
	cpu_elapsed = "CPU";
#    else
#      ifdef HAS_FTIME
	struct timeb timenow;
	int sec, msec;
	ftime(&timenow);
	timediff(&timenow, &timebegin, &total, &totalu);
	totalu /= 1000;
	cpu_elapsed = "elapsed";
#      else
#        define NO_RUDATA
#      endif
#    endif
#  endif
#endif


#ifndef NO_RUDATA
	if (!name || eq(name, "totalcputime")) {
	    total += totalu / 1000;
	    totalu %= 1000;
	    fprintf(cp_out, "Total %s time: %lu.%03lu seconds.\n",
		cpu_elapsed, total, totalu);
	}

	if (!name || eq(name, "cputime")) {
	    lastusec = totalu - lastusec;
	    lastsec = total - lastsec;
	    while (lastusec < 0) {
		lastusec += 1000;
		lastsec -= 1;
	    }
	    while (lastusec > 1000) {
		lastusec -= 1000;
		lastsec += 1;
	    }

	    out_printf("%s time since last call: %lu.%03lu seconds.\n",
		cpu_elapsed, lastsec, lastusec);

	    lastsec = total;
	    lastusec = totalu;
	}

	yy = true;
#else
	if (!name || eq(name, "totalcputime"))
	    fprintf(cp_out, "Total CPU time: ??.??? seconds.\n");
	if (!name || eq(name, "cputime"))
	    fprintf(cp_out, "CPU time since last call: ??.??? seconds.\n");
	yy = true;
#endif

    }

    if (!name || eq(name, "space")) {
	long usage = 0, limit = 0;
#ifdef ipsc
	NXINFO cur = nxinfo, start = nxinfo_snap;

	usage = cur.dataend - cur.datastart;
	limit = start.availmem;
#else
#  ifdef HAS_BSDRLIMIT
        struct rlimit rld;
        char *hi;

        getrlimit(RLIMIT_DATA, &rld);
	limit = rld.rlim_cur - (enddata - startdata);
        hi = sbrk(0);
	usage = (long) (hi - enddata);
#  else
#    ifdef HAS_SYSVRLIMIT
        char *hi;

	limit = ulimit(3, 0L) - (enddata - startdata);
        hi = sbrk(0);
	usage = (long) (hi - enddata);
#    else
#      ifdef HAS_MEMAVL

	size_t	mem_avail_now;
	mem_avail_now = _memavl( );
	limit = mem_avail;
	usage = mem_avail - mem_avail_now;

#      endif
#    endif
#  endif
#endif
        fprintf(cp_out, "Current dynamic memory usage = %ld,\n", usage);
        fprintf(cp_out, "Dynamic memory limit = %ld.\n", limit);
        yy = true;
    }

    if (!name || eq(name, "faults")) {
#ifdef HAS_BSDRUSAGE
        struct rusage ruse;

        (void) getrusage(RUSAGE_SELF, &ruse);
        fprintf(cp_out, 
        "%lu page faults, %lu vol + %lu invol = %lu context switches.\n",
                ruse.ru_majflt, ruse.ru_nvcsw, ruse.ru_nivcsw, 
                ruse.ru_nvcsw + ruse.ru_nivcsw);
        yy = true;
#endif
    } 

    /* Now get all the spice resource stuff. */
    if (ft_curckt && ft_curckt->ci_ckt) {
	if (name && eq(name, "task"))
	    v = if_getstat(ft_curckt->ci_ckt, NULL);
	else
	    v = if_getstat(ft_curckt->ci_ckt, name);
        if (name && v) {
            fprintf(cp_out, "%s = ", v->va_name);
            wl_print(cp_varwl(v), cp_out);
            (void) putc('\n', cp_out);
            yy = true;
        } else if (v) {
            (void) putc('\n', cp_out);
            while (v) {
                fprintf(cp_out, "%s = ", v->va_name);
                wl_print(cp_varwl(v), cp_out);
                (void) putc('\n', cp_out);
                v = v->va_next;
            }
            yy = true;
        }
    }

    if (!yy) {
        fprintf(cp_err, "Note: no resource usage information for '%s',\n",
		name);
        fprintf(cp_err, "\tor no active circuit available\n");
        
    }
    return;
}

#ifdef HAS_UNIX_SEGMENT_HACK

#include <signal.h>
#include <setjmp.h>

/*
 * baseaddr( ) returns the base address of the data segment on most Unix
 * systems.  It's an ugly hack for info that should be provided by the OS.
 */

/* Does anyone use a pagesize < 256 bytes??  I'll bet not;
 * too small doesn't hurt
 */

#define LOG2_PAGESIZE	8

static jmp_buf	env;

static RETSIGTYPE
fault(void)
{
	signal(SIGSEGV, (SIGNAL_FUNCTION) fault);	/* SysV style */
	longjmp(env, 1);
}

static void *
baseaddr( )
{
	char *low, *high, *at;
	char *sbrk( );
	long x;
	SIGNAL_TYPE	(*orig_signal)( );

	if (getenv("SPICE_NO_DATASEG_CHECK"))
		return 0;

	low = 0;
	high = (char *) ((unsigned long) sbrk(0) & ~((1 << LOG2_PAGESIZE) - 1));

	orig_signal = signal(SIGSEGV, (SIGNAL_FUNCTION) fault);

	do {

		at = (char *) ((((long)low >> LOG2_PAGESIZE)
			+ ((long)high >> LOG2_PAGESIZE))
			<< (LOG2_PAGESIZE - 1));
#  ifdef notdef
		at = (char *) ((((int) low + (int) high) / 2 + 0x7ff)
			& ~(long) 0xfff);
			/* nearest page */
#  endif
#  ifdef notdef
		printf(
		"high = %#8x    low = %#8x     at = %#8x\n",
			high, low, at);
#  endif

		if (at == low || at == high) {
			break;
		}

		if (setjmp(env)) {
			low = at;
			continue;
		} else
			x = *at;

		if (setjmp(env)) {
			low = at;
			continue;
		} else
			*at = x;

		high = at;

	} while (1);

#  ifdef notdef
	printf ("start is at %#x, end is at %#x\n", high, sbrk(0));
#  endif
	(void) signal(SIGSEGV, (SIGNAL_FUNCTION) orig_signal);
	return (void *) high;
}

#  ifdef notdef
main( )
{
	printf("testing\n");
	printf("baseaddr: %#8x  topaddr: %#8x\n", baseaddr( ), sbrk(0));
}
#  endif

#else

static void *
baseaddr( )
{
	return 0;
}

#endif
