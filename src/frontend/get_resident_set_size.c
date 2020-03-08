/*
 * Author:   David Robert Nadeau
 * Site:     http://NadeauSoftware.com/
 * License:  Creative Commons Attribution 3.0 Unported License
 *           http://creativecommons.org/licenses/by/3.0/deed.en_US
 * Modified: Holger Vogt, 2019
 */

#include "ngspice/ngspice.h"
#include "resource.h"

#if defined(_WIN32)
#undef BOOLEAN
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
unsigned long long getPeakRSS(void)
{
#if defined(HAVE__PROC_MEMINFO)
    /* Linux ---------------------------------------------------- */
    unsigned long long rss = 0L;
    FILE* fp = NULL;
    if ( (fp = fopen( "/proc/self/statm", "r" )) == NULL )
            return (unsigned long long) 0L; /* Can't open? */
    if ( fscanf( fp, "%llu", &rss ) != 1 )
    {
        fclose( fp );
        return 0L;      /* Can't read? */
    }
    fclose( fp );
        return rss * (unsigned long long) sysconf(_SC_PAGESIZE);
        
#elif defined(HAVE_GETRUSAGE)
    /* BSD, Linux, and OSX -------------------------------------- 
     * not (yet) available with CYGWIN */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (unsigned long long) rusage.ru_maxrss;
#else
    return (unsigned long long) (rusage.ru_maxrss * 1024L);
#endif

#elif defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
        return (unsigned long long) info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ( (fd = open( "/proc/self/psinfo", O_RDONLY )) == -1 )
        return 0L;      /* Can't open? */
    if ( read( fd, &psinfo, sizeof(psinfo) ) != sizeof(psinfo) )
    {
        close( fd );
        return 0L;      /* Can't read? */
    }
    close( fd );
        return (unsigned long long) (psinfo.pr_rssize * 1024L);

#else
    /* Unknown OS ----------------------------------------------- */
    return 0L;          /* Unsupported. */
#endif
}





/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
unsigned long long getCurrentRSS(void)
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
        return (unsigned long long) info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if ( task_info( mach_task_self( ), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount ) != KERN_SUCCESS )
        return 0L;      /* Can't access? */
        return (unsigned long long) info.resident_size;

//#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#elif defined(HAVE__PROC_MEMINFO)
    /* Linux ---------------------------------------------------- */
    unsigned long long rss = 0L;
    FILE* fp = NULL;
    if ( (fp = fopen( "/proc/self/statm", "r" )) == NULL )
            return (unsigned long long) 0L; /* Can't open? */
    if ( fscanf( fp, "%*s%llu", &rss ) != 1 )
    {
        fclose( fp );
        return 0L;      /* Can't read? */
    }
    fclose( fp );
        return rss * (unsigned long long) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (unsigned long long) 0L; /* Unsupported. */
#endif
}
