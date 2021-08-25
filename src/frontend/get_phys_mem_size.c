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

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/types.h>
#include <sys/param.h>
#if defined(BSD) && defined(HAVE_SYS_SYSCTL_H)
#include <sys/sysctl.h>
#endif

#else
#error "Unable to define getMemorySize( ) for an unknown OS."
#endif



/**
 * Returns the size of physical memory (RAM) in bytes.
 */
unsigned long long getMemorySize(void)
{
#if defined(HAVE__PROC_MEMINFO)
    /* Cygwin , Linux--------------------------------- */
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
    return mem_got * 1024L;

#elif defined(_WIN32)
    /* Windows. ------------------------------------------------- */
    /* Use new 64-bit MEMORYSTATUSEX, not old 32-bit MEMORYSTATUS */
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx( &status );
    return (unsigned long long) status.ullTotalPhys;

#elif defined(__unix__) || defined(__unix) || defined(unix) ||  \
        (defined(__APPLE__) && defined(__MACH__))
    /* UNIX variants. ------------------------------------------- */
    /* Prefer sysctl() over sysconf() except sysctl() HW_REALMEM and HW_PHYSMEM */

#if defined(CTL_HW) && (defined(HW_MEMSIZE) || defined(HW_PHYSMEM64))
    int mib[2];
    mib[0] = CTL_HW;
#if defined(HW_MEMSIZE)
    mib[1] = HW_MEMSIZE;            /* OSX. --------------------- */
#elif defined(HW_PHYSMEM64)
    mib[1] = HW_PHYSMEM64;          /* NetBSD, OpenBSD. --------- */
#endif
    int64_t size = 0;               /* 64-bit */
    size_t len = sizeof( size );
    if ( sysctl( mib, 2, &size, &len, NULL, 0 ) == 0 )
        return (unsigned long long) size;
    return 0L;          /* Failed? */

#elif defined(_SC_AIX_REALMEM)
    /* AIX. ----------------------------------------------------- */
    return (unsigned long long) sysconf(_SC_AIX_REALMEM) * (size_t) 1024L;

#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    /* FreeBSD, Linux, OpenBSD, and Solaris. -------------------- */
    return (unsigned long long) sysconf(_SC_PHYS_PAGES) *
            (unsigned long long) sysconf(_SC_PAGESIZE);

#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    /* Legacy. -------------------------------------------------- */
    return (unsigned long long) sysconf(_SC_PHYS_PAGES) *
            (unsigned long long) sysconf(_SC_PAGE_SIZE);

#elif defined(CTL_HW) && (defined(HW_PHYSMEM) || defined(HW_REALMEM)) && defined(HAVE_SYS_SYSCTL_H)
    /* DragonFly BSD, FreeBSD, NetBSD, OpenBSD, and OSX. -------- */
    int mib[2];
    mib[0] = CTL_HW;
#if defined(HW_REALMEM)
    mib[1] = HW_REALMEM;        /* FreeBSD. ----------------- */
#elif defined(HW_PYSMEM)
    mib[1] = HW_PHYSMEM;        /* Others. ------------------ */
#endif
    unsigned long long size = 0; /* 32-bit */
    size_t len = sizeof( size );
    if ( sysctl( mib, 2, &size, &len, NULL, 0 ) == 0 )
        return (unsigned long long) size;
    return 0L;          /* Failed? */
#endif /* sysctl and sysconf variants */

#else
    return 0L;          /* Unknown OS. */
#endif
}
