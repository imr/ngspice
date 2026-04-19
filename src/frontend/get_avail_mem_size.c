/*
 * Author:  Holger Vogt
 * License: 3-clause BSD License
 * 
 */

#include "ngspice/ngspice.h"
#include "resource.h"

#if defined(_WIN32)
#include <windows.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/types.h>
#include <sys/param.h>
#if defined(BSD) && defined(HAVE_SYS_SYSCTL_H)
#include <sys/sysctl.h>
#endif
#if defined(__APPLE__) && defined(__MACH__)
#import <mach/mach.h>
#import <mach/mach_host.h>
#endif
#else
#error "Unable to define getMemorySize( ) for an unknown OS."
#endif


/**
 * Returns the size of available memory (RAM) in bytes.
 */
unsigned long long getAvailableMemorySize(void)
{
#if defined(HAVE__PROC_MEMINFO)
    /* Cygwin , Linux--------------------------------- */
    /* Search for string "MemFree" */
    FILE *fp;
    char buffer[2048];
    size_t bytes_read;
    char *match;
    unsigned long long mem_got;

    if ((fp = fopen("/proc/meminfo", "r")) == NULL) {
        perror("fopen(\"/proc/meminfo\")");
        return 0L;
    }

    bytes_read = fread(buffer, 1, sizeof(buffer), fp);
    fclose(fp);
    if (bytes_read == 0 || bytes_read == sizeof(buffer))
        return 0L;
    buffer[bytes_read] = '\0';
    match = strstr(buffer, "MemFree");
    if (match == NULL) /* not found */
        return 0L;
    sscanf(match, "MemFree: %llu", &mem_got);
    return mem_got * 1024L;

#elif defined(_WIN32)
    /* Windows. ------------------------------------------------- */
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx( &status );
    return status.ullAvailPhys;

#elif defined(__APPLE__) && defined(__MACH__)

#include <stdio.h>
#include <stdlib.h>
#include <sys/sysctl.h>
#include <mach/mach.h>

    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

    vm_statistics64_data_t vmstats;

    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                    (host_info64_t)&vmstats, &count) == KERN_SUCCESS) {
        // Calculate available memory (free + inactive)
        const int page_size = PAGE_SIZE;
        uint64_t free_memory = (uint64_t)vmstats.free_count * page_size;
        uint64_t inactive_memory = (uint64_t)vmstats.inactive_count * page_size;
        uint64_t available_memory = free_memory + inactive_memory;

        return available_memory;
    }
    else
        return 0;

#elif defined(__unix__) || defined(__unix) || defined(unix)
    /* Linux/UNIX variants. ------------------------------------------- */
    /* Prefer sysctl() over sysconf() except sysctl() HW_REALMEM and HW_PHYSMEM */

#if defined(CTL_HW) && (defined(HW_MEMSIZE) || defined(HW_PHYSMEM64)) && defined(HAVE_SYS_SYSCTL_H)
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
        return (size_t)size;
    return 0L;          /* Failed? */

#elif defined(_SC_AIX_REALMEM)
    /* AIX. ----------------------------------------------------- */
    return (size_t)sysconf( _SC_AIX_REALMEM ) * (size_t)1024L;

#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    /* FreeBSD, Linux, OpenBSD, and Solaris. -------------------- */
    return (size_t)sysconf( _SC_PHYS_PAGES ) *
        (size_t)sysconf( _SC_PAGESIZE );

#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    /* Legacy. -------------------------------------------------- */
    return (size_t)sysconf( _SC_PHYS_PAGES ) *
        (size_t)sysconf( _SC_PAGE_SIZE );

#elif defined(CTL_HW) && (defined(HW_PHYSMEM) || defined(HW_REALMEM)) && defined(HAVE_SYS_SYSCTL_H)
    /* DragonFly BSD, FreeBSD, NetBSD, OpenBSD, and OSX. -------- */
    int mib[2];
    mib[0] = CTL_HW;
#if defined(HW_REALMEM)
    mib[1] = HW_REALMEM;        /* FreeBSD. ----------------- */
#elif defined(HW_PYSMEM)
    mib[1] = HW_PHYSMEM;        /* Others. ------------------ */
#endif
    unsigned int size = 0;      /* 32-bit */
    size_t len = sizeof( size );
    if ( sysctl( mib, 2, &size, &len, NULL, 0 ) == 0 )
        return (size_t)size;
    return 0L;          /* Failed? */
#endif /* sysctl and sysconf variants */

#else
    return 0L;          /* Unknown OS. */
#endif
}
