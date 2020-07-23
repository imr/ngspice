 /* Provide system information

   LINUX: /proc file system
   Windows: GlobalMemoryStatusEx, GetSystemInfo, GetVersionExA, RegQueryValueExA

   Authors: Holger Vogt,  Hendrik Vogt

 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/fteext.h"
#include "com_commands.h"

#ifdef _WIN32
#undef BOOLEAN
#include <windows.h>
#include <psapi.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ngspice/dstring.h"

/* system info */
typedef struct TSI {
    char *cpuModelName;
    char *osName;
    unsigned int numPhysicalProcessors;
    unsigned int numLogicalProcessors;
} TesSystemInfo;

/* Struture with info about system */
static TesSystemInfo system_info;

/* memory info */
struct sys_memory {
    unsigned long long size_m;  /* Total memory size */
    unsigned long long free_m;  /* Free memory */
    unsigned long long swap_t;  /* Swap total */
    unsigned long long swap_f;  /* Swap free */
};

static void fprintmem(FILE *stream, unsigned long long memory);
static void free_static_system_info(void);
static int get_sysmem(struct sys_memory *memall);
static void set_static_system_info(void);

#ifdef _WIN32
static inline void get_logical_processor_count(void);
static void get_os_info(void);
static void get_physical_processor_count(void);
static void get_processor_name(void);
#endif



/* Print the available system info */
void com_sysinfo(wordlist *wl)
{
    NG_IGNORE(wl);

    /* Invariant system data such as OS name */
    {
        /* Flag that have at least some system info */
        bool f_have_system_info = FALSE;

        static bool f_first_call = TRUE;
        if (f_first_call) {
            /* Obtain the system info when this function is called the
             * first time */
            set_static_system_info();

            /* Free the allocations on exit. Not really necessary since they
             * will be cleaned up then, but it may be useful when checking for
             * memory leaks */
            if (atexit(&free_static_system_info) != 0) {
                fprintf(cp_err,
                        "Unable to set handler to clean up system info.\n");
            }

            /* Mark that first-call init is done. Note that since the calls to
             * set_static_system_info() and atexit define sequence points, the
             * flag will not be set until after they complete, so the code is
             * safe for reentrant calls. */
            f_first_call = FALSE;
        }

        if (system_info.osName != (char *) NULL) {
            fprintf(cp_out, "\nOS: %s\n", system_info.osName);
            f_have_system_info = TRUE;
        }

        if (system_info.cpuModelName != (char *) NULL) {
            fprintf(cp_out, "CPU: %s\n", system_info.cpuModelName);
            f_have_system_info = TRUE;
        }

        if (system_info.numPhysicalProcessors > 0) {
            fprintf(cp_out, "Physical processors: %u, ",
                    system_info.numPhysicalProcessors);
            f_have_system_info = TRUE;
        }

        if (system_info.numLogicalProcessors > 0) {
            fprintf(cp_out, "Logical processors: %u\n",
                    system_info.numLogicalProcessors);
            f_have_system_info = TRUE;
        }

        /* Print something if no system info available */
        if (!f_have_system_info) {
            fprintf(cp_err, "No system info available!\n");
        }
    } /* end of block getting invariant system info */

    /* Get memory information */
    {
        struct sys_memory mem_t_act;
        if (get_sysmem(&mem_t_act) == 0) {
            /* get_sysmem returns bytes */
            fprintf(cp_out, "Total DRAM available = ");
            fprintmem(cp_out, mem_t_act.size_m);
            fprintf(cp_out, ".\n");

            fprintf(cp_out, "DRAM currently available = ");
            fprintmem(cp_out, mem_t_act.free_m);
            fprintf(cp_out, ".\n\n");
        }
        else {
            fprintf(cp_err, "Memory info is unavailable! \n");
        }
    }

    return;
} /* end of function com_sysinfo */



/* This function frees the buffers used to store system allocation strings */
static void free_static_system_info(void)
{
    tfree(system_info.cpuModelName);
    tfree(system_info.osName);
} /* end of fuction free_system_info */



/* Print to stream the given memory size in a human friendly format */
static void fprintmem(FILE *stream, unsigned long long memory)
{
    if (memory > 1048576) {
        fprintf(stream, "%8.6f MB", (double) memory /1048576.);
    }
    else if (memory > 1024) {
        fprintf(stream, "%5.3f kB", (double) memory / 1024.);
    }
    else {
        fprintf(stream, "%u bytes", (unsigned) memory);
    }
} /* end of funtion fprintmem */



/*** Get processor and memory information as appropriate for the system ***/
#ifdef HAVE__PROC_MEMINFO

/* Get memory information */
static int get_sysmem(struct sys_memory *memall)
{
    FILE *fp;
    char buffer[2048];
    size_t bytes_read;
    char *match;
    unsigned long mem_got;

    if ((fp = fopen("/proc/meminfo", "r")) == NULL) {
        perror("fopen(\"/proc/meminfo\")");
        return -1;
    }

    bytes_read = fread(buffer, 1, sizeof(buffer), fp);
    fclose(fp);
    if (bytes_read == 0 || bytes_read == sizeof(buffer))
        return -1;
    buffer[bytes_read] = '\0';

    /* Search for string "MemTotal" */
    match = strstr(buffer, "MemTotal");
    if (match == NULL) /* not found */
        return -1;
    sscanf(match, "MemTotal: %ld", &mem_got);
    memall->size_m = mem_got*1024; /* 1MB = 1024KB */
    /* Search for string "MemFree" */
    match = strstr(buffer, "MemFree");
    if (match == NULL) /* not found */
        return -1;
    sscanf(match, "MemFree: %ld", &mem_got);
    memall->free_m = mem_got*1024; /* 1MB = 1024KB */
    /* Search for string "SwapTotal" */
    match = strstr(buffer, "SwapTotal");
    if (match == NULL) /* not found */
        return -1;
    sscanf(match, "SwapTotal: %ld", &mem_got);
    memall->swap_t = mem_got*1024; /* 1MB = 1024KB */
    /* Search for string "SwapFree" */
    match = strstr(buffer, "SwapFree");
    if (match == NULL) /* not found */
        return -1;
    sscanf(match, "SwapFree: %ld", &mem_got);
    memall->swap_f = mem_got*1024; /* 1MB = 1024KB */
    return 0;
}


/* Return length of first line in a string */
static inline size_t getLineLength(const char *str)
{
    const char *p = str;

    while (*p  &&  (*p != '\n')) {
        p++;
    }

    return (size_t) (p - str);
}


/* Checks if number 'match' is found in a vector 'set' of size 'size'
   Returns 1 if yes, otherwise, 0 */
static int searchInSet(const int *set, unsigned size, int match)
{
    unsigned index;
    for (index = 0; index < size; index++)
        if (match == set[index])
            return 1;
    return 0;
}


/* Get system information */
static void set_static_system_info(void)
{
    FILE *file;

    /* Init to all information unailable */
    system_info.cpuModelName = (char *) NULL;
    system_info.osName = (char *) NULL;
    system_info.numLogicalProcessors = system_info.numPhysicalProcessors = 0;

    /* get kernel version string */
    file = fopen("/proc/version", "rb");
    if (file != NULL) {
        size_t size;

        /* read bytes and find end of file */
        for (size = 0; ; size++) {
            if (EOF == fgetc(file)) {
                break;
            }
        }

        system_info.osName = TMALLOC(char, size + 1);
        rewind(file);
        if (fread(system_info.osName, sizeof(char), size, file) != size) {
            (void) fprintf(cp_err, "Unable to read \"/proc/version\".\n");
            fclose(file);
            tfree(system_info.osName);
            return;
        }


        fclose(file);

        system_info.osName[size] = '\0';
    }

    /* get cpu information */
    file = fopen("/proc/cpuinfo", "rb");
    if (file != NULL) {
        size_t size;
        char *inStr;

        /* read bytes and find end of file */
        for (size = 0; ; size++) {
            if (EOF == fgetc(file)) {
                break;
            }
        }

        /* get complete string */
        inStr = TMALLOC(char, size+1);
        rewind(file);
        if (fread(inStr, sizeof(char), size, file) != size) {
            (void) fprintf(cp_err, "Unable to read \"/proc/cpuinfo\".\n");
            fclose(file);
            txfree(inStr);
            return;
        }
        inStr[size] = '\0';

        {
            const char *matchStr = "model name";
            /* pointer to first occurrence of model name*/
            const char *modelStr = strstr(inStr, matchStr);
            if (modelStr != NULL) {
                /* search for ':' */
                const char *modelPtr = strchr(modelStr, ':');
                if (modelPtr != NULL) {
                    /*length of string from ':' till end of line */
                    size_t numToEOL = getLineLength(modelPtr);
                    if (numToEOL > 2) {
                        /* skip ": "*/
                        numToEOL -= 2;
                        system_info.cpuModelName = TMALLOC(char, numToEOL+1);
                        memcpy(system_info.cpuModelName, modelPtr+2, numToEOL);
                        system_info.cpuModelName[numToEOL] = '\0';
                    }
                }
            }
        }

        {
            const char *matchStrProc = "processor";
            const char *matchStrPhys = "physical id";
            char *strPtr = inStr;
            unsigned numProcs = 0;
            int *physIDs;

            /* get number of logical processors */
            while ((strPtr = strstr(strPtr, matchStrProc)) != NULL) {
                // numProcs++;
                strPtr += strlen(matchStrProc);
                if (isblank_c(*strPtr)) numProcs++;
            }
            system_info.numLogicalProcessors = numProcs;
            physIDs = TMALLOC(int, numProcs);

            /* get number of physical CPUs */
            numProcs = 0;
            strPtr = inStr;
            while ((strPtr = strstr(strPtr, matchStrProc)) != NULL) {

                /* search for first occurrence of physical id */
                strPtr = strstr(strPtr, matchStrPhys);
                if (strPtr != NULL) {
                    /* go to ';' */
                    strPtr = strchr(strPtr, ':');
                    if (strPtr != NULL) {
                        int buffer = 0;
                        /* skip ": " */
                        strPtr += 2;
                        /* get number */
                        sscanf(strPtr, "%d", &buffer);
                        /* If this  physical id is unique,
                           we have another physically available CPU */
                        if (searchInSet(physIDs, numProcs, buffer) == 0) {
                            physIDs[numProcs] = buffer;
                            numProcs++;
                        }
                    }
                    else {
                        break;
                    }
                }
                else {
                    break;
                }
            }
            system_info.numPhysicalProcessors = numProcs;
            tfree(physIDs);
        }

        /* another test to get number of logical processors
         * if (system_info.numLogicalProcessors == 0) {
         *     char *token;
         *     char *cpustr = copy(inStr);
         *     while (cpustr && !*cpustr)
         *         if (cieq(gettok(&cpustr), "processor")) {
         *             gettok(&cpustr);
         *             token = gettok(&cpustr);
         *         }
         *
         *     system_info.numLogicalProcessors = atoi(token) + 1;
         *     tfree(cpustr);
         * }
         */

        txfree(inStr);
        fclose(file);
    } /* end of case that file was opened OK */

    return;
} /* end of function set_static_system_info */

#elif defined(__APPLE__) && defined(__MACH__)
/* Get memory information */
static int get_sysmem(struct sys_memory *memall)
{
    fprintf(stderr, "System memory info is not available\n");
    return -1;
}
/* Get system information */
static void set_static_system_info(void)
{
}

#elif defined(_WIN32)

/* Get memory information */
static int get_sysmem(struct sys_memory *memall)
{
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&ms) == FALSE) {
        return -1;
    }
    memall->size_m = ms.ullTotalPhys;
    memall->free_m = ms.ullAvailPhys;
    memall->swap_t = ms.ullTotalPageFile;
    memall->swap_f = ms.ullAvailPageFile;
    return 0;
} /* end of function get_sysmem */



/* This function gets system information about the version of Windows and
 * the number processors available, and save this information in the static
 * TesSystemInfo structure. If an item cannot be obtained, it is set to
 * 0/NULL. This allows callers to check for valid data since neither of these
 * values are valid */
static void set_static_system_info(void)
{
    get_processor_name(); /* name of processor */
    get_os_info(); /* name of OS with build and service pack, if any */
    get_logical_processor_count(); /* Get number of logical cores */
    get_physical_processor_count(); /* # hardware components */
    return;
} /* end of function set_static_system_info */



/* Copy data at HKLM/sz_subkey/sz_val_name to an allocated buffer that is
 * 1 byte longer and always null-termianted, possibly with 2 nulls
 *
 * Parameters
 * sz_subkey: Subkey string
 * sz_val_name: Name of value to get
 * p_ds: Address of dstring to receive data
 *
 * Return codes
 * 0: Data obtained OK
 * -1: Data not obtained.
 */
static int registry_value_to_ds(const char *sz_subkey,
        const char *sz_val_name, DSTRING *p_ds)
{
    int xrc = 0;
    DWORD n_byte_data = 0;
    HKEY hk;
    bool f_key_open = FALSE;

    /* Opwn the key with the processor details */
    {
        DWORD rc;
        if ((rc = RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                sz_subkey, 0, KEY_READ, &hk)) != ERROR_SUCCESS) {
            fprintf(cp_err,
                    "Unable to open key for registry data \"%s\". "
                    "System code = %lu\n",
                    sz_subkey, rc);
            xrc = -1;
            goto EXITPOINT;
        }
    }
    f_key_open = TRUE;

    /* Get size of the name string. Strings in the registry need not be
     * null-terminated, but if they are, the null is included in the
     * size. */
    {
        DWORD rc;
        if ((rc = RegQueryValueExA(hk, sz_val_name,
                0, 0, NULL, &n_byte_data)) != ERROR_SUCCESS) {
            fprintf(cp_err,
                    "Unable to get the size of value for \"%s\". "
                    "System code = %lu\n",
                    sz_val_name, rc);
            xrc = -1;
            goto EXITPOINT;
        }
    }

    /* Ensure dstring buffer is large enough for the data + 1 byte to add
     * a null to the end */
    {
        size_t n_byte_reserve = (size_t) n_byte_data + 1;
        if (ds_reserve(p_ds, n_byte_reserve) != 0) {
            (void) fprintf(cp_err,
                    "Unable to reserve a buffer of %u bytes for data.\n",
                    n_byte_reserve);
            xrc = -1;
            goto EXITPOINT;
        }
    }

    /* Retrieve the value using the dstring buffer to receive it */
    {
        DWORD rc;
        char *p_buf = ds_get_buf(p_ds);
        if ((rc = RegQueryValueExA(hk, sz_val_name, 0, 0,
                (LPBYTE) p_buf, &n_byte_data)) != ERROR_SUCCESS) {
            (void) fprintf(cp_err,
                    "Unable to get the value for \"%s\". "
                    "System code = %lu\n",
                    sz_val_name, rc);
            xrc = -1;
            goto EXITPOINT;
        }
    }

    /* Set the dstring length */
    (void) ds_set_length(p_ds, n_byte_data);

EXITPOINT:
    /* Indicate error if failure */
    if (xrc != 0) {
        ds_clear(p_ds);
    }

    if (f_key_open) { /* close key if opened */
        RegCloseKey(hk);
    }

    return xrc;
} /* end of function registry_value_to_ds */



/* Gets the name of the processor from the registry and sets field
 * cpuModelName in system_info. On failure, the field is set to NULL */
static void get_processor_name(void)
{
    DS_CREATE(ds, 200);

    system_info.cpuModelName = NULL; /* init in case of failure */
    if (registry_value_to_ds(
            "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            "ProcessorNameString",
            &ds) != 0) {
        (void) fprintf(cp_err,
                "Unable to get processor name data from the registry.\n");
        return;
    }

    /* Step past any leading blanks and copy name to cpuModelName */
    {
        const char *proc_name = ds_get_buf(&ds);

        while (*proc_name == ' ') {
            ++proc_name;
        } /* end of loop finding first non-blank of processor name */

        /* Make a copy of the string at cpuModelName field of system_info */
        system_info.cpuModelName = copy(proc_name);
    }

    ds_free(&ds); /* Free resources */

    return;
} /* end of function get_processor_name */



/* This function gets the release details to distinguish between
 * 2016 and 2019 servers. If necessary, it can be extended to return
 * codes for other servers in the future.
 *
 * See
 * https://techcommunity.microsoft.com/t5/Windows-Server-Insiders/Windows-Server-2019-version-info/m-p/234472
 *
 * Return codes
 * -1: Failure
 * +1: 2016 server
 * +2: 2019 server (probably)
 *
 * Remarks
 * Calling this function alone is not sufficient to identify a server.
 * Rather it should be called given that a server OS is present to identify
 * the serer version.
 */
static int get_server_id(void)
{
    DS_CREATE(ds, 25);

    if (registry_value_to_ds(
            "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
            "ReleaseId",
            &ds) != 0) {
        (void) fprintf(cp_err,
                "Unable to get release ID data from the registry.\n");
        return -1;
    }

    int id_code = -1; /* Set to failure until found */
    /* Convert the release ID to a number */
    {
        char *p_end;
        errno = 0;
        const char *p_buf = ds_get_buf(&ds);
        unsigned long id_val = strtoul(p_buf, &p_end, 10);
        if (errno || *p_end != '\0') {
            fprintf(cp_err,
                "Unable to convert \"%s\" to a release ID number.\n",
                p_buf);
            goto EXITPOINT;
        }

        if (id_val == 1607ul) { /* code for 2016 server */
            id_code = 1;
        }
        else if (id_code > 1607ul) { /* Probably 2019 server */
            id_code = 2;
        }
        /* Else unknown ID */
    }

EXITPOINT:
    ds_free(&ds); /* Free resources */

    return id_code;
} /* end of function get_server_id */



/* This function creates a name of the form <OS name> ' ' <Service pack>,
 * allocates a buffer for it, and stores it in system_info.osNname. On
 * failure an error is reported and the string is set to NULL.
 *
 * Remarks
 * Getting the version has been complicated greatly in later versions of
 * Windows. A good discussion of the issue can be found at
 * https://stackoverflow.com/questions/47581146/getting-os-build-version-from-win32-api-c
 *
 * First, the function GetVersionEx() has been deprecated, so the
 * straightforward call to retrieve the version is not the recommended
 * approach any longer and will output a message to this effect during
 * compilation. Also, it may be removed at some later time. Even if it is
 * called, since Windows 8.0, the value returned depends not on the version
 * of the OS, but the manifested version of the calling program.
 *
 * As an alternative function RtlGetVersion() always returns version info
 * the same version as GetVersionEx() prior to Windows 8.1, and it
 * is not deprecated. Unfortunately, the simple solution is made less
 * simple because the header providing a prototype for RtlGetVersion()
 * is part of the Windows DDK and the function is not directly exposed
 * by ntdll.lib. Also, the DDK only works with UTF-16, so the name string
 * must be converted.
 *
 * The following link has a table showing how to determine the all operating
 * systems from Windows 2000 through Windows 10/Windows Server 2016.
 * https://web.archive.org/web/20190501082653/https://docs.microsoft.com/en-us/windows/desktop/api/winnt/ns-winnt-_osversioninfoexa
 *
 *     OS                   ver  Other OSV=OSVERSIONINFOEX
 * Windows Server 2016     10.0 OSV.wProductType != VER_NT_WORKSTATION
 * Windows 10              10.0 OSV.wProductType == VER_NT_WORKSTATION
 * Windows Server 2008      6.0 OSV.wProductType != VER_NT_WORKSTATION
 * Windows Vista            6.0 OSV.wProductType == VER_NT_WORKSTATION
 * Windows Server 2008 R2   6.1 OSV.wProductType != VER_NT_WORKSTATION
 * Windows 7                6.1 OSV.wProductType == VER_NT_WORKSTATION
 * Windows Server 2012      6.2 OSV.wProductType != VER_NT_WORKSTATION
 * Windows 8                6.2 OSV.wProductType == VER_NT_WORKSTATION
 * Windows Server 2012 R2   6.3 OSV.wProductType != VER_NT_WORKSTATION
 * Windows 8.1              6.3 OSV.wProductType == VER_NT_WORKSTATION
 * Windows 2000             5.0 Not applicable
 * Windows XP               5.1 Not applicable
 * Windows Home Server      5.2 OSV.wSuiteMask & VER_SUITE_WH_SERVER
 * Windows XP Professional
 * x64 Edition              5.2 (OSV.wProductType == VER_NT_WORKSTATION) &&
 *                              (SYSTEM_INFO.wProcessorArchitecture ==
 *                                  PROESSOR_ARCHITECTURE_AMD64)
 * Windows Server 2003      5.2 GetSystemMetrics(SM_SERVERR2) == 0
 * Windows Server 2003 R2   5.2 GetSystemMetrics(SM_SERVERR2) != 0

 * Information on distinguishing between Windows Server 2016 and 2019 does
 * not appear to have been provided as of early 2019:
 * https://stackoverflow.com/questions/53393150/c-how-to-detect-windows-server-2019
 * Hopefully this issue will be resolved in the future.
 */
static void get_os_info(void)
{
    OSVERSIONINFOEXW ver_info;

    /* the name of the OS. Init to prevent compiler warning */
    const char *sz_os_name = NULL;

    /* Load library containing RtlGetVersion()  */
    HMODULE lib = LoadLibraryExW(L"ntdll.dll", NULL, 0);
    if (lib == (HMODULE) NULL) { /* Not loaded OK */
        (void) fprintf(cp_err,
                "Unable to load ntdll.dll. "
                "System code = %lu\n",
                (unsigned long) GetLastError());
        system_info.osName = (char *) NULL;
        return;
    }

    /* Locate RtlGetVersion() */
    FARPROC p_get_ver = GetProcAddress(lib, "RtlGetVersion");
    if (p_get_ver == (FARPROC) NULL) { /* Did not get function addr OK */
        (void) fprintf(cp_err,
                "Unable to locate function RtlGetVersion. "
                "System code = %lu\n",
                (unsigned long) GetLastError());
        system_info.osName = (char *) NULL;
        return;
    }

    /* Get version info. RtlGetVersion cannot fail. */
    ver_info.dwOSVersionInfoSize = sizeof(ver_info);
    (void) ((DWORD (WINAPI *)(OSVERSIONINFOEXW *)) p_get_ver)(
            &ver_info);

    switch (ver_info.dwMajorVersion) {
    case 10: {
        static const char OS_srvr[] = "Windows Server 2016/2019/other";
        static const char OS_10[] = "Windows 10";
        static const char OS_2016[] = "Windows Server 2016";
        static const char OS_2019[] = "Windows Server 2019";
        static const char * const p_str[] = {
            OS_srvr, OS_10, OS_2016, OS_2019
        };

        if (ver_info.dwMinorVersion != 0) { /* only know 10.0 */
            system_info.osName = (char *) NULL;
            return;
        }
        sz_os_name = p_str[ver_info.wProductType == VER_NT_WORKSTATION ?
                1 : get_server_id() + 1];
        break;
    }
    case 6: {
        static const char OS_2008[] = "Windows Server 2008";
        static const char OS_vista[] = "Windows Vista";
        static const char OS_2008R2[] = "Windows Server 2008 R2";
        static const char OS_7[] = "Windows 7";
        static const char OS_2012[] = "Windows Server 2012";
        static const char OS_8[] = "Windows 8";
        static const char OS_2012R2[] = "Windows Server 2012 R2";
        static const char OS_8_1[] = "Windows 8.1";
        static const char * const p_str[] = {
            OS_2008, OS_vista,
            OS_2008R2, OS_7,
            OS_2012, OS_8,
            OS_2012R2, OS_8_1
        };
        if (ver_info.dwMinorVersion > 3) { /* know 6.0 through 6.3 */
            (void) fprintf(cp_err, "Unknown Windows version 6.%lu. ",
                    (unsigned long) ver_info.dwMinorVersion);
            system_info.osName = (char *) NULL;
            return;
        }
        sz_os_name = p_str[2 * ver_info.dwMinorVersion +
                ver_info.wProductType == VER_NT_WORKSTATION];
        break;
    }
    case 5: { /* an assortment of other conditions must be checked */

        switch (ver_info.dwMinorVersion) { /* filter by minor verson */
        case 0: {
            static const char OS_2k[] = "Windows 2000";
            sz_os_name = OS_2k;
            break;
        }
        case 1: {
            static const char OS_xp[] = "Windows XP";
            sz_os_name = OS_xp;
            break;
        }
        case 2:
            if (ver_info.wSuiteMask & VER_SUITE_WH_SERVER) {
                static const char OS_home_server[] = "Windows Home Server";
                sz_os_name = OS_home_server;
            }
            else if (ver_info.wProductType == VER_NT_WORKSTATION) {
                    SYSTEM_INFO si;
                    GetSystemInfo(&si);
                    if (si.wProcessorArchitecture ==
                            PROCESSOR_ARCHITECTURE_AMD64) {
                        static const char OS_xp64[] =
                                "Windows XP Professional x64 Edition";
                        sz_os_name = OS_xp64;
                    }
                }
            else { /* Server 2003 or 2003 R2 */
                static const char OS_2003R2[] = "Windows Server 2003 R2";
                static const char OS_2003[] = "Windows Server 2003";
                static const char * const p_str[] = {OS_2003R2, OS_2003};
                sz_os_name = p_str[!GetSystemMetrics(SM_SERVERR2)];
            }
            break;
        default:
            (void) fprintf(cp_err, "Unknown Windows version 5.%lu. ",
                    (unsigned long) ver_info.dwMinorVersion);
            system_info.osName = (char *) NULL;
            return;
        } /* end of switch over minor version for major version 5 */
        break;
    }
    case 4:
        switch (ver_info.dwMinorVersion) {
        case 0: {
            static const char OS_95[] = "Windows 95";
            static const char OS_nt4[] = "Windows NT 4.0";
            static const char * const p_str[] = {OS_95, OS_nt4};
            sz_os_name = p_str[ver_info.wProductType == VER_NT_WORKSTATION];
        }
        case 10: {
            static const char OS_98[] = "Windows 98";
            sz_os_name = OS_98;
            break;
        }
        case 90: {
            static const char OS_me[] = "Windows ME";
            sz_os_name = OS_me;
            break;
        }
        default:
            (void) fprintf(cp_err, "Unknown Windows version 4.%lu. ",
                    (unsigned long) ver_info.dwMinorVersion);
            system_info.osName = (char *) NULL;
            return;
        } /* end of switch over minor version for major version 4 */
    default:
        (void) fprintf(cp_err, "Unknown Windows version %lu.%lu. ",
                (unsigned long) ver_info.dwMajorVersion,
                (unsigned long) ver_info.dwMinorVersion);
        system_info.osName = (char *) NULL;
        return;
    }/* end of switch over major version */

    /* Have the base version name. Now must add service pack, if any */
    if (ver_info.wServicePackMajor == 0) { /* no service pack */
        system_info.osName = tprintf("%s, Build %lu",
                sz_os_name, (unsigned long) ver_info.dwBuildNumber);
    }
    else if (ver_info.wServicePackMinor == 0) { /* major # only */
        system_info.osName = tprintf("%s, Build %lu, Service Pack %u",
                sz_os_name, (unsigned long) ver_info.dwBuildNumber,
                (unsigned) ver_info.wServicePackMajor);
    }
    else { /* service pack has major and minor versions */
        system_info.osName = tprintf("%s, Build %lu, Service Pack %u.%u",
                sz_os_name, (unsigned long) ver_info.dwBuildNumber,
                (unsigned) ver_info.wServicePackMajor,
                (unsigned) ver_info.wServicePackMinor);
    }

    return;
} /* end of function get_os_info */



/* This function sets the number of processors field in system_info */
 static inline void get_logical_processor_count(void)
 {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    system_info.numLogicalProcessors = si.dwNumberOfProcessors;
} /* end of function get_logical_processor_count */



 /* This funtion sets the field storing the number of physical processors
  * in system_info */
typedef BOOL (WINAPI *glp_t)(LOGICAL_PROCESSOR_RELATIONSHIP,
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, PDWORD);
static void get_physical_processor_count(void)
{
    DWORD n_byte_buf = 0;
    system_info.numPhysicalProcessors = 0; /* Init to 0 until found */

    /* Get a handle to the DLL with the required function. Since the
     * functdion GetModuleHandleW() is in kernel32.dll, it is safe to
     * assume that kernel32.dll is already loaded. Not using
     * LoadLibraryExW() simplifies error handling. */
    HMODULE lib = GetModuleHandleW(L"kernel32.dll");
    if (lib == (HMODULE) NULL) { /* Handle not obtained */
        (void) fprintf(cp_err,
                "Unable to obtain a handle to kernel32.dll. "
                "System code = %lu\n",
                (unsigned long) GetLastError());
        return;
    }

    /* Locate GetLogicalProcessorInformationEx(). It must be
     * dynamically loaded since it is only present in
     * Windows 7/Server 2008 R2 and later OS versions */
    FARPROC p_glp = GetProcAddress(lib,
            "GetLogicalProcessorInformationEx");
    if (p_glp == (FARPROC) NULL) { /* Did not get function addr OK */
        (void) fprintf(cp_err,
                "Unable to locate function "
                "GetLogicalProcessorInformationEx. "
                "System code = %lu\n",
                (unsigned long) GetLastError());
        return;
    }

    /* Find requried size. Should return FALSE/ERROR_INSUFFICIENT_BUFFER if
     * working properly */
    if (((glp_t) (*p_glp))(RelationProcessorPackage,
            NULL, &n_byte_buf) != 0) {
        fprintf(cp_err,
                "Unexpected error getting logical processor buffer size.\n");
        return;
    }

    {
        DWORD rc;
        if ((rc = GetLastError()) != ERROR_INSUFFICIENT_BUFFER) {
            fprintf(cp_err,
                    "Unable to get the logical processor bufer size. "
                    "System code = %lu.\n",
                    (unsigned long) rc);
            return;
        }
    }

    /* Allocate buffer to get the info */
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX * const buf =
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)TMALLOC(char, n_byte_buf);
    if (buf == (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *) NULL) {
        fprintf(cp_err,
                "Unable to allocate a buffer of %lu bytes "
                "for logical processor information.\n",
                n_byte_buf);
        return;
    }

    /* Try again with a buffer and the size obtained before */
    {
        DWORD rc;
        if ((rc = ((glp_t) (*p_glp))(RelationProcessorPackage,
                buf, &n_byte_buf)) == 0) {
            fprintf(cp_err,
                    "Unable to get the logical processor info. "
                    "System code = %lu.\n",
                    (unsigned long) rc);
            return;
        }
    }

    /* Count the number of processor packages */
    {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX * p_buf_cur = buf;
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX * const p_buf_end =
                (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)
                ((char *) buf + n_byte_buf);
        unsigned int n_processor_package = 0;
        for ( ; p_buf_cur < p_buf_end;
                p_buf_cur = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)
                        ((char *) p_buf_cur + p_buf_cur->Size)) {
            ++n_processor_package;
        }
        system_info.numPhysicalProcessors = n_processor_package;
    }

    return;
} /* end of function get_physical_processor_count */



#else /* no Windows OS, no proc info file system */
static int get_sysmem(struct sys_memory *memall)
{
    return -1; // Return N/A
}

void set_static_system_info(void)
{
    /* Set to no data available */
    system_info.osName = (char *) NULL;
    system_info.cpuModelName = (char *) NULL;
    system_info.numPhysicalProcessors = 0;
    system_info.numLogicalProcessors = 0;
    return;
} /* end of function set_static_system_info */

#endif
