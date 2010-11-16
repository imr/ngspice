 /* Provide system information
   
   LINUX: /proc file system
   Windows: GlobalMemoryStatusEx, GetSystemInfo, GetVersionExA, RegQueryValueExA

   Authors: Holger Vogt,  Hendrik Vogt
   
   $Id$
 */
 
#include "config.h"
#include "ngspice.h" 
#include "cpdefs.h"
#include "fteext.h"
#include "com_commands.h"

/* We might compile for Windows, but only as a console application (e.g. tcl) */
#if defined(HAS_WINDOWS) || defined(__MINGW32__) || defined(_MSC_VER)
#define HAVE_WIN32
#endif

#ifdef HAVE_WIN32

#define WIN32_LEAN_AND_MEAN
 
#ifdef __MINGW32__  /* access to GlobalMemoryStatusEx in winbase.h:1558 */
#define WINVER 0x0500
#endif

#undef BOOLEAN
#include "windows.h"
#if defined(__MINGW32__) || (_MSC_VER > 1200) /* Exclude VC++ 6.0 from using the psapi */
#include <psapi.h>
#endif
#endif

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define tInt int
#define TesError int
#define TES_FAIL 1
#define TES_SUCCESS 0
#define TES_INVALID_PARAMS 1

/* system info */
typedef struct TSI {
	char* cpuModelName;
	int numPhysicalProcessors;
	int numLogicalProcessors;
	char* osName;
} TesSystemInfo;

/* memory info */
struct sys_memory {
   long long size_m;  /* Total memory size */
   long long free_m;  /* Free memory */
   long long swap_t;  /* Swap total */
   long long swap_f;  /* Swap free */
};

static struct sys_memory mem_t_act; 

TesError tesCreateSystemInfo(TesSystemInfo *info);
static size_t get_sysmem(struct sys_memory *memall);

/* Print to stream the given memory size in a human friendly format */
static void
fprintmem(FILE* stream, unsigned long long memory) {
    if (memory > 1048576)
      fprintf(stream, "%8.6f MB", memory/1048576.);
    else if (memory > 1024) 
      fprintf(stream, "%5.3f kB", memory/1024.);
    else
      fprintf(stream, "%lu bytes", (unsigned long)memory);
}


static void tesFreeSystemInfo(TesSystemInfo *info) {
	if(info != NULL) {
		free(info->cpuModelName);
		free(info->osName);
	}
}

/* print system info */
void com_sysinfo(wordlist *wl)
{
   int errorcode;
   TesSystemInfo* info;

   NG_IGNORE(wl);

   info = TMALLOC(TesSystemInfo, 1);

   errorcode = tesCreateSystemInfo(info);   
   if (errorcode)
      fprintf(cp_err, "No system info available! \n");
   else {
      fprintf(cp_out, "\nOS: %s\n", info->osName);
      fprintf(cp_out, "CPU: %s\n", info->cpuModelName);
      if (info->numPhysicalProcessors > 0) 
         fprintf(cp_out, "Physical processors: %d, ", info->numPhysicalProcessors);
      fprintf(cp_out, "Logical processors: %d\n", 
         info->numLogicalProcessors);
   }
#if defined(HAVE_WIN32) || defined(HAVE__PROC_MEMINFO) 	
	
	get_sysmem(&mem_t_act);
	
        /* get_sysmem returns bytes */
	fprintf(cp_out, "Total DRAM available = ");
	fprintmem(cp_out, mem_t_act.size_m);
	fprintf(cp_out, ".\n");
	
	fprintf(cp_out, "DRAM currently available = ");
	fprintmem(cp_out, mem_t_act.free_m);
	fprintf(cp_out, ".\n\n");

#endif

   tesFreeSystemInfo(info);
   tfree(info);
}


#ifdef HAVE__PROC_MEMINFO 

/* Get memory information */
static size_t get_sysmem(struct sys_memory *memall) {
   FILE *fp;
   char buffer[2048];
   size_t bytes_read;
   char *match;
   unsigned long mem_got;

   if((fp = fopen("/proc/meminfo", "r")) == NULL) {
      perror("fopen(\"/proc/meminfo\")");
      return 0;
   }
   
   bytes_read = fread (buffer, 1, sizeof (buffer), fp);
   fclose (fp);
   if (bytes_read == 0 || bytes_read == sizeof (buffer))
      return 0;
   buffer[bytes_read] = '\0';

   /* Search for string "MemTotal" */
   match = strstr (buffer, "MemTotal");
   if (match == NULL) /* not found */
      return 0;
   sscanf (match, "MemTotal: %ld", &mem_got);
   memall->size_m = mem_got*1024; /* 1MB = 1024KB */
   /* Search for string "MemFree" */
   match = strstr (buffer, "MemFree");
   if (match == NULL) /* not found */
      return 0;
   sscanf (match, "MemFree: %ld", &mem_got);
   memall->free_m = mem_got*1024; /* 1MB = 1024KB */  
   /* Search for string "SwapTotal" */
   match = strstr (buffer, "SwapTotal");
   if (match == NULL) /* not found */
      return 0;
   sscanf (match, "SwapTotal: %ld", &mem_got);
   memall->swap_t = mem_got*1024; /* 1MB = 1024KB */
   /* Search for string "SwapFree" */
   match = strstr (buffer, "SwapFree");
   if (match == NULL) /* not found */
      return 0;
   sscanf (match, "SwapFree: %ld", &mem_got);
   memall->swap_f = mem_got*1024; /* 1MB = 1024KB */
   return 1;     
}



/* Return length of first line in a string */
static tInt getLineLength(const char *str) {
	tInt length = strlen(str);
	char c = str[0];
	tInt index = 0;
	
	while((c != '\n') && (index < length)) {
		index++;
		c = str[index];
	}
	return index;
}

/* Checks if number 'match' is found in a vector 'set' of size 'size'
   Returns 1 if yes, otherwise, 0 */
static tInt searchInSet(const tInt *set, tInt size, tInt match) {
	tInt index;
	for(index = 0; index < size; index++) {
		if(match == set[index]) {
			return 1;
		}
	}
	return 0;
}

/* Get system information */
TesError tesCreateSystemInfo(TesSystemInfo *info) {
	FILE *file;
	TesError error = TES_SUCCESS;
	
	if(info == NULL)
		return TES_INVALID_PARAMS;
	info->cpuModelName = NULL;
	info->osName = NULL;
	info->numLogicalProcessors = info->numPhysicalProcessors = 0;
	
	/* get kernel version string */
	file = fopen("/proc/version", "rb");
	if(file != NULL) {	
		tInt size = 0;
		char buf;
		
		/* read bytes and find end of file */
		buf = fgetc(file);	
		while(buf != EOF) {
			size++;
			buf = fgetc(file);
		}

		info->osName = (char*) malloc((size) * sizeof(char));
		rewind(file);
		fread(info->osName, sizeof(char), size, file);
		fclose(file);
		
		info->osName[size-1] = '\0';
	}
	else {
		error = TES_FAIL;
	}
	
	/* get cpu information */
   file = fopen("/proc/cpuinfo", "rb");
	if(file != NULL) {	
		tInt size = 0;
		char buf, *inStr;
		
		/* read bytes and find end of file */
		buf = fgetc(file);	
		while(buf != EOF) {
			size++;
			buf = fgetc(file);
		}
		/* get complete string */
		inStr = (char*) malloc((size+1) * sizeof(char));
		rewind(file);
		fread(inStr, sizeof(char), size, file); 
		inStr[size] = '\0';
		
		{	
			const char *matchStr = "model name";	
			/* pointer to first occurrence of model name*/		
			const char *modelStr = strstr(inStr, matchStr);
			if(modelStr != NULL) {
            /* search for ':' */
				const char *modelPtr = strchr(modelStr, ':');
				if(modelPtr != NULL) {
               /*length of string from ':' till end of line */
					tInt numToEOL = getLineLength(modelPtr);
					if(numToEOL > 2) {
                  /* skip ": "*/
						numToEOL-=2;
						info->cpuModelName = (char*) malloc(numToEOL+1);
						memcpy(info->cpuModelName, modelPtr+2, numToEOL);
						info->cpuModelName[numToEOL] = '\0';					
					} 
				}
				else {
					error = TES_FAIL;
				}
			}
			else {
				error = TES_FAIL;
			}
		}
		{
			const char *matchStrProc = "processor";
			const char *matchStrPhys = "physical id";
			char *strPtr = inStr;
			tInt numProcs = 0;
			tInt *physIDs;
			
			/* get number of logical processors */
			while((strPtr = strstr(strPtr, matchStrProc)) != NULL) {
//				numProcs++;
				strPtr += strlen(matchStrProc);
				if (isblank(*strPtr)) numProcs++;
			}
			info->numLogicalProcessors = numProcs;
			physIDs = (tInt*) malloc(numProcs * sizeof(tInt));
						
			/* get number of physical CPUs */
			numProcs = 0;
			strPtr = inStr;
			while((strPtr = strstr(strPtr, matchStrProc)) != NULL) {
				
				/* search for first occurrence of physical id */
				strPtr = strstr(strPtr, matchStrPhys);
				if(strPtr != NULL) {
					/* go to ';' */
					strPtr = strchr(strPtr, ':');
					if(strPtr != NULL) {
						tInt buffer = 0;
                  /* skip ": " */
						strPtr += 2;
						/* get number */
						sscanf(strPtr, "%d", &buffer);
						/* If this  physical id is unique,
						   we have another physically available CPU */
						if(searchInSet(physIDs, numProcs, buffer) == 0) {
							physIDs[numProcs] = buffer;
							numProcs++;
						}
					}
					else {
//						error = TES_FAIL;
						break;
					}
				}
				else {
//					error = TES_FAIL;
					break;
				}
			}	
			info->numPhysicalProcessors = numProcs;
			free(physIDs);		
		}

      /* another test to get number of logical processors 
      if (info->numLogicalProcessors == 0) {
         char* token;
         char* cpustr = copy(inStr);
         while ((cpustr) && !*cpustr)
            if (cieq(gettok(&cpustr), "processor")) {
               gettok(&cpustr);
               token = gettok(&cpustr);
            }
            
         info->numLogicalProcessors = atoi(token) + 1;
         tfree(cpustr);
      }*/
		
		free(inStr);
		fclose(file);
	}
	else {
		error = TES_FAIL;
	}

	return error;
}

#elif defined(HAVE_WIN32)

/* get memory information */
static size_t get_sysmem(struct sys_memory *memall) {
#if ( _WIN32_WINNT >= 0x0500)
   MEMORYSTATUSEX ms;
   ms.dwLength = sizeof(MEMORYSTATUSEX);
   GlobalMemoryStatusEx( &ms);
   memall->size_m = ms.ullTotalPhys; 
   memall->free_m = ms.ullAvailPhys;
   memall->swap_t = ms.ullTotalPageFile;
   memall->swap_f = ms.ullAvailPageFile;
#else
   MEMORYSTATUS ms;
   ms.dwLength = sizeof(MEMORYSTATUS);
   GlobalMemoryStatus( &ms);
   memall->size_m = ms.dwTotalPhys; 
   memall->free_m = ms.dwAvailPhys;
   memall->swap_t = ms.dwTotalPageFile;
   memall->swap_f = ms.dwAvailPageFile;
#endif /*_WIN32_WINNT 0x0500*/
   return 1;     
}

/* get system information */
TesError tesCreateSystemInfo(TesSystemInfo *info) {
	OSVERSIONINFOA version;
	char *versionStr = NULL, *procStr, *freeStr;
	tInt major, minor;
	DWORD dwLen;
   HKEY hkBaseCPU;
	LONG lResult;

    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);


	info->numPhysicalProcessors =  0;
	info->numLogicalProcessors = sysinfo.dwNumberOfProcessors; //atoi(getenv("NUMBER_OF_PROCESSORS"));
	info->osName = NULL;

	version.dwOSVersionInfoSize = sizeof(OSVERSIONINFOA);
	if(GetVersionExA(&version) == 0) {
		return TES_FAIL;
	}
	major = version.dwMajorVersion;
	minor = version.dwMinorVersion;
	switch(major) {
	case 4:
		if(minor == 0) {
			versionStr = "Windows 95/NT4.0";
		}
		else if(minor == 10) {
			versionStr = "Windows 98";
		}
		else if (minor == 90) {
			versionStr = "Windows ME";
		}
		break;
	case 5:
		if(minor == 0) {
			versionStr = "Windows 2000";
		}
		else if(minor == 1) {
			versionStr = "Windows XP";
		}
		else if(minor == 2) {
			versionStr = "Windows Server 2003";
		}
		break;
	case 6:
		if(minor == 0) {
			versionStr = "Windows Vista";
		}
		else if(minor == 1) {
			versionStr = "Windows 7";
		}		
		break;
	default:
		break;
	}

	if(versionStr != NULL) {
		tInt lengthCSD = strlen(version.szCSDVersion);
		tInt lengthVer = strlen(versionStr);

		info->osName = malloc(lengthVer + lengthCSD + 2);
		memcpy(info->osName, versionStr, lengthVer);
		memcpy(info->osName + lengthVer + 1, version.szCSDVersion, lengthCSD);
		info->osName[lengthVer] = ' ';
		info->osName[lengthVer + lengthCSD + 1] = '\0';
	}


    lResult = RegOpenKeyExA(HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        0,KEY_READ,&hkBaseCPU);
    if(lResult != ERROR_SUCCESS) {
		info->cpuModelName = NULL;
		return TES_FAIL;
    }

    RegQueryValueExA(hkBaseCPU,"ProcessorNameString",0,0,NULL,&dwLen);
    freeStr = procStr = TMALLOC(char, dwLen + 1);
    RegQueryValueExA(hkBaseCPU,"ProcessorNameString",0,0,(LPBYTE)procStr,&dwLen);
    procStr[dwLen] = '\0';
    while (*procStr == ' ') procStr++;
    info->cpuModelName = copy(procStr);
    tfree(freeStr);

    RegCloseKey(hkBaseCPU); 

	return TES_SUCCESS;
}

#else
/* no Windows OS, no proc info file system */
TesError tesCreateSystemInfo(TesSystemInfo *info) {
   return 1;
}

#endif

