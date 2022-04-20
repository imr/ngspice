/* NG-SPICE -- An electrical circuit simulator
 *
 * Copyright (c) 1990 University of California
 * Copyright (c) 2000 Arno W. Peters
 *
 * Permission to use, copy, modify, and distribute this software and
 * its documentation without fee, and without a written agreement is
 * hereby granted, provided that the above copyright notice, this
 * paragraph and the following three paragraphs appear in all copies.
 *
 * This software program and documentation are copyrighted by their
 * authors. The software program and documentation are supplied "as
 * is", without any accompanying services from the authors. The
 * authors do not warrant that the operation of the program will be
 * uninterrupted or error-free. The end-user understands that the
 * program was developed for research purposes and is advised not to
 * rely exclusively on the program for any reason.
 *
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE TO ANY PARTY FOR DIRECT,
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
 * LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
 * DOCUMENTATION, EVEN IF THE AUTHORS HAVE BEEN ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE. THE AUTHORS SPECIFICALLY DISCLAIMS ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
 * SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE AUTHORS
 * HAVE NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
 * ENHANCEMENTS, OR MODIFICATIONS. */

#include "ngspice/ngspice.h"
#include "assert.h"

#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"

#include "dev.h"
#include "ngspice/memory.h" /* to alloc, realloc devices*/


#ifdef XSPICE
/*saj headers for xspice*/
#include <string.h> /* for strcpy, strcat*/
#if (!defined HAS_WINGUI) && (!defined __MINGW32__) && (!defined _MSC_VER)
#include <dlfcn.h> /* to load libraries*/
typedef void * funptr_t;
#define FREE_DLERR_MSG(msg)
#else /* ifdef HAS_WINGUI */
#undef BOOLEAN
#include <windows.h>
typedef FARPROC funptr_t;
void *dlopen(const char *, int);
funptr_t dlsym(void *, const char *);
char *dlerror(void);
#define FREE_DLERR_MSG(msg) free_dlerr_msg(msg)
static void free_dlerr_msg(char *msg);
#define RTLD_LAZY   1 /* lazy function call binding */
#define RTLD_NOW    2 /* immediate function call binding */
#define RTLD_GLOBAL 4 /* symbols in this dlopen'ed obj are visible to other
                       * dlopen'ed objs */
#endif /* ifndef HAS_WINGUI */

#include "ngspice/dllitf.h" /* the coreInfo Structure*/
#include "ngspice/evtudn.h" /*Use defined nodes */

Evt_Udn_Info_t  **g_evt_udn_info = NULL;
int g_evt_num_udn_types = 0;

/*The digital node type */
extern Evt_Udn_Info_t idn_digital_info;
int add_device(int n, SPICEdev **devs, int flag);
int add_udn(int,Evt_Udn_Info_t **);

extern struct coreInfo_t  coreInfo; /* cmexport.c */
#endif


#include "asrc/asrcitf.h"
#include "bjt/bjtitf.h"
#include "bsim1/bsim1itf.h"
#include "bsim2/bsim2itf.h"
#include "bsim3/bsim3itf.h"
#include "bsim3v0/bsim3v0itf.h"
#include "bsim3v1/bsim3v1itf.h"
#include "bsim3v32/bsim3v32itf.h"
#include "bsim4/bsim4itf.h"
#include "bsim4v5/bsim4v5itf.h"
#include "bsim4v6/bsim4v6itf.h"
#include "bsim4v7/bsim4v7itf.h"
#include "bsim3soi_pd/b3soipditf.h"
#include "bsim3soi_fd/b3soifditf.h"
#include "bsim3soi_dd/b3soidditf.h"
#include "bsimsoi/b4soiitf.h"
#include "cap/capitf.h"
#include "cccs/cccsitf.h"
#include "ccvs/ccvsitf.h"
#include "csw/cswitf.h"
#include "dio/dioitf.h"
#include "hfet1/hfetitf.h"
#include "hfet2/hfet2itf.h"
#include "hicum2/hicum2itf.h"
#include "hisim2/hsm2itf.h"
#include "hisimhv1/hsmhvitf.h"
#include "hisimhv2/hsmhv2itf.h"
#include "ind/inditf.h"
#include "isrc/isrcitf.h"
#include "jfet/jfetitf.h"
#include "jfet2/jfet2itf.h"
#include "ltra/ltraitf.h"
#include "mes/mesitf.h"
#include "mesa/mesaitf.h"
#include "mos1/mos1itf.h"
#include "mos2/mos2itf.h"
#include "mos3/mos3itf.h"
#include "mos6/mos6itf.h"
#include "mos9/mos9itf.h"
#include "cpl/cplitf.h"
#include "res/resitf.h"
#include "soi3/soi3itf.h"
#include "sw/switf.h"
#include "tra/traitf.h"
#include "txl/txlitf.h"
#include "urc/urcitf.h"
#include "vbic/vbicitf.h"
#include "vccs/vccsitf.h"
#include "vcvs/vcvsitf.h"
#include "vsrc/vsrcitf.h"
#include "vdmos/vdmositf.h"
#ifdef ADMS
#include "adms/hicum0/hicum0itf.h"
#include "adms/mextram/bjt504titf.h"
#include "adms/ekv/ekvitf.h"
#include "adms/psp102/psp102itf.h"
#include "adms/psp103/psp103itf.h"
#include "adms/bsimbulk/bsimbulkitf.h"
#include "adms/bsimcmg/bsimcmgitf.h"
#include "adms/r2_cmc/r2_cmcitf.h"
#endif
#ifdef CIDER
/* Numerical devices (Cider integration) */
#include "nbjt/nbjtitf.h"
#include "nbjt2/nbjt2itf.h"
#include "numd/numditf.h"
#include "numd2/numd2itf.h"
#include "numos/numositf.h"
#endif
#ifdef NDEV
#include "ndev/ndevitf.h"
#endif

static SPICEdev *(*static_devices[])(void) = {
    /* URC device MUST precede both resistors and capacitors */
    get_urc_info,
    get_asrc_info,
    get_bjt_info,
    get_bsim1_info,
    get_bsim2_info,
    get_bsim3_info,
    get_bsim3v0_info,
    get_bsim3v1_info,
    get_bsim3v32_info,
    get_b4soi_info,
    get_bsim4_info,
    get_bsim4v5_info,
    get_bsim4v6_info,
    get_bsim4v7_info,
    get_b3soipd_info,
    get_b3soifd_info,
    get_b3soidd_info,
    get_cap_info,
    get_cccs_info,
    get_ccvs_info,
    get_cpl_info,
    get_csw_info,
    get_dio_info,
    get_hfeta_info,
    get_hfet2_info,
    get_hicum_info,
    get_hsm2_info,
    get_hsmhv_info,
    get_hsmhv2_info,
    get_ind_info,
    get_mut_info,
    get_isrc_info,
    get_jfet_info,
    get_jfet2_info,
    get_ltra_info,
    get_mes_info,
    get_mesa_info,
    get_mos1_info,
    get_mos2_info,
    get_mos3_info,
    get_mos6_info,
    get_mos9_info,
    get_res_info,
    get_soi3_info,
    get_sw_info,
    get_tra_info,
    get_txl_info,
    get_vbic_info,
    get_vccs_info,
    get_vcvs_info,
    get_vsrc_info,
    get_vdmos_info,

#ifdef CIDER
    get_nbjt_info,
    get_nbjt2_info,
    get_numd_info,
    get_numd2_info,
    get_numos_info,
#endif

#ifdef ADMS
    (SPICEdev *(*)(void)) get_hicum0_info,
    (SPICEdev *(*)(void)) get_bjt504t_info,
    (SPICEdev *(*)(void)) get_ekv_info,
    (SPICEdev *(*)(void)) get_psp102_info,
    (SPICEdev *(*)(void)) get_psp103_info,
    (SPICEdev *(*)(void)) get_bsimbulk_info,
    (SPICEdev *(*)(void)) get_bsimcmg_info,
    (SPICEdev *(*)(void)) get_r2_cmc_info,
#endif

#ifdef NDEV
    get_ndev_info,
#endif

};

static int DEVNUM = NUMELEMS(static_devices);

/*Make this dynamic for later attempt to make all devices dynamic*/
SPICEdev **DEVices=NULL;

/*Flag to indicate what device type it is,
 *0 = normal spice device
 *1 = xspice device
 */
#ifdef XSPICE
int *DEVicesfl=NULL;
int DEVflag(int type){
  if(type < DEVNUM && type >= 0)
    return DEVicesfl[type];
  else
    return -1;
}
#endif



void
spice_destroy_devices(void)
{
#ifdef XSPICE
    tfree(g_evt_udn_info);
    tfree(DEVicesfl);
#endif
    tfree(DEVices);
    DEVNUM = 0;
}


void
spice_init_devices(void)
{
    int i;
    /* Safeguard against double initialization */
    DEVNUM = NUMELEMS(static_devices);

#ifdef XSPICE
    /* Initialize the structs and add digital node type */
    g_evt_udn_info = TMALLOC(Evt_Udn_Info_t  *, 1);
    g_evt_num_udn_types = 1;
    g_evt_udn_info[0] =  &idn_digital_info;

    DEVicesfl = TMALLOC(int, DEVNUM);
    /* tmalloc should automatically zero the array! */
#endif

    DEVices = TMALLOC(SPICEdev *, DEVNUM);

    for (i = 0; i < DEVNUM; i++)
        DEVices[i] = static_devices[i]();
}

int num_devices(void)
{
    return DEVNUM;
}

IFdevice ** devices_ptr(void)
{
    return (IFdevice **) DEVices;
}

SPICEdev ** devices(void)
{
    return DEVices;
}


#ifdef DEVLIB
/*not yet usable*/

#ifdef ADMS
#define DEVICES_USED {"asrc", "bjt", "vbic", "bsim1", "bsim2", "bsim3", "bsim3v32", "bsim3v2", "bsim3v1", "bsim4", "bsim4v5", "bsim4v6", "bsim4v7", \
                      "bsim4soi", "bsim3soipd", "bsim3soifd", "bsim3soidd", "hisim2", "hisimhv1",  "hisimhv2", \
                      "cap", "cccs", "ccvs", "csw", "dio", "hfet", "hfet2", "ind", "isrc", "jfet", "ltra", "mes", "mesa" ,"mos1", "mos2", "mos3", \
                      "mos6", "mos9", "res", "soi3", "sw", "tra", "urc", "vccs", "vcvs", "vsrc", "hicum0", "bjt504t", "ekv", "psp102", "psp103", "bsimbulk", "bsimcmg"}
#else
#define DEVICES_USED {"asrc", "bjt", "vbic", "bsim1", "bsim2", "bsim3", "bsim3v32", "bsim3v2", "bsim3v1", "bsim4", "bsim4v5", "bsim4v6", "bsim4v7", \
                      "bsim4soi", "bsim3soipd", "bsim3soifd", "bsim3soidd", "hisim2", "hisimhv1", "hisimhv2", \
                      "cap", "cccs", "ccvs", "csw", "dio", "hfet", "hfet2", "ind", "isrc", "jfet", "ltra", "mes", "mesa" ,"mos1", "mos2", "mos3", \
                      "mos6", "mos9", "res", "soi3", "sw", "tra", "urc", "vccs", "vcvs", "vsrc", "hicum2"}
#endif
int load_dev(char *name) {
  char *msg;
  char libname[50];
  void *lib;
  funptr_t fetch;
  SPICEdev *device;

  strcpy(libname, "lib");
  strcat(libname,name);
  strcat(libname,".so");

  lib = dlopen(libname,RTLD_NOW);
  if(!lib){
    msg = dlerror();
    printf("%s\n", msg);
    return 1;
  }

  strcpy(libname, "get_");
  strcat(libname,name);
  strcat(libname,"_info");
  fetch = dlsym(lib,libname);

  if(!fetch){
    msg = dlerror();
    printf("%s\n", msg);
    return 1;
  }
  device = ((SPICEdev * (*)(void)) fetch) ();
  add_device(1,&device,0);
  return 0;
}

void load_alldevs(void){
  char *devs[] = DEVICES_USED;
  int num = NUMELEMS(devs);
  int i;
  for(i=0; i< num;i++)
    load_dev(devs[i]);
  return;
}
#endif

#if defined(XSPICE) || defined(OSDI)
#include "ngspice/fteext.h"  /* for ft_sim */
#include "ngspice/cktdefs.h" /* for DEVmaxnum */

static void relink(void) {

/*
 * This replacement done by SDB on 6.11.2003
 *
 * ft_sim->numDevices = num_devices();
 * DEVmaxnum = num_devices();
 */
  ft_sim->numDevices = DEVNUM;
  DEVmaxnum = DEVNUM;

  ft_sim->devices = devices_ptr();
  return;
}

#endif

/*--------------------   XSPICE additions below  ----------------------*/
#ifdef XSPICE
#include "ngspice/cm.h"
#include "ngspice/cpextern.h"
#include "ngspice/mif.h"
int add_device(int n, SPICEdev **devs, int flag) {
  int i;
  int dnum = DEVNUM + n;
  DEVices = TREALLOC(SPICEdev *, DEVices, dnum);
  DEVicesfl = TREALLOC(int, DEVicesfl, dnum);
  for(i = 0; i < n;i++){
#ifdef TRACE
      printf("Added device: %s\n",devs[i]->DEVpublic.name);
#endif
    DEVices[DEVNUM+i] = devs[i];

    DEVices[DEVNUM+i]->DEVinstSize = &MIFiSize;
    DEVices[DEVNUM+i]->DEVmodSize = &MIFmSize;

    DEVicesfl[DEVNUM+i] = flag;
  }
  DEVNUM += n;
  relink();
  return 0;
}

int add_udn(int n,Evt_Udn_Info_t **udns){
  int i;
  int utypes = g_evt_num_udn_types + n;
  g_evt_udn_info = TREALLOC(Evt_Udn_Info_t  *, g_evt_udn_info, utypes);
  for(i = 0; i < n;i++){
#ifdef TRACE
      printf("Added udn: %s\n",udns[i]->name);
#endif
    g_evt_udn_info[g_evt_num_udn_types+i] = udns[i];
  }
  g_evt_num_udn_types += n;
  return 0;
}


int load_opus(const char *name)
{
    void *lib;
    char *msg;
    int num;
    SPICEdev **devs;
    Evt_Udn_Info_t **udns;
    funptr_t fetch;

    lib = dlopen(name, RTLD_NOW);
    if (!lib) {
        msg = dlerror();
        printf("Error opening code model \"%s\": %s\n", name, msg);
        FREE_DLERR_MSG(msg);
        return 1;
    }


    /* Get code models defined by the library */
    if ((fetch = dlsym(lib, "CMdevNum")) != (funptr_t) NULL) {
        num = *(*(int * (*)(void)) fetch)();
        fetch = dlsym(lib, "CMdevs");
        if (fetch != (funptr_t) NULL) {
            devs = (*(SPICEdev ** (*)(void)) fetch)();
        }
        else {
            msg = dlerror();
            printf("Error getting the list of devices: %s\n",
                    msg);
            FREE_DLERR_MSG(msg);
            return 1;
        }
    }
    else {
        msg = dlerror();
        printf("Error finding the number of devices: %s\n", msg);
        FREE_DLERR_MSG(msg);
        return 1;
    }

    add_device(num, devs, 1);

#ifdef TRACE
        printf("Got %d devices.\n", num);
#endif


    /* Get user-defined ndes defined by the library */
    if ((fetch = dlsym(lib, "CMudnNum")) != (funptr_t) NULL) {
        num = *(*(int * (*)(void)) fetch)();
        fetch = dlsym(lib, "CMudns");
        if (fetch != (funptr_t) NULL) {
            udns = (*(Evt_Udn_Info_t ** (*)(void)) fetch)();
        }
        else {
            msg = dlerror();
            printf("Error getting the list of user-defined types: %s\n",
                    msg);
            FREE_DLERR_MSG(msg);
            return 1;
        }
    }
    else {
        msg = dlerror();
        printf("Error finding the number of user-defined types: %s\n", msg);
        FREE_DLERR_MSG(msg);
        return 1;
    }

    add_udn(num, udns);
#ifdef TRACE
    printf("Got %d udns.\n", num);
#endif

    /* Give the code model access to facilities provided by ngspice. */
    if ((fetch = dlsym(lib,"CMgetCoreItfPtr")) != (funptr_t) NULL) {
        const struct coreInfo_t ** const core =
                (const struct coreInfo_t **const)
                (*(struct coreInfo_t ** (*)(void)) fetch)();
        *core = &coreInfo;
    }
    else {
        msg = dlerror();
        printf("Error getting interface pointer: %s\n", msg);
        FREE_DLERR_MSG(msg);
        return 1;
    }

    return 0;
} /* end of function load_opus */



#if defined(__MINGW32__) || defined(HAS_WINGUI) || defined(_MSC_VER)

/* For reporting error message if formatting fails */
static const char errstr_fmt[] =
        "Unable to find message in dlerr(). System code = %lu";
static char errstr[sizeof errstr_fmt - 3 + 3 * sizeof(unsigned long)];

/* Emulations of POSIX dlopen(), dlsym(), and dlerror(). */
void *dlopen(const char *name, int type)
{
    NG_IGNORE(type);
    return LoadLibrary(name);
}

funptr_t dlsym(void *hDll, const char *funcname)
{
    return GetProcAddress(hDll, funcname);
}

char *dlerror(void)
{
    LPVOID lpMsgBuf;

    DWORD rc = FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            GetLastError(),
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR) &lpMsgBuf,
            0,
            NULL
            );

    if (rc == 0) { /* FormatMessage failed */
        (void) sprintf(errstr, errstr_fmt, (unsigned long) GetLastError());
        return errstr;
    }

    return lpMsgBuf; /* Return the formatted message */
} /* end of function dlerror */



/* Free message related to dynamic loading */
static void free_dlerr_msg(char *msg)
{
    if (msg != errstr) { /* msg is an allocation */
        LocalFree(msg);
    }
} /* end of function free_dlerr_msg */



#endif /* Windows emulation of dlopen, dlsym, and dlerr */
#endif
/*--------------------   end of XSPICE additions  ----------------------*/

#ifdef OSDI
#include "ngspice/osdiitf.h"

static int osdi_add_device(int n, OsdiRegistryEntry *devs) {
  int i;
  int dnum = DEVNUM + n;
  DEVices = TREALLOC(SPICEdev *, DEVices, dnum);
#ifdef XSPICE
  DEVicesfl = TREALLOC(int, DEVicesfl, dnum);
#endif
  for (i = 0; i < n; i++) {
#ifdef TRACE
    printf("Added device: %s\n", devs[i]->DEVpublic.name);
#endif
    DEVices[DEVNUM + i] = osdi_create_spicedev(&devs[i]);
  }
  DEVNUM += n;
  relink();
  return 0;
}

int load_osdi(const char *path) {
  OsdiObjectFile file = load_object_file(path);
  if (file.num_entries < 0) {
    return file.num_entries;
  }

  osdi_add_device(file.num_entries, file.entrys);
  return 0;
}
#endif
