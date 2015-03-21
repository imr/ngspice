#ifndef ngspice_DGEN_H
#define ngspice_DGEN_H

typedef struct st_dgen dgen;

struct st_dgen {
	CKTcircuit	*ckt;
	wordlist	*dev_list;
	int		flags;
	int		dev_type_no;
	int		dev;
	GENinstance	*instance;
	GENmodel	*model;
};


#define	DGEN_ALL	0x00e
#define	DGEN_TYPE	0x002
#define	DGEN_MODEL	0x004
#define	DGEN_INSTANCE	0x008
#define	DGEN_INIT	0x010

#define	DGEN_DEFDEVS	0x020
#define	DGEN_ALLDEVS	0x040

#define	DGEN_DEFPARAMS	0x001
#define	DGEN_ALLPARAMS	0x002

extern dgen	*dgen_init(CKTcircuit *ckt, wordlist *wl, int nomix, int flag, int model);

#endif
