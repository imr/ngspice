typedef struct s_sgen	sgen;
struct s_sgen {
	CKTcircuit	*ckt;
	GENmodel	**devlist;
	GENmodel	*model, *next_model, *first_model;
	GENinstance	*instance, *next_instance, *first_instance;
	IFparm		*ptable;
	double		value;
	int		dev;
	int		istate;
	int		param, max_param;
	int		is_dc;
	int		is_instparam;
	int		is_q;
	int		is_principle;
	int		is_zerook;
};

extern sgen *sgen_init(CKTcircuit *ckt, int is_dc);
extern int sgen_next(sgen **xsg);
extern int sgen_setp(sgen*, CKTcircuit*, IFvalue* ); /* AlansFixes */
extern int sens_getp(sgen *, CKTcircuit *, IFvalue *);
