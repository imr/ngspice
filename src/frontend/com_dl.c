#include <ngspice.h> /* for wl */
#include "ftedefs.h"
#include <devdefs.h> /* solve deps in dev.h*/
#include <../spicelib/devices/dev.h> /*for load library commands*/

#ifdef XSPICE

#include "mifproto.h"
#include "inpdefs.h"
#include "fteext.h"
#include "ifsim.h"
#include "dllitf.h"
#include "iferrmsg.h"
#include <iferrmsg.h>




void com_codemodel(wordlist *wl){
  wordlist *ww;
  for(ww = wl;ww;ww = ww->wl_next)
    if(load_opus(wl->wl_word))
      fprintf(cp_err,"Error: Library %s couldn't be loaded!\n",ww->wl_word);
  return;
}


	
void com_xaltermod(wordlist *wl){
  wordlist *ww, *eqword, *words;
  int i, err, typecode=-1;
  GENinstance *devptr=(GENinstance *)NULL;
  GENmodel *modptr=(GENmodel *)NULL;
  IFdevice *device;
  IFparm *opt;
  IFvalue nval;
  bool found;
  char *dev, *p;
  char *param;
  int id;
  int numParams =0;

  if (!ft_curckt) {
        fprintf(cp_err, "Error: no circuit loaded\n");
        return;
  }

 
  if (!wl) {
    fprintf(cp_err, "usage: xaltermod dev param = expression\n");
    return;
  }


  words = wl;
  while (words) {
    p = words->wl_word;
    eqword = words;
    words = words->wl_next;
    if (eq(p, "=")) {
      break;
    }
  }
  if (!words) {
    fprintf(cp_err, "usage: xaltermod dev param = expression\n");
    return;
  }

  dev = NULL;
  param = NULL;
  words = wl;
  while (words != eqword) {
    p = words->wl_word;
    if (param) {
      fprintf(cp_err, "Error: excess parameter name \"%s\" ignored.\n",p);
    } else if (dev) {
      param = words->wl_word;
    } else {
      dev = p;
    }
    words = words->wl_next;
  }
  if (!dev) {
    fprintf(cp_err, "Error: no model name provided.\n" );
    return;
  }

  words = eqword->wl_next;
 
  INPretrieve(&dev,(INPtables *)ft_curckt->ci_symtab);
 
  err = (*(ft_sim->findInstance))(ft_curckt->ci_ckt,&typecode,(void **)&devptr,dev,NULL,NULL);
    
  if (err != OK) {
    typecode = -1;
    devptr   = (void *)NULL;
    err = (*(ft_sim->findModel))(ft_curckt->ci_ckt,&typecode,(void **)&modptr,dev);
  }
    
  if (err != OK) {
    printf("Error: no such model '%s'.\n",dev);
    printf(" use spice::showmod all, to display all models\n");
    return;
  }

  device = ft_sim->devices[typecode];
  found = FALSE;

  for(ww = words;ww;ww = ww->wl_next)
    numParams ++;

  for (i = 0; i < *(device->numModelParms); i++) {
    opt = &device->modelParms[i];
    
    if (strcmp(opt->keyword,param) == 0) {
      id = opt->id;
      nval.v.numValue = 0;

      if (opt->dataType & IF_VECTOR) {
	switch (opt->dataType & (IF_VARTYPES & ~IF_VECTOR)) 
	  {
	  case IF_STRING:
	    found = TRUE;
	    nval.v.vec.sVec = malloc((numParams +1) * sizeof(char *));
	    for(ww = words;ww;ww = ww->wl_next) {
	      nval.v.vec.sVec[nval.v.numValue] = ww->wl_word;
	      nval.v.numValue++;
	    }
	    break;
	    
	  default:
	    printf("Error: xaltermod only supports vectors of strings.\n");
	    printf(" use spice::altermod\n");
	    return;
	    break;	    
	  }
      } else {
 	printf("Error: xaltermod only supports vectors of strings.\n");
	printf(" use spice::altermod\n");
	return;
      }
    }
  }
    
  if (found == FALSE) {
    printf("Error: no parameter '%s' in model '%s'.\n", param, dev);
    printf(" use spice::showmod all, to list all model parameters\n");
    return;
  }
 
  err = (*(ft_sim->setModelParm))((void *)ft_curckt->ci_ckt, (void *)modptr, 
  				  id, &nval, (IFvalue *)NULL);
  
  return;
}

#endif
#ifdef DEVLIB
void com_use(wordlist *wl){
    wordlist *ww;  
  for(ww = wl;ww;ww = ww->wl_next)
    if(load_dev(wl->wl_word))
      fprintf(cp_err,"Error: Library %s couldn't be loaded!\n",ww->wl_word);
  return;
}
#endif

