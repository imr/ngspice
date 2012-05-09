#include "ngspice/ngspice.h" /* for wl */
#include "ngspice/ftedefs.h"
#include "ngspice/devdefs.h" /* solve deps in dev.h*/
#include <../spicelib/devices/dev.h> /*for load library commands*/
#include "com_dl.h"

#ifdef XSPICE
void com_codemodel(wordlist *wl){
  wordlist *ww;
  for(ww = wl;ww;ww = ww->wl_next)
    if(load_opus(wl->wl_word))
      fprintf(cp_err,"Error: Library %s couldn't be loaded!\n",ww->wl_word);
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

