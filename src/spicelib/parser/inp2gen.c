/*******************************************************************************
 * Copyright 2020 Florian Ballenegger, Anamosic Ballenegger Design
 *******************************************************************************
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "inpxx.h"

#include "ngspice/cktdefs.h"

#if 0
int INP2gentype(char* line)
{
   int i;
   char c;
   for(i=0;c=line[i];i++)
   if(!isblank(c))
      break;
   if(line[i]=='.') return 0;
   if(line[i]=='*') return 0;
   if(line[i]=='$') return 0;

   for(;c=line[i];i++)
   {
      if(isblank(c))
        return 0;
      if(c=='#')
        return i+1;
   }
   return 0;
}
#endif
int INP2GEN(CKTcircuit *ckt, INPtables * tab, struct card *current)
{

/* :devtype:inst node1 node2 ... noden param=val */

    int type;			/* the type the model says it is */
    int idx, len;
    char *line;			/* the part of the current line left to parse */
    char *name;			/* a name */
    CKTnode *node;		/* a node pointer */
    int error;			/* error code temporary */
    GENinstance *fast;		/* pointer to the actual instance */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFuid uid;			/* uid of default model to be created */
    GENmodel* genmod;
    
    line = current->line;
    if(*line != ':')
       return 0; /* not a new generic instance format */
    
    for(len=1;line[len];len++) {
    	if(line[len]==':')
	    break;
    }
    if(line[len] != ':')
       return 0; /* not a new generic instance format */
              
    /* look if find this device */
    for (type = 0; type < ft_sim->numDevices; type++) {
       if(ft_sim->devices[type])
       if(strlen(ft_sim->devices[type]->name)==(len-1))
       if(strncasecmp(line+1, ft_sim->devices[type]->name, len-1)==0)
          break;
    }
    if(type==ft_sim->numDevices)  { /* not found */
	LITERR("Device type not supported by this binary\n");
	return 0;
    }
    
    /* TODO check if legacy device a new type of device */
    
    if (1) {
	/* create default model if needed */
	name = dup_string(ft_sim->devices[type]->name,len-1);
	IFnewUid(ckt, &uid, NULL, name, UID_MODEL, NULL);
	genmod = ft_sim->findModel(ckt, uid);
	if(!genmod)
	   IFC(newModel, (ckt, type, &genmod, uid));
    }
    line = &line[len+1];
    INPgetTok(&line,&name,1);
    IFC(newInstance, (ckt, genmod, &fast, name));
    if(0) printf("Instance name: %s\n", name);
        
    /* parse node connections and bind nodes */
    for(idx=0;idx<*(ft_sim->devices[type]->terms);idx++)
    {
        INPgetNetTok(&line, &name, 1);
        INPtermInsert(ckt, &name, tab, &node);
	IFC(bindNode, (ckt, fast, idx+1, node));
	if(0) printf("bind term %d to %s\n", idx, node->name);
    }
    
    /* parse parameters and assign parameters */
    while(*line)
    {
       INPgetTok(&line, &name, 0);
       if(*line!='=') {
          printf("Expect param=val format for %s, got %c\n", name, *line);
          LITERR("Expect param=val format\n");
       	  return(0);
       }
       line++; /* skip '=' */
       for(idx=0;idx<*(ft_sim->devices[type]->numInstanceParms);idx++)
         if(strcmp(name, ft_sim->devices[type]->instanceParms[idx].keyword)==0)
           break;
       tfree(name); /* don't need it anymore, we still know the keyword */
       if(idx<*(ft_sim->devices[type]->numInstanceParms))
       { /* found */
          int dataType;
	  IFvalue *parm;
	  dataType = ft_sim->devices[type]->instanceParms[idx].dataType;
	  parm = INPgetValue(ckt, &line, dataType, tab);
	  GCA(INPpName, (ft_sim->devices[type]->instanceParms[idx].keyword, parm, ckt, type, fast)); 
	  if(0) printf("Assign %s\n", ft_sim->devices[type]->instanceParms[idx].keyword);
       }
       else
       {
          LITERR("Unrecognized parameter\n");
       	  return(0);
       }
       
    }
    return (int) ':';
}


#if 0
int INP2GEN_old(CKTcircuit *ckt, INPtables * tab, struct card *current)
{

/* <inst>#<devtype> node1 node2 ... noden param=val */

    int type;			/* the type the model says it is */
    int idx, len;
    char *line;			/* the part of the current line left to parse */
    char *name;			/* a name */
    CKTnode *node;		/* a node pointer */
    int error;			/* error code temporary */
    GENinstance *fast;		/* pointer to the actual instance */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFuid uid;			/* uid of default model to be created */
    GENmodel* genmod;
    
    line = current->line;
    idx=INP2gentype(line);
    if(!idx)
      return 0; /* not a new generic instance format */
    
    /* calc length of device type string */
    for(len=0;line[idx+len];len++)
    if(isblank(line[idx+len]))
       break;
    if(len==0) return 0;
           
    /* look if find this device */
    for (type = 0; type < ft_sim->numDevices; type++) {
       if(ft_sim->devices[type])
       if(strlen(ft_sim->devices[type]->name)==len)
       if(strncasecmp(&line[idx], ft_sim->devices[type]->name, len)==0)
          break;
    }
    if(type==ft_sim->numDevices)  { /* not found */
	LITERR("Device type not supported by this binary\n");
	return 0;
    }
    
    if (1) {
	/* create default model */
	name = dup_string(ft_sim->devices[type]->name,len);
	IFnewUid(ckt, &uid, NULL, name, UID_MODEL, NULL);
	genmod = ft_sim->findModel(ckt, uid);
	if(!genmod)
	   IFC(newModel, (ckt, type, &genmod, uid));
    }
    name = dup_string(line, idx-1);
    IFC(newInstance, (ckt, genmod, &fast, name));
    if(0) printf("Instance name: %s\n", name);
    
    line = &line[idx+len];
    
    /* parse node connections and bind nodes */
    for(idx=0;idx<*(ft_sim->devices[type]->terms);idx++)
    {
        INPgetNetTok(&line, &name, 1);
        INPtermInsert(ckt, &name, tab, &node);
	IFC(bindNode, (ckt, fast, idx+1, node));
	if(0) printf("bind term %d to %s\n", idx, node->name);
    }
    
    /* parse parameters and assign parameters */
    while(*line)
    {
       INPgetTok(&line, &name, 0);
       if(*line!='=') {
          printf("Expect param=val format for %s, got %c\n", name, *line);
          LITERR("Expect param=val format\n");
       	  return(0);
       }
       line++; /* skip '=' */
       for(idx=0;idx<*(ft_sim->devices[type]->numInstanceParms);idx++)
         if(strcmp(name, ft_sim->devices[type]->instanceParms[idx].keyword)==0)
           break;
       tfree(name); /* don't need it anymore, we still know the keyword */
       if(idx<*(ft_sim->devices[type]->numInstanceParms))
       { /* found */
          int dataType;
	  IFvalue *parm;
	  dataType = ft_sim->devices[type]->instanceParms[idx].dataType;
	  parm = INPgetValue(ckt, &line, dataType, tab);
	  GCA(INPpName, (ft_sim->devices[type]->instanceParms[idx].keyword, parm, ckt, type, fast)); 
	  if(0) printf("Assign %s\n", ft_sim->devices[type]->instanceParms[idx].keyword);
       }
       else
       {
          LITERR("Unrecognized parameter\n");
       	  return(0);
       }
       
    }
}
#endif
