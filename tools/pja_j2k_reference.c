/* Development-only original PJA J2K primitive oracle; not runtime linkage.
 * PJA MIT revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd; HT tables
 * OpenJPH BSD-2-Clause, see crates/rs-jxl-plus/NOTICE-PJA.
 * clang -O2 -fwrapv tools/pja_j2k_reference.c -o out/pja_j2k_reference.exe
 * argv: mq|tier1|ht descriptor output. All integers little endian.
 * mq: u32 n,npe,reset; u8 ctx[38],sym[n]; u32 pe[npe].
 * output u32 nf,ne,committed; bytes F,E; u32 triplets[npe]; ctx[38].
 * tier1: u32 w,h,orient,style,passes,bitpos,nbits,cap,nbytes; bits[nbytes].
 * output u32 ns,used; symbols[ns]; u32 passEnds[passes].
 * ht: u32 w,h,causal,np,cleanupRule,refRule; i32 q[w*h].
 * output u32 cleanupLen,refLen; cleanup,refinement; i32 decoded[w*h].
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../out/pja-port-20261117/source/pjaj2kt1.c"
#undef NW_
#undef N_
#undef NE_
#undef W_
#undef E_
#undef SW_
#undef S_
#undef SE_
#include "../out/pja-port-20261117/source/pjaj2kht.c"
/* Build with -DPJA_CLASSIC_MODEL or -DPJA_HT_MODEL for coefficient-model oracles. */
#if defined(PJA_CLASSIC_MODEL) || defined(PJA_HT_MODEL)
#ifdef PJA_CLASSIC_MODEL
#define NW_ 1
#define N_ 2
#define NE_ 4
#define W_ 8
#define E_ 16
#define SW_ 32
#define S_ 64
#define SE_ 128
#define sign_step km_sign_step
#define pass_spp km_pass_spp
#define pass_mrp km_pass_mrp
#define pass_cup km_pass_cup
#include "../out/pja-port-20261117/source/pjaj2kmc.c"
#undef sign_step
#undef pass_spp
#undef pass_mrp
#undef pass_cup
#else
#define bitlen32 hm_bitlen32
#define sigat hm_sigat
#define nbsig hm_nbsig
#include "../out/pja-port-20261117/source/pjaj2khm.c"
#undef bitlen32
#undef sigat
#undef nbsig
#endif
#endif
#ifdef PJA_PIXEL_T1
#define NW_ 1
#define N_ 2
#define NE_ 4
#define W_ 8
#define E_ 16
#define SW_ 32
#define S_ 64
#define SE_ 128
#include "../out/pja-port-20261117/source/pjaj2kpm.c"
#endif
static void fail(const char *s){fprintf(stderr,"%s\n",s);exit(1);}
static uint32_t rd(FILE*f){uint32_t v=0;for(int i=0;i<4;i++){int b=fgetc(f);if(b<0)fail("short descriptor");v|=(uint32_t)b<<(8*i);}return v;}
static void wr(FILE*f,uint32_t v){for(int i=0;i<4;i++)if(fputc((v>>(8*i))&255,f)<0)fail("write");}
static void *mem(size_t n){if(n>32000000)fail("allocation bound");void*p=calloc(n?n:1,1);if(!p)fail("allocation");return p;}
static void readbytes(FILE*f,void*p,size_t n){if(fread(p,1,n,f)!=n)fail("short bytes");}
#if defined(PJA_CLASSIC_MODEL) || defined(PJA_HT_MODEL)
static void model_put(void*ctx,const uint8_t*b,intptr_t n){if(fwrite(b,1,(size_t)n,(FILE*)ctx)!=(size_t)n)fail("model write");}
/* model descriptor: u32 hb,nb,mapbytes,nblocks; each band ten signed i32
 * comp,res,orient,nres,w,h,mb,parent,cross0,cross1 then signed i64 map;
 * each classic block b,x,y,w,h,style,passes,zbp,n then symbols[n];
 * each HT block b,x,y,w,h,causal,np,B then signed i32 q[w*h]. */
static void model_run(FILE*f,FILE*o){uint32_t hb=rd(f),nb=rd(f),mapsize=rd(f),ncbs=rd(f);if(nb>1024||ncbs>10000||mapsize>10000000)fail("model bounds");
#ifdef PJA_CLASSIC_MODEL
j2kmband_t*bands=mem((size_t)nb*sizeof(*bands));
#else
hmband_t*bands=mem((size_t)nb*sizeof(*bands));
#endif
for(uint32_t b=0;b<nb;b++){int32_t *v=&bands[b].comp;for(int k=0;k<10;k++)v[k]=(int32_t)rd(f);uint64_t lo=rd(f),hi=rd(f);bands[b].map=(int64_t)(lo|(hi<<32));}
uint8_t*maps=mem(mapsize);
#ifdef PJA_CLASSIC_MODEL
int64_t size=pjaj2km_size(hb);if(size<0)fail("model size");void*state=calloc(1,(size_t)size);if(!state)fail("state allocation");if(pjaj2km_init(state,size,hb,1,model_put,o,NULL,0))fail("model init");pjaj2km_tile(state,bands,nb,maps,mapsize);
#else
int64_t size=pjaj2khm_size(hb);if(size<0)fail("model size");void*state=calloc(1,(size_t)size);if(!state)fail("state allocation");if(pjaj2khm_init(state,size,hb,1,model_put,o,NULL,0))fail("model init");pjaj2khm_tile(state,bands,nb,maps,mapsize);
#endif
for(uint32_t cb=0;cb<ncbs;cb++){int32_t b=rd(f),x=rd(f),y=rd(f),w=rd(f),h=rd(f),sty=rd(f),np=rd(f),plane=rd(f);if(w<1||h<1||w>1024||h>1024||(uint64_t)w*h>4096||np<0||np>255)fail("model block");
#ifdef PJA_CLASSIC_MODEL
uint32_t n=rd(f);uint8_t*sym=mem(n);readbytes(f,sym,n);int32_t*ends=mem((size_t)np*4);if(pjaj2km_cb(state,b,x,y,w,h,sty,np,plane,sym,n,ends)!=(int64_t)n)fail("model cb");
#else
int32_t*q=mem((size_t)w*h*4);for(int32_t i=0;i<w*h;i++)q[i]=(int32_t)rd(f);if(pjaj2khm_cb(state,b,x,y,w,h,sty,np,plane,q))fail("ht model cb");
#endif
}
#ifdef PJA_CLASSIC_MODEL
pjaj2km_finish(state);
#else
pjaj2khm_finish(state);
#endif
}
#endif
#ifdef PJA_PIXEL_T1
/* u32 w,h,orient,style,passes,P; i32 coefficients[w*h]. Output u32 n,full;
 * symbols[n], u32 passend[passes], i32 decoded coefficients[w*h]. */
static void pixel_run(FILE*f,FILE*o){uint32_t w=rd(f),h=rd(f),orient=rd(f),style=rd(f),np=rd(f),P=rd(f);if(!w||!h||w>1024||h>1024||(uint64_t)w*h>4096||np>255)fail("pixel tier1 bounds");int32_t*q=mem((size_t)w*h*4),*d=mem((size_t)w*h*4);for(uint32_t i=0;i<w*h;i++)q[i]=(int32_t)rd(f);void*t=mem(pjaj2kp_t1_size());uint8_t*sym=mem(1000000);int32_t*ends=mem((size_t)np*4);int64_t n=pjaj2kp_cb_sym(t,w,h,orient,style,np,P,q,w,sym,1000000,ends);if(n<0)fail("pixel cb sym");int full=pjaj2kp_cb_full(t,w,h,orient,style,P,q,w);if(full<0)fail("pixel cb full");if(pjaj2kp_cb_coef(t,w,h,orient,style,np,P,sym,n,ends,d,w))fail("pixel cb coef");wr(o,(uint32_t)n);wr(o,(uint32_t)full);fwrite(sym,1,n,o);for(uint32_t i=0;i<np;i++)wr(o,(uint32_t)ends[i]);for(uint32_t i=0;i<w*h;i++)wr(o,(uint32_t)d[i]);}
#endif
int main(int argc,char**argv){if(argc!=4)fail("usage mode descriptor output");FILE*f=fopen(argv[2],"rb"),*o=fopen(argv[3],"wb");if(!f||!o)fail("open");
#if defined(PJA_CLASSIC_MODEL) || defined(PJA_HT_MODEL)
if(!strcmp(argv[1],"model")){model_run(f,o);}else
#endif
#ifdef PJA_PIXEL_T1
if(!strcmp(argv[1],"pixel_tier1")){pixel_run(f,o);}else
#endif
if(!strcmp(argv[1],"mq")){uint32_t n=rd(f),np=rd(f),reset=rd(f);uint8_t ctx[38];readbytes(f,ctx,38);uint8_t*s=mem(n);readbytes(f,s,n);int64_t *pe=mem((size_t)np*8),*kp=mem((size_t)np*24);for(uint32_t i=0;i<np;i++)pe[i]=rd(f);size_t cap=(size_t)n*3+32;uint8_t*F=mem(cap),*E=mem(cap);int64_t out[3];if(pjaj2k_mq_encode(s,n,pe,np,reset,ctx,F,E,cap,out,kp))fail("mq encode");for(int i=0;i<3;i++)wr(o,(uint32_t)out[i]);fwrite(F,1,out[0],o);fwrite(E,1,out[1],o);for(uint32_t i=0;i<3*np;i++)wr(o,(uint32_t)kp[i]);fwrite(ctx,1,38,o);
}else if(!strcmp(argv[1],"tier1")){uint32_t w=rd(f),h=rd(f),ori=rd(f),sty=rd(f),np=rd(f),pos=rd(f),nb=rd(f),cap=rd(f),len=rd(f);if(w>1024||h>1024||w*(uint64_t)h>4096||np>255)fail("block");uint8_t*b=mem(len);readbytes(f,b,len);uint8_t*s=mem(cap),*st=mem(2*(size_t)(w+2)*(h+2));int32_t*ends=mem((size_t)np*4);int64_t used=0,n=pjaj2k_t1_replay(w,h,ori,sty,np,b,pos,nb,&used,s,cap,ends,st);if(n<0)fail("replay");wr(o,(uint32_t)n);wr(o,(uint32_t)used);fwrite(s,1,n,o);for(uint32_t i=0;i<np;i++)wr(o,ends[i]);
}else if(!strcmp(argv[1],"ht")){uint32_t w=rd(f),h=rd(f),causal=rd(f),np=rd(f),cup=rd(f),ref=rd(f);if(w>1024||h>1024||w*(uint64_t)h>4096)fail("block");int32_t*q=mem((size_t)w*h*4),*d=mem((size_t)w*h*4);for(uint32_t i=0;i<w*h;i++)q[i]=(int32_t)rd(f);void*ws=mem(pjaj2kh_wsize());uint8_t*out=mem(65536);int n1=pjaj2kh_cup(ws,w,h,np,q,cup,out,32768);if(n1<0)fail("cleanup");int n2=np==1?0:pjaj2kh_ref(ws,w,h,causal,np,q,ref,out+n1,32768);if(n2<0)fail("refinement");if(pjaj2kh_decode(ws,w,h,causal,np,out,n1,n2,d))fail("ht decode");wr(o,n1);wr(o,n2);fwrite(out,1,n1+n2,o);for(uint32_t i=0;i<w*h;i++)wr(o,(uint32_t)d[i]);
}else fail("mode");if(fgetc(f)!=EOF)fail("trailing descriptor");if(fclose(o))fail("close");fclose(f);return 0;}
