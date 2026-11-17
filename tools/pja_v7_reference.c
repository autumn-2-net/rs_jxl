/* Development-only PJA v7 independent coefficient oracle.
 * Original included C remains unmodified; PJA MIT attribution: see
 * crates/rs-jxl-plus/LICENSE-PJA and NOTICE-PJA. Not linked to Rust runtime.
 * clang -O2 -fwrapv -ffp-contract=off tools/pja_v7_reference.c -o out/pja_v7_reference.exe
 * Usage: PROGRAM encode descriptor output | PROGRAM decode descriptor arithmetic output
 * Descriptor little-endian protocol PJA7REF1:
 * 13 [u32 length, u8 values] arrays: interior,interiorLeft,ibNa,ibNb,ibWa,
 * ibWb,ibNone,ratioOf,unzig,zzTable,nzTable,remTable,kBucket;
 * 6 [u32 length, i64 values] arrays: w3,e3,kerLeft,kerTop,kerRight,kerBottom;
 * u64 interiorZz, u64 edgeZz; u32 component_count (1..3);
 * each component: u32 bw,bh,sx,sy; u16 qt[64]; i16 zigzag_coef[bw*bh*64].
 * One whole band; component zero is luma, components 1 and 2 share a model.
 * Encode output is bare arithmetic stream including pjaarith's four-byte tail.
 * Decode output concatenates component coefficients as little-endian i16.
 * Descriptor prediction tables are exported by Rust, so this tests arithmetic
 * and coefficient loops independently, but does not independently verify those
 * prediction constants. Probability tables are generated independently here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define PJA_V7_PLAIN 1
#include "../out/pja-port-20261117/source/pjav7c.c"
static void die(const char *s) { fprintf(stderr, "%s\n", s); exit(1); }
static void *alloc(size_t n, size_t z) { if (n > 100000000 || z > 128) die("allocation bound"); void *p=calloc(n,z); if(!p) die("allocation failed"); return p; }
static uint64_t readle(FILE *f,int n) { uint64_t x=0; for(int i=0;i<n;i++){int b=fgetc(f);if(b==EOF)die("short descriptor");x|=(uint64_t)b<<(i*8);}return x; }
static void writele(FILE *f,uint64_t x,int n) { for(int i=0;i<n;i++)if(fputc((int)(x>>(i*8))&255,f)==EOF)die("write failed"); }
static uint8_t *bytes(FILE *f,size_t minimum) { size_t n=(size_t)readle(f,4);if(n<minimum)die("short table");uint8_t *p=alloc(n,1);if(fread(p,1,n,f)!=n)die("short table data");return p; }
static int64_t *longs(FILE *f,size_t minimum) { size_t n=(size_t)readle(f,4);if(n<minimum)die("short long table");int64_t *p=alloc(n,8);for(size_t i=0;i<n;i++)p[i]=(int64_t)readle(f,8);return p; }
static abit_t *counters(size_t n) { abit_t *p=alloc(n,sizeof(*p));for(size_t i=0;i<n;i++)p[i].p=32768;return p; }
static int32_t *weights(int n,int sets) { int32_t *p=alloc(sets*8,4);for(int s=0;s<sets;s++)for(int i=0;i<n-1;i++)p[s*8+i]=65536*4/(3*n);return p; }
static void model(v7m_t *m) {
  m->dcExpo = counters(28 * 16);
  m->dcMant = counters((MAX_EXP_DC + 1) * 16);
  m->dcMant1 = counters((MAX_EXP_DC + 1) * 28);
  m->dcZero = counters(28);
  m->dsX = counters(9 * 9 * 3);
  m->dsf = counters((28 / 2) * 9 * 3 * 3);
  m->dx2 = counters(20 * 16);
  m->dx3 = counters(NZ_BUCKETS * 8 * 16);
  m->dx4 = counters(28 * 3 * 16);
  m->dxX = counters(8 * 17 * 16);
  m->dz3 = counters(NZ_BUCKETS * 8);
  m->dz4 = counters(28 * 3);
  m->dzX = counters(28 * 17);
  m->dzf = counters(20 * 5);
  m->eMant = counters(EDGE_POS * 1280);
  m->eSgn = counters(EDGE_POS * 3 * 3 * 3 * 3 * 3);
  m->ec1 = counters(2 * NZ_BUCKETS * 9 * 8);
  m->ec2 = counters(2 * 9 * 9 * 8);
  m->ec3 = counters(2 * 9 * PRED_BUCKETS * 8);
  m->ecX = counters(2048 + 2 * 9 * 18 * 8);
  m->ecp = counters(2 * 16 * NZ_BUCKETS * 8);
  m->eexpo1 = counters(EDGE_POS * PQ_BUCKETS * 16);
  m->eexpo2 = counters(EDGE_POS * MAG_BUCKETS * 16);
  m->eexpo3 = counters(PQ_BUCKETS * 8 * 16);
  m->eexpo4 = counters(EDGE_POS * NZ_BUCKETS * 16);
  m->eexpo5 = counters(EDGE_POS * 12 * 16);
  m->emant1 = counters((MAX_EXP_AC + 1) * PQ_BUCKETS);
  m->esr = counters(2 * 3 * PQ_BUCKETS * 9);
  m->exCb = counters(EDGE_POS * 26 * 16);
  m->exLum = counters(EDGE_POS * 26 * 16);
  m->expo3 = counters(K_BUCKETS * REM_BUCKETS * 16);
  m->ez6 = counters(EDGE_POS * 8 * 8 * 4);
  m->ezLum = counters(EDGE_POS * 26 * 4);
  m->ezero4 = counters(EDGE_POS * MAG_BUCKETS);
  m->iSign = counters(64 * 9);
  m->mFirst = counters((MAX_EXP_AC + 1) * MAG_BUCKETS * 16);
  m->mRest = counters(K_BUCKETS * 256);
  m->mant1 = counters((MAX_EXP_AC + 1) * K_BUCKETS);
  m->nz2 = counters(NZ_BUCKETS * NZ_BUCKETS * 64);
  m->nz3 = counters((NZ_BUCKETS + 1) * NZ_BUCKETS * 64);
  m->nz4 = counters((NZ_BUCKETS + 1) * 12 * 8);
  m->nzm1 = counters(NZ_BUCKETS * 8 * 8);
  m->nzm2 = counters((NZ_BUCKETS + 1) * 8 * 8);
  m->xCb = counters(K_BUCKETS * 28 * 16);
  m->xLum = counters(K_BUCKETS * 28 * 16);
  m->xPos = counters(64 * MAG_BUCKETS * 16);
  m->xRow = counters(K_BUCKETS * 16 * 16);
  m->zCb = counters(64 * 28 * REM_BUCKETS);
  m->zLum = counters(64 * 28 * REM_BUCKETS);
  m->zero2 = counters(64 * MAG_BUCKETS * 4);
  m->zero4 = counters(K_BUCKETS * MAG_BUCKETS * REM_BUCKETS);
  m->wNz = weights(4, 6);
  m->wZero = weights(5, 64);
  m->wExpo = weights(6, 1485);
  m->wEZero = weights(4, 896);
  m->wEExpo = weights(8, 840);
  m->wDcZero = weights(6, 28);
  m->wDcExpo = weights(6, 17);
  m->wECnt = weights(6, 176);
}
static void put(void *ctx,const uint8_t *b,intptr_t n) { if(fwrite(b,1,(size_t)n,(FILE*)ctx)!=(size_t)n)die("write failed"); }
static uint32_t get(void *ctx,intptr_t p) { (void)p;int b=fgetc((FILE*)ctx);return b==EOF?0:(uint32_t)b; }
static uint16_t sq[4096]; static int16_t st[4096]; static int32_t rc[256];
static void probability(void) {
  for(int i=0;i<4096;i++){long q=lrint(4096.0/(1.0+exp(-(i-2048)/256.0)));sq[i]=(uint16_t)(q<1?1:q>4095?4095:q);}
  int pi=0;for(int x=-2048;x<2048;x++)while(pi<=4095&&pi<=sq[x+2048])st[pi++]=(int16_t)x;
  while(pi<=4095)st[pi++]=2047;st[0]=st[1];
  for(int i=0;i<256;i++)rc[i]=(int32_t)lrint(65536.0/(i+1.5));
}
int main(int argc,char **argv) {
  int enc=argc>=2&&!strcmp(argv[1],"encode");
  if(argc!=(enc?4:5)||(!enc&&strcmp(argv[1],"decode")))die("usage: encode descriptor output | decode descriptor arithmetic output");
  FILE *f=fopen(argv[2],"rb"),*out=fopen(argv[enc?3:4],"wb");if(!f||!out)die("open failed");
  char magic[8];if(fread(magic,1,8,f)!=8||memcmp(magic,"PJA7REF1",8))die("descriptor magic");
  probability();v7t_t t={0};t.sq=sq;t.st=st;t.rc=rc;t.put=put;t.get=get;
#define B(field,min) t.field=bytes(f,min)
  B(interior,49); B(interiorLeft,49); B(ibNa,49); B(ibNb,49); B(ibWa,49); B(ibWb,49); B(ibNone,49); B(ratioOf,49*49);
  B(unzig,64); B(zzTable,64); B(nzTable,64); B(remTable,64); B(kBucket,64);
#undef B
#define L(field,min) t.field=longs(f,min)
  L(w3,8); L(e3,8); L(kerLeft,512); L(kerTop,512); L(kerRight,512); L(kerBottom,512);
#undef L
  t.interiorZz=readle(f,8);t.edgeZz=readle(f,8);pjav7_tables(&t);
  uint32_t count=(uint32_t)readle(f,4);if(!count||count>3)die("component count");
  v7m_t models[2]={{0}};model(&models[0]);if(count>1)model(&models[1]);
  aenc_t e={0};e.x2=UINT32_MAX;e.ctx=out;
  adec_t d={0};FILE *input=NULL;
  if(!enc){input=fopen(argv[3],"rb");if(!input)die("open arithmetic failed");d.x2=UINT32_MAX;d.live=input;for(int i=0;i<4;i++)d.x=(d.x<<8)|get(input,i);d.p=4;}
  int16_t *coef[3]={0};uint8_t *nzc[3]={0},*dcr[3]={0};uint32_t widths[3]={0},heights[3]={0};
  for(uint32_t c=0;c<count;c++) {
    uint32_t bw=(uint32_t)readle(f,4),bh=(uint32_t)readle(f,4),sx=(uint32_t)readle(f,4),sy=(uint32_t)readle(f,4);
    if(!bw||!bh||bw>4096||bh>4096||bw*(uint64_t)bh>1000000||!sx||!sy||sx>4||sy>4)die("component dimensions");
    if(c&&((bw-1)*(uint64_t)sx>=widths[0]||(bh-1)*(uint64_t)sy>=heights[0]))die("luma mapping");
    widths[c]=bw;heights[c]=bh;size_t blocks=(size_t)bw*bh;
    uint16_t qt[64];for(int k=0;k<64;k++){qt[k]=(uint16_t)readle(f,2);if(!qt[k])die("zero quantum");}
    coef[c]=alloc(blocks*64,2);for(size_t k=0;k<blocks*64;k++){int16_t v=(int16_t)readle(f,2);if(enc)coef[c][k]=v;}
    nzc[c]=alloc(blocks,1);dcr[c]=alloc(blocks,1);
    v7work_t wk={0};wk.rowC=alloc(bw,sizeof(ecache_t));wk.dcr=dcr[c];wk.rcnt=alloc(blocks,1);wk.ccnt=alloc(blocks,1);
    if(c){wk.lumaC=coef[0];wk.lumaD=dcr[0];}
    if(c==2&&bw==widths[1]&&bh==heights[1]){wk.prevC=coef[1];wk.prevD=dcr[1];}
    int ok=enc?pjav7_component_e(&e,&models[c!=0],coef[c],bw,bh,qt,c?nzc[0]:NULL,widths[0],sx,sy,0,bh,nzc[c],&wk):pjav7_component_d(&d,&models[c!=0],coef[c],bw,bh,qt,c?nzc[0]:NULL,widths[0],sx,sy,0,bh,nzc[c],&wk);
    if(!ok)die("original C rejected coefficient stream");
    if(!enc)for(size_t k=0;k<blocks*64;k++)writele(out,(uint16_t)coef[c][k],2);
  }
  if(enc)for(int i=0;i<4;i++){writele(out,e.x1>>24,1);e.x1<<=8;}
  if(fgetc(f)!=EOF)die("trailing descriptor bytes");
  if(fclose(out))die("close output failed");fclose(f);if(input)fclose(input);return 0;
}
