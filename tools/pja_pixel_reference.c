/* Development-only oracle; original PJA source is MIT/NOTICE-PJA attributed.
 * Never linked into the Rust runtime. Build against hash-verified source. */
#define _CRT_SECURE_NO_WARNINGS
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../out/pja-port-20261117/source/pjaj2kpm.c"
static void die(const char *s){fprintf(stderr,"pixel oracle: %s\n",s);exit(2);}
static uint32_t get32(FILE*f){uint8_t b[4];if(fread(b,1,4,f)!=4)die("read32");return b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;}
static void put_bytes(void*ctx,const uint8_t*b,intptr_t n){if(fwrite(b,1,n,(FILE*)ctx)!=(size_t)n)die("write");}
int main(int argc,char**argv){
  if(argc!=4)die("mode input output");FILE*in=fopen(argv[2],"rb"),*out=fopen(argv[3],"wb");if(!in||!out)die("open");
  if(!strcmp(argv[1],"model")){
    int hb=get32(in),n=get32(in);if(n<1||n>4)die("planes");int32_t w[4],h[4],depth[4];uint16_t*planes[4];int maxw=0;
    for(int i=0;i<n;i++){w[i]=get32(in);h[i]=get32(in);depth[i]=get32(in);if(w[i]>4096||h[i]>4096)die("shape");size_t count=(size_t)w[i]*h[i];planes[i]=calloc(count,2);if(!planes[i]||fread(planes[i],2,count,in)!=count)die("pixels");if(w[i]>maxw)maxw=w[i];}
    int64_t size=pjaj2kp_model_size(hb);void*mem=calloc(1,size);int32_t*res=calloc(18*(size_t)maxw,4);if(!mem||!res)die("allocation");
    if(pjaj2kp_code(mem,size,hb,1,put_bytes,out,0,0,n,w,h,depth,planes,res,18*(int64_t)maxw,0))die("model");
    free(mem);free(res);for(int i=0;i<n;i++)free(planes[i]);
  }else if(!strcmp(argv[1],"dwt")){
    int32_t w=get32(in),h=get32(in),x=get32(in),y=get32(in),levels=get32(in),fwd=get32(in);size_t count=(size_t)w*h;if(count>1<<20)die("shape");int32_t*a=calloc(count,4);int64_t*line=calloc(2*(size_t)(w>h?w:h),8);if(!a||!line||fread(a,4,count,in)!=count)die("coefficients");if(pjaj2kp_dwt(a,w,w,h,x,y,levels,fwd,line))die("transform");fwrite(a,4,count,out);free(a);free(line);
  }else die("unknown mode");
  fclose(in);if(fclose(out))die("close");return 0;
}
