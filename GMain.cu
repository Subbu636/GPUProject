#include "util.h"

int main(int argc,char **argv){
    char *fname = argv[1];
    int *len = (int*)malloc(sizeof(int)), d = 2;
    double *points = read_file_array(fname, d,len);
    int iter = 20,k = 20, l = len[0];
    double *cprob = init_prob(l,k);
    double *gprob = (double*)malloc(sizeof(double)*l*k);
    copy(cprob,cprob + l*k, gprob);
    double *ctime = create(iter,2),  *gtime = create(iter,2);
    // cpu
    gmix_cpu(points,cprob,k,iter,l,d, ctime);
    char cfname[30]; ; strcpy(cfname,"out_cpu_"); strcat(cfname,&fname[11]);
    fprint_mat(cfname,cprob,l,k);
    char ctfname[30]; ; strcpy(ctfname,"time_cpu_"); strcat(ctfname,&fname[11]);
    fprint_mat(ctfname,ctime,iter,2);
    // gpu
    gmix_gpu(points,gprob,k,iter,l,d, gtime);
    char gfname[30]; ; strcpy(gfname,"out_gpu_"); strcat(gfname,&fname[11]);
    fprint_mat(gfname,gprob,l,k);
    char gtfname[30]; ; strcpy(gtfname,"time_gpu_"); strcat(gtfname,&fname[11]);
    fprint_mat(gtfname,gtime,iter,2);
    return 0;
}


