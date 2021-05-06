#include "util.h"

int main(int argc,char **argv){
    char *fname = argv[1];
    int *len = (int*)malloc(sizeof(int)), d = 2;
    int iter = 10,k = 20;
    if(argc >= 3) k = atoi(argv[2]);
    if(argc >= 4) iter = atoi(argv[3]);
    if(argc >= 5) d = atoi(argv[4]);
    double *points = read_file_array(fname, d,len);
    int l = len[0];
    if(argc >= 6) l = atoi(argv[5]);
    double *cprob = init_prob(l,k);
    double *gprob = (double*)malloc(sizeof(double)*l*k);
    copy(cprob,cprob + l*k, gprob);
    double *ctime = create(iter,2),  *gtime = create(iter,2);
    // cpu
    gmix_cpu(points,cprob,k,iter,l,d, ctime);
    char cfname[30]; ; strcpy(cfname,"out_cpu_");strcat(cfname,&fname[12]);
    fprint_mat(cfname,cprob,l,k);
    char ctfname[30]; ; strcpy(ctfname,"time_cpu_"); strcat(ctfname,&fname[12]);
    fprint_mat(ctfname,ctime,iter,2);
    // gpu
    gmix_gpu(points,gprob,k,iter,l,d, gtime);
    char gfname[30]; ; strcpy(gfname,"out_gpu_");strcat(gfname,&fname[12]);
    fprint_mat(gfname,gprob,l,k);
    char gtfname[30]; ; strcpy(gtfname,"time_gpu_");strcat(gtfname,&fname[12]);
    fprint_mat(gtfname,gtime,iter,2);
    return 0;
}

// strcat(cfname,argv[2]);strcat(cfname,".txt");
