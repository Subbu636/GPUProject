#include <cuda.h>
#include <cuComplex.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <sys/time.h> 
#include "util.h"
using namespace std;
Point* read(char *filename, int* num){
    FILE *file;
    file = fopen(filename, "r");
    if (!file)  {
        printf("#cannot open input file\n");
        return NULL;
    }
    vector<Point> points;
    float x,y;
    int n = 0;
    while(fscanf(file,"%f %f",&x,&y) != EOF){
        Point p;
        p.x = x;
        p.y = y;
        points.push_back(p);
        n++;
    }
     Point* ps = (Point*) malloc(n*sizeof(Point));
     for(int i=0;i<n;i++)
     {
         ps[i] = points[i];
     }
     num[0] = n;
     return ps;
}

int main(int argc,char **argv){
    int seed = 1234;
    srand(1234);
    struct timeval t1, t2;
    char *fname = argv[1];

    if(fname == NULL)
    {
        printf("give the input file\n");
        return 0;
    }
    char *oname = argv[2];
    
    if (oname == NULL)
    {
        printf("give the output file\n");
        return 0;
    }
    
    FILE *outfile;
    outfile = fopen(oname, "w");
    char* kchar = argv[3];
    if(kchar == NULL) 
    {
        printf("Input K\n");
        return 0;
    } 
    int k = atoi(kchar);

    Point* points;
    int p[1];
    p[0] = 0;
    points = read(fname,p);
    
    // for(int i=0;i<100;i++)
    // {
    //     printf("%f %f\n", points[i].x,points[i].y);
    // }

    int num_points = p[0];
    printf("Points Allocated\n");

    vector<int> ind;
    for(int i=0;i<k;i++) ind.push_back(i*(num_points/k));
    // for(int i=0;i<num_points;i++) random_shuffle(0, n);   random shuffle taking a lot time for 10^5 so used above method 
    Point* gpupoints;
    cudaMalloc(&gpupoints, num_points*sizeof(Point));
    printf("GPU Points Allocated\n");
    
    cudaMemcpy(gpupoints,points,num_points*sizeof(Point),cudaMemcpyHostToDevice); 
    printf("GPU Points Copied\n");

    Point* means = (Point*) malloc(k*sizeof(Point));
    int* labels = (int*) malloc(num_points*sizeof(int));
    printf("Means and Labels Allocated\n");

    
    // Naive Sharding Initialization
    for(int i=0;i<num_points;i++)
    {
        // cout<<(i%k)<<endl;
        labels[i] = i%k;
    }
    for(int i=0;i<k;i++)
    {
        means[i].x = points[ind[i]].x;
        means[i].y = points[ind[i]].y;
    }

    Point* imeans = (Point*) malloc(k*sizeof(Point));
    int* ilabels = (int*) malloc(num_points*sizeof(int));
    for(int i=0;i<num_points;i++) ilabels[i] = labels[i];
    for(int i=0;i<k;i++) { imeans[i].x = means[i].x; imeans[i].y = means[i].y;} 
    printf("imeans and ilabels Allocated\n");

    Point* gpumeans;
    cudaMalloc(&gpumeans, k*sizeof(Point));
    cudaMemcpy(gpumeans,means,k*sizeof(Point),cudaMemcpyHostToDevice); 
    printf("GPU means Allocated\n");
    
    int* gpulabels;
    cudaMalloc(&gpulabels, num_points*sizeof(int));
    cudaMemcpy(gpulabels,labels,num_points*sizeof(int),cudaMemcpyHostToDevice); 
    printf("GPU labels Allocated\n");

    int* gpuilabels;
    cudaMalloc(&gpuilabels, num_points*sizeof(int));
    cudaMemcpy(gpuilabels,ilabels,num_points*sizeof(int),cudaMemcpyHostToDevice); 
    printf("GPU ilabels Allocated\n");

    int iter = 1000; // maximum iteration
    float prob = 0.0001; // converging criteria
    // cpu
    float *dist = (float*)malloc(num_points*k*sizeof(float));
    printf("Dist Allocated\n");
    float* gpudist;
    cudaMalloc(&gpudist, num_points*k*sizeof(float));
    printf("GPU Dist Allocated\n");

    ///////////////////////// *************** STANDARD CPU ALGORITHM *************:228******* 
    int it = 0;
    float time = 0;
    float avgout[3];
    for(int j=0;j<3;j++) avgout[j]=0;

    while(it < iter){
        float out[3];
        float ct = kmeans_cpu( points,means,labels,dist,iter,num_points,k,out);
        
        time+=ct;
        it+=1;
        
        for(int j=0;j<3;j++) avgout[j]+=out[j];
        float ch = (float) out[1];
        if( ch/num_points < prob ) break;
    }

    printf("CPU Time taken: %.6f ms\n", time/(float)it);

    for(int j=0;j<3;j++) avgout[j]/=it;
    //// Writing to file ////
    fprintf(outfile,"%f %f %f\n",avgout[0],avgout[1],avgout[2]);
    for(int i=0;i<k;i++)
    {
        fprintf(outfile,"%f %f\n",means[i].x,means[i].y);
    }
    for(int i=0;i<num_points;i++){ fprintf(outfile,"%d ",labels[i]);}
    fprintf(outfile,"\n");

    for(int i=0;i<num_points;i++){ printf("%d ",labels[i]);}
    printf("\n");
    
    ///////////////////////// *************** STANDARD GPU ALGORITHM ******************** 
    it = 0;
    time = 0;
    for(int j=0;j<3;j++) avgout[j]=0;
    while(it < iter){
            float out[3];
            float gt = kmeans_gpu(gpupoints,points,gpumeans,gpulabels,gpudist,dist,iter,num_points,k,out);

            time+=gt;
            it++;
            
            for(int j=0;j<3;j++) avgout[j]+=out[j];
            float ch = (float) out[1];
             if( ch/num_points < prob) break;
    }

     printf("GPU Time taken: %.6f ms\n", time/it);
     printf("iterations %d\n",it);
    cudaMemcpy(labels,gpulabels,num_points*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(means,gpumeans,k*sizeof(Point),cudaMemcpyDeviceToHost);

    for(int j=0;j<3;j++) avgout[j]/=it;
     //// Writing to file ////
    for(int i=0;i<num_points;i++){ printf("%d ",labels[i]);}
    printf("\n");
    fprintf(outfile,"%f %f %f\n",avgout[0],avgout[1],avgout[2]);
    for(int i=0;i<k;i++)
    {
        fprintf(outfile,"%f %f\n",means[i].x,means[i].y);
    }
    for(int i=0;i<num_points;i++){ fprintf(outfile,"%d ",labels[i]);}
    fprintf(outfile,"\n");
    
    ///////////////////////// *************** Reinforced CPU ALGORITHM ******************** 
    it = 0;
    time = 0;
    float* icd = (float*) malloc(k*k*sizeof(float));
    int* rid = (int*) malloc(k*k*sizeof(int));
    printf("RID and ICD Allocated\n");
    for(int j=0;j<3;j++) avgout[j]=0;

    while(it < iter){
        float out[3];
        float ct = kmeans_cpu_ineq( points,imeans,ilabels,icd,rid,iter,num_points,k,out);

        time+=ct;
        it+=1;
        for(int j=0;j<3;j++) avgout[j]+=out[j];
        float ch = (float) out[1];
        if( ch/num_points < prob) break;
    }

    printf("CPU Time taken for ineq cpu: %.6f ms\n", time/it);
    for(int i=0;i<num_points;i++){ printf("%d ",ilabels[i]);}
    printf("\n");

    for(int j=0;j<3;j++) avgout[j]/=it;
     //// Writing to file ////
    fprintf(outfile,"%f %f %f\n",avgout[0],avgout[1],avgout[2]);
    for(int i=0;i<k;i++)
    {
        fprintf(outfile,"%f %f\n",imeans[i].x,imeans[i].y);
    }
    for(int i=0;i<num_points;i++){ fprintf(outfile,"%d ",ilabels[i]);}
    fprintf(outfile,"\n");


    // Replacing imeans and ilabels as random for the GPU Algo
    for(int i=0;i<num_points;i++)
    {
        // cout<<(i%k)<<endl;
        ilabels[i] = i%k;
    }
   
    for(int i=0;i<k;i++)
    {
        imeans[i].x = points[ind[i]].x;
        imeans[i].y = points[ind[i]].y;
    }
   
    ///////////////////////// *************** Reinforced GPU ALGORITHM (Inefficient) ******************** 
    int* mcpu;
    mcpu = (int*) malloc(num_points*sizeof(int));
   
    it = 0;
    time = 0;
    for(int j=0;j<3;j++) avgout[j]=0;
    int* m;
    cudaMalloc(&m,num_points*(sizeof(int)));
   
    while(it<iter){
        float out[3];
        cudaMemset(m,0,num_points*sizeof(int));
        float gt = kmeans_gpu_ineq( gpupoints, imeans, gpuilabels, icd, rid, iter, num_points, k,out,m);
        cudaMemcpy(mcpu,m,num_points*sizeof(int),cudaMemcpyDeviceToHost);
        double c = 0;
        for(int i=0;i<num_points;i++) c+=mcpu[i];
        // printf("%f \n", c/num_points);
        time+=gt;
        it++;
        for(int j=0;j<3;j++) avgout[j]+=out[j];
        float ch = (float) out[1];
        //  printf("%f\n",ch);
        if( ch/num_points < prob) break;
    }

    printf("GPU Time taken for ineq gpu inefficient version: %.6f ms\n", time/it);
     printf("iterations %d\n",it);
    cudaMemcpy(ilabels,gpuilabels,num_points*sizeof(int),cudaMemcpyDeviceToHost); 
    for(int i=0;i<num_points;i++){ printf("%d ",ilabels[i]);}
    printf("\n");
    for(int j=0;j<3;j++) avgout[j]/=it;
    
     //// Writing to file ////
    fprintf(outfile,"%f %f %f\n",avgout[0],avgout[1],avgout[2]);
    for(int i=0;i<k;i++)
    {
        fprintf(outfile,"%f %f\n",imeans[i].x,imeans[i].y);
    }
    for(int i=0;i<num_points;i++){ fprintf(outfile,"%d ",ilabels[i]);}
    fprintf(outfile,"\n");

    // // Replacing imeans and ilabels as random
    // for(int i=0;i<k;i++) { num[i] = 0;}
    for(int i=0;i<num_points;i++)
    {
        // cout<<(i%k)<<endl;
        ilabels[i] = i%k;
    }
    for(int i=0;i<k;i++)
    {
        imeans[i].x = points[ind[i]].x;
        imeans[i].y = points[ind[i]].y;
    }
    // restoring gpu variables
    cudaMemcpy(gpuilabels,ilabels,num_points*sizeof(int),cudaMemcpyHostToDevice);
    ///////////////////////// *************** Efficient Reinforced GPU ALGORITHM ******************** 
    // Epoch 1
    printf("Efficient implementation\n");
    printf("Epoch 1\n");
    it = 0;
    time = 0;
    for(int j=0;j<3;j++) avgout[j]=0;
    
    while( it < 3)
    {
        float out[3];
        
        cudaMemset(m,0,num_points*sizeof(int));

        float gt = kmeans_gpu_ineq( gpupoints, imeans, gpuilabels, icd, rid, iter, num_points, k,out,m);
        cudaMemcpy(mcpu,m,num_points*sizeof(int),cudaMemcpyDeviceToHost);

        // printf("mcpus\n");
        // for(int i=0;i<num_points;i++) printf("%d ",mcpu[i]);
        // printf("\n");

        float c = 0;
        
        for(int i=0;i<num_points;i++) c+=mcpu[i];
        
        time+=gt;
        it++;

        for(int j=0;j<3;j++) avgout[j]+=out[j];
        float ch = (float) out[1];

        if( ch/num_points < prob) break;
    }
    printf("GPU Time taken for ineq gpu Epoch 1: %.6f ms\n", time/it);

    vector<pair<int,int>> v;
    for(int i=0;i<num_points;i++) v.push_back(make_pair(-mcpu[i],i));
    sort(v.begin(),v.end());
    
    printf("Rearranging Points\n");
    int* inds = (int*)malloc(num_points*sizeof(int));
    int* gpuinds;
    cudaMalloc(&gpuinds,num_points*sizeof(int));
    for(int i=0;i<num_points;i++)
    {
        inds[i] = v[i].second;
    }
    cudaMemcpy(gpuinds,inds,num_points*sizeof(int),cudaMemcpyHostToDevice);

    Point* points2 = (Point*) malloc(num_points*sizeof(Point));
    for(int i=0;i<num_points;i++) { points2[i].x = points[inds[i]].x; points2[i].y = points[inds[i]].y; }
    cudaMemcpy(gpupoints,points2,num_points*sizeof(Point),cudaMemcpyHostToDevice); 

    printf("GPU Points Copied\n");
    printf("Epoch 2\n");
    // it = 0;
    // time = 0;
    // for(int j=0;j<3;j++) avgout[j]=0;
    printf("Indices\n");
    for(int i=0;i<num_points;i++) printf("%d ",inds[i]);
    printf("\n");
    
    time = 0;
    it = 0;
    for(int j=0;j<3;j++) avgout[j]=0;

    while(it<iter)
    {
        float out[3];
        float gt = kmeans_gpu_ineq_eff( gpupoints,gpuinds, imeans, gpuilabels, icd, rid, iter, num_points, k,out);
        time+=gt;
        it++;
        for(int j=0;j<3;j++) avgout[j]+=out[j];
        float ch = (float) out[1];
        //  printf("ch : %f\n",ch);
        if( ch/num_points < prob) break;
    }
    printf("GPU Time taken for ineq gpu effective: %.6f ms\n", time/it);
     printf("iterations %d\n",it);
    cudaMemcpy(ilabels,gpuilabels,num_points*sizeof(int),cudaMemcpyDeviceToHost); 

    for(int i=0;i<num_points;i++){ printf("%d ",ilabels[i]);}
    printf("\n");
    for(int j=0;j<3;j++) avgout[j]/=it;
    
     //// Writing to file ////
    fprintf(outfile,"%f %f %f\n",avgout[0],avgout[1],avgout[2]);
    for(int i=0;i<k;i++)
    {
        fprintf(outfile,"%f %f\n",imeans[i].x,imeans[i].y);
    }
    for(int i=0;i<num_points;i++){ fprintf(outfile,"%d ",ilabels[i]);}
    fprintf(outfile,"\n");
    return 0;
}


