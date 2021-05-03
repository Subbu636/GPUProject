#include "util.h"

__host__ __device__ double *create(int m, int n){
    return (double*)malloc(sizeof(double)*n*m);
}

__host__ __device__ void init(double *v, int m, int n, double val){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;++j){
            v[i*n+j] = val;
        }
    }
}

__host__ __device__ void copy(double *a, double *b, int m, int n){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;++j){
            a[i*m+j] = b[i*m+j];
        }
    }
}


__host__ __device__ double *transpose(double *trans, double *v, int m, int n){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;j++){
            trans[j*m+i] = v[i*n+j];
        }
    }
    return trans;
}

__host__ __device__ double *matsub(double *res, double *v1, double *v2, int m, int n){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;++j){
            res[i*n+j] = v1[i*n+j]-v2[i*n+j];
        }
    }
    return res;
}

__host__ __device__ double *matmul(double *res, double *a, double *b, int m, int s, int n){
    init(res, m, n, 0.0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < s; k++) res[i*n+j] += a[i*s+k]*b[k*n+j];
        }
    }
    return res;
}

__host__ __device__ double *matadd(double *res, double *v1, double *v2, int m, int n){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;++j){
            res[i*n+j] = v1[i*n+j]+v2[i*n+j];
        }
    }
    return res;
}

__host__ __device__ double *matsmul(double *res, double *v, double val, int m, int n){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;++j){
            res[i*n+j] = val*v[i*n+j];
        }
    }
    return res;
}

__host__ __device__ double *cofactor(double *res, double *a, int p, int q, int n)
{
    int r = 0, c = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i != p && j != q)
            {
                res[r*(n-1)+(c++)] = a[i*n+j];
                if (c == n - 1)
                {
                    c = 0;
                    r++;
                }
            }
        }
    }
    return res;
}

__host__ __device__ double matdet(double *a, int n)
{
    double res = 0.0;
    if (n == 1) return a[0];
    double s = 1.0, *temp = create(n-1,n-1);
    for (int i = 0; i < n; i++)
    {
        res += (s*a[i]*matdet(cofactor(temp,a,0,i,n),n-1));
        s = -1.0*s;
    }
    // free(temp);
    return res;
}

__host__ __device__ double *matadj(double *res, double *a, int n)
{
    if(n == 1){
        res[0] = 1.0;
        return res;
    }
    double s = 1.0, *temp = create(n,n);
    for(int i = 0;i < n;i++)
    {
        for(int j = 0;j < n;++j)
        {
            if((i+j)%2 == 0) s = 1.0;
            else s = -1.0;
            res[j*n+i] = (s)*(matdet(cofactor(temp,a,i,j,n), n-1));
        }
    }
    free(temp);
    return res;
}

__host__ __device__ double *matinv(double *res, double *a, int n)
{
    double d = matdet(a,n);
    if (d == 0.0)
    {
        printf("#singular - inverse not possible\n");
        assert(false);
    }
    double *adj = create(n,n);
    matadj(adj,a,n);
    for(int i = 0;i < n;++i){
        for(int j = 0;j < n;++j){
            res[i*n+j] = adj[i*n+j]/d;
        }
    }
    free(adj);
    return res;
}

__host__ __device__ double norm(double *x, double *mu, double *sigma, int d){
    double deno = sqrt(pow((double)(2.0*3.14159),(double)d)*matdet(sigma,d));
    double *dif = create(d,1),*temp = create(d,1),*inv = create(d,d),val[1],*trans = create(d,1);
    matsub(dif,x,mu,d,1);
    double expo = (-0.5)*(matmul(val,matmul(temp,transpose(trans,dif,d,1),matinv(inv,sigma,d),1,d,d),dif,1,d,1)[0]);
    // printf("%lf %lf\n",matdet(sigma,d),inv[0]);
    free(dif); free(temp); free(inv); free(trans);
    return exp(expo)/deno;
}

__host__ __device__ void matprint(double *a, int m, int n){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;j++){
            printf("%lf ",a[i*n+j]);
        }
        printf ("\n");
    }
}


void gmix_cpu(double *p,double *r, int k, int iter, int l, int d, double *ctime){

    // Debugging
    // double arr[4] = {2.0,1.0,1.0,2.0};
    // double mu[2] = {1.0, 1.0}, x[2] = {1.0, 0.0};
    // cout<<norm(x,mu,arr,2)<<endl;

    double n[k],pi[k],mu[k*d],sigma[k*d*d];
    double temp[d], var[d*d],trans[d];
    struct timeval t1, t2;

    for(int t = 0;t < iter;++t){
        gettimeofday(&t1, 0);
        // M-step
        init(n,1,k,0.0);
        for(int i = 0;i < k;++i){
            for(int j = 0;j < l;++j){
                n[i] += r[j*k+i];
            }
        }
        // matprint(n,1,k);
        // cout<<"n--------------------------------"<<endl;
        for(int i = 0;i < k;++i){
            pi[i] = n[i]/(double)l;
        }
        // matprint(pi,1,k);
        // cout<<"pi--------------------------------"<<endl;
        init(mu,k,d,0.0);
        for(int i = 0;i < k;++i){
            for(int j = 0;j < l;++j){
                matadd(&mu[i*d],&mu[i*d],matsmul(temp,&p[j*d],r[j*k+i],d,1),d,1);
            }
        }
        for(int i = 0;i < k;++i){
            matsmul(&mu[i*d],&mu[i*d],1/n[i],d,1);
        }
        // matprint(mu,1,k*d);
        // cout<<"mu--------------------------------"<<endl;
        init(sigma,k,d*d,0.0);
        for(int i = 0;i < k;++i){
            for(int j = 0;j < l;++j){
                matsub(temp,&p[j*d],&mu[i*d],d,1);
                matmul(var,temp,transpose(trans,temp,d,1),d,1,d);
                matsmul(var,var,r[j*k+i],d,d);
                matadd(&sigma[i*d*d],&sigma[i*d*d],var,d,d);
            }
        }
        for(int i = 0;i < k;++i){
            matsmul(&sigma[i*d*d],&sigma[i*d*d],1/n[i],d,d);
        }
        // matprint(sigma,1,k*d*d);
        // cout<<"sigma--------------------------------"<<endl;
        gettimeofday(&t2, 0);
        ctime[t*2] = ((double)(1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0);
        gettimeofday(&t1, 0);
        // E-step
        for(int i = 0;i < l; ++i){
            double s = 0;
            for(int j = 0;j < k; ++j){
                r[i*k+j] = pi[j]*norm(&p[i*d],&mu[j*d],&sigma[j*d*d],d);
                s+=r[i*k+j];
            }
            for(int j = 0;j < k;++j){
                r[i*k+j] = r[i*k+j]/s;
            }
        }
        // matprint(r,1,k);
        // cout<<"r--------------------------------"<<endl;
        gettimeofday(&t2, 0);
        ctime[t*2+1] = ((double)(1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0);
    }
    return;  
} 

__global__ void compute_n_pi(double *pi, double *n, double *r, int l, int k){
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= k) return;
    for(int i = 0;i < l;++i){
        n[id] += r[i*k+id];
    }
    pi[id] = n[id]/(double)l;
}

__global__ void compute_pi(double *pi, double *n, int l, int k){
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= k) return;
    pi[id] = n[id]/(double)l; 
}

__global__ void compute_div(double *m, double *n, int k, int d){
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= k) return;
    for(int i = 0;i < d;i++){
        m[id*d+i] = m[id*d+i]/n[id];
    }
}

__global__ void compute_v(double *v, double *p, double *mu, double *r, int l, int k, int d){
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= k*l) return;
    int i = id/k, j = id%k;
    double *temp = create(d,1), *var = create(d,d), *trans = create(d,1);
    matsub(temp,&p[i*d],&mu[j*d],d,1);
    matmul(var,temp,transpose(trans,temp,d,1),d,1,d);
    for(int s = 0;s < d*d;++s){
        v[l*(j*d*d + s) + i] = var[s]*r[i*k+j];
    }
    free(temp); free(var); free(trans);
}

__global__ void compute_r(double *r, double *pi, double *p, double *mu, double *sigma, int l, int k, int d){
    int i = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(i >= l) return;
    double s = 0.0;
    for(int j = 0;j < k; ++j){
        r[i*k+j] = pi[j]*norm(&p[i*d],&mu[j*d],&sigma[j*d*d],d);
        s+=r[i*k+j];
    }
    for(int j = 0;j < k;++j){
        r[i*k+j] = r[i*k+j]/s;
    }
}

__global__ void gmatprint(double *a, int m, int n){
    matprint(a,m,n);
}

void cublas_atb(double *res, const double *a, const double *b, const int m, const int k, const int n, cublasHandle_t &handle) {
    int lda=n,ldb=m,ldc=n;
    const double alf = 1.0;
    const double bet = 0.0;
    const double *alpha = &alf;
    const double *beta = &bet;
    cublasStatus_t stat;
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, alpha, b, lda, a, ldb, beta, res, ldc);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("#cublas multiply error\n");
        exit(1);
    }
}

void cublas_ab(double *res, const double *a, const double *b, const int m, const int k, const int n,cublasHandle_t &handle) {
    int lda=n,ldb=k,ldc=n;
    const double alf = 1.0;
    const double bet = 0.0;
    const double *alpha = &alf;
    const double *beta = &bet;
    cublasStatus_t stat;
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, b, lda, a, ldb, beta, res, ldc);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("#cublas multiply error\n");
        exit(1);
    }
}
 
void gmix_gpu(double *cp,double *cr, int k, int iter, int l, int d, double *gtime){
    cudaDeviceSetLimit(cudaLimitStackSize, 1 << 25);
    // debugging
    // double *res, *a, *b;
    // cudaMalloc(&res, 6*sizeof(double));
    // cudaMalloc(&a, 6*sizeof(double));
    // cudaMalloc(&b, 6*sizeof(double));
    // double ca[6] = {1.0,2.0,3.0,4.0,5.0,6.0}, cb[6] = {1.0,0.0,0.0,1.0,1.0,0.0};
    // cudaMemcpy(a,ca,6*sizeof(double),cudaMemcpyHostToDevice);
    // cudaMemcpy(b,cb,6*sizeof(double),cudaMemcpyHostToDevice);
    // cublas_atb(res, a, b, 3, 2, 3);
    // gmatprint<<<1,1>>>(res,3,3);
    

    double *n,*pi,*mu,*sigma;
    cudaMalloc(&n,k*sizeof(double));
    cudaMalloc(&pi,k*sizeof(double));
    cudaMalloc(&mu,k*d*sizeof(double));
    cudaMalloc(&sigma,k*d*d*sizeof(double));
    double *temp,*var,*trans,*lones;
    cudaMalloc(&temp,d*sizeof(double));
    cudaMalloc(&var,d*d*sizeof(double));
    cudaMalloc(&trans,d*sizeof(double));
    cudaMalloc(&lones,l*sizeof(double));
    double clo[l]; fill_n(clo, l, 1.0);
    cudaMemcpy(lones,clo,l*sizeof(double),cudaMemcpyHostToDevice);

    double *p,*r,*v;
    cudaMalloc(&p,l*d*sizeof(double));
    cudaMalloc(&r,l*k*sizeof(double));
    cudaMalloc(&v,l*d*d*k*sizeof(double));
    cudaMemcpy(p,cp,l*d*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(r,cr,l*k*sizeof(double),cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("#handle create error\n");
        exit(1);
    }
    struct timeval t1, t2;

    // rik -> P(aprior), pi -> alpha, 

    for(int t = 0; t < iter;++t){
        gettimeofday(&t1, 0);
        // M-step
        cublas_atb(n,r,lones,k,l,1,handle);
        // gmatprint<<<1,1>>>(n, 1, k);
        // cudaDeviceSynchronize();
        // cout<<"n-----------------------------"<<endl;
        compute_pi<<<k,1>>>(pi,n,l,k);
        cudaDeviceSynchronize();
        // gmatprint<<<1,1>>>(pi, 1, k);
        // cudaDeviceSynchronize();
        // cout<<"pi-----------------------------"<<endl;
        cublas_atb(mu, r, p, k, l, d,handle);
        // gmatprint<<<1,1>>>(mu, 1, k*d);
        // cudaDeviceSynchronize();
        // cout<<"mu-----------------------------"<<endl;
        compute_div<<<k,1>>>(mu,n,k,d);
        cudaDeviceSynchronize();
        // gmatprint<<<1,1>>>(mu, 1, k*d);
        // cudaDeviceSynchronize();
        // cout<<"mu-----------------------------"<<endl;
        compute_v<<<l*k,1>>>(v,p,mu,r,l,k,d);
        cudaDeviceSynchronize();
        cublas_ab(sigma, v, lones, d*d*k,l,1,handle);
        // gmatprint<<<1,1>>>(sigma, k, d*d);
        // cudaDeviceSynchronize();
        // cout<<"sigma-----------------------------"<<endl;
        compute_div<<<k,1>>>(sigma,n,k,d*d);
        cudaDeviceSynchronize();
        // gmatprint<<<1,1>>>(sigma, k, d*d);
        // cudaDeviceSynchronize();
        // cout<<"sigma-----------------------------"<<endl;
        gettimeofday(&t2, 0);
        gtime[t*2] = ((double)(1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0);
        gettimeofday(&t1, 0);
        // E-step
        compute_r<<<l,1>>>(r, pi, p, mu, sigma, l, k, d);
        // gmatprint<<<1,1>>>(r, 1, k);
        // cudaDeviceSynchronize();
        // cout<<"r0-----------------------------"<<endl;
        gettimeofday(&t2, 0);
        gtime[t*2+1] = ((double)(1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0);

    }
    stat = cublasDestroy(handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("#handle destroy error\n");
        exit(1);
    }
    cudaMemcpy(cr,r,l*k*sizeof(double),cudaMemcpyDeviceToHost);
    return;
}


