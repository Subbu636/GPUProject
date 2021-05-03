#include<bits/stdc++.h>
#include <stdio.h>
#include "util.h"
using namespace std;


void read_file(char *filename, vector <Point> &points){
    FILE *file;
    file = fopen(filename, "r");
    if (!file)  {
        printf("#cannot open input file");
        return;
    }
    float x,y;
    while(fscanf(file,"%f %f",&x,&y) != EOF){
        Point p;
        p.x = x;
        p.y = y;
        points.push_back(p);
    }
}

double** make_vector(Point &p){
    double *x = new double(p.x),*y = new double(p.y);
    double ** vec = new double*[2];
    vec[0] = x;
    vec[1] = y;
    return vec;
}

double *read_file_array(char *fname, int d,int *l){
    FILE *file;
    file = fopen(fname, "r");
    if (!file)  {
        cout<<"#cannot open input file"<<endl;
        assert(false);
    }
    vector <double> vec;
    double x;
    while(fscanf(file,"%lf",&x) != EOF){
        vec.push_back(x);
    }
    *l = vec.size()/d;
    double *arr = create(*l,d);
    int k = 0;
    for(int i = 0;i < *l; i++){
        for(int j = 0;j < d;++j){
            arr[i*d+j] = vec[k];
            k++;
        }
    }
    return arr;
}

double *init_prob(int l, int k){
    double *prob = create(l,k);
    for(int i = 0;i < l;++i){
        double s = 0.0;
        for(int j = 0;j < k;++j){
            prob[i*k+j] = (double) ((rand()%10)+1.0);
            s += prob[i*k+j];
        }
        // cout<<s<<endl;
        for(int j = 0;j < k;++j){
            prob[i*k+j] = prob[i*k+j]/s;
            // cout<<i<<":"<<j<<":"<<prob[i*k+j]<<endl;
        }
    }
    return prob;
}

void fprint_mat(char *fname, double *prob, int l, int k){
    FILE *op = fopen(fname,"w");
    for(int i = 0;i < l;++i){
        for(int j = 0;j < k;++j) fprintf(op,"%.6f ",prob[i*k+j]);
        fprintf(op,"\n");
    }
}