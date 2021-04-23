#include<bits/stdc++.h>
#include <stdio.h>
#include "util.h"
using namespace std;

double Point::distance(Point p){
    return (p.x - x)*(p.x - x) + (p.y - y)*(p.y - y);
}

void read_file(char *filename, vector <Point> &points){
    FILE *file;
    file = fopen(filename, "r");
    if (!file)  {
        printf("#cannot open input file");
        return;
    }
    double x,y;
    while(fscanf(file,"%lf %lf",&x,&y) != EOF){
        Point p;
        p.x = x;
        p.y = y;
        points.push_back(p);
    }
}