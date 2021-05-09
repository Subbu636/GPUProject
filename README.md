# Clustering Algorithms using CUDA

  a) K Means 
  1. To compile use : make kmeans
  2. To run : ./kmeans (path to input file) (path to output file) (k value) 
  


  b) Gaussian Mixture Model
  1. To compile use : make gmix (for normal matrix multiplication) or make gmix-cublas(matrix maultiplication using cublas)
  2. To run : ./(gmix or gmix-cublas) <path to input file> <k value> <num_iterations> <dim_value> <num_points>
  
  Note : Use make clean to clear compilation files
