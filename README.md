# Clustering Algorithms using CUDA

  a) K Means 
  1. To compile use : make kmeans
  2. To run : ./kmeans <path_to_input_file> <path_to_output_file> <k_value> 
  


  b) Gaussian Mixture Model
  1. To compile use : make gmix (for normal matrix multiplication) or make gmix-cublas(matrix maultiplication using cublas)
  2. To run : ./(gmix or gmix-cublas) <path_to_input_file> <k_value> <num_iterations> <dim_value> <num_points>
  
  Note : Use make clean to clear compilation files
