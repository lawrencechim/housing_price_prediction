# Housing price prediction using k-NN

### Questions
***

1.  Please refer to the file knn.py to find the implementation of the k-NN model
    which takes care of the issue of time leakage.

2.  Keeping k=4, the MRAE is approximately 0.1986 using a 5-fold cross
    validation with TimeSeriesSplit.

3. The general rule of thumb is to select k as the square root of n, where n
   is the number of rows in the dataset at hand. Another way to find the optimal
   k is to use the elbow method. Construct a graph with error on the y-axis
   and k on the x-axis. Identify the 'elbow' of the graph where the error dips
   significantly from k-1 to k and where the decreases from k+1 onwards are
   small and not worth the tradeoff of extra computational time

4. Spacial trends - From EDA, it seems that majority of the houses have latitude
   between 35 to 38 and longitude between -100 to -96. Price predictions outside
   this geographical cluster will not be as accurate due to major data imbalance.

   Spacial trends - There doesn't seem to be any major temporal trends. MRAE
   remains quite consistent through each of the 5 fold cross validation.

5. The current implementation of the model is naive and computationally
   expensive. Here are a few ways the model could be improved:
   *  choosing an optimal number k as specified in Question 3
   *  enhance the run-time of the k-NN algorithm by making it parallelizable
      using Hadoop/MapReduce
   *  use a more efficient algorithm such as K-D tree to speed up run time. Such
      algorithms are faster as they don't compare every single point to find the
      closest neighbors, but rather they eliminate clusters of points that are
      unlikely to be a nearest neighbor

6. To productionize the model, create an interactive app which displays the
   predicted prices, location (geospatial), and k-nearest neighbors in real-time
   , taking into account the points raised in Question 5. Also consider changing
   the programming language when productionizing the model to a more low-level
   language in an effort to reduce run-time.
