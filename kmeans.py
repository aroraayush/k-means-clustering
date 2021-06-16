from cluster import cluster
from sklearn.datasets._samples_generator import make_blobs
import numpy as np
import random
import math
import statistics

class KMeans(cluster):
    
    # initialize cluster size and maximum iterations
    def __init__(self, k = 5, max_iterations = 100) -> None:
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = list()

    #====================== KMeans (Fit) ===========================
    
    def fit(self, X) -> None:
        """
        X is the list of n instances with several features that overall describe the n instances
        returns cluster labels and cluster centroids
        """
        # Initializing centroids in self.centroids
        self.place_k_centroids(X)

        print("Initial centroids = ", self.centroids)

        # Cluster Assignment & Move centroid step
        for _ in range(self.max_iterations):
            centroids = self.centroids
            distance = self.compute_eucl_dist_centroid_X(X, centroids)
            self.labels = self.finding_closest_cluster_to_X(distance, X)
            # Updating the centroids
            self.calculate_new_centroids(X, self.labels)
            if centroids == self.centroids:
                break;

        print("New centroids = ", self.centroids)
        return self.labels, self.centroids

    #================ KMeans Fit Extended ===========================
    
    def fit_extended(self, X, balanced = False) -> None:
        """
        X is the list of n instances with several features that overall describe the n instances
        balanced checks if each cluster have almost equal number of data points i.e. clusters must be balanced.
        returns cluster labels and cluster centroids
        """
        if(balanced == False):
            return self.fit(X)
        else:
            # Initializing centroids in self.centroids
            self.place_k_centroids(X)
            for _ in range(self.max_iterations):
                centroids = self.centroids
                distance = self.compute_eucl_dist_centroid_X(X, centroids)
                self.labels = self.finding_closest_cluster_balanced(distance, X)

                # Updating the centroids
                self.calculate_new_centroids(X, self.labels)
                if centroids == self.centroids:
                    break;
            return self.labels, self.centroids
        
    #================ Helper Functions ===========================

    # Initializing centroid as points fromt the data
    def place_k_centroids(self, X) -> None:
        random_init = np.random.randint(len(X), size=len(X))
        self.centroids = [X[s] for s in random.sample(random_init.tolist(), self.k)]
    
    # Computing the eucleadean distance from each data point to each centroid
    def compute_eucl_dist_centroid_X(self, X, centroids):
        distance = []
        # Storing the centroid distance for each X in collection of 4, but together
        #[dist(X1C1),dist(X1C2),dist(X1C3),dist(X1C4),dist(X2C1),dist(X2C2),dist(X2C3),dist(X2C4).....]
        for x in X:
            for k in centroids:
                distance.append(math.sqrt(((x[0] - k[0])**2 + (x[1] - k[1])**2)))
        return distance


    # finding the closest cluster to each data point based on the calculated distance values
    def finding_closest_cluster_to_X(self, distance, X) -> None:
        closest_cluster_indices = []
        # Note : Centroid distance for each X were stored as follows:
        #[dist(X1C1),dist(X1C2),dist(X1C3),dist(X1C4),dist(X2C1),dist(X2C2),dist(X2C3),dist(X2C4).....]
        # Extracting 'k' (4) at a time
        for i in range(len(X)):
            centr_dist_for_x_i = distance[( i * self.k ) : ( i * self.k ) + self.k]
            min_dis = min(centr_dist_for_x_i)
            centroid_chosen = centr_dist_for_x_i.index(min_dis)
            closest_cluster_indices.append(centroid_chosen)
        return closest_cluster_indices
    

    # Finding the closest cluster to each data point with 
    # Added new optional condition to assign equal data points to each cluster
    def finding_closest_cluster_balanced(self, distance, X):
        closest_cluster_indices = []
        balance_number = len(X)/self.k
        
        # Counting the number of points per centroid
        centroids_dict = {}
        for i in range(len(X)):
            # Centroid distance for x(i)
            centr_dist_for_x_i = distance[( i * self.k ) : ( i * self.k ) + self.k]
            min_dis = min(centr_dist_for_x_i)
            centroid_chosen = centr_dist_for_x_i.index(min_dis)

            while centroid_chosen in centroids_dict and centroids_dict[centroid_chosen] >= balance_number:
                # Increasing the centroid distance from point
                centr_dist_for_x_i[centroid_chosen] = 9999999
                centroid_chosen = centr_dist_for_x_i.index(min(centr_dist_for_x_i))

            centroids_dict[centroid_chosen] = centroids_dict.get(centroid_chosen, 0) + 1
            closest_cluster_indices.append(centroid_chosen)

        return closest_cluster_indices

    # calculating the new centroids for X
    def calculate_new_centroids(self, X, labels):
        centroid_dict = dict()
        new_centroids = []
        for k in range(0,self.k):
            data_points_list = list()
            for index,label in enumerate(labels):
                # Checking for a specific centroid label
                if(label == k):
                    data_points_list.append(X[index])

            # If none found, assigning a default centroid value
            if(len(data_points_list) == 0):
                data_points_list.append([0.0,0.0])
            
            centroid_dict[k] = data_points_list

        # Computing new centroid by taking a mean of the points
        for k_val in range(0,self.k):
            new_centroids.append([statistics.mean(i) for i in zip(*centroid_dict[k_val])])
            
        self.centroids = new_centroids



km_object = KMeans(4,100)
# Generating 200 instances of data points i.e. coordinates for the 100 instances
X, cluster_assignments = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=0)
x_list = X.tolist()

# Passing n instances
result_kmeans = km_object.fit(x_list)
print("\n\n\n",result_kmeans)

# Writing output to file
print(result_kmeans, file=open("kmeans_output.txt", "w"))

# Passing additional boolean value
result_extended = km_object.fit_extended(x_list, True)
print("\n\n\n",result_extended)

# Writing output to file
print(result_extended, file=open("extended_kmeans_output.txt", "w"))
