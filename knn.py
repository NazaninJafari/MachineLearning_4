
import numpy as np

class kNearestNeighbors:
    def __init__(self, k):
        self.k = k
    
    #train    
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
         
    #calculate distance
    def euclidianDistance(self, a, b):
        distance = np.sqrt(np.sum((a-b)**2))
        return distance
    
    # return array of nearest neighbors
    def nearNeighbors(self, X_test):
        dists= []
        for x_train in self.X_train:
            dist = self.euclidianDistance(x_train, X_test)
            dists.append(dist)
            
        index_sorted = np.argsort(dists)
        #Gender of index
        gender_sorted = self.Y_train[index_sorted]
        # namayesh gender az avalin ta k omin hamsayeh
        return gender_sorted[0:self.k]
        
    def predict(self, X_test):
        neighbors = self.nearNeighbors(X_test)
        Y_test = np.argmax(np.bincount(neighbors))
        
        return Y_test
    
    def evaluate(self, X_test, Y_test):
        correct = 0
        test_array = []
        for x in X_test:
            test = self.predict(x)
            test_array.append(test)
        for i in range(len(test_array)):
            if test_array[i] == Y_test[i]:
                correct +=1
        
        accuracy = correct/len(Y_test)
        return accuracy