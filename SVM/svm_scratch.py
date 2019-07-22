import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


# A very naive approach to SVM with a lot of assumptions and intuitions
# Real SVM shouldn't work like this

class Support_Vector_Machine:
    def __init__(self, visualization=True):        
        self.visualization = visualization
        # it's good to add the colors beforehand
        # r = red, b = blue
        self.colors = {1:'r',-1:'b'}        
        if self.visualization:            
            self.fig = plt.figure()
            # adds a subplot of 1x1 in the location 1 of the main figure
            self.ax = self.fig.add_subplot(1,1,1)                        
            
    def fit(self,data):
        self.data = data

        #( ||w||: [w,b]}
        # it will store the optimized [w, b] pairs and then select the one with the least ||w||
        opt_dict = {}

        # used for finding the most apt vector w by observing it in all the directions
        # the assumption is that the vector w is along the x-y=0 line
        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                	# feature is the individual number
                	# used just to get the max and the min from the while list of numbers
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        # self.max_feature_value = 8
        self.min_feature_value = min(all_data)
        # self.min_feature_value = -1
        all_data = None

        # it is preferred to use the max_feature_value to set the step_size as 10%, 1%, 0.1%, etc.
        # len(step_sizes) = max_iter in our scratch code
        step_sizes = [self.max_feature_value * 0.1, # = 8
                      self.max_feature_value * 0.01, # = 0.8
                      # point of expense:
                      self.max_feature_value * 0.001 # = 0.08
                      ]
        #             More steps more refined answer
        #             self.max_features_value * 0.0001,]

        #This makes yi(xi*w+b)=1 closer to 1
        #e.g 1.01 or 1.0001 never >1 as the loop yi(xi*w+b) >=1 
        #extermely expensive
        #more optimized greater costs time

        # this multiplier helps setting the range of b
        # from where to where it will traverse
        # by manipulating max_feature_value 
        b_range_multiple = 5
        
        #We do need need to take as small of steps with b as we do with w 
        # the multiplier is used to manipulate the max_feature_value to fit in the step_size of b
        b_step_multiple = 5

        # just a basic value of vector w to start with
        latest_optimum = self.max_feature_value*10 # = 80

        for step in step_sizes:
            # made that assumption
            w=np.array([latest_optimum,latest_optimum])
            optimized = False
            while not optimized:
                # this part of the loop can be threaded ---
                # for all values of b in the list of arithmetic numbers
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), # = -8*5 = -40
                                   self.max_feature_value*b_range_multiple, # = 8*5 = 40
                                   step*b_step_multiple): # = step*5
                   for transformation in transforms:
                        w_t=w*transformation
                        #print(w_t)
                        found_option = True
                        #weakest like in the SVM fundamentally
                        #SMO attempts to fix this a bit
                        #yi(xi*w+b) >=1

                        # check all the data
                        # for the values of w and b that fit the main criteria of 
                        # yi(xi*w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                # this below condition is the stopping criteria
                                if not yi*(np.dot(w_t,xi)+b)>=1:
                                    #print(w_t,b)
                                    found_option = False
                                    #break
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                            #Threads end

                if w[0]<0:
                    # <0 because we are already testing all the possible positive and negative of w from latest_optimum to 0
                    print(w)
                    optimized=True
                    print('optimized a step')
                else:
                    # the reduction of vector w for all the values of w to be tested
                    #w=[5,5]
                    #step=1
                    #w-[step,step]
                    w= w-step

            #||w||:[w,b]
            # the list of all ||w||
            norms = sorted([n for n in opt_dict])
            # choosing the minimum ||w||
            opt_choice = opt_dict[norms[0]]
            print(opt_choice)
            self.w=opt_choice[0]
            self.b=opt_choice[1]
            #This number should be soft coded later *2
            # this is just a guess for updating the latest_optimum by intuition
            latest_optimum = opt_choice[0][0]+step*2

            # just to print and check the data_points and their values returned by the pair [w, b]
            # the assurance that we have all the values
            # here comes the tolerance value,and this is used to terminate it
            for i in self.data:
                for xi in self.data[i]:
                    yi=i
                    print(xi,':',yi*(np.dot(self.w,xi)+self.b))
            
    def predict(self,features):
        #sign(x.w+b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0],features[1],s=200,marker='*', c=self.colors[classification])            
        return classification
    
	# Humans like this function and the value(v)
    def visualize(self):
        # plotted the data
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        #hyperplane = x*w+b and we are seeking value (v)
        #v=x.w+b
        #psv = 1
        #nsv = -1
        #decition line = 0
        def hyperplane(x,w,b,v):
            # this function returns the y-coordinate fo the data_point using the equation
            # x*w[0] + y*w[1] + b = v
            return(-w[0]*x-b+v)/ w[1]
        datarange = (self.min_feature_value*1.1,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        
        
        #w.x+b = 1        
        #positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min,self.w,self.b,1)
        psv2 = hyperplane(hyp_x_max,self.w,self.b,1)
        # this is just plotting the positive support vector line
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k') # k = black
                     
        #w.x+b = -1
        #negitive support vector hyperplane
        nsv1 = hyperplane(hyp_x_min,self.w,self.b,-1)
        nsv2 = hyperplane(hyp_x_max,self.w,self.b,-1)
        # this is just plotting the negative support vector line
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k') # k = black

        #w.x+b = 0
        #decision_boundary support vector hyperplane
        db1 = hyperplane(hyp_x_min,self.w,self.b,0)
        db2 = hyperplane(hyp_x_max,self.w,self.b,0)
        # this is just plotting the decision_boundary support vector line
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--') # y-- = yellow dashed
        
        plt.show()

# test data        
data_dict = {-1:np.array([[1,7],[2,8],[3,8]]),
             1:np.array([[5,1],[6,-1],[7,3]])}

svm = Support_Vector_Machine()
svm.fit(data_dict)

predict_us = [[0,10],
			  [1,3],
			  [3,4],
			  [3,5],
			  [5,5],
			  [5,6],
			  [6,-5],
			  [5,8]]

for p in predict_us:
	svm.predict(p)

svm.visualize()