
from sklearn.metrics import precision_score, recall_score, classification_report,roc_auc_score ,confusion_matrix ,accuracy_score ,mean_absolute_error,f1_score
from sklearn.preprocessing import StandardScaler

import math
from math import sqrt

import numpy
from numpy import mean, std, dstack

from pandas import read_csv
import pandas as pd


from keras.models import Sequential
from keras.layers import Dense, Flatten,Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import to_categorical

import csv
import random
import time
from matplotlib import pyplot

import sklearn
from numpy import asarray
from sklearn.preprocessing import normalize

def Calc_Fitness(config,trainX, trainy, testX, testy):
    score_list , accuracy = evaluation(config,trainX, trainy, testX, testy)
    return score_list , accuracy


class solution:
    def __init__(self):
        self.best_error = 0
        self.bestIndividual = []
        self.convergence = []
        self.startTime = 0
        self.endTime = 0
        self.executionTime = 0
        self.accuracy = 0
        self.prediction = list()


best_solution = solution()

#Dataset Flder
data_path = 'C:/'
#save result folder
result_path = 'C:/result/'


# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix +  data_path)
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix +  data_path)
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
# 	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy



# standardize data
def scale_data(trainX, testX, standardize):
	# remove overlap
	cut = int(trainX.shape[1] / 2)
	longX = trainX[:, -cut:, :]
	# flatten windows
	longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
	# flatten train and test
	flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
	flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
	# standardize
	if standardize:
		s = StandardScaler()
		# fit on training data
		s.fit(longX)
		# apply to training and test data
		longX = s.transform(longX)
		flatTrainX = s.transform(flatTrainX)
		flatTestX = s.transform(flatTestX)
	# reshape
	flatTrainX = flatTrainX.reshape((trainX.shape))
	flatTestX = flatTestX.reshape((testX.shape))
	return flatTrainX, flatTestX

# cnn model
def evaluate_model(config,trainX, trainy, testX, testy):
	verbose = 0
	filters, kernel_size, epochs, batch_size, pool_size = config
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# trainX, testX = scale_data(trainX, testX, True)
	model = Sequential()
	model.add(Conv1D(filters, kernel_size, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters, kernel_size, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D(pool_size))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# run an experiment
def evaluation(config,trainX, trainy, testX, testy):
	model = evaluate_model(config,trainX, trainy, testX, testy) 
	yhat_probs = model.predict(testX)
	yhat_classes = model.predict_classes(testX)
 
 #=======================[ Calculate metrics ]====================
	rounded_labels = numpy.argmax(testy, axis=1)
	accuracy = accuracy_score(rounded_labels, yhat_classes)
 #===========================[ END ]==============================

	score_list = [ yhat_probs , yhat_classes ]
	return(score_list , accuracy)


# GWO

def GWO(lb, ub, dim, SearchAgents_no, Max_iter,trainX, trainy, testX, testy):

    # initialize alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]

    Convergence_curve = numpy.zeros(Max_iter)

    # Loop counter
    print("GWO is optimizing")

    # Main loop
    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = math.floor(0.5 + (ub[j] - lb[j] + 1) * ((Positions[i, j] - lb[j]) / (ub[j] - lb[j])))
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            n_filter = int(Positions[i, 0])
            n_kernel = int(Positions[i, 1])
            n_epoch = int(Positions[i, 2])
            n_batch = int(Positions[i, 3])
            pool_size = int(Positions[i, 4])
            config = [ n_filter, n_kernel, n_epoch, n_batch, pool_size]
            
            score_list , accuracy = Calc_Fitness(config,trainX, trainy, testX, testy)
            fitness = 1-accuracy

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()
                best_solution.accuracy = accuracy
                best_solution.bestIndividual = config.copy()
                best_solution.best_error = fitness
                best_solution.prediction = score_list
                
            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        a = 2 - l * ((2) / Max_iter);  # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  # Equation (3.3)
                C1 = 2 * r2;  # Equation (3.4)

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # Equation (3.5)-part 1
                X1 = Alpha_pos[j] - A1 * D_alpha;  # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  # Equation (3.3)
                C2 = 2 * r2;  # Equation (3.4)

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);  # Equation (3.5)-part 2
                X2 = Beta_pos[j] - A2 * D_beta;  # Equation (3.6)-part 2

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;  # Equation (3.3)
                C3 = 2 * r2;  # Equation (3.4)

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);  # Equation (3.5)-part 3
                X3 = Delta_pos[j] - A3 * D_delta;  # Equation (3.5)-part 3

                Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)


        if (l % 1 == 0):
            print(['At iteration ' + str(l) 
            + ' the best fitness is ' + str(Alpha_score)
            + ' acc is ' + str(best_solution.accuracy)
            + ' individial is ' + str(best_solution.bestIndividual)])

        Convergence_curve[l] = Alpha_score #accuracy


    best_solution.convergence = Convergence_curve

    return best_solution 

# ================================ Load and Split input data ===========================================================

trainX, trainy, testX, testy = load_dataset()
trainX, testX = scale_data(trainX, testX, True)

dim = 5

lb_filter = 1
ub_filter = 300

lb_kernel = 1
ub_kernel = 20

lb_epoch = 1
ub_epoch = 300

lb_batch = 10
ub_batch = 100

lb_poolsize = 1
ub_poolsize = 20

lb = list()
lb.append(lb_filter)
lb.append(lb_kernel)
lb.append(lb_epoch)
lb.append(lb_batch)
lb.append(lb_poolsize)
ub = list()
ub.append(ub_filter)
ub.append(ub_kernel)
ub.append(ub_epoch)
ub.append(ub_batch)
ub.append(ub_poolsize)

SearchAgents_no = 25
Max_iter = 20
Num_runs = 10
best_acc = 0 
best_result = list()
best_config = []
best_convergence = []
best_confusion = []
# ======================================== Perform Algorithm ===========================================================
all_acc          = list()
all_acc_seperate = list()
all_precision    = list()
all_recall       = list()
all_f1           = list()
all_conf_mat     = list()
all_bestind      = list()
all_class_report = list()

meta = ['GWO']


for z in range(len(meta)):
    

    with open(result_path+str(meta[z])+'/Results.csv', 'w') as re:
        
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@[' + str(meta[z]) + ']@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@' )        
        for i in range(Num_runs):
            
            print('=========== Results of Run ' + str(i) + ' ==================')        
            res = globals()[str(meta[z])](lb, ub, dim, SearchAgents_no, Max_iter,trainX, trainy, testX, testy)      
            print('The evaluation metrics:')
    
            #prediction
            yhat_probs , yhat_classes = res.prediction
            rounded_labels = numpy.argmax(testy, axis=1)
    
            #ACC overall
            acc = res.accuracy
            all_acc.append(acc)
    
    
            #precision
            precision = precision_score(rounded_labels, yhat_classes , average=None)
            all_precision.append(precision)
            
    
            #recall
            recall = recall_score(rounded_labels , yhat_classes , average=None)
            all_recall.append(recall)
    
    
            #f1_score
            f1 = f1_score(rounded_labels , yhat_classes , average=None)
            all_f1.append(f1)   
    
    
            #confusion_matrix
            conf_mat = confusion_matrix(rounded_labels , yhat_classes )
            all_conf_mat.append(conf_mat)
            
            #best individial
            all_bestind.append(res.bestIndividual)
    
            #classification_report
            class_report = classification_report(rounded_labels , yhat_classes)
            all_class_report.append(class_report)
    
            #Print result
            print('ACCURACY OVERALL= ' + str(acc) ,'\n',
                  'precision = ' + str(precision) ,'\n',
                  'recall = ' + str(recall) ,'\n',
                  'f1 = ' + str(f1) ,'\n')

    
            # ================================================ Save convergence values =====================================
            convergence_path = result_path+str(meta[z])+'/Convergence' + str(i) + '.csv'
            with open(convergence_path, 'w') as f:
                f.write("Iteration,ACCURACY")
                f.write("\n")
                for j in range(Max_iter):
                    f.write(str(j) + "," + str(res.convergence[j]))
                    f.write("\n")
                f.close()
    
            if acc > best_acc:
                best_acc = acc
                best_result_classes = yhat_classes
                best_result_probs   = yhat_probs 
    
                best_config = res.bestIndividual
                best_convergence = res.convergence
                best_confusion = confusion_matrix(rounded_labels, yhat_classes)
                best_Num_runs = i
                best_per = precision
                best_acc = acc
                best_recall = recall
                best_f1 = f1
    
    
            str1 = '================================== Results of Run ' + str(i) + ' ======================================'
            re.write(str1)
            re.write("\n")
            
            re.write("acc = " + str(acc))
            re.write("\n")
            
            re.write("precision = " + str(precision))
            re.write("\n")
    
            re.write("recall = " + str(recall))
            re.write("\n")
    
            re.write("f1 = " + str(f1))
            re.write("\n")
    
            re.write("best individial = " + str(res.bestIndividual))
            re.write("\n")
    
            re.write("confusion_matrix = " +'\n'+ str(conf_mat))
            re.write("\n")
    
            re.write("class_report = " +'\n'+ str(class_report))
            re.write("\n")
            
            str1 = '========================================================================================'
            re.write(str1)
            re.write("\n")


        str1 = '====================================== Average of Results ========================================='
        re.write(str1)
        re.write("\n")

        scores_m = mean(all_acc)
        re.write("Mean acc = " + str(scores_m))
        re.write("\n")

        scores_p = mean(all_precision, axis=0)
        re.write("Mean precision = " + str(scores_p))
        re.write("\n")

        scores_r = mean(all_recall, axis=0)
        re.write("Mean recall = " + str(scores_r))
        re.write("\n")

        scores_f = mean(all_f1, axis=0)
        re.write("Mean f1-score = " + str(scores_f))
        re.write("\n")

        re.write("best individial = " + str(all_bestind))
        re.write("\n")

        re.write("\n")


        str1 = '========================================= Best Obtained Config ============================================'
        re.write(str1)
        re.write("\n")
        re.write("Number of filters = " + str(best_config[0]))
        re.write("\n")
        re.write("Number of kernels = " + str(best_config[1]))
        re.write("\n")
        re.write("Number of epochs = " + str(best_config[2]))
        re.write("\n")
        re.write("Number of batch = " + str(best_config[3]))
        re.write("\n")
        re.write("Pool size = " + str(best_config[4]))
        re.write("\n")
        re.write("\n")
        re.write("\n")
        re.write("\n")

        re.write("best_acc = "+str(best_acc ))
        re.write("\n")

        re.write("best_per = "+str(best_per ))
        re.write("\n")

        re.write("best_recall = "+str(best_recall ))
        re.write("\n")

        re.write("best_f1 = "+str(best_f1 ))
        re.write("\n")
        
        re.write("best_confusion = " +'\n'+ str(best_confusion))
        re.write("\n")
        re.write("best_Num_runs = "+str(best_Num_runs ))
        re.close()

  # ================================================ Save Best convergence values ========================================
    with open(result_path+str(meta[z])+'/Best_Convergence.csv', 'w') as f:
        f.write("Iteration,accuracy")
        f.write("\n")
        for i in range(Max_iter):
            f.write(str(i) + "," + str(best_convergence[i]))
            f.write("\n")
        f.close()
  # ================================================ Save best prediction and trust values ========================================

    with open(result_path+str(meta[z])+'/pred_trust.csv', 'w') as f:
        f.write("Actual,yhat_classes,yhat_probs")

        f.write("\n")
        for i in range(len(testy)):
                probs = max(best_result_probs[i])
                f.write(str(rounded_labels[i]) + "," + str(best_result_classes[i])+ "," + str(probs))
                f.write("\n")
        f.close()