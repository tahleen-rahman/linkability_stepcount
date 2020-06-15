Project setup:
1. Create a virtualenv:
* pip3 install virtualenv
* virtualenv -p /usr/bin/python3 stepenv
* source stepenv/bin/activate
* pip3 install -r requirements.txt
* deactivate
2. Setup PyCharm:
New Project
* Location: /path/to/project/stepcount/src
* Interpreter: /path/to/project/stepcount/stepenv/bin/python3
* go to Preferences -> Project -> Project Structure
* + Add Content Root
* /path/to/project/stepcount/data


Working on project (in stepenv):
* source stepenv/bin/activate
* ... your work ...
* deactivate


USAGE:

* python3 main.py # runs everything serially from processing raw data until plotting results
or
* To manually run different  steps from their separate files (to control parameters etc) follow the following steps:
1. python3 prep_features.py
2. python3 link_siam.py exp cl1 server weekend
3. python3 link_baseline.py  exp  cl2 server weekend
4. pyhton 3 link_unsupr.py  metric server weekend
5. python3 plots.py


* Explanation for Parameters exp, cl, server, weekend:

1. exp1 - Choose from the keys 0...4 , for the siamese attack, according to the dimensions of features (less epoch patience,  more variance threshold for files with lesser features).

Values are in the format (patience parameter for early stopping, subdir containing features, variance threshold)
              
            0: (30, 'linkdata_0/', 0.005),
            1: (20, 'linkdata_1/', 0.001),
            2: (10, 'linkdata_2/', 0.0),
            3: (5,  'linkdata_3/', 0.0),
            4: (20, 'linkdata_dist/', 0.0)


2. cl1 - Choose a classifier for the siamese attack 

         'lstm1' : ([[0.5, 0.2], [0.25, 0.2]]),  # list of size = num of lstm layers [lstm units as frac of inputsize, dropout]
         'lstm2' : ([[16, 0.2]]), #for medium files
          'lstm3' : ([[8, 0.2]]),  #for the big files
          'cnn1'   : ((16, 6), (16, 6), 8, 1), # layer i (filt size, kernel size) , max poolsize
          'dense'  : [0.5, 0.25],  #[frac of inputsize]
          'cnn2'   : ((16, 6), (16, 6), 8, 2)
           }


3. cl2 - Choose a classifier for the supervised baseline attack 

               'rf': RandomForestClassifier(n_estimators=trees, random_state=0),
               'lr': LinearRegression(),
               'svm': svm.SVC(gamma='scale', decision_function_shape='ovo'),
               'lsvc': svm.LinearSVC(max_iter=2000),  # May not converge if training data is not normalized
               'dense1': BinaryDNN(num_layers=1, layer_params=[[0.25, 0.2]], num_epochs=100, batch_size=64, verbose=0),
               'dense2': BinaryDNN(num_layers=2, layer_params=[[0.5, 0.2], [0.5, 0.2]], num_epochs=100, batch_size=64, verbose=0),
               'dense3': BinaryDNN(num_layers=3, layer_params=[[0.25, 0.2], [0.25, 0.2], [0.25, 0.2]], num_epochs=100, batch_size=64, verbose=0)

            
4. server = '1' 
5. weekend = '1' if include weekends else '0'
6. metric = 'cosine' or 'eucl'


