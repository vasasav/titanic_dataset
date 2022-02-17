# titanic_dataset

The aim of this work is to compare the results of XGBoost and TF.BoostedTrees performance on the familiar Titanic dataset

## Directory `prelims`

Downloading the initial dataset, exploring it, and preparing it for machine learning models.

* `exploration_load.ipynb` - load the data from `opendatasoft.com` and store locally
* `explore.ipynb` - exploratory data analysis of the dataset
* `preproc_titanic.ipynb` - encoding features in such a way as to make further progress with XGBoost and BoosterTrees simple

## Directory `XGBoost`

Once data has been explored and packaged as pickle files, applying the XGBoost classifier. Exploring feature importance and hyper-parameter tuning
* `titanic_XGboost.ipynb` - load data and train basic XBoost model on it. Explore feature importance as well as visualization of the inner structure of the trained model
* `titanic_XGBoost_Hyperparam.ipynb` - explore hyperparameter tuning (sklearn). Analyze leaf weight statistics of the trained trees

## Directory `BoostedTrees`

Applying tensorflow's BoostedTrees model to titanic dataset. Checking saving/loading model. Global feature importance. Some work on checking the model graphs
* `titanic_BoostedTrees.ipynb` - load data and train the basic TF.BT model. Explore feature importance.
* `titanic_BoostedTrees_HyperParameters.ipynb` - explore impact of tuning the hyperparameters on TF.BT model
* `boosted_trees_graph.ipynb` - explore the internal structure of the TF.BT. Look into the computational graph it corresponds to
* `titanic_BostedTrees_following_tutorial.ipynb` - warm-up, using TF.BT tutorial from Google
* `Tensors_to_Examples_Graph.ipynb` - experiments on coding TF.Tensors to protobuf Examples, within TF.
