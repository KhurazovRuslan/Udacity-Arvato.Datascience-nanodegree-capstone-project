<h1>Udacity - Arvato customer segmentation and mailout campaign prediction.</h1>


This is the final project for Data science nanodegree by <a href='https://www.udacity.com/'>udacity.com</a>. The first goal of the project is to clusterize general population in Germany and current customers of the company trying to find features that make up a potential customer. The second goal is to build a classification model trying to predict responses to a marketing campaign. As the result of the project a <a href='https://khurazovruslan.medium.com/customer-segmentation-and-prediction-for-arvato-financial-services-b7bea75380fd'>blogpost was posted on medium.com</a>


<h2>Data:</h2>
<li>Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).</li>
<li>Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).</li>
<li>Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).</li>
<li>Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).</li>
<li>DIAS_Information_Levels-Attributes_2017.xlsx: a top-level list of attributes and descriptions, organized by informational category.</li>
<li>DIAS_Attributes-Values_2017.xlsx: a detailed mapping of data values for each feature in alphabetical order.</li><br>

The data provided by Arvato Financial Services.<br>
NOTE! All data files are deleted from local computer as required by terms and conditions of the project.


<h2>Files:</h2>
<li><a href='https://github.com/KhurazovRuslan/Udacity-Arvato.Datascience-nanodegree-capstone-project/blob/main/Capstone.ipynb'>Capstone.ipynb</a> - contains python code for both unsupervised and supervised learning parts of the project.</li>
<li><a href='https://github.com/KhurazovRuslan/Udacity-Arvato.Datascience-nanodegree-capstone-project/blob/main/functions_for_project.py'>functions_for_project.py</a> - contains python code with custom function for the project.</li>
<li><a href='https://github.com/KhurazovRuslan/Udacity-Arvato.Datascience-nanodegree-capstone-project/blob/main/README.md'>Readme file</a></li>

<h2>Technologies used in the project:</h2>
<li>python 3.7.8</li>
<li>numpy 1.18.5</li>
<li>pandas 1.1.0</li>
<li>matplotlib 3.3.0</li>
<li>seaborn 0.10.1</li>
<li>pickle 4.0</li>
<li>scikit-learn 0.23.2</li>
<li>tqdm 4.48.2</li>
<li>xgboost 1.2.0</li>
<li>lightgbm 3.2.1</li>
<li>hyperopt 0.2.5</li>

<h2></h2>
<div>In the project unsupervised learning algorithms were used to cluster general population and customers into 10 distinctive segments which should help the company to prioritize their marketing campaigns on a particular group. Also supervised learning techniques were used to build a model that predicts response to the marketing campaign with ROC AUC score of 0.80251.</div>

<h2>Thanks to:</h2>
<li><a href='https://towardsdatascience.com/customer-segmentation-with-python-31dca5d5bdad'>This post by Natassha Selvaraj</a> for explaining basics of customer segmentation.</li>
<li><a href='https://towardsdatascience.com/how-to-create-a-radar-chart-in-python-36b9ebaa7a64'>This post by Abhijith Chandradas</a> for teaching to plot radar charts.</li>
<li><a href='https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e'>And this post by Wai</a> for helping with hyperparameters tuning.</li>
