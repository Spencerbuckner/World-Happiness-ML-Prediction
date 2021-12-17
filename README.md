# WORLD HAPPINESS REPORT

## Problem Statement:

We have been hired by the UN to build a model that can predict the happiness of an average person in each country based on scores for 8 categories provided by the UN and an additional 4 factors that we added. Secondly, can we predict what region of the world this average person is from, based on the scores from the features.

## Data Gathering

Data was collected from the UN Happiness Report for the years 2015-2017 [Happiness Report](https://www.kaggle.com/unsdsn/world-happiness)
Data was also gathered from [Our World in Data](https://ourworldindata.org/)

## Data Dictionary

|Feature|Type|Description|
|---|---|---|
|**country**|*object*|Name of the country.|
|**region**|*object*|Region the country belongs to.|
|**year**|*int*|Year of the report.|
|**happiness_score**|*float*|A metric measured by asking the sampled people the question: "How would you rate your happiness on a scale of 1-10"|
|**economy_(gdp_per_capita)**|*float*|The extent to which GDP contributes to the calculation of the Happiness Score.|
|**family**|*float*|The extent to which Family contributes to the calculation of the Happiness Score.|
|**health_(life_expectancy)**|*float*|The extent to which Life expectancy contributed to the calculation of the Happiness Score.|
|**freedom**|*float*|The extent to which Freedom contributed to the calculation of the Happiness Score.|
|**trust_(government_corruption)**|*float*|The extent to which Perception of Corruption contributes to Happiness Score.|
|**generosity**|*float*|The extent to which Generosity contributed to the calculation of the Happiness Score.|
|**dystopia_residual**|*float*|Dystopia is the worst possible country based on the lowest score for each category. The residual is the difference between a specific countries score and the dystopia score. Example: A residual of 0 would reflect being equal to the worst country.|
|**food_supply_(kcal/capita/day)**|*float*|The number of calories available per capita per day for each country.|
|**crude_birth_rate_(births_per_1,000_population)**|*float*|Birth rate per 1,000 people in each country.|
|**deaths_unsafe_water_source_per_100k_people**|*float*|Deaths due to unsafe drinking water per 100,000 people in each country.|
|**deaths_conflict_and_terrorism**|*float*|Total deaths from conflict terrorism a country occured within each country.|


## Data Cleaning

- The UN data was clean, nothing needed to be changed.
- For the data from Our World in Data, there were a few NaN values, for which we imputed the mean value for the region that the country was located in.

## EDA

The first thing that we explored was the relationship between happiness, food, and freedom. Using Tableau, we created a visualization to help show how these features relate to each other. [Tableau](https://public.tableau.com/app/profile/spencer.buckner/viz/HappinessData_16396021350410/Dash1) 

Secondly, we created a heatmap that displays the correlation between happiness and all other features. Then, we created a table to show the average happiness score by region, countries, and also by the year. 

![](./visuals/avg_happiness_by_region.png)

We checked the top 10 average scores by country: 

![](./visuals/top_10_by_country.png)

We checked the bottom 10 scores by country:

![](./visuals/bottom_10_by_country.png)

Next, we wanted to look at the relationship between the economy of a country and the happiness score: 

![](./visuals/birth_rate_vs_happiness.png)

We also then wanted to look at each region individually. We created dataframes for each region and then created heatmaps for each one. The heatmaps showed that each region had different correlations between the different features and the happiness score, which makes intuitive sense, assuming that people in different regions of the world view happiness differently. There were three regions that show these distinctions most clearly:

North America:

![](../visuals/heatmap_NA.png)

Sub-Saharan Africa:

![](../visuals/heatmap_SSA.png)

Southeast Asia:

![](../visuals/heatmap_SeA.png)

We lastly wanted to check the principle component scores for the different features to see which ones had the highest explained variance within the model. We found that the first principle component could account for roughly 50% of the explained variance:

![](../visuals/cummulative_explained_difference.png)

It also appeared that deaths from unsafe water and crude birthrate had the most influence on the first principle component:

![](../visuals/primary_component_map.png)

After completing the EDA, it was time to start building the models.

## Models

For this project we chose to do two separate paths of model creation. The first path was to use different regressors in order to predict the happiness score based on 13 of the features. The second path was to do multi-classifier models to predict the region. 

![](./visuals/model%20paths.PNG)


### Linear Regression

#### Multivariate Regression Model (MLR)

The first model that we explored was a Multivariate Regression Model. In finding what features we wanted to select, we thought that using 2 feature engineered features could provide some insight into our model.

For our first feature, we multiplied the economy_(gdp_per_capita) by food_supply_(kcal/capita/day) for each respective country. Our second feature was created by multiplying the health_(life_expectancy) by family score for each respective country as well. Once we had our new features, we ran a Linear Regression and produced the following model:

![](./visuals/3D%20Lin%20Reg%20Model.gif)

Our model produced an R2 score of ~ 0.72. Not exactly amazing, but a good starting point for our regression models.

#### MLR with Polynomial Features

Our linear regression model with Polynomial features and Lasso features performed both well with an R2 score of 0.9999. We also got 0.98 after adding PCA. We also discovered that food supply had the highest coefficient score after regularizing using Lasso.

![](./visuals/lasso_coefficients.png)

#### MLR using Neural Networks

After building the model with polynomial features, we wanted to see if a neural network could match the R2 score from the previous model. We therefore used the same polynomial features from the last model, then built a Sequential NN model. We also used GridSearch to find the best parameters for the sequential model, eventually ending up two hidden layers that contained 128 neurons each, along with two dropout phases with a dropout percentage of 25%. This model produced an R2 score of .95, worse off than the non neural net regressor. We then checked the models to see where the predictions are off:

![](./visuals/nnr_preds_vs_actual.png)

We also wanted to visualize how the model performed on the training data vs. the test data:

![](./visuals/loss_and_mse.png)

We decided that since the neural network could not out perform the other model, it was time to move onto the classifiers.

### Multi-Classification

As we move from Regression towards classification, we have two schools of thought when it comes to model selection. We will first work on 3 supervised models (KNN, Logistic Regression, Random Forest), compare those findings in a confusion matrix and classification report, and then create a model stack with all three (Second level will be the model with the greatest accuracy). Once we have that finished, we will then investigate whether a Neural Network Classifier can produce a better model.

As we compare accuracy, keep in mind that our baseline will be the region representing the most total countries (Sub-Saharan Africa). This region represents roughly 25% of all countries in this data set.

**Disclaimer**: Due to Regions North America and Australia/New Zealand only having two countries in each, the precision/recall/f1/accuracy metrics are slightly skewed since there is not enough data to group into separate buckets for train/test splitting. Regions with higher country count provide a greater account into how well model performs overall.

Let’s start with KNN

#### KNN Model

![](./visuals/KNNConfMatr.png)     ![](./visuals/knn%20classification%20report.PNG)


Our KNN model did not perform that great. We achieved a total accuracy score of ~71%, and it was not able to correctly predict any country correct from North America or Australia/New Zealand. The model had a hard time disassociating between Middle East/Northern Africa and Central/Eastern Europe. The model also had trouble with Central and Eastern Europe and Latin America and Caribbean. KNN produced better than our baseline accuracy (25%), but still has more room to improve upon to become competitive.

As we move to our next model lets look at Logistic Regression

#### Logistic Regression:

![](./visuals/LogRegl1ConfMatr.png)      ![](./visuals/logreg%20classification%20report.PNG)

Our Logistic Regression model provided a better picture into our data with a better accuracy score of ~75%. Most of the larger errors were grouped towards Central and Eastern Europe countries being misclassified as Latin America and Caribbean/Western Europe/Middle East and Northern Africa. The only other interesting error was that 3 countries in Sub-Saharan Africa were all classified as Southern Asia.

Our final model in this set of three will be a Random Forest

#### Random Forest:

![](./visuals/RFConfMatr.png)      ![](./visuals/rf%20classification%20report.PNG)

Our Random Forests produced our best model yet with an accuracy score of ~92%. Most errors are gone now, and while some do remain, they are concentrated towards the same errors as the Log Reg model above with Central and Eastern Europe countries being misclassified. Something to take note of here is that 3 of those 10 regions had 100% precision and recall scores by classifying them all correctly (Southern Asia, Eastern Asia, and Australia/New Zealand).

We also wanted to create a model stack of all 3 models as our initial level 1, with a Random Forest as our second level since that produced the highest accuracy score of our 3 classification models so far.

#### Model Stack

![](./visuals/StackConfMatr.png)       ![](./visuals/stack%20classification%20report.PNG)

Our model stack did not produce better results that our random forest and resulted in an accuracy score of ~90%. There are some minor differences and compared to our Random Forest from above, only produced 2 perfect regions (Eastern Asia and Australia/New Zealand).

#### NN Multi-Classifier

![](./visuals/nn%20class%20confmatr.PNG)

Our neural network multi-classifier was built with two hidden layers, each with 128 neurons, and two dropout layers, each with a 25% dropout rate. This model had an accuracy score of 85%, well under the best random forest model. The confusion matrix showed that the model did not perfectly predicted any of the regions. 


## Conclusion

### Regression Models

Since all of the features were known and that that there were no unknown variables that would affect happiness, our polynomial model produced the highest R2 score. When reducing features with a Lasso, we were able to reduce down from 90 to 27 features. We also discovered that the most important coefficient was the food calorie supply, which was not information supplied by the UN. It was also interesting that the neural network performed worse than the Linear Regression, even though the Neural Network was also given polynomial features. We believe that this is likely due to the fact that this is a small data set, and neural networks were designed to handle larger data sets. If we were to add more years, and therefore more data, we believe it is likely that the Neural Network would match the effectiveness of the simple linear regression model.

### Classification Models

The worst model was the KNN model with 71%, followed by the Logistic Regression model with an accuracy of 75%. The best model was the Random Forest model with an accuracy of 92%, which was even better than when we stacked all of the models. The random forest was able to perfectly predict 3 of the regions with 100% precision and recall scores. All models struggled to predict Central and Eastern Europe countries. The Neural Network again was out performed by the simpler models, and again we believe this is the case because we had a small data set. It should noted that all models out performed the baseline accuracy, which was 25%. 



