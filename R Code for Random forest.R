rfd17_18 <- read.csv(file.choose(), header = TRUE)
View(rfd17_18)
str(rfd17_18)
library(randomForest)
library(randomForestExplainer)
library(party)
library(rsample)
library(ranger)
library(caret)
library(h2o)
library(caTools)
library(writexl)
# Data Partition 
set.seed(123)
sample <- sample.split(rfd17_18,SplitRatio = 0.75)
train1 <- subset(rfd17_18,sample ==TRUE)
test1 <- subset(rfd17_18, sample ==FALSE)
set.seed(222)
rfn <- randomForest(ES~., data = train1, localImp = TRUE)
print(rfn)
#Distribution of minimal depth
min_depth_frame <- min_depth_distribution(rfn)
save(min_depth_frame, file = "min_depth_frame.rda")
load("min_depth_frame.rda")
head(min_depth_frame, n = 10)
# plot_min_depth_distribution(forest)
plot_min_depth_distribution(min_depth_frame)
#mean minimal depth is calculated using only non-missing values.
plot_min_depth_distribution(min_depth_frame, mean_sample = "relevant_trees", k = 15)
importance_frame <- measure_importance(rfn)
save(importance_frame, file = "importance_frame.rda")
load("importance_frame.rda")
importance_frame
write_xlsx(importance_frame, "E:\\Random Forest\\New Analysis\\import.xlsx")
# plot_multi_way_importance(forest, size_measure = "no_of_nodes")
plot_multi_way_importance(importance_frame, size_measure = "no_of_nodes")
plot_multi_way_importance(importance_frame, x_measure = "mse_increase", y_measure = "node_purity_increase", size_measure = "p_value", no_of_labels = 5)
plot_multi_way_importance(importance_frame, x_measure = "mse_increase", y_measure = "node_purity_increase", size_measure = "p_value", no_of_labels = 5)

# plot_importance_ggpairs(forest) # gives the same result as below but takes longer
plot_importance_ggpairs(importance_frame)
# plot_importance_rankings(forest) # gives the same result as below but takes longer
plot_importance_rankings(importance_frame)
# (vars <- important_variables(forest, k = 5, measures = c("mean_min_depth", "no_of_trees"))) # gives the same result as below but takes longer
vars <- important_variables(importance_frame, k = 5, measures = c("mean_min_depth", "no_of_trees"))
vars
interactions_frame <- min_depth_interactions(rfn, vars)
save(interactions_frame, file = "interactions_frame.rda")
load("interactions_frame.rda")
mindepath <- head(interactions_frame[order(interactions_frame$occurrences, decreasing = TRUE), ])
write_xlsx(mindepath, "E:\\Random Forest\\New Analysis\\depth.xlsx")
plot_min_depth_interactions(interactions_frame)
#Prediction of the forest on a grid
plot_predict_interaction(rfn, rfd17_18, "CS", "AS")
explain_forest(rfn, interactions = TRUE, data = rfd17_18)

which.min(rfn$mse)
plot(rfn)
imp <- as.data.frame(sort(importance(rfn)[,1],decreasing = TRUE),optional = T)
imp
test.pred.forest <- predict(rfn,test1)
test.pred.forest
RMSE.forest <- sqrt(mean((test.pred.forest-test1$ES)^2))
RMSE.forest
MAE.forest <- mean(abs(test.pred.forest-test1$ES))
MAE.forest 
summary(rfn)
attributes(rfn)
rfn$call
rfn$predicted
rfn$oob.times
rfn$ntree
rfn$forest
predict(rfn)
# Create a data frame with the error metrics for RF method
accuracy <- data.frame((Method = "Random Forest"), RMSE = RMSE.forest, MAE = MAE.forest)
# Round the values and print the table
accuracy$RMSE <- round(accuracy$RMSE,2)
accuracy$MAE <- round(accuracy$MAE,2) 
accuracy
# Create a data frame with the predictions for each method
all.prediction <- data.frame(actual = test1$ES, random.forest = test.pred.forest)
head(all.prediction)
# Predicted vs. actual for each model
all.prediction <- gather(all.prediction,actual = test1$ES , key = model,value = predictions)
head(all.prediction)
