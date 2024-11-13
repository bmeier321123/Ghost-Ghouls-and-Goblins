library(tidymodels)
library(vroom)
library(embed)
library(discrim)
library(themis) 

# Read in Data----------------------------
trainData <- vroom("train.csv")
testData <- vroom("test.csv")

predict_export <- function(workFlow, fileName) {
  predictions <- predict(final_wf, new_data = testData, type = "class") %>%
    bind_cols(testData) %>%
    select(id, .pred_class) %>%  # Adjust according to your actual prediction columns
    rename(type = .pred_class)  # Adjust if needed
  vroom_write(predictions, file = fileName, delim = ",")
}

my_recipe <- recipe(type ~ ., data = trainData) %>%
  step_mutate(color = as.factor(color)) %>% 
  step_mutate(id, feature=id) %>% 
  step_range(all_numeric_predictors(), min=0, max=1)  #scale to [0,1] 


nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here
grid_of_tuning_params <- grid_regular(Laplace(), smoothness(), levels = 3)

folds <- vfold_cv(trainData, v = 10, repeats = 1)

CV_results <- nb_wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(roc_auc)
  )

bestTune <- CV_results %>% 
  select_best(metric="roc_auc")

## Finalize workflow and predict
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)

predict_export(final_wf, "NB_16.csv")

