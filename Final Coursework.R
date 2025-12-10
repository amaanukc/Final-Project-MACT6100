# ==============================================================================
# F1 CHAMPIONSHIP PREDICTION PROJECT
# ==============================================================================

# AUTHOR: Amaan Kara
# MODULE: MAST6100
# GOAL: Predict F1 Points points using GLM (Lasso), Random Forest, Boosting and Deep Learning.
# RESEARCH QUESTIONS:
# 1. Can we predict points using pre-race data?
# 2. Is F1 "Linear" (GLM) or "Complex" (Deep Learning)?
# 3. Do context features (Form, Skill) matter more than raw car stats?

# ==============================================================================
#1. LIBRARIES & SETUP 
# ==============================================================================
# Loading all required libraries.

install.packages('tidyverse')
library(tidyverse)  # For data wrangling (dplyr) and plotting (ggplot2)

install.packages("lubridate")
library(lubridate)  # For handling dates (Driver Age calculation)

install.packages("zoo")
library(zoo)        # For rolling averages (creating the "Recent Form" feature)

install.packages("h2o")
library(h2o)        # The library for Random Forest & Deep Learning

install.packages("pROC")
library(pROC)       # For calculating AUC scores (Evaluation Metric)

install.packages("glmnet")
library(glmnet)     # For Lasso Regression (Regularized GLM)

install.packages("caret")
library(caret)      # For data splitting functions

install.packages('corrplot')
library(corrplot)
# Initialize H2O Cluster
h2o.init(nthreads = -1, max_mem_size = "4g")

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
# Loading the CSV files that we will use for the model. 
# In the data set we have '\N', which is transformed as 'NA'

results      <- read_csv("results.csv", na = c("\\N", "NA"), show_col_types = FALSE)
races         <- read_csv("races.csv", na = c("\\N", "NA"), show_col_types = FALSE)
drivers     <- read_csv("drivers.csv", na = c("\\N", "NA"), show_col_types = FALSE)
constructors  <- read_csv("constructors.csv", na = c("\\N", "NA"), show_col_types = FALSE)
qualifying    <- read_csv("qualifying.csv", na = c("\\N", "NA"), show_col_types = FALSE)
lap_times     <- read_csv("lap_times.csv", show_col_types = FALSE)

#We will now inspect our data sets

summary(results)
str(results)

summary(races)
str(races)

summary(drivers)
str(drivers)

summary(constructors)
str(constructors)

summary(qualifying)
str(qualifying)

summary(lap_times)
str(lap_times)

# ==============================================================================
# SECTION 3: FEATURE ENGINEERING (EXPANDED DATASET VERSION)
# ==============================================================================

# 3.1 PREPARE & MERGE
results_prep <- results
results_prep$PointsScored <- as.factor(ifelse(results_prep$points > 0, "Yes", "No"))
results_prep <- select(results_prep, raceId, driverId, constructorId, points, positionOrder, grid, PointsScored)

races_prep <- select(races, raceId, year, round, circuitId, date)
races_prep$race_date <- as_date(races_prep$date)
races_prep <- arrange(races_prep, year, round)

base <- left_join(results_prep, races_prep, by = "raceId")
base <- left_join(base, select(drivers, driverId, dob, nationality), by = "driverId")
base <- left_join(base, select(constructors, constructorId, const_nat = nationality), by = "constructorId")
base <- left_join(base, select(qualifying, raceId, driverId, quali_pos = position), by = c("raceId", "driverId"))

# 3.2 NEW: ADD SEASON STANDINGS (The Missing Link)
# We join the standings tables to get points before the race
# Note: Standings files track points after the race, so we need to be careful with logic.
# A simpler way for a report: Use the 'driver_form_3' we made as the dynamic metric,
# and use 'constructor_standings' for Team Strength.

# Let's aggregate cumulative points manually to be safe and accurate
base <- arrange(base, driverId, year, round)
base <- group_by(base, driverId, year)
base <- mutate(base,
                  # Cumulative sum of points in the season SO FAR (lagged by 1 to exclude today)
                  driver_season_points = cumsum(lag(points, default = 0))
)
base <- ungroup(base)

# Now for Constructors (Team Strength)
base <- arrange(base, constructorId, year, round)
base <- group_by(base, constructorId, year)
base <- mutate(base,
                  constructor_season_points = cumsum(lag(points, default = 0))
)
base <- ungroup(base)


# 3.3 FEATURE: DRIVER FORM (Rolling Average)
base <- arrange(base, driverId, year, round)
base_grouped <- group_by(base, driverId)
base <- mutate(base_grouped,
                  prev_points_1 = lag(points, 1, default = 0),
                  prev_points_2 = lag(points, 2, default = 0),
                  prev_points_3 = lag(points, 3, default = 0),
                  driver_form_3 = (prev_points_1 + prev_points_2 + prev_points_3) / 3,
                  career_races  = row_number()
)
base <- ungroup(base)


# 3.4 FEATURE: TEAMMATE COMPARISON
stats_grouped <- group_by(base, raceId, constructorId)
teammate_stats <- summarise(stats_grouped, team_avg_quali = mean(quali_pos, na.rm = TRUE), .groups = "drop")
base <- left_join(base, teammate_stats, by = c("raceId", "constructorId"))

base <- mutate(base,
                  quali_pos_imputed = ifelse(is.na(quali_pos), 25, quali_pos),
                  team_avg_quali    = ifelse(is.na(team_avg_quali), 25, team_avg_quali),
                  teammate_quali_diff = quali_pos_imputed - team_avg_quali
)


# 3.5 FINAL CLEANUP
F1_Final <- filter(base, year >= 2000)
F1_Final <- mutate(F1_Final,
                   DriverAge = as.numeric(difftime(race_date, as_date(dob), units = "days")) / 365.25,
                   nationality = as.factor(nationality),
                   const_nat   = as.factor(const_nat)
)

# SELECT THE EXPANDED FEATURE SET
F1_Final <- select(F1_Final, 
                   PointsScored, year, round, 
                   grid, quali_pos_imputed, DriverAge, 
                   driver_form_3,              # Driver Momentum
                   driver_season_points,       # Driver Hierarchy
                   constructor_season_points,  # Team Hierarchy
                   teammate_quali_diff,        # Skill Isolation
                   career_races,               # Experience
                   nationality, const_nat
)

F1_Final <- na.omit(F1_Final)
print(paste("Final Dataset Size:", nrow(F1_Final), "rows"))

# ==============================================================================
# 4. EDA: FEASIBILITY STUDY
# =============================================================================
# This section will help us further understand our dataset, and help us decide
# on which models to use in our project

# PLOT 1: TARGET DISTRIBUTION
p_uni_1 <- ggplot(F1_Final, aes(x = PointsScored, fill = PointsScored)) +
  geom_bar(width = 0.6) + scale_fill_manual(values = c("gray", "#E10600")) +
  labs(title = "Target Distribution", subtitle = "57% 'No Points' (Imbalanced)", y = "Count") + theme_minimal()
print(p_uni_1)

# PLOT 2: AGE DISTRIBUTION
p_uni_2 <- ggplot(F1_Final, aes(x = DriverAge)) +
  geom_histogram(binwidth = 2, fill = "steelblue", color = "white") +
  labs(title = "Driver Age Distribution", x = "Age", y = "Count") + theme_minimal()
print(p_uni_2)

# PLOT 3: MULTICOLLINEARITY
numeric_vars <- select(F1_Final, grid, quali_pos_imputed, DriverAge, driver_form_3, career_races)
cor_matrix <- cor(numeric_vars)
corrplot(cor_matrix, method="color", type="upper", addCoef.col = "black", 
         tl.col="black", title="Correlation Matrix (Multicollinearity Check)")

# PLOT 4: LINEARITY CHECK
p1 <- ggplot(F1_Final, aes(x = grid, y = as.numeric(PointsScored)-1)) +
  geom_jitter(height = 0.05, alpha = 0.1, color = "gray") +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), color = "blue") +
  labs(title = "Effect of Grid on Scoring (Linear Signal)", x = "Grid Position", y = "Probability") + theme_minimal()
print(p1)

# PLOT 5: NON-LINEARITY CHECK
p2 <- ggplot(F1_Final, aes(x = DriverAge, y = as.numeric(PointsScored)-1)) +
  geom_smooth(method = "loess", color = "darkgreen") +
  labs(title = "Age Performance Curve (Non-Linear Signal)", x = "Age", y = "Probability") + theme_minimal()
print(p2)

# PLOT 6: INTERACTION CHECK
top_5_nations <- names(sort(table(F1_Final$const_nat), decreasing = TRUE)[1:5])
interaction_data <- F1_Final %>% filter(const_nat %in% top_5_nations)

p_int <- ggplot(interaction_data, aes(x = grid, y = as.numeric(PointsScored)-1, color = const_nat)) +
  geom_smooth(method = "lm", se = FALSE, size=1.2) + 
  scale_y_continuous(limits = c(-0.2, 1.2), breaks = c(0, 0.5, 1)) +
  labs(title = "Interaction Effects: Grid x Team Nationality",
       subtitle = "Slopes differ by nation (e.g., Austrian/Red Bull recover best from poor grid)",
       x = "Starting Grid Position", 
       y = "Probability of Scoring",
       color = "Constructor Nation") +
  theme_minimal() +
  theme(legend.position = "right")
print(p_int)


# From these graphs we can see GLM (Lasso), Random Forest, Boosting and Deep Learning
# are the best models for us to use.

# ==============================================================================
# 5. DATA SPLIT
# ==============================================================================

# Now we need to split our dataset to training and testing data (70/30 split)
set.seed(123)
train_index <- createDataPartition(F1_Final$PointsScored, p = 0.7, list = FALSE)
train_data <- F1_Final[train_index, ]
test_data  <- F1_Final[-train_index, ]

# ==============================================================================
# 6. MODEL 1 - LASSO GLM
# ==============================================================================

x_train <- model.matrix(PointsScored ~ ., train_data)[, -1]
y_train <- train_data$PointsScored
x_test <- model.matrix(PointsScored ~ ., test_data)[, -1]
cv_lasso <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
lasso_probs <- predict(cv_lasso, newx = x_test, s = "lambda.min", type = "response")
print(paste("Lasso AUC:", auc(roc(test_data$PointsScored, as.numeric(lasso_probs)))))

# The high AUC score demonstrates our model performed very well, and 
# also that qualifying position is quite important for scoring points.

# ==============================================================================
# 7. H2O LIBRARY SETUP
# ==============================================================================

train_h2o <- as.h2o(train_data)
test_h2o  <- as.h2o(test_data)
y <- "PointsScored"
x <- setdiff(names(train_data), y)

# ==============================================================================
# 8. MODEL 2: RANDOM FOREST
# ==============================================================================

rf_params <- list(ntrees = c(50, 100), max_depth = c(10, 20))
rf_grid <- h2o.grid("randomForest", x = x, y = y, training_frame = train_h2o,
                    grid_id = "rf", hyper_params = rf_params, seed = 123)
best_rf <- h2o.getModel(h2o.getGrid("rf", sort_by = "auc", decreasing = TRUE)@model_ids[[1]])
print("Best RF Performance:")
h2o.performance(best_rf, newdata = test_h2o)

# ==============================================================================
# 9. MODEL 3: GRADIENT BOOSTING
# ==============================================================================
gbm_model <- h2o.gbm(x = x, y = y, training_frame = train_h2o,
                     ntrees = 100, max_depth = 5, learn_rate = 0.1, 
                     balance_classes = TRUE, seed = 123)
gbm_perf <- h2o.performance(gbm_model, newdata = test_h2o)
print(paste("GBM AUC:", h2o.auc(gbm_perf)))

# ==============================================================================
# 10. MODEL 4: DEEP LEARNING
# ==============================================================================
dl_model <- h2o.deeplearning(x = x, y = y, training_frame = train_h2o,
                             hidden = c(200, 200), epochs = 20,
                             activation = "RectifierWithDropout", balance_classes = TRUE, seed = 123)
print(paste("DL AUC:", h2o.auc(h2o.performance(dl_model, newdata = test_h2o))))

# ==============================================================================
# 12. MODEL 4: DEEP LEARNING
# ==============================================================================

# Variable importance plot
par(mfrow=c(1,3))
h2o.varimp_plot(best_rf, num_of_features = 10)
h2o.varimp_plot(gbm_model, num_of_features = 10)
h2o.varimp_plot(dl_model, num_of_features = 10)

# Comparisons of the models are below

# ==============================================================================
# SECTION 12: FINAL MODEL EVALUATION & COMPARISON
# ==============================================================================
print("--- Generating Final Comparison ---")

# 1. EXTRACT PREDICTIONS FROM ALL MODELS
# ---------------------------------------------------------
# MODEL 1: Lasso
lasso_pred <- predict(cv_lasso, newx = x_test, s = "lambda.min", type = "response")
lasso_auc  <- auc(roc(test_data$PointsScored, as.numeric(lasso_pred)))

# M2: Random Forest
rf_perf <- h2o.performance(best_rf, newdata = test_h2o)
rf_auc  <- h2o.auc(rf_perf)

# M3: GBM
gbm_perf <- h2o.performance(gbm_model, newdata = test_h2o)
gbm_auc  <- h2o.auc(gbm_perf)

# M4: Deep Learning
dl_perf <- h2o.performance(dl_model, newdata = test_h2o)
dl_auc  <- h2o.auc(dl_perf)


# 2. CREATE A COMPARISON TABLE
# ---------------------------------------------------------
# We gather the AUCs into a dataframe
model_comparison <- data.frame(
  Model = c("Lasso GLM", "Random Forest", "GBM", "Deep Learning"),
  Type  = c("Linear", "Bagging", "Boosting", "Neural Net"),
  AUC   = c(lasso_auc, rf_auc, gbm_auc, dl_auc)
)

# Sort by best performance
model_comparison <- model_comparison %>% arrange(desc(AUC))

print(model_comparison)


# 3. VISUALIZE THE WINNER (Bar Chart)
# ---------------------------------------------------------
p_res <- ggplot(model_comparison, aes(x = reorder(Model, AUC), y = AUC, fill = Type)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = round(AUC, 3)), hjust = -0.2) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "Final Model Comparison (Test Set)",
       subtitle = "Which model predicted F1 points best?",
       x = "", y = "AUC Score") +
  theme_minimal()
print(p_res)


# 4. COMBINED ROC CURVE
# ---------------------------------------------------------
# This plots the actual curves. If they overlap, the models are similar.

# 1. Helper function to extract ROC data from H2O models
get_roc_data <- function(model, name) {
  perf <- h2o.performance(model, newdata = test_h2o)
  # Extract the dataframe of True Positive/False Positive rates
  df <- h2o.metric(perf) %>% select(tpr, fpr)
  df$Model <- name
  return(df)
}

# 2. Extract data for all 3 Advanced Models
roc_rf  <- get_roc_data(best_rf, "Random Forest")
roc_gbm <- get_roc_data(gbm_model, "GBM")
roc_dl  <- get_roc_data(dl_model, "Deep Learning")

# 3. Combine into one dataframe
roc_all <- bind_rows(roc_rf, roc_gbm, roc_dl)

# 4. Plot using ggplot2 (Thinner lines to see overlap)
p_roc <- ggplot(roc_all, aes(x = fpr, y = tpr, color = Model)) +
  geom_line(size = 0.8, alpha = 0.8) + # Alpha makes them slightly transparent
  geom_abline(linetype = "dashed", color = "gray") + # Diagonal "Random Guess" line
  labs(title = "ROC Curve Comparison",
       subtitle = "Overlapping curves indicate similar predictive performance",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal() +
  scale_color_manual(values = c("Deep Learning"="green3", "GBM"="red", "Random Forest"="blue")) +
  theme(legend.position = "bottom")

print(p_roc)

#########################################################################################################
# END OF SCRIPT
#########################################################################################################