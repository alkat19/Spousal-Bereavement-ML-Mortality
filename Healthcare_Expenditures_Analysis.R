# Code for paper 'Machine learning models of healthcare expenditures predicting mortality: A cohort study of spousal bereaved Danish individuals'

# Loading the required packages

library(tidyverse) # Data handling
library(janitor) # Data handling
library(DSTora) # Access to database
library(ROracle)# Access to database
library(survival) # Survival tools
library(riskRegression) # Main package for developing and evaluating risk prediction models
library(lubridate) # Date handling
library(Hmisc) # Modelling Tools
library(fpp3) # Time Series Feature Extraction
library(dtplyr) # Fasted data wrangling
library(ggthemes) # Custom ggplot2 themes
library(rsample) # Splitting data tool
library(Publish) # Creating of Table 1
library(patchwork) # Graph Customization
library(scales) # Further graph customization
library(rmda) # Decision Curve Analysis
library(data.table) # Data Pre-processing
library(tidymodels) # XGBoost
library(finetune) # Fine-tuning
library(DALEXtra) # Explainability of Models


# Population Dataset (conn is an R object that has the stored data)
# For data security reasons, access to conn is not illustrated here since it requires the use of personal credentials.
pop_data <- tbl(conn, 'pop') %>% collect()


#Construct the dataset with the variables that we need

pop_data <- pop_data %>% select(ID ='PERSON_ID',Date_Of_Death = 'DODDATO', Date_Of_Birth = 'FOED_DAG', Sex = 'KOEN',Bereavement_Date = 'BEREAVEMENT_DATE')


# See how many have died until the end of 2016

pop_data %>% 
  count(lubridate::year(Date_Of_Death) <= 2016)


# Let's include only people who have experienced bereavement

pop_data_bereaved <- pop_data %>% filter(!is.na(Bereavement_Date))


# Let's make the Dates and Sex variables prettier

pop_data_bereaved$Date_Of_Death <- strftime(pop_data_bereaved$Date_Of_Death,format = '%Y-%m-%d')

pop_data_bereaved$Date_Of_Birth <- strftime(pop_data_bereaved$Date_Of_Birth,format = '%Y-%m-%d')

pop_data_bereaved$Bereavement_Date <- strftime(pop_data_bereaved$Bereavement_Date, format = '%Y-%m-%d')

pop_data_bereaved$Date_Of_Birth <- as_date(pop_data_bereaved$Date_Of_Birth)

pop_data_bereaved$Date_Of_Death <- as_date(pop_data_bereaved$Date_Of_Death)

pop_data_bereaved$Bereavement_Date <- as_date(pop_data_bereaved$Bereavement_Date)

pop_data_bereaved$Sex <- as.factor(pop_data_bereaved$Sex)


# Further analysis (Creation of Age at Death, Age at Start and Age At Bereavement)

int <- lubridate::interval(pop_data_bereaved$Date_Of_Birth,pop_data_bereaved$Date_Of_Death)

pop_data_bereaved$Age_At_Death <- trunc(time_length(int,'year'))

int_bereaved <- lubridate::interval(pop_data_bereaved$Date_Of_Birth,pop_data_bereaved$Bereavement_Date)

pop_data_bereaved$Age_At_Bereavement <- trunc(time_length(int_bereaved,'year'))

int_age <- lubridate::interval(pop_data_bereaved$Date_Of_Birth,as_date('2011-01-01'))

pop_data_bereaved$Age_At_Start <- trunc(time_length(int_age,'year'))


# Let's look at Bereavement date

# We should exclude those individuals who have died within a week after bereavement, because we have no expenditures for 
# them

int_death_bereavement <- lubridate::interval(pop_data_bereaved$Bereavement_Date, pop_data_bereaved$Date_Of_Death)

pop_data_bereaved$Dead_After_Ber <- trunc(time_length(int_death_bereavement, 'day'))

pop_data_bereaved <- pop_data_bereaved %>% 
  filter(Dead_After_Ber > 7 | is.na(Dead_After_Ber))


# Change Sex variable levels to Male & Female
levels(pop_data_bereaved$Sex)[levels(pop_data_bereaved$Sex) == '1'] <- 'Males'
levels(pop_data_bereaved$Sex)[levels(pop_data_bereaved$Sex) == '2'] <- 'Females'

summary(pop_data_bereaved$Age_At_Bereavement)

# Group Age at Start
pop_data_bereaved <- pop_data_bereaved %>% 
  mutate(Age_Group_Start = case_when(Age_At_Start %in%  65:69 ~ '65-69',
                                     Age_At_Start %in%  70:74 ~ '70-74',
                                     Age_At_Start %in%  75:79 ~ '75-79',
                                     Age_At_Start %in%  80:84 ~ '80-84',
                                     Age_At_Start %in%  85:105 ~ '85plus'
  ))

# Group Age at Bereavement
pop_data_bereaved <- pop_data_bereaved %>% 
  mutate(Age_Group_Bereavement = case_when(Age_At_Bereavement %in%  65:69 ~ '65-69',
                                           Age_At_Bereavement %in%  70:74 ~ '70-74',
                                           Age_At_Bereavement %in%  75:79 ~ '75-79',
                                           Age_At_Bereavement %in%  80:84 ~ '80-84',
                                           Age_At_Bereavement %in%  85:105 ~ '85plus'))

# Make them as factors

pop_data_bereaved$Age_Group_Start <- as.factor(pop_data_bereaved$Age_Group_Start)
pop_data_bereaved$Age_Group_Bereavement <- as.factor(pop_data_bereaved$Age_Group_Bereavement)


# Rename the ID variable

pop_data_bereaved <- pop_data_bereaved %>% rename(PERSON_ID = 'ID')

###########################################################
###########Exploration of Healthcare Expenditures #########
###########################################################

# Hospital costs

hospital_costs <- tbl(conn,'costs_drg') %>% collect()

# Prescription Drugs:

prescription_drugs <- tbl(conn, 'costs_lmdb') %>% collect()

# Home Care Costs:

home_care <- tbl(conn, 'costs_home_care') %>% collect()

# Residential Care Costs:

resid_care <- tbl(conn, 'costs_residential') %>% collect()

# Primary Care Costs:

primary_care <-  tbl(conn, 'costs_sssy') %>% collect()

# Hospital Costs (Inpatient):

inpatient_costs <- hospital_costs %>% 
  filter(SOURCE == 'DRGHEL') %>%
  select(-SOURCE)

# Hospital Costs(Outpatient):
outpatient_costs <- hospital_costs %>%
  filter(SOURCE == 'DRGAMB') %>%
  select(-SOURCE)

# Let's explore the expenditures data

summary(hospital_costs$COST)
summary(inpatient_costs$COST)
summary(prescription_drugs$COST)
summary(home_care$COST)
summary(resid_care$COST)
summary(primary_care$COST)
summary(outpatient_costs$COST)

# First let's see the NA values of inpatient costs

inpatient_costs %>% filter(is.na(COST)) %>% view()

# We will impute the NAs with zeros, since estimation could not be performed. 
# But we have values for other kinds of costs for those individuals

inpatient_costs <- inpatient_costs %>% replace_na(list(COST = 0))


# Making sure primary care costs are strictly non negative

primary_care <- primary_care %>% mutate(COST = ifelse(COST < 0 , 0, COST))

# Now we should merge the costs for individuals

merged_costs <- bind_rows(prescription_drugs, home_care, resid_care, primary_care,inpatient_costs,outpatient_costs)

# I will create a dtplyr object for faster computation

merged_costs_lazy <- lazy_dt(merged_costs)

# We sum all the costs per person and time 

merged_costs_test <- merged_costs_lazy %>% 
  group_by(PERSON_ID, TIME) %>% 
  mutate(Total_Costs = sum(COST)) %>% 
  ungroup()

merged_costs_test <- as.data.frame(merged_costs_test)


# The dataframe has duplicated time points, we do not need that. We will just keep one time point

merged_costs_test <- merged_costs_test %>% 
  group_by(PERSON_ID) %>% 
  filter(!duplicated(TIME)) %>% 
  ungroup()

# Let's also arrange the time and remove the COST column

merged_costs_test <- merged_costs_test %>% 
  arrange(TIME) %>% 
  select(-COST)

# We will merge the healthcare expenditures data with the bereaved population data

length(unique(pop_data_bereaved$PERSON_ID))

df <- pop_data_bereaved %>% inner_join(merged_costs_test, by = 'PERSON_ID')


# We will replace the NA dates of death (NA in that case means they are still alive)
# We will just replace the NA with '2018-01-01'.

df <- df %>% replace_na(list(Date_Of_Death = as_date('2018-01-01')))

length(unique(df$PERSON_ID))

# I would like to investigate two years of expenditures before bereavement to predict mortality the years after bereavement

# That means that I do not care for expenditures after bereavement

# Let's specify the date of two years before bereavement

df$Two_Years_Before_Bereavement <- df$Bereavement_Date - 2*365.25


# Let's also exclude those individuals who had bereavement at the 2011 or 2012.
# We want the individuals to be non-bereaved for at least 2 years.

df_new <- df %>% filter(lubridate::year(Bereavement_Date) > 2012 )

length(unique(df_new$PERSON_ID))


# Create an Survival variable

summary(df_new$Date_Of_Death)

df_new$Survival <- ifelse(df_new$Date_Of_Death > '2016-12-31', 'Alive','Deceased')  

df_new$Survival <- as.factor(df_new$Survival)


# Let's create a date variable corresponding to the TIME variable (Dates corresponding to week numbers)

Dates <-seq(as_date('2011/01/01'),as_date('2016/12/31'),by = 'week')

Dates <- as.data.frame(Dates)

Dates <- Dates %>% mutate(TIME = seq(0,313,1))

df_new <- df_new %>% inner_join(Dates, by = 'TIME')


# Specify the Bereavement Status (if needed)

df_new <- df_new %>% mutate(Bereavement_Status = ifelse(Bereavement_Date >= Dates,'Pre-Bereavement','Post-Bereavement'))

df_new$Bereavement_Status <- as.factor(df_new$Bereavement_Status)

length(unique(df_new$PERSON_ID))


# We have matched the time and date variables

##################################################
############# Filling the missing weeks ##########
##################################################

df_fill <- df_new

# Missing costs for weeks indicate that the person did not spend any amount on medical services.
# We fill those missing values with zeros.

df_fill <- tidyr::complete(df_fill,PERSON_ID,Dates = seq(as.Date('2011-01-01'), as.Date('2016-12-31'),by = 'week'), fill = list(Total_Costs = 0 ))


# Fill the NA values of different columns due to the filling

df_fill <- df_fill %>% 
  group_by(PERSON_ID) %>%
  tidyr::fill(Date_Of_Death,Date_Of_Birth,Sex,Bereavement_Date, Age_At_Death, Age_At_Bereavement,Age_At_Start,
              Age_Group_Start, Age_Group_Bereavement, Two_Years_Before_Bereavement, Survival, Dead_After_Ber, 
              Bereavement_Status,
              .direction = 'downup') %>% 
  ungroup()

df_fill <- df_fill %>% 
  select(-TIME) %>% 
  filter(Dates >= Two_Years_Before_Bereavement & Dates <= Bereavement_Date)

df_fill <- df_fill %>% 
  group_by(PERSON_ID) %>% 
  mutate(Weeks = row_number() - 1) %>% 
  ungroup()


# Now we have everything we need


#######################################################
############### Creation of DIORs #####################
#######################################################

df_fill <- as.data.frame(df_fill)

# Let's arrange our dataframe in a format for extraction of DIORs

ts_diors <- df_fill %>% 
  group_by(PERSON_ID,Weeks) %>%
  arrange(Weeks) %>%
  dplyr::summarise(PreB_Cost = mean(Total_Costs)) %>% 
  ungroup()

ts_diors <- as.data.frame(ts_diors)

# Let's use the fpp3 package to arrange our dataframe as tsibble (time series format)

ts_diors <- as_tsibble(ts_diors, key = c(PERSON_ID),index = Weeks)

# I would like to extract the remainder component of the time series (useful for the calculation of some DIORs)

remainder <- ts_diors %>% 
  model(stl = STL(PreB_Cost))

comps <- components(remainder)

comps_selected <- comps %>% select(PERSON_ID, remainder)


# Let's merge the remainder data to the original df

df_fill <- df_fill %>% inner_join(comps_selected, by = c('PERSON_ID','Weeks'))


# Now we need to compute the DIORs of healthcare expenditures before bereavement
# We compute the Mean Squared error, Autocorrelation of Detrended-Remainder data
# We also compute the Average of healthcare expenditures

DIORs <- df_fill %>% 
  mutate(Deviations_Squared = remainder^2) %>% 
  dplyr::group_by(PERSON_ID,Dates) %>%
  arrange(Dates) %>%
  dplyr::summarise(Cost = mean(Total_Costs), Detrended_Cost = mean(remainder), 
                   Detrended_Squared = mean(Deviations_Squared)) %>%
  dplyr::group_by(PERSON_ID) %>% 
  dplyr::summarise(AC_Detrended = round(acf(Detrended_Cost,plot = F, lag.max = 1)$acf[2],3), 
                   MSE = round(mean(Detrended_Squared),3), Average = round(mean(Cost),3)) %>% 
  ungroup()


# We use the fpp3 package, to extract additional features from our time series of medical usage

# Extract some autocorrelation features first

# The features command demands from us to specify from which variable we extract the metrics 
# We also need to specify what kind of features we need to extract

autocor_features <- ts_diors %>% features(PreB_Cost, feat_acf)

autocor_selected <- autocor_features %>% select(PERSON_ID, acf10, diff1_acf1)


# Extract some Variability features, similar strategy as before

stl_features <- ts_diors %>% features(PreB_Cost, feat_stl)

stl_selected <- stl_features %>%  select(trend_strength, spikiness, stl_e_acf1, linearity)


# Extract additional features using a list of pre-selected ones.

extra_features <- ts_diors %>% features(PreB_Cost, list(coef_hurst,shift_level_max,n_crossing_points,longest_flat_spot))

extra_selected <- extra_features %>% select(-PERSON_ID)


# Let's merge the 3 different data frames with the features extracted above

bonus_features <- bind_cols(autocor_selected, stl_selected, extra_selected)


# Let's add all the DIORs to the original DIORs dataframe

DIORs <- DIORs %>% inner_join(bonus_features, by = 'PERSON_ID')


# Let's add the rest of the demographics to our DIORs dataframe

df_fill_distinct <- df_fill %>% filter(!duplicated(PERSON_ID))

df_fill_distinct <- df_fill_distinct %>% select(-Dates, -Total_Costs, -Bereavement_Status, -Weeks, -remainder)

DIORs <- df_fill_distinct %>% inner_join(DIORs, by = 'PERSON_ID')


# Let's also calculate the permutation entropy of the time series

# Let's calculate entropy of time series
entropy_df <- df_fill %>% 
  group_by(PERSON_ID,Weeks) %>%
  arrange(Weeks) %>%
  dplyr::summarise(PreB_Cost = mean(Total_Costs)) %>%
  group_by(PERSON_ID) %>%
  summarise(entropy= list(od = statcomp::ordinal_pattern_distribution(x = PreB_Cost,ndemb = 3))) %>% 
  ungroup()

entropy_df <- as.data.frame(entropy_df)

# We loop over all individuals to calculate the permutation entropy

init <- matrix(0,nrow = 50245,ncol = 1) #50245 is the length of the entropy_df dataframe

for (i in 1:50245) {
  init[i]= as.numeric(statcomp::permutation_entropy(entropy_df$entropy[[i]]))
}

# We store the values
init <- as.data.frame(init)

colnames(init) <- c('Entropy')

init <- bind_cols(entropy_df,init)

#That's the dataframe with the requested value

init <- init %>% 
  select(-entropy)

# We merge it back to the original dataframe with the rest of the DIORs
DIORs <- DIORs %>%  inner_join(init, by = 'PERSON_ID')


# We now have most of the DIORs implemented in the data frame, we need to add a couple more
# We use the feasts package which is part of the fpp3 package family

# Let's add more features to our model


total_features <- ts_diors %>% features(PreB_Cost, feature_set(pkgs = 'feasts'))

bit_features <- total_features %>% select(PERSON_ID, zero_run_mean, nonzero_squared_cv, zero_start_prop, var_tiled_var,
                                          var_tiled_mean, acf1 , pacf5)

DIORs <- DIORs %>% inner_join(bit_features, by = 'PERSON_ID')

##########################################
#########The dataframe is almost ready ###
##########################################


# Additional cleaning of the DIORs dataframe (DIORs + plus other metadata)

# Let's use the clean na omitted data
# Some individuals with zero medical usage across the entire follow up are omitted
# For them the computation of many DIORs is not possible

DIORs_clean <- DIORs %>% select(-Age_At_Death, -Dead_After_Ber) %>% na.omit()

# Create a variable indicating survival status the year after spousal bereavement

DIORs_clean <- DIORs_clean %>% mutate(Survival2 = ifelse(Date_Of_Death > Bereavement_Date + 365.25, 'Alive', 'Deceased'))

DIORs_clean$Survival2 <- as.factor(DIORs_clean$Survival2)

levels(DIORs_clean$Survival2)

# Make Sex variable as a binary variable (might be of use for some models that do not handle factors)

DIORs_clean$Sex2 <- ifelse(DIORs_clean$Sex == 'Males', 0, 1)

# Create a Status binary variable, similar reasoning as above (not necessarily needed)

DIORs_clean <- DIORs_clean %>% mutate(Status2 = ifelse(Survival2 == 'Alive', 0, 1))
length(unique(DIORs_clean$PERSON_ID))

# We also need to add some additional sociodemographics to our DIORs_clean dataframe

# Collect the immigration status of individuals

demos <- tbl(conn, 'pop') %>% collect()

demos <- demos %>% select(PERSON_ID,IE_TYPE)

# Merge them in the DIORs_clean data frame

DIORs_clean1 <- DIORs_clean %>% inner_join(demos,by = 'PERSON_ID')

DIORs_clean1$IE_TYPE <- as.factor(DIORs_clean1$IE_TYPE)


# Additional sociodemographics

affluence <- tbl(conn, 'affluence_index') %>% collect() # Affluence index

n_child <- tbl(conn, 'number_of_children') %>% collect() # Number of offsprings

multimorb <- tbl(conn, 'multimorbidity') %>% collect() # Number of multimorbidities

# Keep the quantiles of affluence index
affluence_new <- affluence %>% select(AFFLUENCE_GROUP,PERSON_ID)


# Let's put the affluence on the new dataframe

DIORs_clean2 <- DIORs_clean1 %>% inner_join(affluence_new, by = 'PERSON_ID')

# Now for morbitities

# Create a dataframe specifying the total number of diseases a person has accumulated until baseline
morb_new <- 
  multimorb %>% group_by(PERSON_ID) %>% summarise(Total_Number = sum(n())) %>% ungroup()


DIORs_clean2 <- DIORs_clean2 %>% left_join(morb_new, by = 'PERSON_ID')

# Let's replace NAs with zeros

DIORs_clean2 <- DIORs_clean2 %>% replace_na(list(Total_Number = 0))


# Let's implement the number of children in the dataframe

children <- n_child %>% 
  replace_na(list(N_MOTHER = 0, N_FATHER = 0)) %>% 
  group_by(PERSON_ID) %>% 
  mutate(Children = sum(N_MOTHER + N_FATHER)) %>% 
  ungroup() %>% 
  select(PERSON_ID,Children)

# Merge into the DIORs dataframe

DIORs_clean2 <- DIORs_clean2 %>% left_join(children,by = 'PERSON_ID')

DIORs_clean2 <- DIORs_clean2 %>% replace_na(list(Children = 0))


# Again we will have to introduce some more DIORs to our DIORs_clean1 dataframe

memory_features <- total_features %>% select(PERSON_ID, dplyr::contains('acf'),shift_level_index,shift_var_index,shift_kl_max,shift_kl_index)

memory_features <- memory_features %>% select(-stl_e_acf1,-acf1,-diff1_acf1,-pacf5,-acf10,-shift_level_index)

# Now add those new memory features in the dataframe and create a new dataframe.
DIORs_clean3 <- DIORs_clean2 %>% left_join(memory_features,by = 'PERSON_ID')


##################################################
##### DataFrame is now ready for analysis ########
##################################################


# Let's create a summary table (Table 1) :

#####################################################################################################
# Now we need some descriptive statistics of the final sample at the time of bereavement ############
#####################################################################################################

DIORs_clean3_table <- DIORs_clean3 %>%  mutate(Children_F = case_when(Children == 0 ~ 'Zero',
                                                                      Children == 1 ~ 'One',
                                                                      Children == 2 ~ 'Two',
                                                                      Children == 3 ~ 'Three',
                                                                      Children >= 4 ~ '4 or more')) %>% 
  mutate(Total_Number_F = case_when(Total_Number == 0 ~ 'Zero',
                                    Total_Number == 1 ~ 'One',
                                    Total_Number == 2 ~ 'Two',
                                    Total_Number == 3 ~ 'Three',
                                    Total_Number >= 4 ~ '4 or more')) %>% 
  mutate(Affluence = case_when(AFFLUENCE_GROUP %in% seq(0,25,1) ~ 'Lowest',
                               AFFLUENCE_GROUP %in% seq(26,50,1) ~ 'Second',
                               AFFLUENCE_GROUP %in% seq(51,75,1) ~ 'Third',
                               AFFLUENCE_GROUP %in% seq(76,100,1) ~ 'Highest')) %>% 
  mutate(Immigration_Status = case_when(IE_TYPE == 1 ~ 'Danish',
                                        IE_TYPE == 2 ~ 'Immigrants',
                                        IE_TYPE == 3 ~ 'Descendants'))


DIORs_clean3_table$Immigration_Status <- as.factor(DIORs_clean3_table$Immigration_Status)
DIORs_clean3_table$Total_Number_F <- as.factor(DIORs_clean3_table$Total_Number_F)
DIORs_clean3_table$Affluence <- as.factor(DIORs_clean3_table$Affluence)
DIORs_clean3_table$Children_F <- as.factor(DIORs_clean3_table$Children_F)


summary(utable(Sex ~ Age_At_Bereavement + Affluence + Total_Number_F + Immigration_Status + Children_F, 
               data = DIORs_clean3_table,show.totals = T))


# First let's split the data into train and test-sets

set.seed(234)

DIORs_clean3_split <- initial_split(DIORs_clean3)

DIORs_clean3_train <- training(DIORs_clean3_split)

DIORs_clean3_test <- testing(DIORs_clean3_split)


# Now we will create a cross-validated set which will be used for the fine-tuning of our XGBoost model

set.seed(234)

DIORs_clean3_folds <- vfold_cv(data = DIORs_clean3_train,v = 5)


##############################################################
####### Recipe for xgboost classifier (FULL MODEL)############
##############################################################

# Define metrics

metrics <- metric_set(mn_log_loss, roc_auc)

xg_rec <- recipe(Survival2 ~ Age_At_Bereavement + Average + Sex2 + MSE + coef_hurst + acf10 + 
                   diff1_acf1 + trend_strength + spikiness + linearity + longest_flat_spot + shift_level_max +
                   n_crossing_points + Entropy + acf1 + shift_var_index + shift_level_index + shift_kl_index +
                   diff1_acf10 + stl_e_acf1 + stl_e_acf10 + diff2_acf1 + diff2_acf10 + diff1_pacf5 + diff2_pacf5 +
                   zero_run_mean + nonzero_squared_cv + zero_start_prop + var_tiled_var + var_tiled_mean + pacf5 + 
                   IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_train)


# Tune the hyperparameters

xgb_spec <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf <- workflow(xg_rec, xgb_spec)
xgb_wf

# Run a race anova to find out the best hyperparameter configuration

xgb_rs <-
  tune_race_anova(
    xgb_wf,
    DIORs_clean3_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs)

# We show the best grid of hyperparameters
show_best(xgb_rs)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last <- xgb_wf %>% 
  finalize_workflow(select_best(xgb_rs, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_split)

# Collect metrics
xgb_last %>% collect_metrics()

# Collect predictions
xgb_last %>% collect_predictions()


# Let's now group the predictions   

xgb_full_preds <- xgb_last$.predictions

xgb_full_preds <- xgb_full_preds[[1]][2]

xgb_full_preds <- as.matrix(xgb_full_preds)


# Evaluate Performance on testing set using riskregression's package function Score

full_score <- Score(list('Xgboost' = xgb_full_preds),formula = Survival2 ~ 1, data = DIORs_clean3_test, summary = 'ipa',
                    plots = 'calibration', metrics = c('AUC','brier'), se.fit = T, seed = 9)

summary(full_score)

plotCalibration(full_score,round = F,rug = F)

##############################################################
####### Recipe for xgboost classifier (Overall Dynamics)######
##############################################################


# Define metrics
xg_rec_overall <- recipe(Survival2 ~ Age_At_Bereavement + Average + Sex2 +trend_strength + linearity +
                           zero_run_mean + zero_start_prop + 
                           IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_train)


# Tune the hyperparameters

xgb_spec_overall <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_overall <- workflow(xg_rec_overall, xgb_spec_overall)
xgb_wf_overall

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_overall <-
  tune_race_anova(
    xgb_wf_overall,
    DIORs_clean3_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_overall)

# We show the best grid of hyperparameters
show_best(xgb_rs_overall)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_overall <- xgb_wf_overall %>% 
  finalize_workflow(select_best(xgb_rs_overall, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_split)

# Collect metrics
xgb_last_overall %>% collect_metrics()

# Collect predictions
xgb_last_overall %>% collect_predictions()


# Let's now group the predictions

xgb_overall_preds <- xgb_last_overall$.predictions

xgb_overall_preds <- xgb_overall_preds[[1]][2]

xgb_overall_preds <- as.matrix(xgb_overall_preds)


# Evaluate performance using riskRegression's package function Score

full_score1 <- Score(list('Xgboost' = xgb_overall_preds),formula = Survival2 ~ 1, data = DIORs_clean3_test, summary = 'ipa',
                     plots = 'calibration', metrics = c('AUC','brier'), se.fit = T, seed = 9)

summary(full_score1)

plotCalibration(full_score1,round = F,rug = F)


##############################################################
####### Recipe for xgboost classifier (Dispersion Dynamics)###
##############################################################


# Define recipe
xg_rec_dispersion <- recipe(Survival2 ~ Age_At_Bereavement + MSE + Sex2 + spikiness + shift_level_max +
                              shift_level_index + shift_var_index + shift_kl_index + n_crossing_points + Entropy +
                              nonzero_squared_cv + var_tiled_var + var_tiled_mean +
                              IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_train)


# Tune the hyperparameters

xgb_spec_dispersion <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_dispersion <- workflow(xg_rec_dispersion, xgb_spec_dispersion)
xgb_wf_dispersion

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_dispersion <-
  tune_race_anova(
    xgb_wf_dispersion,
    DIORs_clean3_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_dispersion)

# We show the best grid of hyperparameters
show_best(xgb_rs_dispersion)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_dispersion <- xgb_wf_dispersion %>% 
  finalize_workflow(select_best(xgb_rs_dispersion, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_split)

# Collect metrics
xgb_last_dispersion %>% collect_metrics()

# Collect predictions
xgb_last_dispersion %>% collect_predictions()


# Let's now group the predictions

xgb_dispersion_preds <- xgb_last_dispersion$.predictions

xgb_dispersion_preds <- xgb_dispersion_preds[[1]][2]

xgb_dispersion_preds <- as.matrix(xgb_dispersion_preds)


##############################################################
####### Recipe for xgboost classifier (Memory Dynamics)#######
##############################################################


# Define metrics
xg_rec_memory <- recipe(Survival2 ~ Age_At_Bereavement + stl_e_acf1 + Sex2 + coef_hurst + acf10 +
                          longest_flat_spot  + acf1 + diff1_acf10 + diff1_acf1 +stl_e_acf10 +
                          diff2_acf1 + diff2_acf10 + diff1_pacf5 + diff2_pacf5 + pacf5 + 
                          IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_train)


# Tune the hyperparameters

xgb_spec_memory <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_memory <- workflow(xg_rec_memory, xgb_spec_memory)
xgb_wf_memory

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_memory <-
  tune_race_anova(
    xgb_wf_memory,
    DIORs_clean3_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_memory)

# We show the best grid of hyperparameters
show_best(xgb_rs_memory)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_memory <- xgb_wf_memory %>% 
  finalize_workflow(select_best(xgb_rs_memory, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_split)

# Collect metrics
xgb_last_memory %>% collect_metrics()

# Collect predictions
xgb_last_memory %>% collect_predictions()


# Let's now group the predictions

xgb_memory_preds <- xgb_last_memory$.predictions

xgb_memory_preds <- xgb_memory_preds[[1]][2]

xgb_memory_preds <- as.matrix(xgb_memory_preds)



##############################################################
####### Recipe for xgboost classifier (Basic 4 DIORs)#########
##############################################################


# Define metrics
xg_rec_basic <- recipe(Survival2 ~ Age_At_Bereavement + Sex2 + Average + MSE +
                         AC_Detrended  + linearity + 
                         IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_train)


# Tune the hyperparameters

xgb_spec_basic <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_basic <- workflow(xg_rec_basic, xgb_spec_basic)
xgb_wf_basic

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_basic <-
  tune_race_anova(
    xgb_wf_basic,
    DIORs_clean3_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_basic)

# We show the best grid of hyperparameters
show_best(xgb_rs_basic)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_basic <- xgb_wf_basic %>% 
  finalize_workflow(select_best(xgb_rs_basic, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_split)

# Collect metrics
xgb_last_basic %>% collect_metrics()

# Collect predictions
xgb_last_basic %>% collect_predictions()


# Let's now group the predictions

xgb_basic_preds <- xgb_last_basic$.predictions

xgb_basic_preds <- xgb_basic_preds[[1]][2]

xgb_basic_preds <- as.matrix(xgb_basic_preds)



##############################################################
####### Recipe for xgboost classifier(Benchmark)##############
##############################################################


# Define metrics
xg_rec_benchmark <- recipe(Survival2 ~ Age_At_Bereavement + Sex2 +
                             IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_train)


# Tune the hyperparameters

xgb_spec_benchmark <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_benchmark <- workflow(xg_rec_benchmark, xgb_spec_benchmark)
xgb_wf_benchmark

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_benchmark <-
  tune_race_anova(
    xgb_wf_benchmark,
    DIORs_clean3_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_benchmark)

# We show the best grid of hyperparameters
show_best(xgb_rs_benchmark)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_benchmark <- xgb_wf_benchmark %>% 
  finalize_workflow(select_best(xgb_rs_benchmark, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_split)

# Collect metrics
xgb_last_benchmark %>% collect_metrics()

# Collect predictions
xgb_last_benchmark %>% collect_predictions()


# Let's now group the predictions

xgb_benchmark_preds <- xgb_last_benchmark$.predictions

xgb_benchmark_preds <- xgb_benchmark_preds[[1]][2]

xgb_benchmark_preds <- as.matrix(xgb_benchmark_preds)


##############################################################
####### Recipe for xgboost classifier(Age + Sex)##############
##############################################################


# Define metrics
xg_rec_simple <- recipe(Survival2 ~ Age_At_Bereavement + Sex2, data = DIORs_clean3_train)


# Tune the hyperparameters

xgb_spec_simple <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_simple <- workflow(xg_rec_simple, xgb_spec_simple)
xgb_wf_simple

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_simple <-
  tune_race_anova(
    xgb_wf_simple,
    DIORs_clean3_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_simple)

# We show the best grid of hyperparameters
show_best(xgb_rs_simple)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_simple <- xgb_wf_simple %>% 
  finalize_workflow(select_best(xgb_rs_simple, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_split)

# Collect metrics
xgb_last_simple %>% collect_metrics()

# Collect predictions
xgb_last_simple %>% collect_predictions()


# Let's now group the predictions

xgb_simple_preds <- xgb_last_simple$.predictions

xgb_simple_preds <- xgb_simple_preds[[1]][2]

xgb_simple_preds <- as.matrix(xgb_simple_preds)

# Let's repeat the same process for males and females separately

# For males

# Split now

set.seed(234)

DIORs_clean3_males_split <- initial_split(DIORs_clean3_males)

DIORs_clean3_males_training <- training(DIORs_clean3_males_split)

DIORs_clean3_males_testing <- testing(DIORs_clean3_males_split)

# Let's create the folds dataset

set.seed(234)
DIORs_clean3_males_folds <- vfold_cv(DIORs_clean3_males_training,v = 5)


# Start the modelling of XGBoost

##############################################################
####### Recipe for xgboost classifier (FULL MODEL)############
##############################################################

# Define metrics

metrics <- metric_set(mn_log_loss, roc_auc)

xg_rec_males <- recipe(Survival2 ~ Age_At_Bereavement + Average + MSE + coef_hurst + acf10 + 
                         diff1_acf1 + trend_strength + spikiness + linearity + longest_flat_spot + shift_level_max +
                         n_crossing_points + Entropy + acf1 + shift_var_index + shift_level_index + shift_kl_index +
                         diff1_acf10 + stl_e_acf1 + stl_e_acf10 + diff2_acf1 + diff2_acf10 + diff1_pacf5 + diff2_pacf5 +
                         zero_run_mean + nonzero_squared_cv + zero_start_prop + var_tiled_var + var_tiled_mean + pacf5 + 
                         IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_males_training)


# Tune the hyperparameters

xgb_spec_males <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_males <- workflow(xg_rec_males, xgb_spec_males)
xgb_wf_males

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_males <-
  tune_race_anova(
    xgb_wf_males,
    DIORs_clean3_males_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_males)

# We show the best grid of hyperparameters
show_best(xgb_rs_males)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_males <- xgb_wf_males %>% 
  finalize_workflow(select_best(xgb_rs_males, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_males_split)

# Collect metrics
xgb_last_males %>% collect_metrics()

# Collect predictions
xgb_last_males %>% collect_predictions()


# Let's now group the predictions   

xgb_full_preds_males <- xgb_last_males$.predictions

xgb_full_preds_males <- xgb_full_preds_males[[1]][2]

xgb_full_preds_males <- as.matrix(xgb_full_preds_males)


# Evaluate Performance on testing set using riskRegression's package function Score

full_score_males <- Score(list('Xgboost' = xgb_full_preds_males),formula = Survival2 ~ 1, data = DIORs_clean3_males_testing, summary = 'ipa',
                          plots = 'calibration', metrics = c('AUC','brier'), se.fit = T, seed = 9)

summary(full_score_males)

plotCalibration(full_score_males,round = F,rug = F)

##############################################################
####### Recipe for xgboost classifier (Overall Dynamics)######
##############################################################


# Define metrics
xg_rec_overall_males <- recipe(Survival2 ~ Age_At_Bereavement + Average +trend_strength + linearity +
                                 zero_run_mean + zero_start_prop + 
                                 IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_males_training)


# Tune the hyperparameters

xgb_spec_overall_males <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_overall_males <- workflow(xg_rec_overall_males, xgb_spec_overall_males)
xgb_wf_overall_males

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_overall_males <-
  tune_race_anova(
    xgb_wf_overall_males,
    DIORs_clean3_males_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_overall_males)

# We show the best grid of hyperparameters
show_best(xgb_rs_overall_males)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_overall_males <- xgb_wf_overall_males %>% 
  finalize_workflow(select_best(xgb_rs_overall_males, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_males_split)

# Collect metrics
xgb_last_overall_males %>% collect_metrics()

# Collect predictions
xgb_last_overall_males %>% collect_predictions()


# Let's now group the predictions

xgb_overall_preds_males <- xgb_last_overall_males$.predictions

xgb_overall_preds_males <- xgb_overall_preds_males[[1]][2]

xgb_overall_preds_males <- as.matrix(xgb_overall_preds_males)


# Evaluate performance using riskregression's package function Score

full_score1_males <- Score(list('Xgboost' = xgb_overall_preds_males),formula = Survival2 ~ 1, data = DIORs_clean3_males_testing, summary = 'ipa',
                           plots = 'calibration', metrics = c('AUC','brier'), se.fit = T, seed = 9)

summary(full_score1_males)

plotCalibration(full_score1_males,round = F,rug = F)


##############################################################
####### Recipe for xgboost classifier (Dispersion Dynamics)###
##############################################################


# Define recipe
xg_rec_dispersion_males <- recipe(Survival2 ~ Age_At_Bereavement + MSE + spikiness + shift_level_max +
                                    shift_level_index + shift_var_index + shift_kl_index + n_crossing_points + Entropy +
                                    nonzero_squared_cv + var_tiled_var + var_tiled_mean +
                                    IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_males_testing)


# Tune the hyperparameters

xgb_spec_dispersion_males <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_dispersion_males <- workflow(xg_rec_dispersion_males, xgb_spec_dispersion_males)
xgb_wf_dispersion_males

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_dispersion_males <-
  tune_race_anova(
    xgb_wf_dispersion_males,
    DIORs_clean3_males_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_dispersion_males)

# We show the best grid of hyperparameters
show_best(xgb_rs_dispersion_males)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_dispersion_males <- xgb_wf_dispersion_males %>% 
  finalize_workflow(select_best(xgb_rs_dispersion_males, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_males_split)

# Collect metrics
xgb_last_dispersion_males %>% collect_metrics()

# Collect predictions
xgb_last_dispersion_males %>% collect_predictions()


# Let's now group the predictions

xgb_dispersion_preds_males <- xgb_last_dispersion_males$.predictions

xgb_dispersion_preds_males <- xgb_dispersion_preds_males[[1]][2]

xgb_dispersion_preds_males <- as.matrix(xgb_dispersion_preds_males)


##############################################################
####### Recipe for xgboost classifier (Memory Dynamics)#######
##############################################################


# Define metrics
xg_rec_memory_males <- recipe(Survival2 ~ Age_At_Bereavement + stl_e_acf1 + coef_hurst + acf10 +
                                longest_flat_spot  + acf1 + diff1_acf10 + diff1_acf1 +stl_e_acf10 +
                                diff2_acf1 + diff2_acf10 + diff1_pacf5 + diff2_pacf5 + pacf5 + 
                                IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_males_training)


# Tune the hyperparameters

xgb_spec_memory_males <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_memory_males <- workflow(xg_rec_memory_males, xgb_spec_memory_males)
xgb_wf_memory_males

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_memory_males <-
  tune_race_anova(
    xgb_wf_memory_males,
    DIORs_clean3_males_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_memory_males)

# We show the best grid of hyperparameters
show_best(xgb_rs_memory_males)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_memory_males <- xgb_wf_memory_males %>% 
  finalize_workflow(select_best(xgb_rs_memory_males, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_males_split)

# Collect metrics
xgb_last_memory_males %>% collect_metrics()

# Collect predictions
xgb_last_memory_males %>% collect_predictions()


# Let's now group the predictions

xgb_memory_preds_males <- xgb_last_memory_males$.predictions

xgb_memory_preds_males <- xgb_memory_preds_males[[1]][2]

xgb_memory_preds_males <- as.matrix(xgb_memory_preds_males)



##############################################################
####### Recipe for xgboost classifier (Basic 4 DIORs)#########
##############################################################


# Define metrics
xg_rec_basic_males <- recipe(Survival2 ~ Age_At_Bereavement + Average + MSE +
                               AC_Detrended  + linearity + 
                               IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_males_training)


# Tune the hyperparameters

xgb_spec_basic_males <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_basic_males <- workflow(xg_rec_basic_males, xgb_spec_basic_males)
xgb_wf_basic_males

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_basic_males <-
  tune_race_anova(
    xgb_wf_basic_males,
    DIORs_clean3_males_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_basic_males)

# We show the best grid of hyperparameters
show_best(xgb_rs_basic_males)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_basic_males <- xgb_wf_basic_males %>% 
  finalize_workflow(select_best(xgb_rs_basic_males, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_males_split)

# Collect metrics
xgb_last_basic_males %>% collect_metrics()

# Collect predictions
xgb_last_basic_males %>% collect_predictions()


# Let's now group the predictions

xgb_basic_preds_males <- xgb_last_basic_males$.predictions

xgb_basic_preds_males <- xgb_basic_preds_males[[1]][2]

xgb_basic_preds_males <- as.matrix(xgb_basic_preds_males)



##############################################################
####### Recipe for xgboost classifier(Benchmark)##############
##############################################################


# Define metrics
xg_rec_benchmark_males <- recipe(Survival2 ~ Age_At_Bereavement + 
                                   IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_males_training)


# Tune the hyperparameters

xgb_spec_benchmark_males <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_benchmark_males <- workflow(xg_rec_benchmark_males, xgb_spec_benchmark_males)
xgb_wf_benchmark_males

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_benchmark_males <-
  tune_race_anova(
    xgb_wf_benchmark_males,
    DIORs_clean3_males_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_benchmark_males)

# We show the best grid of hyperparameters
show_best(xgb_rs_benchmark_males)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_benchmark_males <- xgb_wf_benchmark_males %>% 
  finalize_workflow(select_best(xgb_rs_benchmark_males, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_males_split)

# Collect metrics
xgb_last_benchmark_males %>% collect_metrics()

# Collect predictions
xgb_last_benchmark_males %>% collect_predictions()


# Let's now group the predictions

xgb_benchmark_preds_males <- xgb_last_benchmark_males$.predictions

xgb_benchmark_preds_males <- xgb_benchmark_preds_males[[1]][2]

xgb_benchmark_preds_males <- as.matrix(xgb_benchmark_preds_males)


##############################################################
####### Recipe for xgboost classifier(Age Only)###############
##############################################################


# Define metrics
xg_rec_simple_males <- recipe(Survival2 ~ Age_At_Bereavement, data = DIORs_clean3_males_training)


# Tune the hyperparameters

xgb_spec_simple_males <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_simple_males <- workflow(xg_rec_simple_males, xgb_spec_simple_males)
xgb_wf_simple_males

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_simple_males <-
  tune_race_anova(
    xgb_wf_simple_males,
    DIORs_clean3_males_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_simple_males)

# We show the best grid of hyperparameters
show_best(xgb_rs_simple_males)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_simple_males <- xgb_wf_simple_males %>% 
  finalize_workflow(select_best(xgb_rs_simple_males, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_males_split)

# Collect metrics
xgb_last_simple_males %>% collect_metrics()

# Collect predictions
xgb_last_simple_males %>% collect_predictions()


# Let's now group the predictions

xgb_simple_preds_males <- xgb_last_simple_males$.predictions

xgb_simple_preds_males <- xgb_simple_preds_males[[1]][2]

xgb_simple_preds_males <- as.matrix(xgb_simple_preds_males)


# For females

# Split now

set.seed(234)

DIORs_clean3_females_split <- initial_split(DIORs_clean3_females)

DIORs_clean3_females_training <- training(DIORs_clean3_females_split)

DIORs_clean3_females_testing <- testing(DIORs_clean3_females_split)

# Let's create the folds dataset

set.seed(234)
DIORs_clean3_females_folds <- vfold_cv(DIORs_clean3_females_training,v = 5)


# Start the modelling of XGBoost

##############################################################
####### Recipe for XGBboost classifier (FULL MODEL)############
##############################################################

# Define metrics

metrics <- metric_set(mn_log_loss, roc_auc)

xg_rec_females <- recipe(Survival2 ~ Age_At_Bereavement + Average + MSE + coef_hurst + acf10 + 
                           diff1_acf1 + trend_strength + spikiness + linearity + longest_flat_spot + shift_level_max +
                           n_crossing_points + Entropy + acf1 + shift_var_index + shift_level_index + shift_kl_index +
                           diff1_acf10 + stl_e_acf1 + stl_e_acf10 + diff2_acf1 + diff2_acf10 + diff1_pacf5 + diff2_pacf5 +
                           zero_run_mean + nonzero_squared_cv + zero_start_prop + var_tiled_var + var_tiled_mean + pacf5 + 
                           IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_females_training)


# Tune the hyperparameters

xgb_spec_females <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_females <- workflow(xg_rec_females, xgb_spec_females)
xgb_wf_females

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_females <-
  tune_race_anova(
    xgb_wf_females,
    DIORs_clean3_females_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_females)

# We show the best grid of hyperparameters
show_best(xgb_rs_females)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_females <- xgb_wf_females %>% 
  finalize_workflow(select_best(xgb_rs_females, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_females_split)

# Collect metrics
xgb_last_females %>% collect_metrics()

# Collect predictions
xgb_last_females %>% collect_predictions()


# Let's now group the predictions   

xgb_full_preds_females <- xgb_last_females$.predictions

xgb_full_preds_females <- xgb_full_preds_females[[1]][2]

xgb_full_preds_females <- as.matrix(xgb_full_preds_females)


# Evaluate Performance on testing set (using riskRegression's package function Score)

full_score_females <- Score(list('Xgboost' = xgb_full_preds_females),formula = Survival2 ~ 1, data = DIORs_clean3_females_testing, summary = 'ipa',
                            plots = 'calibration', metrics = c('AUC','brier'), se.fit = T, seed = 9)

summary(full_score_females)

plotCalibration(full_score_females,round = F,rug = F)

##############################################################
####### Recipe for xgboost classifier (Overall Dynamics)######
##############################################################


# Define metrics
xg_rec_overall_females <- recipe(Survival2 ~ Age_At_Bereavement + Average +trend_strength + linearity +
                                   zero_run_mean + zero_start_prop + 
                                   IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_females_training)


# Tune the hyperparameters

xgb_spec_overall_females <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_overall_females <- workflow(xg_rec_overall_females, xgb_spec_overall_females)
xgb_wf_overall_females

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_overall_females <-
  tune_race_anova(
    xgb_wf_overall_females,
    DIORs_clean3_females_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_overall_females)

# We show the best grid of hyperparameters
show_best(xgb_rs_overall_females)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_overall_females <- xgb_wf_overall_females %>% 
  finalize_workflow(select_best(xgb_rs_overall_females, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_females_split)

# Collect metrics
xgb_last_overall_females %>% collect_metrics()

# Collect predictions
xgb_last_overall_females %>% collect_predictions()


# Let's now group the predictions

xgb_overall_preds_females <- xgb_last_overall_females$.predictions

xgb_overall_preds_females <- xgb_overall_preds_females[[1]][2]

xgb_overall_preds_females <- as.matrix(xgb_overall_preds_females)


# Evaluate performance

full_score1_females <- Score(list('Xgboost' = xgb_overall_preds_females),formula = Survival2 ~ 1, data = DIORs_clean3_females_testing, summary = 'ipa',
                             plots = 'calibration', metrics = c('AUC','brier'), se.fit = T, seed = 9)

summary(full_score1_females)

plotCalibration(full_score1_females,round = F,rug = F)


##############################################################
####### Recipe for xgboost classifier (Dispersion Dynamics)###
##############################################################


# Define recipe
xg_rec_dispersion_females <- recipe(Survival2 ~ Age_At_Bereavement + MSE + spikiness + shift_level_max +
                                      shift_level_index + shift_var_index + shift_kl_index + n_crossing_points + Entropy +
                                      nonzero_squared_cv + var_tiled_var + var_tiled_mean +
                                      IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_females_testing)


# Tune the hyperparameters

xgb_spec_dispersion_females <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_dispersion_females <- workflow(xg_rec_dispersion_females, xgb_spec_dispersion_females)
xgb_wf_dispersion_females

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_dispersion_females <-
  tune_race_anova(
    xgb_wf_dispersion_females,
    DIORs_clean3_females_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_dispersion_females)

# We show the best grid of hyperparameters
show_best(xgb_rs_dispersion_females)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_dispersion_females <- xgb_wf_dispersion_females %>% 
  finalize_workflow(select_best(xgb_rs_dispersion_females, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_females_split)

# Collect metrics
xgb_last_dispersion_females %>% collect_metrics()

# Collect predictions
xgb_last_dispersion_females %>% collect_predictions()


# Let's now group the predictions

xgb_dispersion_preds_females <- xgb_last_dispersion_females$.predictions

xgb_dispersion_preds_females <- xgb_dispersion_preds_females[[1]][2]

xgb_dispersion_preds_females <- as.matrix(xgb_dispersion_preds_females)


##############################################################
####### Recipe for xgboost classifier (Memory Dynamics)#######
##############################################################


# Define metrics
xg_rec_memory_females <- recipe(Survival2 ~ Age_At_Bereavement + stl_e_acf1 + coef_hurst + acf10 +
                                  longest_flat_spot  + acf1 + diff1_acf10 + diff1_acf1 +stl_e_acf10 +
                                  diff2_acf1 + diff2_acf10 + diff1_pacf5 + diff2_pacf5 + pacf5 + 
                                  IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_females_training)


# Tune the hyperparameters

xgb_spec_memory_females <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_memory_females <- workflow(xg_rec_memory_females, xgb_spec_memory_females)
xgb_wf_memory_females

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_memory_females <-
  tune_race_anova(
    xgb_wf_memory_females,
    DIORs_clean3_females_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_memory_females)

# We show the best grid of hyperparameters
show_best(xgb_rs_memory_females)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_memory_females <- xgb_wf_memory_females %>% 
  finalize_workflow(select_best(xgb_rs_memory_females, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_females_split)

# Collect metrics
xgb_last_memory_females %>% collect_metrics()

# Collect predictions
xgb_last_memory_females %>% collect_predictions()


# Let's now group the predictions

xgb_memory_preds_females <- xgb_last_memory_females$.predictions

xgb_memory_preds_females <- xgb_memory_preds_females[[1]][2]

xgb_memory_preds_females <- as.matrix(xgb_memory_preds_females)



##############################################################
####### Recipe for xgboost classifier (Basic 4 DIORs)#########
##############################################################


# Define metrics
xg_rec_basic_females <- recipe(Survival2 ~ Age_At_Bereavement + Average + MSE +
                                 AC_Detrended  + linearity + 
                                 IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_females_training)


# Tune the hyperparameters

xgb_spec_basic_females <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_basic_females <- workflow(xg_rec_basic_females, xgb_spec_basic_females)
xgb_wf_basic_females

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_basic_females <-
  tune_race_anova(
    xgb_wf_basic_females,
    DIORs_clean3_females_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_basic_females)

# We show the best grid of hyperparameters
show_best(xgb_rs_basic_females)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_basic_females <- xgb_wf_basic_females %>% 
  finalize_workflow(select_best(xgb_rs_basic_females, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_females_split)

# Collect metrics
xgb_last_basic_females %>% collect_metrics()

# Collect predictions
xgb_last_basic_females %>% collect_predictions()


# Let's now group the predictions

xgb_basic_preds_females <- xgb_last_basic_females$.predictions

xgb_basic_preds_females <- xgb_basic_preds_females[[1]][2]

xgb_basic_preds_females <- as.matrix(xgb_basic_preds_females)



##############################################################
####### Recipe for xgboost classifier(Benchmark)##############
##############################################################


# Define metrics
xg_rec_benchmark_females <- recipe(Survival2 ~ Age_At_Bereavement + 
                                     IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_females_training)


# Tune the hyperparameters

xgb_spec_benchmark_females <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_benchmark_females <- workflow(xg_rec_benchmark_females, xgb_spec_benchmark_females)
xgb_wf_benchmark_females

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_benchmark_females <-
  tune_race_anova(
    xgb_wf_benchmark_females,
    DIORs_clean3_females_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_benchmark_females)

# We show the best grid of hyperparameters
show_best(xgb_rs_benchmark_females)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_benchmark_females <- xgb_wf_benchmark_females %>% 
  finalize_workflow(select_best(xgb_rs_benchmark_females, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_females_split)

# Collect metrics
xgb_last_benchmark_females %>% collect_metrics()

# Collect predictions
xgb_last_benchmark_females %>% collect_predictions()


# Let's now group the predictions

xgb_benchmark_preds_females <- xgb_last_benchmark_females$.predictions

xgb_benchmark_preds_females <- xgb_benchmark_preds_females[[1]][2]

xgb_benchmark_preds_females <- as.matrix(xgb_benchmark_preds_females)


##############################################################
####### Recipe for xgboost classifier(Age Only)###############
##############################################################


# Define metrics
xg_rec_simple_females <- recipe(Survival2 ~ Age_At_Bereavement, data = DIORs_clean3_females_training)


# Tune the hyperparameters

xgb_spec_simple_females <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_simple_females <- workflow(xg_rec_simple_females, xgb_spec_simple_females)
xgb_wf_simple_females

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_simple_females <-
  tune_race_anova(
    xgb_wf_simple_females,
    DIORs_clean3_females_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_simple_females)

# We show the best grid of hyperparameters
show_best(xgb_rs_simple_females)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_simple_females <- xgb_wf_simple_females %>% 
  finalize_workflow(select_best(xgb_rs_simple_females, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_females_split)

# Collect metrics
xgb_last_simple_females %>% collect_metrics()

# Collect predictions
xgb_last_simple_females %>% collect_predictions()


# Let's now group the predictions

xgb_simple_preds_females <- xgb_last_simple_females$.predictions

xgb_simple_preds_females <- xgb_simple_preds_females[[1]][2]

xgb_simple_preds_females <- as.matrix(xgb_simple_preds_females)


# Now let's create the final scores for ALL

final_score_non_stratified <- Score(list('Benchmark + Aggregated Dynamics' = xgb_full_preds,
                                         'Benchmark + Four Basic DIORs' = xgb_basic_preds,
                                         'Benchmark + Overall Dynamics' = xgb_overall_preds,
                                         'Benchmark + Dispersion Dynamics' = xgb_dispersion_preds,
                                         'Benchmark + Memory Dynamics' = xgb_memory_preds,
                                         'Benchmark' = xgb_benchmark_preds,
                                         'Age and Sex' = xgb_simple_preds),formula = Survival2 ~ 1, data = DIORs_clean3_test, summary = 'ipa',
                                    plots = 'calibration', metrics = c('AUC','brier'), se.fit = T, seed = 9)

summary(final_score_non_stratified)

plotCalibration(final_score_non_stratified,models = c('Age and Sex', 'Benchmark', 
                                                      'Benchmark + Four Basic DIORs', 'Benchmark + Aggregated Dynamics'),
                rug = F, round = F, auc.in.legend = F, brier.in.legend = F)


#################################
######### Graph Creation ########
#################################

# Create a barplot which shows the AUCs and Brier Scores of the models

# Keep the AUC scores as a dataframe
plot_df <- final_score_non_stratified$AUC$score

# Grab the LB and UB of the AUC
plot_df <- plot_df %>% rename(Lower_AUC = 'lower', 'Upper_AUC' = 'upper')
plot_df <- as.data.frame(plot_df)


# Keep the Brier scores as a dataframe
plot_df_b <- final_score_non_stratified$Brier$score

# Grab the LB and UB of the Brier Scores
plot_df_b <- plot_df_b %>% 
  select(-IPA) %>% 
  rename(Lower_Brier = 'lower', Upper_Brier = 'upper')

plot_df_b <- as.data.frame(plot_df_b)


# Merge the two dataframes into one
plot_df_merged <- plot_df %>% inner_join(plot_df_b,by = 'model')


# Create the first plot for the AUCs
first_plot <- 
  plot_df_merged %>% 
  select(-se.x,-se.y) %>% 
  mutate(model = factor(model,levels = c('Age and Sex','Benchmark','Benchmark + Four Basic DIORs','Benchmark + Overall Dynamics',
                                         'Benchmark + Dispersion Dynamics','Benchmark + Memory Dynamics',
                                         'Benchmark + Aggregated Dynamics'))) %>% 
  ggplot() +
  geom_bar(aes(x = model, y = AUC), stat = 'identity',fill = 'skyblue', alpha = 0.5,width = 0.3) +
  geom_pointrange(aes(x = model, y = AUC, ymin = Lower_AUC,ymax = Upper_AUC),width = 0.3,
                  colour = 'orange', alpha = 0.9, size = 1.8,fatten = T) +
  theme_minimal() +
  scale_y_continuous(breaks = seq(0,0.9,0.1)) +
  scale_x_discrete(labels = scales::label_wrap(15)) + 
  geom_hline(yintercept = 0.5, color = 'black',linetype = 'dashed')  +
  xlab(label = NULL) + 
  theme(axis.text.x = element_blank())


# Create the second plot for the Brier Scores
second_plot <- plot_df_merged %>% 
  mutate(model = factor(model,levels = c('Age and Sex','Benchmark','Benchmark + Four Basic DIORs','Benchmark + Overall Dynamics',
                                         'Benchmark + Dispersion Dynamics','Benchmark + Memory Dynamics',
                                         'Benchmark + Aggregated Dynamics'))) %>% 
  ggplot() +
  geom_bar(aes(x = model, y = Brier), stat = 'identity',fill = 'skyblue', alpha = 0.5,width = 0.3) +
  geom_pointrange(aes(x = model, y = Brier ,ymin = Lower_Brier, ymax = Upper_Brier),width = 0.3,
                  colour = 'orange', alpha = 0.9, size = 1.8,fatten = T) +
  scale_y_continuous(breaks = seq(0,0.08,0.01),minor_breaks = 8) +
  scale_x_discrete(labels = scales::label_wrap(15)) +
  theme_minimal() +
  xlab(label = NULL)

# We use the library patchwork to stack the two plots on top of each other

(first_plot / second_plot) # It's as easy as that



# Now let's create the final scores for Males (Using Score Function of riskRegression package)

final_score_non_stratified_males <- Score(list('Benchmark + Aggregated Dynamics' = xgb_full_preds_males,
                                               'Benchmark + Four Basic DIORs' = xgb_basic_preds_males,
                                               'Benchmark + Overall Dynamics' = xgb_overall_preds_males,
                                               'Benchmark + Dispersion Dynamics' = xgb_dispersion_preds_males,
                                               'Benchmark + Memory Dynamics' = xgb_memory_preds_males,
                                               'Benchmark' = xgb_benchmark_preds_males,
                                               'Age Only' = xgb_simple_preds_males),formula = Survival2 ~ 1, data = DIORs_clean3_males_testing, summary = 'ipa',
                                          plots = 'calibration', metrics = c('AUC','brier'), se.fit = T, seed = 9)

summary(final_score_non_stratified_males)

plotCalibration(final_score_non_stratified_males,models = c('Age Only', 'Benchmark', 
                                                            'Benchmark + Four Basic DIORs', 'Benchmark + Aggregated Dynamics'),
                rug = F, round = F, auc.in.legend = F, brier.in.legend = F)


# Now let's create the final scores for Females

final_score_non_stratified_females <- Score(list('Benchmark + Aggregated Dynamics' = xgb_full_preds_females,
                                                 'Benchmark + Four Basic DIORs' = xgb_basic_preds_females,
                                                 'Benchmark + Overall Dynamics' = xgb_overall_preds_females,
                                                 'Benchmark + Dispersion Dynamics' = xgb_dispersion_preds_females,
                                                 'Benchmark + Memory Dynamics' = xgb_memory_preds_females,
                                                 'Benchmark' = xgb_benchmark_preds_females,
                                                 'Age Only' = xgb_simple_preds_females),formula = Survival2 ~ 1, data = DIORs_clean3_females_testing, summary = 'ipa',
                                            plots = 'calibration', metrics = c('AUC','brier'), se.fit = T, seed = 9)

summary(final_score_non_stratified_females)

plotCalibration(final_score_non_stratified_females,models = c('Age Only', 'Benchmark', 
                                                              'Benchmark + Four Basic DIORs', 'Benchmark + Aggregated Dynamics'),
                rug = F, round = F, auc.in.legend = F, brier.in.legend = F)


# We have our predictions, now we can focus on the prediction stratified on age groups

# Split the datasets

DIORs_75_split <- initial_split(DIORs_75_84)
DIORs_75_train <- training(DIORs_75_split)
DIORs_75_test <- testing(DIORs_75_split)
DIORs_75_folds <- vfold_cv(DIORs_75_train, v = 5)

DIORs_65_split <- initial_split(DIORs_65_75)
DIORs_65_train <- training(DIORs_65_split)
DIORs_65_test <- testing(DIORs_65_split)
DIORs_65_folds <- vfold_cv(DIORs_65_train, v = 5)

DIORs_85_split <- initial_split(DIORs_85)
DIORs_85_train <- training(DIORs_85_split)
DIORs_85_test <- testing(DIORs_85_split)
DIORs_85_folds <- vfold_cv(DIORs_85_train, v = 5)


# For the 75-84
# Model
xg_rec_75 <- recipe(Survival2 ~ Age_At_Bereavement + Average + Sex2 + MSE + coef_hurst + acf10 + 
                      diff1_acf1 + trend_strength + spikiness + linearity + longest_flat_spot + shift_level_max +
                      n_crossing_points + Entropy + acf1 + shift_var_index + shift_level_index + shift_kl_index +
                      diff1_acf10 + stl_e_acf1 + stl_e_acf10 + diff2_acf1 + diff2_acf10 + diff1_pacf5 + diff2_pacf5 +
                      zero_run_mean + nonzero_squared_cv + zero_start_prop + var_tiled_var + var_tiled_mean + pacf5 + 
                      IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_75_train)


# Tune the hyperparameters

xgb_spec_75 <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_75 <- workflow(xg_rec_75, xgb_spec_75)
xgb_wf_75

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_75 <-
  tune_race_anova(
    xgb_wf_75,
    DIORs_75_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_75)

# We show the best grid of hyperparameters
show_best(xgb_rs_75)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_75 <- xgb_wf_75 %>% 
  finalize_workflow(select_best(xgb_rs_75, 'mn_log_loss')) %>% 
  last_fit(DIORs_75_split)

# Collect metrics
xgb_last_75 %>% collect_metrics()

# Collect predictions
xgb_last_75 %>% collect_predictions()


# Let's now group the predictions   

xgb_full_preds_75 <- xgb_last_75$.predictions

xgb_full_preds_75 <- xgb_full_preds_75[[1]][2]

xgb_full_preds_75 <- as.matrix(xgb_full_preds_75)


# Evaluation of model
old_score75 <- Score(list('XGBoost_75' = xgb_full_preds_75),
                     formula = Survival2 ~ 1, metrics = c('AUC','brier'), 
                     summary = c('ipa'), plots = c('cal'), 
                     se.fit = T, data = DIORs_75_test,
                     seed = 9)

# Extraction of Results
summary(old_score75)


# For the 65-74
# Model

xg_rec_65 <- recipe(Survival2 ~ Age_At_Bereavement + Average + Sex2 + MSE + coef_hurst + acf10 + 
                      diff1_acf1 + trend_strength + spikiness + linearity + longest_flat_spot + shift_level_max +
                      n_crossing_points + Entropy + acf1 + shift_var_index + shift_level_index + shift_kl_index +
                      diff1_acf10 + stl_e_acf1 + stl_e_acf10 + diff2_acf1 + diff2_acf10 + diff1_pacf5 + diff2_pacf5 +
                      zero_run_mean + nonzero_squared_cv + zero_start_prop + var_tiled_var + var_tiled_mean + pacf5 + 
                      IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_65_train)


# Tune the hyperparameters

xgb_spec_65 <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_65 <- workflow(xg_rec_65, xgb_spec_65)
xgb_wf_65

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_65 <-
  tune_race_anova(
    xgb_wf_65,
    DIORs_65_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_65)

# We show the best grid of hyperparameters
show_best(xgb_rs_65)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_65 <- xgb_wf_65 %>% 
  finalize_workflow(select_best(xgb_rs_65, 'mn_log_loss')) %>% 
  last_fit(DIORs_65_split)

# Collect metrics
xgb_last_65 %>% collect_metrics()

# Collect predictions
xgb_last_65 %>% collect_predictions()


# Let's now group the predictions   

xgb_full_preds_65 <- xgb_last_65$.predictions

xgb_full_preds_65 <- xgb_full_preds_65[[1]][2]

xgb_full_preds_65 <- as.matrix(xgb_full_preds_65)


# Evaluation of model
old_score65 <- Score(list('XGBoost_65' = xgb_full_preds_65),
                     formula = Survival2 ~ 1, metrics = c('AUC','brier'), 
                     summary = c('ipa'), plots = c('cal'), 
                     se.fit = T, data = DIORs_65_test,
                     seed = 9)

# Extraction of results
summary(old_score65)



# For the 85plus
# Model

xg_rec_85 <- recipe(Survival2 ~ Age_At_Bereavement + Average + Sex2 + MSE + coef_hurst + acf10 + 
                      diff1_acf1 + trend_strength + spikiness + linearity + longest_flat_spot + shift_level_max +
                      n_crossing_points + Entropy + acf1 + shift_var_index + shift_level_index + shift_kl_index +
                      diff1_acf10 + stl_e_acf1 + stl_e_acf10 + diff2_acf1 + diff2_acf10 + diff1_pacf5 + diff2_pacf5 +
                      zero_run_mean + nonzero_squared_cv + zero_start_prop + var_tiled_var + var_tiled_mean + pacf5 + 
                      IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_85_train)


# Tune the hyperparameters

xgb_spec_85 <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_85 <- workflow(xg_rec_85, xgb_spec_85)
xgb_wf_85

# Run a race anova to find out the best hyperparameter configuration

xgb_rs_85 <-
  tune_race_anova(
    xgb_wf_85,
    DIORs_85_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(xgb_rs_85)

# We show the best grid of hyperparameters
show_best(xgb_rs_85)

# We finalize our workflow using the best combination of hyperparameters based on mn_log_loss metric
xgb_last_85 <- xgb_wf_85 %>% 
  finalize_workflow(select_best(xgb_rs_85, 'mn_log_loss')) %>% 
  last_fit(DIORs_85_split)

# Collect metrics
xgb_last_85 %>% collect_metrics()

# Collect predictions
xgb_last_85 %>% collect_predictions()


# Let's now group the predictions   

xgb_full_preds_85 <- xgb_last_85$.predictions

xgb_full_preds_85 <- xgb_full_preds_85[[1]][2]

xgb_full_preds_85 <- as.matrix(xgb_full_preds_85)


# Evaluation of model
old_score85 <- Score(list('XGBoost_85' = xgb_full_preds_85),
                     formula = Survival2 ~ 1, metrics = c('AUC','brier'), 
                     summary = c('ipa'), plots = c('cal'), 
                     se.fit = T, data = DIORs_85_test,
                     seed = 9)

# Extraction of results
summary(old_score85)


##############################################################
#################### Decision Curve Analysis #################
##############################################################

# Extract Predictions

# Predictions of the full model

DIORs_clean3_test$predd <- as.vector(xgb_full_preds)

# Predictions of the Benchmark model
DIORs_clean3_test$predds_simple <- as.vector(xgb_benchmark_preds)

# Predictions of the Benchmark + Four Basic Diors model
DIORs_clean3_test$predds_basic_diors <- as.vector(xgb_basic_preds)

# Predictions of the Benchmark + Overall Dynamics DIORs
DIORs_clean3_test$predds_overall_dynamics <- as.vector(xgb_overall_preds)

# Predictions of the Benchmark + Dispersion Dynamics DIORs
DIORs_clean3_test$predds_dispersion_dynamics <- as.vector(xgb_dispersion_preds)

# Predictions of the Benchmark + Memory Dynamics DIORs
DIORs_clean3_test$predds_memory_dynamics <- as.vector(xgb_memory_preds)


# Accordingly for each model we need to calculate the Proportion of Accurately Diagnosed and Treated Individuals
# This has been mainly termed as Net Benefit

dc1 <- decision_curve(Status2 ~ predd,fitted.risk = T,data = DIORs_clean3_test, thresholds = seq(0,0.5,0.05))

dc_simple1 <- decision_curve(Status2 ~ predds_simple,fitted.risk = T,data = DIORs_clean3_test, thresholds = seq(0,0.5,0.05))

dc_basic_one1 <- decision_curve(Status2 ~ predds_basic_diors,fitted.risk = T,data = DIORs_clean3_test, thresholds = seq(0,0.5,0.05))

dc_overall1 <- decision_curve(Status2 ~ predds_overall_dynamics,fitted.risk = T,data = DIORs_clean3_test, thresholds = seq(0,0.5,0.05))

dc_dispersion1 <- decision_curve(Status2 ~ predds_dispersion_dynamics,fitted.risk = T,data = DIORs_clean3_test, thresholds = seq(0,0.5,0.05))

dc_memory1 <- decision_curve(Status2 ~ predds_memory_dynamics,fitted.risk = T,data = DIORs_clean3_test, thresholds = seq(0,0.5,0.05))


# Now we plot the decision curves for the models

plot_decision_curve(list(dc1,dc_simple1,dc_basic_one1,dc_overall1,dc_dispersion1,dc_memory1),
                    curve.names = c('All DIORs + Sociodemographics', 'Sociodemographics Only',
                                    'Sociodemographics + 4 Basic DIORs','Sociodemographics + Overall Trend Dynamics',
                                    'Sociodemographics + Dispersion Dynamics','Sociodemographics + Memory Dynamics'),standardize = F,
                    confidence.intervals = F,cost.benefit.axis = F,ylim = c(-0.02,0.06),
                    ylab = 'Accurately Diagnosed and Treated (Proportion)', xlab = 'Risk Threshold')


######################################
######## Interpretation of ML ########
######################################


# We have the Decision Curves now, now let's look at some feature importance plots


##################################################################
################### DALEXtra for explainability ##################
##################################################################


library(DALEXtra)

vip_features <- c('Age_At_Bereavement', 'Average', 'Sex2' , 'MSE' , 'coef_hurst' , 'acf10' , 
                  'diff1_acf1' , 'trend_strength' , 'spikiness' , 'linearity' , 'longest_flat_spot' , 'shift_level_max' ,
                  'n_crossing_points' , 'Entropy' , 'acf1' , 'shift_var_index' , 'shift_level_index' , 'shift_kl_index' ,
                  'diff1_acf10' , 'stl_e_acf1' , 'stl_e_acf10' , 'diff2_acf1' , 'diff2_acf10' , 'diff1_pacf5' , 'diff2_pacf5' ,
                  'zero_run_mean' , 'nonzero_squared_cv' , 'zero_start_prop' , 'var_tiled_var' , 'var_tiled_mean' , 'pacf5' , 
                  'IE_TYPE' , 'Children' , 'AFFLUENCE_GROUP' , 'Total_Number')


vip_train <- DIORs_clean3_train %>% 
  select(all_of(vip_features))


xgb_fit <- xgb_last$.workflow[[1]]

explainer_xg <- explain_tidymodels(xgb_fit, data = vip_train, y = DIORs_clean3_train$Status2, 
                                   label = 'XGBoost', verbose = FALSE)


# Permutation importance by permuting only once
fimp <- feature_importance(explainer_xg,B = 1,
                           variable_groups = list('Age and Sex' = c('Age_At_Bereavement', 'Sex2'),
                                                  'Average Medical Spending' = c('Average'),
                                                  'Other Sociodemographics' = c('IE_TYPE', 'Children','AFFLUENCE_GROUP','Total_Number'),
                                                  'Overall Dynamics Except Average' = c('trend_strength','linearity','zero_run_mean','zero_start_prop'),
                                                  'Dispersion Dynamics' = c('MSE','spikiness','Entropy','shift_level_max','shift_level_index','shift_kl_index','nonzero_squared_cv','var_tiled_var','var_tiled_mean','shift_var_index','n_crossing_points'),
                                                  'Memory Dynamics' = c('coef_hurst','acf10','diff1_acf1','longest_flat_spot','acf1','diff1_acf10','stl_e_acf10','diff2_acf1','diff2_acf10','diff1_pacf5','diff2_pacf5','pacf5')))
plot(fimp)


# Permutation importance by permuting 1000 times
fimpp <- feature_importance(explainer_xg,B = 1000,
                            variable_groups = list('Age and Sex' = c('Age_At_Bereavement', 'Sex2'),
                                                   'Average Medical Spending' = c('Average'),
                                                   'Other Sociodemographics' = c('IE_TYPE', 'Children','AFFLUENCE_GROUP','Total_Number'),
                                                   'Overall Dynamics Except Average' = c('trend_strength','linearity','zero_run_mean','zero_start_prop'),
                                                   'Dispersion Dynamics' = c('MSE','spikiness','Entropy','shift_level_max','shift_level_index','shift_kl_index','nonzero_squared_cv','var_tiled_var','var_tiled_mean','shift_var_index','n_crossing_points'),
                                                   'Memory Dynamics' = c('coef_hurst','acf10','diff1_acf1','longest_flat_spot','acf1','diff1_acf10','stl_e_acf10','diff2_acf1','diff2_acf10','diff1_pacf5','diff2_pacf5','pacf5')))

plot(fimpp)

plot(fimpp) + ggtitle('Mean variable-importance over 1000 permutations', '')


###########################################################
############### Full Expenditure Analysis #################
###########################################################


# Restructure the original dataset so as we have one column per weekly-expenditure variable

full_expenditures <- df_fill %>% select(PERSON_ID, Age_At_Bereavement, Sex, Total_Costs, Weeks, Date_Of_Death, Bereavement_Date)

full_expenditures

full_expenditures <- full_expenditures %>% mutate(Survival2 = ifelse(Date_Of_Death > Bereavement_Date + 365.25, 'Alive', 'Deceased'))

full_expenditures$Survival2 <- as.factor(full_expenditures$Survival2)

full_wide <- full_expenditures %>% pivot_wider(names_from = Weeks, values_from = Total_Costs,names_prefix = 'Week_')

full_wide <- full_wide %>% select(-Date_Of_Death, -Bereavement_Date)

full_wide$Sex2 <- ifelse(full_wide$Sex == 'Males', 0, 1)

full_wide <- full_wide %>% select(-Sex)

full_wide <- full_wide %>% select(-Week_104)

full_wide$Status2 <- ifelse(full_wide$Survival2 == 'Alive', 0, 1)

full_wide <- full_wide %>% relocate(Survival2, .before = Age_At_Bereavement) %>% relocate(Status2, .before = Survival2)

full_wide1 <- full_wide %>% filter(PERSON_ID %in% DIORs_clean3$PERSON_ID)


# Let's fit an xgboost model (after splitting)

set.seed(234)

full_wide1 <- full_wide1 %>% select(-Status2,-PERSON_ID)

full_wide1_split <- initial_split(full_wide1)

full_wide1_Train <- training(full_wide1_split)

full_wide1_Test <- testing(full_wide1_split)

wide1_folds <- vfold_cv(full_wide1_Train, v = 5)


# Recipe for xgboost classifier

xg_wide_rec <- recipe(Survival2 ~ ., data = full_wide1_Train)


# Tune the hyperparameters

xgb_spec_wide <-
  boost_tree(
    trees = tune(),
    mtry = tune(),
    min_n = tune(),
    tree_depth = tune(),
    stop_iter = tune(),
    learn_rate = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf_wide <- workflow(xg_wide_rec, xgb_spec_wide)
xgb_wf_wide

# Run a race anova to find out the best hyperparameter configuration

xgb_wide_rs <-
  tune_race_anova(
    xgb_wf_wide,
    wide1_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

plot_race(xgb_wide_rs)

show_best(xgb_wide_rs)



# Now let's fit on the testing set

metrics <- metric_set(roc_auc, mn_log_loss)


# End of implementation, gather the metrics

xgb_last_wide <- xgb_wf_wide %>% 
  finalize_workflow(select_best(xgb_wide_rs)) %>% 
  last_fit(full_wide1_split)

# Collect metrics
xgb_last_wide %>% collect_metrics()

# Collect predictions

xgb_last_wide %>% collect_predictions()


# Some more details

xgb_preds_wide <- xgb_last_wide$.predictions

xgb_preds_wide_matrix <- xgb_preds_wide[[1]][2]

xgb_preds_wide_matrix <- as.matrix(xgb_preds_wide_matrix)


# Let's check its performance

xgb_wide_score <- Score(list('XGBoost' = xgb_preds_wide_matrix), formula = Survival2 ~ 1, 
                        data = full_wide1_Test,metrics = c('AUC', 'brier'), summary = 'IPA', se.fit = T, plots = 'calibration' )

summary(xgb_wide_score)

# Let's check the calibration of both (Full Expenditures and DIORs)
plotCalibration(xgb_wide_score)
plotCalibration(full_score)

# Put them together

xgb_wide_score <- Score(list('XGBoost (Full Expenditures)' = xgb_preds_wide_matrix,
                             'XGBoost (DIORs)' = xgb_full_preds), formula = Survival2 ~ 1, 
                        data = full_wide1_Test,metrics = c('AUC', 'brier'), summary = 'IPA', se.fit = T, plots = 'calibration' )

plotCalibration(xgb_wide_score,rug = F, round = F, auc.in.legend = F, brier.in.legend = F)

##########################################################################
######## Variable importance for the full expenditures model #############
##########################################################################


exp_features <- full_wide1_Train %>% select(-Survival2)

xgb_exp_fit <- xgb_last_wide$.workflow[[1]]

explainer_xg_wide <- explain_tidymodels(xgb_exp_fit, data = exp_features, y = DIORs_clean3_train$Status2, 
                                        label = 'XGBoost', verbose = FALSE)

fimp_wide <- feature_importance(explainer_xg_wide,B = 1000,variables = c('Week_103','Age_At_Bereavement','Sex2'))

plot(fimp_wide)

plot(fimp_wide) + ggtitle('Mean variable-importance over 1000 permutations', '')



# Decision curve analysis for full expenditures model

# Predictions of the Full Expenditures model
DIORs_clean3_test$full_exp_preds <- as.vector(xgb_preds_wide_matrix)


# Accordingly we need to calculate the Proportion of Accurately Diagnosed and Treated Individuals
# This has been mainly termed as Net Benefit

dc_full_exp <- decision_curve(Status2 ~ full_exp_preds,fitted.risk = T,data = DIORs_clean3_test, thresholds = seq(0,0.5,0.05))


###########################################
# Using a GLM(elastic net) to compare #####
###########################################


glm_rec <- recipe(Survival2 ~ Age_At_Bereavement + Average + Sex2 + MSE + coef_hurst + acf10 + 
                    diff1_acf1 + trend_strength + spikiness + linearity + longest_flat_spot + shift_level_max +
                    n_crossing_points + Entropy + acf1 + shift_var_index + shift_level_index + shift_kl_index +
                    diff1_acf10 + stl_e_acf1 + stl_e_acf10 + diff2_acf1 + diff2_acf10 + diff1_pacf5 + diff2_pacf5 +
                    zero_run_mean + nonzero_squared_cv + zero_start_prop + var_tiled_var + var_tiled_mean + pacf5 + 
                    IE_TYPE + Children + AFFLUENCE_GROUP + Total_Number, data = DIORs_clean3_train)

# Normalize the predictors and add natural splines  
glm_rec <- glm_rec %>% step_normalize(all_numeric_predictors(), -Sex2) %>% step_ns(all_numeric_predictors(),-Sex2,deg_free = 3)

# Tune the hyper-parameters (penalty and mixture)

glm_spec <-
  logistic_reg(penalty = tune(),
               mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

glm_wf <- workflow(glm_rec, glm_spec)
glm_wf



# Run a race anova to find out the best hyper-parameter configuration

glm_rs <-
  tune_race_anova(
    glm_wf,
    DIORs_clean3_folds,
    grid = 30,
    metrics = metric_set(mn_log_loss),
    control = control_race(verbose_elim = TRUE)
  )

# We plot the race to visualize the results
plot_race(glm_rs)

# We show the best grid of hyper-parameters
show_best(glm_rs)

# We finalize our workflow using the best combination of hyper-parameters based on mn_log_loss metric
glm_last <- glm_wf %>% 
  finalize_workflow(select_best(glm_rs, 'mn_log_loss')) %>% 
  last_fit(DIORs_clean3_split)

# Collect metrics
glm_last %>% collect_metrics()

# Collect predictions
glm_last %>% collect_predictions()


# Let's now group the predictions   

glm_full_preds <- glm_last$.predictions

glm_full_preds <- glm_full_preds[[1]][2]

glm_full_preds <- as.matrix(glm_full_preds)


# Evaluate Performance on testing set

full_score_glm <- Score(list('Regularized (Elastic Net) Logistic Regression' = glm_full_preds,'XGBoost' = xgb_full_preds),formula = Survival2 ~ 1, data = DIORs_clean3_test, summary = 'ipa',
                        plots = 'calibration', metrics = c('AUC','brier'), se.fit = T, seed = 9)

summary(full_score_glm)
summary(final_score_non_stratified)

plotCalibration(full_score_glm,round = F,rug = F,auc.in.legend = F, brier.in.legend = F,
                xlab = 'Predicted 1-year mortality risk after spousal loss', ylab = 'Observed Mortality Proportion')


