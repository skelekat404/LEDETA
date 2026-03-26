# tests/run_rq1_predictive_validity.py

## DESCRIPTION: This script operationalizes RQ1 by building case-level units from the cleaned 
# Enron dataset, extracting engineered features, generating rubric-derived case scores as the 
# target variable, training a LightGBM regressor on an 80/20 split, evaluating predictive 
# performance with MAE, RMSE, and R², testing whether prediction differences are systematic 
# using either a paired t-test or Wilcoxon test depending on normality, and then saving the 
# results and raw validation outputs for reporting.

import argparse #allows script to accept command-line arguments
import numpy as np #numerical operations
import pandas as pd #reading CSV and dfs

GLOBAL_RANDOM_STATE = 42 #define global random seed value

from sklearn.model_selection import train_test_split #splits your dataset into training and validation
from sklearn.impute import SimpleImputer #a preprocessing tool that fills in missing values
from sklearn.pipeline import Pipeline #chain preprocessing and modeling into one workflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #three main regression eval metrics

from ledeta.case_builder import build_cases #func converts raw emails into case-level units
from ledeta.rubric import score_case_rubric #rubric scoring func - target score model is learning
from ledeta.features import extract_engineered_features, DEFAULT_KEYWORDS #turns each case into structure numeric feature + default keyword list

# Stats
from scipy.stats import shapiro, ttest_rel, wilcoxon #shapiro-wilk normality, paired t--test, wilcoxon


def rmse(y_true, y_pred): #helper func takes true target values and predicted values
    return float(np.sqrt(mean_squared_error(y_true, y_pred))) #calculates MSE b/n true and predicted and takes sqrt to convert to RMSE -> float

def paired_cohens_d(diff): #helper func that computes Cohen's d effect size for paired samples
    # Cohen's d for paired samples = mean(diff)/sd(diff)
    diff = np.asarray(diff, dtype=float) #converts input diff into a NumPy array of floats
    sd = diff.std(ddof=1) #computes sample standard deviation of paired differences. ddof=1 means it uses sample version, not pop version
    return float(diff.mean() / sd) if sd > 0 else float("nan") #if std>0, calc Cohen's d as the mean diff divided by std. if std=0, retyrn NaN because undefined effect size

def bootstrap_ci(metric_fn, y_true, y_pred, n_boot=1000, seed=42): #defines func that estimates a bootstrao conf interval for any metric
                #metric_fn - metric function to evaluate, y_true=actual values, y_pred=predicted values, n_boot=# of bootstrap samples, seed=random seed
    rng = np.random.default_rng(seed) #random num generator using seed so bootstrao process is reproducible
    y_true = np.asarray(y_true) #converts true values into NumPy array
    y_pred = np.asarray(y_pred) #converts predicted values into NumPy array
    n = len(y_true) #stores # of observations
    stats = [] #empty list to hold metric value from each sample
    for _ in range(n_boot): #loop that will repeat n_boot times
        idx = rng.integers(0, n, size=n) #creates a bootstrap sample of indices by randomly drawing n indices with replacement from 0 - n-1
        stats.append(metric_fn(y_true[idx], y_pred[idx])) #use sampled indices to created resmapled dataset, computes metric on the resample and adds metric val to stats list
    lo, hi = np.percentile(stats, [2.5, 97.5]) #after all bootstrap runs, calculate the 2.5th and 97.5th percentiles of metric values. those are now low/upper bounds of approximate 95% confidence interval
    return float(lo), float(hi) #returns lower and upper confidence interbal bounds as floats

def cases_to_Xy(cases): #defines a func that converts list of case objs to
    # x: a feature matrix, y: a target vector
    rows = [] #empty list that holds one feature dict per case
    y = [] #empty list that will hold on feature dict per case
    for c in cases: #loop through every case obj in input list
        feats = extract_engineered_features(c, keywords=DEFAULT_KEYWORDS) #calls feature engineering function on current case using default keyword list,
        rows.append({k: float(v) if v is not None else np.nan for k, v in feats.items()}) #builds cleaned feature dict from feats. each key/value pair, convert value to float, use np.nan for missing, append clean feature dict to rows list
        s, _ = score_case_rubric(c)   # scores case with the ribric. rubric triage score (0..100)
        y.append(float(s)) #converts rubric score to float and adds to target list y
    X = pd.DataFrame(rows) #converts list of feature dicts into dfs where each row is a case, column is a feature
    y = np.array(y, dtype=float) #converts target list into NumPy array of floats
    return X, y #returns the feature matrix x and target vector y

def main(): #main entry point for script
    ap = argparse.ArgumentParser() #creates argument parser object
    ap.add_argument("--csv", required=True, help="Path to cleaned Enron CSV") #adds a rwquired command-line argument named --csv. user much provide path to cleaned Enron CSV
    ap.add_argument("--window_days", type=int, default=30) #optional argument specifying case window size in dates, default 30
    ap.add_argument("--body_mode", default="excerpt", choices=["excerpt", "full", "none"]) #optional argument controlling how much body text is stored when cases built. default = exerpt
    ap.add_argument("--excerpt_len", type=int, default=800) #optional arugment specifying how amny chars to keep when body mode is exerpt
    ap.add_argument("--seed", type=int, default=42) #optional argument for random seed
    ap.add_argument("--boot", type=int, default=1000) #opotional argument for # of bootstrap samples. default is 1000
    args = ap.parse_args() #parses the actual command-line arguments and stores them in args

    df = pd.read_csv(args.csv, low_memory=False) #reads the csv file into a df
    df["date"] = pd.to_datetime(df["date"], errors="coerce") #converts date coolumn to datetime objects. invalid becomes NaT instead

    cases = build_cases(df, window_days=args.window_days, body_mode=args.body_mode, excerpt_len=args.excerpt_len) #builds list of case objejcts from raw df using window size, body mode, exerpt length
    X, y = cases_to_Xy(cases) #converts those built cases into: x - engineered feature df, y - rubric-derived target scores

    # LightGBM model
    import lightgbm as lgb #import LightGBM locally in main
    model = lgb.LGBMRegressor( #start defining a LightGBM regression model
        n_estimators=800, ## boosting trees
        learning_rate=0.05, #learning rate, meaning each tree updates predictions in small steps
        num_leaves=63, #max # of leaves, controlling complexity
        min_child_samples=20, #requires at least 20 samples in a child node, prevents overfitting
        subsample=0.9, #use 90% of rows for each tree, adds randomness to avoid overfitting
        colsample_bytree=0.9, #uses 90% of the features for each tree, adding randomness and improving generalization
        reg_alpha=0.0, #sets  L1 regularization to zero
        reg_lambda=0.0, #sets L2 regularization to zero
        random_state=args.seed, #uses provided seed to make mdel training more reproducible
        n_jobs=-1, #tells LightGBM to use all available CPUs
    )

    pipe = Pipeline([ #starts a scikitlearn pipeline so preprocessing and modeling happen together in a consistent sequence
        ("imputer", SimpleImputer(strategy="median")), #imputer step added that fills missing feature values using median of each column
        ("model", model), #adds a second step model, which is the LightGBM regressor defined above
    ])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed) #split datasets into 80% training, 20% validation, reproducible because it uses random seed
    pipe.fit(X_train, y_train) #fits pipeline on training set.. imputer learns medians from X_train. model trains on inputed training features and y_train

    y_pred = np.clip(pipe.predict(X_val), 0, 100) #uses fitted pipeline to predict scores for validation set, then slips those predictions so they stay within valid rubric score range of 0-100

    # Metrics
    mae_v = float(mean_absolute_error(y_val, y_pred)) #computes mean absolute error b/n true validation scores and predicted scores -> float
    rmse_v = rmse(y_val, y_pred) #computes the RMSE using helper func
    r2_v = float(r2_score(y_val, y_pred)) #computes r-squared, which measures variance in the validation targets is explained by model

    # Paired differences test
    diff = y_pred - y_val #computes the elementwise difference between each predicted score and its corresponding rubric score
    sh_w, sh_p = shapiro(diff) #runs the Shapiroo-Wilk normality test on difference scores and stores: test statistic (sh_w) and p-value (sh_p). tells you whether the paired differennces are approx normally distributed

    if sh_p >= 0.05: #if the Shapiro p-value is at least 0.05, treat the differences as not significantly non-normal
        test_name = "paired_t" #stores name of statistical  test chosen: paired t-test
        stat, p = ttest_rel(y_pred, y_val) #runs a paired t-test comparing predicted scores and rubric scores
    else: #if the differences appear non-normal
        test_name = "wilcoxon" #stores chosen test name as Wilcoxon
        stat, p = wilcoxon(y_pred, y_val, zero_method="wilcox", correction=False) #runs the Wilcoxon signed-rank test commpared predicted scores and rubric scores

    d = paired_cohens_d(diff) #computes paired Cohan's d for the difference vector to e stimate effect size

    # Bootstrap CIs
    mae_ci = bootstrap_ci(lambda a,b: float(mean_absolute_error(a,b)), y_val, y_pred, n_boot=args.boot, seed=args.seed) #uses bootstrap helper to estimate 95% conf interval for MAE. lambda func tells bootstrap_ci how to compute MAE on each resample
    rmse_ci = bootstrap_ci(lambda a,b: rmse(a,b), y_val, y_pred, n_boot=args.boot, seed=args.seed) #same for RMSE
    r2_ci = bootstrap_ci(lambda a,b: float(r2_score(a,b)), y_val, y_pred, n_boot=args.boot, seed=args.seed) #same for R-squared

    out = { #starts building a dict to store all final RQ1 output values
        "n_total_cases": int(len(y)), #stores the total # of cases in the full dataset
        "n_train": int(len(y_train)), #stores the number of training cases
        "n_val": int(len(y_val)), #stores the # of validation cases
        "mae": mae_v, "mae_ci_lo": mae_ci[0], "mae_ci_hi": mae_ci[1], #stores MAE and its conf interval bounds
        "rmse": rmse_v, "rmse_ci_lo": rmse_ci[0], "rmse_ci_hi": rmse_ci[1], #stores RMSE and its confidence interval bounds
        "r2": r2_v, "r2_ci_lo": r2_ci[0], "r2_ci_hi": r2_ci[1], #stores R-squared and its confidence interval bounds
        "shapiro_w": float(sh_w), "shapiro_p": float(sh_p), #stores Shapiro-Wilk statistic and p-value
        "paired_test": test_name, "test_stat": float(stat), "test_p": float(p), #stores the name of the paired test used, its test statistic and p-value
        "cohens_d_paired": d, #stores paired Cohen's d
    }

    print("\nRQ1 RESULTS (Predictive Validity)") #section header to terminal
    for k,v in out.items(): #loops through each key/value pair in output dir
        print(f"{k}: {v}") #prints each results on its own line

    pd.DataFrame([out]).to_csv("rq1_results.csv", index=False) #creates a one-row df from the results dict and saves it as rq1_results.csv
    np.savetxt("rq1_y_val.csv", y_val, delimiter=",") #saves the true validation scores into a CSV-style text file
    np.savetxt("rq1_y_pred.csv", y_pred, delimiter=",") #saves the predicted validation scores into a CSV-style text file

    print("\nSaved: rq1_results.csv, rq1_y_val.csv, rq1_y_pred.csv") #prints a confirm message telling the user which files were saved

if __name__ == "__main__": #checks whether files is being run directly from command line instead of imported from another script
    main() #if it is being run directly call main()