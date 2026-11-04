# IBM Applied Data Science Capstone — one-run pipeline
# Downloads IBM Skills Network prepared datasets, performs EDA, SQL, mapping,
# model comparison, and exports all outputs.
from pathlib import Path
import sys, subprocess, importlib.util, warnings, urllib.request, io, json, os
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"; OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

PACKAGES = ["pandas","numpy","matplotlib","seaborn","scikit-learn","requests","folium","sqlalchemy","plotly","nbformat"]
missing = [p for p in PACKAGES if importlib.util.find_spec(p.replace("-","_")) is None]
if missing:
    print("Installing missing packages:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, requests, sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import folium

URLS = {
 "dataset_part_1.csv":"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv",
 "dataset_part_2.csv":"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv",
 "dataset_part_3.csv":"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv",
}
def get_csv(name, url):
    p=DATA/name
    if p.exists() and p.stat().st_size>100: 
        print("Using existing", p.name); return pd.read_csv(p)
    print("Downloading", name)
    r=requests.get(url, timeout=45); r.raise_for_status()
    p.write_bytes(r.content)
    return pd.read_csv(io.BytesIO(r.content))

try:
    raw = get_csv("dataset_part_1.csv", URLS["dataset_part_1.csv"])
    df = get_csv("dataset_part_2.csv", URLS["dataset_part_2.csv"])
    engineered = get_csv("dataset_part_3.csv", URLS["dataset_part_3.csv"])
except Exception as e:
    raise RuntimeError(
        "Dataset download failed. Check internet access, then rerun. "
        "Alternatively download dataset_part_1.csv, dataset_part_2.csv, dataset_part_3.csv "
        "from the IBM Skills Network course and place them in the project's data/ folder. "
        f"Original error: {e}"
    )

# Save source datasets and inspect
print("\nDATA SHAPES", raw.shape, df.shape, engineered.shape)
print("\nCOLUMNS", list(df.columns))
display_df = df.head(8)
display(display_df)

# Standardize target
target_candidates = ["Class","class","LandingSuccess","landing_success"]
target = next((c for c in target_candidates if c in df.columns), None)
if target is None:
    raise ValueError("Could not locate landing outcome target column. Expected 'Class'.")
df[target] = pd.to_numeric(df[target], errors="coerce")
model_df = df.dropna(subset=[target]).copy()
model_df[target] = model_df[target].astype(int)

# Clean and descriptive analysis
summary = model_df.describe(include="all").transpose()
summary.to_csv(OUT/"descriptive_statistics.csv")
model_df.isnull().sum().rename("missing_values").to_csv(OUT/"missing_values.csv")
model_df.to_csv(OUT/"cleaned_launch_data.csv", index=False)
print("\nTarget distribution:\n", model_df[target].value_counts(dropna=False))

# Visualizations
num_cols = model_df.select_dtypes(include=np.number).columns.tolist()
if target in num_cols: num_cols.remove(target)
plots = []
def savefig(name):
    plt.tight_layout(); path=OUT/name; plt.savefig(path,dpi=160,bbox_inches="tight"); plt.show(); plt.close(); plots.append(path)
plt.figure(figsize=(7,4)); sns.countplot(data=model_df, x=target); plt.title("Falcon 9 First-Stage Landing Outcome"); plt.xlabel("Landing class (0 = failure, 1 = success)"); savefig("01_landing_class_distribution.png")
if "PayloadMass" in model_df.columns:
    plt.figure(figsize=(8,4)); sns.histplot(data=model_df,x="PayloadMass",hue=target,kde=True,element="step"); plt.title("Payload Mass Distribution by Landing Outcome"); savefig("02_payload_mass_by_outcome.png")
for col in ["LaunchSite","Orbit","BoosterVersion"]:
    if col in model_df.columns:
        plt.figure(figsize=(10,5)); sns.countplot(data=model_df,y=col,hue=target,order=model_df[col].value_counts().index[:12]); plt.title(f"Launches by {col} and Landing Outcome"); savefig(f"03_{col.lower()}_outcome.png")
if len(num_cols)>=2:
    corr=model_df[num_cols+[target]].corr(numeric_only=True)
    plt.figure(figsize=(10,7)); sns.heatmap(corr,cmap="coolwarm",center=0); plt.title("Correlation Heatmap"); savefig("04_correlation_heatmap.png")

# SQL EDA via SQLite
con=sqlite3.connect(OUT/"spacex_eda.sqlite")
model_df.to_sql("launches",con,if_exists="replace",index=False)
sql_queries = {
 "launches_by_site":"SELECT LaunchSite, COUNT(*) AS launches FROM launches GROUP BY LaunchSite ORDER BY launches DESC",
 "landing_rate_by_site":f"SELECT LaunchSite, COUNT(*) AS launches, AVG([{target}]) AS landing_rate FROM launches GROUP BY LaunchSite ORDER BY landing_rate DESC",
 "landing_rate_by_orbit":f"SELECT Orbit, COUNT(*) AS launches, AVG([{target}]) AS landing_rate FROM launches GROUP BY Orbit ORDER BY landing_rate DESC",
}
with open(OUT/"sql_eda_results.txt","w") as f:
    for title,q in sql_queries.items():
        f.write(f"\n## {title}\n{q}\n")
        try: f.write(pd.read_sql_query(q,con).to_string(index=False)+"\n")
        except Exception as e: f.write(str(e)+"\n")
con.close()

# Folium map using known launch site coordinates when available
site_coords = {
 "CCAFS SLC 40":(28.561857,-80.577366),
 "CCAFS LC-40":(28.561857,-80.577366),
 "KSC LC-39A":(28.6082,-80.6041),
 "KSC LC 39A":(28.6082,-80.6041),
 "VAFB SLC 4E":(34.632093,-120.610829),
 "VAFB SLC-4E":(34.632093,-120.610829),
 "CCSFS SLC-40":(28.561857,-80.577366),
}
m=folium.Map(location=[29.5,-95],zoom_start=4)
if "LaunchSite" in model_df.columns:
    for site,grp in model_df.groupby("LaunchSite"):
        coord=next((v for k,v in site_coords.items() if k in str(site)),None)
        if coord:
            rate=float(grp[target].mean())
            folium.CircleMarker(coord,radius=8,popup=f"{site}: {len(grp)} launches, landing rate {rate:.1%}",tooltip=str(site)).add_to(m)
m.save(str(OUT/"launch_sites_map.html"))

# Interactive Plotly dashboard (self-contained HTML)
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "domain"}]], subplot_titles=("Launches by site", "Landing outcome distribution"))
if "LaunchSite" in model_df.columns:
    site_counts=model_df.groupby("LaunchSite").size().reset_index(name="Launches")
    fig.add_trace(go.Bar(x=site_counts["LaunchSite"],y=site_counts["Launches"],name="Launches"),row=1,col=1)
class_counts=model_df[target].value_counts().sort_index()
fig.add_trace(go.Pie(labels=[f"Class {i}" for i in class_counts.index],values=class_counts.values,name="Outcome"),row=1,col=2)
fig.update_layout(title="SpaceX Falcon 9 — Interactive Launch Dashboard",height=520)
fig.write_html(str(OUT/"interactive_dashboard.html"),include_plotlyjs=True)

# Model-ready features: exclude target, outcome leakage and free-text identifiers
drop_cols=[target]
for c in ["Outcome","LandingPad","Serial","Date","FlightNumber"]:
    if c in model_df.columns: drop_cols.append(c)
X=model_df.drop(columns=drop_cols,errors="ignore")
y=model_df[target]
# Convert categorical columns to one-hot; numeric missing values median; categorical mode
X=pd.get_dummies(X,drop_first=True)
X=X.replace([np.inf,-np.inf],np.nan)
X=X.loc[:,X.notna().any()]
X=X.fillna(X.median(numeric_only=True)).fillna(0)
# remove constant features
X=X.loc[:,X.nunique(dropna=False)>1]
if X.shape[1] == 0: raise ValueError("No usable model features after preprocessing.")
stratify=y if y.nunique()==2 and y.value_counts().min()>=2 else None
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=stratify)

models = {
 "Logistic Regression":(Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000,class_weight="balanced"))]),{"model__C":[0.1,1,10]}),
 "SVM":(Pipeline([("scale",StandardScaler()),("model",SVC(class_weight="balanced"))]),{"model__C":[0.1,1,10],"model__kernel":["linear","rbf"],"model__gamma":["scale","auto"]}),
 "Decision Tree":(Pipeline([("model",DecisionTreeClassifier(class_weight="balanced",random_state=42))]),{"model__max_depth":[2,4,6,None],"model__min_samples_split":[2,5]}),
 "KNN":(Pipeline([("scale",StandardScaler()),("model",KNeighborsClassifier())]),{"model__n_neighbors":[3,5,7],"model__weights":["uniform","distance"]}),
}
rows=[]; best_estimators={}
for name,(pipe,grid) in models.items():
    print("\nTraining",name)
    search=GridSearchCV(pipe,grid,cv=3,scoring="f1",n_jobs=-1,error_score="raise")
    search.fit(X_train,y_train)
    pred=search.predict(X_test)
    row={"Model":name,"Best CV F1":search.best_score_,"Test Accuracy":accuracy_score(y_test,pred),
         "Precision":precision_score(y_test,pred,zero_division=0),"Recall":recall_score(y_test,pred,zero_division=0),
         "F1":f1_score(y_test,pred,zero_division=0),"Best Parameters":str(search.best_params_)}
    rows.append(row); best_estimators[name]=(search.best_estimator_,pred)
    print(row)
results=pd.DataFrame(rows).sort_values("F1",ascending=False)
results.to_csv(OUT/"model_comparison.csv",index=False)
print("\nMODEL COMPARISON\n",results.to_string(index=False))
best_name=results.iloc[0]["Model"]; best_model,best_pred=best_estimators[best_name]
(OUT/"best_model_classification_report.txt").write_text(
    f"Best model by held-out F1: {best_name}\n\n"+classification_report(y_test,best_pred,zero_division=0))
cm=confusion_matrix(y_test,best_pred)
plt.figure(figsize=(5,4)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues"); plt.title(f"Confusion Matrix — {best_name}"); plt.xlabel("Predicted"); plt.ylabel("Actual"); savefig("05_best_model_confusion_matrix.png")

# Reproducibility outputs
(OUT/"run_summary.txt").write_text(
 f"Rows used: {len(model_df)}\nFeatures after encoding: {X.shape[1]}\nTarget: {target}\n"
 f"Best model by held-out F1: {best_name}\n\n{results.to_string(index=False)}\n\n"
 "Important: performance is based on the IBM course-prepared dataset and one random 80/20 split. "
 "Do not describe it as a production-ready model or claim causal relationships.")
# Auto-generated report draft using actual run metrics
report = [
 "# SpaceX Falcon 9 First-Stage Landing Prediction — Results Draft", "",
 "## Objective", "Predict whether a Falcon 9 first stage lands successfully using the IBM course-prepared dataset.", "",
 "## Dataset and preparation", f"The analysis used {len(model_df)} records and {X.shape[1]} encoded predictor columns. The target variable was `{target}`. Rows without a target value were excluded; categorical predictors were one-hot encoded and missing numeric values were imputed with the median.", "",
 "## Model evaluation", "Models were tuned using GridSearchCV with 3-fold cross-validation and evaluated on a held-out 20% test split (random_state=42).", "",
 results.to_markdown(index=False), "",
 f"Best model by held-out F1 score: **{best_name}**.", "",
 "## Limitations", "This is a course-dataset demonstration. Results depend on the prepared dataset, feature availability, random split and chosen metrics. The model is not production-ready and should not be interpreted causally. Inspect the confusion matrix and class balance before drawing conclusions.", "",
 "## Figures", "- `01_landing_class_distribution.png`", "- `02_payload_mass_by_outcome.png` (if payload mass is present)", "- `04_correlation_heatmap.png` (if sufficient numeric columns are present)", "- `05_best_model_confusion_matrix.png`", "- `launch_sites_map.html` and `interactive_dashboard.html`"
]
(OUT/"report_results_draft.md").write_text("\n".join(report))
print("\nDONE. All generated artifacts are in:", OUT)
