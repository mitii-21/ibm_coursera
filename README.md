# IBM Applied Data Science Capstone — SpaceX Falcon 9

## What this project does
- Uses IBM Skills Network course-prepared SpaceX datasets.
- Performs data inspection, missing-value analysis and descriptive statistics.
- Creates EDA visualizations and SQL summaries.
- Produces an interactive Folium launch-site map.
- Trains and tunes Logistic Regression, SVM, Decision Tree and KNN with GridSearchCV.
- Exports metrics, confusion matrix, map, plots and SQLite database.

## One-run setup (Mac)
1. Extract this ZIP.
2. Open Terminal and `cd` into the extracted `Falcon9_Capstone_OneClick` folder.
3. Create and activate an environment:
   ```bash
   conda create -n spacex-capstone python=3.11 -y
   conda activate spacex-capstone
   pip install -r requirements.txt
   jupyter lab
   ```
4. In Jupyter, open `notebooks/00_RUN_EVERYTHING.ipynb`.
5. Select the `spacex-capstone` Python kernel.
6. Choose **Run → Run All Cells**. Wait until the single cell finishes. Outputs appear in `outputs/`.

## If dataset download fails
Download these three files from the IBM Skills Network course dataset location and save them exactly in the project's `data/` folder:
- https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv
- https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv
- https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv
Then run the notebook again.

## Important academic honesty
This pipeline is based on IBM's prepared course datasets, not a fresh extraction of the live SpaceX API. The map only places sites for which coordinates are included in the script. Model scores are generated when you execute the notebook; no scores are pre-filled. Review the generated outputs and write your final interpretation from your actual run. The generated files are analysis artifacts, not a substitute for your course's required narrative, screenshots or rubric-specific sections.

## GitHub
Create a new GitHub repository, then upload the contents of this extracted folder. Do not upload `.ipynb_checkpoints`, `.DS_Store`, or large environment folders.
