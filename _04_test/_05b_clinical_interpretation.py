import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))



# Create clinical interpretation text file

# Not Vital 1 day
merge_type = "not_vital"
days_to_consider = 1
csv_to_read = os.path.join(current_dir, f"stratified_model_performance_{merge_type}_{days_to_consider}Days.csv")
if not os.path.exists(csv_to_read):
    print(f"Nessun file corrispondente al percorso selezionato: {csv_to_read}")
    exit()
df_perf = pd.read_csv(csv_to_read)

interpretation = f"""CLINICAL INTERPRETATION GUIDE ({merge_type.upper()} MERGING, {days_to_consider} DAYS)

1. Overall Performance:
- Best balanced accuracy: 0.65 (ConvTran)
- Best F1 score: 0.63 (ROCKET)

2. Key Findings:
- Non-viable detection: ConvTran (BA=0.95) > ROCKET (BA=0.94) > LSTMFCN (BA=0.91)
- 2.1PN reliability: All models show strong performance (BA=0.88, F1=0.86)
- 3PN challenges: ConvTran has lowest BA=0.59 (manual review recommended)

3. Model Comparison:
- ConvTran: Best for 1PN (BA=0.63) and non-viables
- ROCKET: Most consistent in >3PN (F1=0.5)
- LSTMFCN: Avoid for 3PN (BA=0.66, F1=0.57)

4. Critical Observations:
- 1.1PN F1=0.0 across all models - requires embryologist review
- 2PN performance varies significantly (BA 0.55-0.58)

5. Clinical Recommendations:
- Use ConvTran for initial non-viable screening
- Manual verification required for:
  - All 1.1PN cases
  - ConvTran 3PN predictions
  - LSTMFCN >3PN assessments

Note: F1 Score <0.4 indicates high uncertainty, >0.7 indicates clinical reliability
"""

filename = f"stratified_interpretation_{days_to_consider}Days_{merge_type}.txt"
with open(os.path.join(current_dir, filename), 'w') as f:
    f.write(interpretation)




# Not Vital 3 days
merge_type = "not_vital"
days_to_consider = 3
csv_to_read = os.path.join(current_dir, f"stratified_model_performance_{merge_type}_{days_to_consider}Days.csv")
if not os.path.exists(csv_to_read):
    print(f"Nessun file corrispondente al percorso selezionato: {csv_to_read}")
    exit()
df_perf = pd.read_csv(csv_to_read)

interpretation = f"""CLINICAL INTERPRETATION GUIDE ({merge_type.upper()} MERGING, {days_to_consider} DAYS)

1. Overall Performance:
- Best balanced accuracy: 0.76 (ConvTran)
- Best F1 score: 0.75 (LSTMFCN)

2. Key Findings:
- Non-viable perfection: ConvTran (BA=1.0), LSTMFCN (BA=0.99)
- >3PN excellence: ConvTran (BA/F1=1.0), LSTMFCN (F1=0.8)
- 2PN improvement: 3-day models show +12% BA over 1-day

3. Model Strengths:
- ConvTran: Unmatched in >3PN/anomalies (BA=1.0)
- LSTMFCN: Best 2PN F1=0.75
- ROCKET: Reliable 3PN detection (F1=0.84)

4. Critical Observations:
- 1.1PN remains challenging (ConvTran F1=0.0 despite BA=1.0)
- 1PN performance gap: ConvTran BA=0.82 vs LSTMFCN BA=0.67

5. Clinical Protocol:
- Automated clearance for:
  - ConvTran/LSTMFCN non-viables
  - ConvTran >3PN
- Mandatory manual inspection:
  - All 1.1PN embryos
  - 1PN from LSTMFCN
  - 3PN from ROCKET

Note: F1 Score <0.4 indicates high uncertainty, >0.7 indicates clinical reliability
"""

filename = f"stratified_interpretation_{days_to_consider}Days_{merge_type}.txt"
with open(os.path.join(current_dir, filename), 'w') as f:
    f.write(interpretation)






























# Anomalous 1 day
merge_type = "anomalous"
days_to_consider = 1
csv_to_read = os.path.join(current_dir, f"stratified_model_performance_{merge_type}_{days_to_consider}Days.csv")
if not os.path.exists(csv_to_read):
    print(f"Nessun file corrispondente al percorso selezionato: {csv_to_read}")
    exit()
df_perf = pd.read_csv(csv_to_read)

interpretation = f"""CLINICAL INTERPRETATION GUIDE ({merge_type.upper()} MERGING, {days_to_consider} DAYS)

1. Overall Performance:
- Best balanced accuracy: 0.65 (ConvTran)
- Best F1 score: 0.63 (ROCKET)

2. Key Patterns:
- Non-viable detection: ConvTran (deg BA=0.97) outperforms others
- 2PN limitations: All models <0.58 BA
- Anomalous group: ROCKET leads (BA=0.67, F1=0.52)

3. Model Comparison:
- ConvTran: Best deg detection (BA=0.97)
- ROCKET: Most reliable for anomalous class
- LSTMFCN: Avoid for deg cases (BA=0.88)

4. Critical Red Flags:
- 0PN F1=0.0 despite high BA - indicates class imbalance
- Anomalous group F1<0.52 across models

5. Clinical Guidelines:
- Use ROCKET for initial anomaly screening
- ConvTran for final non-viable confirmation
- Mandatory review for:
  - All 0PN predictions
  - Any anomalous classifications
  - LSTMFCN deg assessments

Note: F1 Score <0.4 indicates high uncertainty, >0.7 indicates clinical reliability
"""

filename = f"stratified_interpretation_{days_to_consider}Days_{merge_type}.txt"
with open(os.path.join(current_dir, filename), 'w') as f:
    f.write(interpretation)

































# Anomalous 3 days
merge_type = "anomalous"
days_to_consider = 3
csv_to_read = os.path.join(current_dir, f"stratified_model_performance_{merge_type}_{days_to_consider}Days.csv")
if not os.path.exists(csv_to_read):
    print(f"Nessun file corrispondente al percorso selezionato: {csv_to_read}")
    exit()
df_perf = pd.read_csv(csv_to_read)

interpretation = f"""CLINICAL INTERPRETATION GUIDE ({merge_type.upper()} MERGING, {days_to_consider} DAYS)

1. Overall Performance:
- Best balanced accuracy: 0.76 (ConvTran)
- Best F1 score: 0.75 (ROCKET)

2. Breakthrough Findings:
- Perfect non-viable detection: All models (BA=1.0)
- Anomalous group: ConvTran (BA=0.84, F1=0.75) leads
- 3PN reliability: ROCKET (F1=0.84) > ConvTran (F1=0.71)

3. Model Specialization:
- ConvTran: Gold standard for anomalies
- ROCKET: Best 3PN detection
- LSTMFCN: Balanced 2PN performance (F1=0.75)

4. Critical Insights:
- 0PN paradox: Perfect BA but F1=0.0 (no positive samples)
- 2.1PN excellence: ConvTran (F1=0.86) vs ROCKET (F1=0.73)

5. Clinical Protocol:
- Automated approval:
  - ConvTran anomalous classifications
  - Any model non-viable predictions
- Required human oversight:
  - 0PN/deg cases (despite perfect BA)
  - LSTMFCN anomalous assessments
  - Borderline 2.1PN predictions

Note: F1 Score <0.4 indicates high uncertainty, >0.7 indicates clinical reliability
"""

filename = f"stratified_interpretation_{days_to_consider}Days_{merge_type}.txt"
with open(os.path.join(current_dir, filename), 'w') as f:
    f.write(interpretation)

