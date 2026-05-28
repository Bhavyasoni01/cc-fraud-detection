
import pandas as pd

df = pd.read_csv('creditcard.csv')

fraud = df[df['Class'] == 1]           
legit = df[df['Class'] == 0].sample(   
    n=4920, random_state=42
)

sample = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
sample.to_csv('creditcard_sample.csv', index=False)
print(f"Sample created: {len(sample)} rows, {sample['Class'].sum()} fraud cases")