# Your Colleague Food Prediction Model 
A small and slightly unnecessary ML project ('cause I got so bored on a holiday!) where I try to predict what food my colleague will bring to the office each day. The goal is not really accuracy (humans are chaotic anyway :)) but to explore how everyday habits can be treated as a tiny behavioral prediction problem. Mostly built for fun, pattern spotting, and practicing a clean ML workflow on a small real-world style dataset.

---

## What this project does

The model looks at simple daily factors like:

- Day of the week  
- Weather  
- Temperature  
- What they brought yesterday  

and tries to guess today's lunch category.

Example output:
```
Predicted food: leftovers

Probabilities:
leftovers: 95.33%
soup: 4.00%
pasta: 0.67%
```

So instead of just guessing one answer, it also shows how confident it is.

---

## Why I built this

Mostly curiosity.

I wanted to see:

- Can daily routines be predictable?
- How much pattern exists in small behavioral data?
- How little data you actually need to train something useful?
- How to structure a tiny end-to-end ML project cleanly

Also it's just fun to treat office lunch like a data science problem! (is it though?)

---

## Project structure

```
your-colleague-food-prediction-model/

data/
│ office_food.csv

train.py
predict.py
food_model.pkl
label_encoder.pkl
```

---

## How it works (simple version)

### Training phase
- Load dataset
- Encode categorical data
- Train Random Forest classifier
- Evaluate accuracy
- Save model

### Prediction phase
- Load trained model
- Input today's conditions
- Predict food category
- Show probability ranking

Basically a tiny real ML pipeline.

---

## How to run it

Install dependencies:
```
pip install pandas scikit-learn joblib
```

Train model:
```
train.py
```

Predict lunch:
```
predict.py
```


---

## Dataset notes

This uses a small synthetic dataset for now. Real data would obviously be much messier and less predictable.

Class balance matters a lot here. Rare food types (like pasta or rice dishes) need more examples for the model to learn anything meaningful.
But as a student having pasta or rice is def NOT rare at all :D 

---

## Things I may add later

Some ideas I'm considering:

- Feature importance visualization
- Prediction history tracking
- Accuracy over time
- Simple dashboard
- Behavior pattern detection
- Maybe a small Streamlit interface

Mostly just evolving this as a playground for ML ideas.

---

## Reality check

This will never perfectly predict what someone eats. Humans are not deterministic systems (sadly for ML).

But it's a nice small experiment in:

- Feature engineering
- Classification
- Probability prediction
- ML workflow structure

And honestly it's a fun excuse to overengineer something trivial.

---

## Future question

Can ML predict office coffee habits too? STAY TUNED! :D
