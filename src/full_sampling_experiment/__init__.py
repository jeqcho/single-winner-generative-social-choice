"""
Full Sampling Experiment module.

Runs comprehensive voting experiments across all 13 topics with:
- Fixed K=20 voters, P=20 alternatives
- 10 replications x 5 samples = 50 samples per topic
- GPT-5-mini for preference ranking, GPT-5.2 for voting methods
- ChatGPT*** method (blind bridging statement generation)
- Likert experiment on all reps
- Mini variant for immigration topic with persona bridging statements
"""
