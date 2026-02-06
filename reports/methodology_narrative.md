# Experiment Methodology: Narrative Description

The experiment begins with two inputs: a pool of 815 adult personas drawn from the SynthLabsAI/PERSONA dataset, and 6 policy topics covering issues such as healthcare, policing, abortion, the environment, electoral reform, and trust in elections.

From the persona pool, 100 voters are sampled for each replication — either uniformly at random or from an ideology-based cluster. Independently, one statement per persona is generated using gpt-5-mini, producing a pool of candidate policy statements for each topic. 100 of these statements are then sampled to serve as the alternatives for the replication.

The 100 sampled voters and 100 sampled statements are fed into an iterative ranking process. Each voter ranks all 100 statements through multiple rounds of top-10/bottom-10 selection, carried out by gpt-5-mini simulating that voter's persona. This produces a full 100-by-100 preference matrix — one complete ranking per voter over all alternatives.

Two things happen with this preference matrix. First, the critical epsilon is precomputed for every one of the 100 alternatives against the full voter profile. Epsilon measures how close a statement is to the Proportional Veto Core: lower epsilon means better consensus, and epsilon of zero means the statement achieves optimal consensus. This precomputation allows instant evaluation of any method that selects from the existing 100 statements.

Second, the preference matrix is subsampled into smaller mini-reps of 20 voters by 20 alternatives. Each replication yields 4 such mini-reps. These mini-reps are the inputs to the winner selection methods.

Three categories of methods operate on each mini-rep. Traditional voting methods — Schulze, Borda, Instant Runoff Voting, Plurality, and Veto by Consumption — select a winner from the 20 subsampled alternatives using classical social choice algorithms, with no LLM involvement. LLM selection methods ask gpt-5-mini to choose the best consensus statement from the existing alternatives. LLM generation methods ask gpt-5-mini to write an entirely new consensus statement.

The evaluation of these methods follows two different paths depending on whether the method selected an existing statement or generated a new one. For traditional methods and LLM selection methods, the winning statement is one of the original 100 alternatives, so its epsilon is simply looked up from the precomputed values. For LLM generation methods, the newly created statement has no precomputed epsilon. Instead, the new statement is batched together with all other generated statements from the replication and ranked alongside the 100 originals through a fresh round of iterative ranking by all 100 voters. The new statement's position relative to the originals is extracted, and its epsilon is computed from that position.

Both evaluation paths converge at a final comparison, where all methods are compared by their epsilon values across topics, replications, and mini-reps. Lower epsilon indicates a method that more consistently finds or creates broadly acceptable consensus statements.
