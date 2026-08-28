-- Load outputs/wr_rankings.csv into a SQLite table named receivers.

-- Biggest model-vs-market sleepers.
SELECT player, draft_year, draft_pick, model_rank, market_rank, sleeper_score, model_grade
FROM receivers
ORDER BY sleeper_score DESC
LIMIT 25;

-- Productive young prospects (available after adding college_wr.csv).
SELECT player, draft_year, age, yards_per_game, yard_share, model_grade
FROM receivers
WHERE age <= 21.5
ORDER BY model_grade DESC
LIMIT 20;

-- Athletic + production intersections.
SELECT player, speed_score, yards_per_game, yard_share, sleeper_score
FROM receivers
WHERE speed_score IS NOT NULL
ORDER BY sleeper_score DESC
LIMIT 20;

-- Potential reaches: market liked them more than the model.
SELECT player, draft_year, draft_pick, model_rank, market_rank, sleeper_score
FROM receivers
ORDER BY sleeper_score ASC
LIMIT 20;
