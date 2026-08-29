# Project Falcon — status, week 34

Owner: Mihaly Fodor. Tech lead: Anna Kovacs. Sponsor: Tomas Bell.

## Status

Falcon is **amber**. The migration of the billing service to the new event bus is code
complete, but the load test on 2026-08-21 showed a p99 latency of 940 ms against a 400 ms
target, so the go-live moved from 2026-09-01 to 2026-09-15.

## Decisions

- We will ship Falcon behind a feature flag and ramp traffic 5% / 25% / 100% over two weeks.
- Anna owns the rollback runbook; it must be signed off before the ramp starts.

## Risks

- The event bus consumer group rebalances under load; mitigation is a fixed partition
  assignment, being tested this week.
- Tomas is on leave from 2026-09-05, so the go/no-go call moves to the 3rd.

## Next steps

1. Fix the rebalance storm (Anna, by 2026-08-29).
2. Re-run the load test and publish the numbers.
3. Go/no-go on 2026-09-03.
