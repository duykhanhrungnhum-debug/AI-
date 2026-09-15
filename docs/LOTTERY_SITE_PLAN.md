# Vietnam Lottery Website — Implementation Plan

## Goal
Build a Vietnam lottery-results website that presents clean lottery results for Miền Bắc, Miền Trung, Miền Nam and Vietlott where permitted.

## Data-ingestion rule
- Fetch only lottery-result data from selected public source pages/endpoints.
- Parse structured result fields (draw date, province, prize tier, numbers, draw status/time).
- Ignore advertisements, banners, tracking pixels, unrelated recommendations, and other non-result content.
- Keep source URL and retrieval timestamp for provenance.
- Validate the parsed result schema before publishing.
- Never claim a result is live/verified unless the fetch and parser actually succeeded.

## Proposed architecture

1. `collector/` — source-specific HTTP clients/fetchers.
2. `parser/` — source-specific parsers that extract only result tables/data.
3. `normalizer/` — normalize provinces, prize tiers, dates and number formats.
4. `validator/` — schema, completeness, duplicate and consistency checks.
5. `storage/` — persist raw source metadata plus normalized results.
6. `api/` — serve latest and historical results.
7. `web/` — responsive Vietnamese UI with live-update status.
8. `scheduler/` — polling around published draw times with backoff/rate limits.
9. `tests/` — parser fixtures, validation tests and source smoke tests.

## Source strategy

Initial candidate sources will be evaluated for reliability, accessibility, terms/robots constraints, and stable result markup before being used in production. A source adapter must be replaceable so the site is not dependent on one provider.

## Verification requirements

A release is not considered complete until:
- source fetch succeeds;
- parser extracts the expected prize tiers;
- normalized data passes validation;
- stored data can be reproduced from the captured source evidence;
- UI/API shows the same verified values;
- automated tests pass.

## Important boundary

This project displays lottery results and does not implement lottery-ticket sales, payment processing, or instructions intended to facilitate gambling activity.
