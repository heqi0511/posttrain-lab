# Parse Failure Examples

Input: `runs/rlvr/gsm8k_frontier_scout_thinking_false/sample_rollouts_for_review.jsonl`
Parse failures in file: `94` / `160`

## Recommendations

- increase max_new_tokens
- strengthen prompt
- run format SFT warmup
- do not run GRPO smoke until parse failures are reduced

## Category Summary

- `no_boxed_answer`: 35 (37.23% of parse failures)
- `truncated_before_final`: 36 (38.30% of parse failures)
- `final_answer_unboxed`: 15 (15.96% of parse failures)
- `malformed_boxed`: 3 (3.19% of parse failures)
- `multiple_conflicting_boxed_answers`: 0 (0.00% of parse failures)
- `non_numeric_boxed_answer`: 0 (0.00% of parse failures)
- `parser_too_strict`: 5 (5.32% of parse failures)
- `thinking_mode_interference`: 0 (0.00% of parse failures)
- `other`: 0 (0.00% of parse failures)

## no_boxed_answer

### gsm8k-train-000005 sample 4

- failure_reason: `no_boxed_answer`
- answer: `35`
- boxed_answers: ``
- estimated_token_count: `114`
- reached_max_new_tokens: `False`

```text
To solve the problem, we need to find the total number of flowers in Mark's garden.  1. **Yellow flowers**: 10   2. **Purple flowers**: 80% more than yellow = $ 10 + 0.8 \times 10 = 18 $   3. **Green flowers**: 25% as many as yellow and purple combined.      - Yellow + Purple = $ 10 + 18 = 28 $      - Green = $ 0.25 \times 28 = 7 $  Now, total flowers = Yellow +
```

### gsm8k-train-000008 sample 1

- failure_reason: `no_boxed_answer`
- answer: `41`
- boxed_answers: ``
- estimated_token_count: `88`
- reached_max_new_tokens: `False`

```text
We start with a budget of $200 and end with $16 left, so the total amount spent is:  $$ 200 - 16 = 184 $$  Now we add up the known expenses:  $$ 30 + 46 + 38 + 11 + 18 = 143 $$  Let $ x $ be the cost of the shoes. Then:  $$ 143 + x = 184 $$  Solving for $ x $:  $$ x = 184 - 143 =
```

### gsm8k-train-000008 sample 2

- failure_reason: `no_boxed_answer`
- answer: `41`
- boxed_answers: ``
- estimated_token_count: `100`
- reached_max_new_tokens: `False`

```text
We start with a total budget of $200. Alexis has $16 left, so she spent:  $$ 200 - 16 = 184 $$  Now we subtract the costs of the items she bought:  - Button-up shirt: $30   - Suit pants: $46   - Suit coat: $38   - Socks: $11   - Belt: $18    Adding these:  $$ 30 + 46 + 38 + 11 + 18 = 143 $$  So the cost of the shoes
```

### gsm8k-train-000008 sample 3

- failure_reason: `no_boxed_answer`
- answer: `41`
- boxed_answers: ``
- estimated_token_count: `112`
- reached_max_new_tokens: `False`

```text
Let's calculate how much Alexis spent on the items she bought, excluding the shoes.  - Button-up shirt: $30   - Suit pants: $46   - Suit coat: $38   - Socks: $11   - Belt: $18    Total spent on these items:   $30 + $46 + $38 + $11 + $18 = $143    Alexis started with a budget of $200 and has $16 left.   So, the amount she spent on the shoes is:   $200 - $16
```

### gsm8k-train-000008 sample 5

- failure_reason: `no_boxed_answer`
- answer: `41`
- boxed_answers: ``
- estimated_token_count: `112`
- reached_max_new_tokens: `False`

```text
To find out how much Alexis paid for the shoes, we start by calculating the total amount she spent on all the other items:  - Button-up shirt: $30   - Suit pants: $46   - Suit coat: $38   - Socks: $11   - Belt: $18    Total spent on these items:   $30 + $46 + $38 + $11 + $18 = $143  Alexis had a budget of $200 and has $16 left, so she spent:   $200 - $16
```

### gsm8k-train-000009 sample 0

- failure_reason: `no_boxed_answer`
- answer: `990`
- boxed_answers: ``
- estimated_token_count: `105`
- reached_max_new_tokens: `False`

```text
Tina earns $18.00 per hour. For overtime, she is paid at 1.5 times her regular wage.  She works 10 hours per day for 5 days:  - Regular hours per day: 8 hours   - Overtime hours per day: 10 - 8 = 2 hours   - Overtime pay rate: $18.00 × 1.5 = $27.00 per hour    Total earnings:  - Regular pay: $18.00 × 8 × 5 = $720.00   - Overtime pay
```

### gsm8k-train-000009 sample 1

- failure_reason: `no_boxed_answer`
- answer: `990`
- boxed_answers: ``
- estimated_token_count: `107`
- reached_max_new_tokens: `False`

```text
Tina's regular pay is $18.00 per hour. For overtime, she earns 1.5 times her regular hourly wage:  $$ 18 + \frac{1}{2} \times 18 = 18 + 9 = 27 \text{ dollars per hour} $$  She works 10 hours per day for 5 days, so:  - 10 hours per day × 5 days = 50 hours total - 8 hours are regular pay, so 50 - 8 = 42 hours of overtime  Total earnings:  $$
```

### gsm8k-train-000009 sample 2

- failure_reason: `no_boxed_answer`
- answer: `990`
- boxed_answers: ``
- estimated_token_count: `109`
- reached_max_new_tokens: `False`

```text
Tina earns $18.00 per hour. For hours over 8, she gets overtime at 1.5 times her regular wage.  She works 10 hours per day for 5 days:   Total hours = $10 \times 5 = 50$ hours.  For the first 8 hours each day, she earns $18.00/hour.   For the remaining 2 hours each day, she earns $18.00 \times 1.5 = 27.00/hour.  Daily earnings:   $ (8 \times 18
```

### gsm8k-train-000009 sample 3

- failure_reason: `no_boxed_answer`
- answer: `990`
- boxed_answers: ``
- estimated_token_count: `83`
- reached_max_new_tokens: `False`

```text
Tina's regular pay is $18.00 per hour.   Overtime pay is $18.00 + ($18.00 × 1/2) = $18.00 + $9.00 = $27.00 per hour.  She works 10 hours per day for 5 days:   - 8 hours are regular pay: $18.00 × 8 = $144.00   - 2 hours are overtime: $27.00 × 2 = $54.00    Total for
```

### gsm8k-train-000009 sample 4

- failure_reason: `no_boxed_answer`
- answer: `990`
- boxed_answers: ``
- estimated_token_count: `115`
- reached_max_new_tokens: `False`

```text
Tina's regular pay is $18.00 per hour. For overtime, she is paid at 1.5 times her regular wage:  $$ 18 + \frac{1}{2} \times 18 = 18 + 9 = 27 \text{ dollars per hour} $$  She works 10 hours per day for 5 days. Of those 10 hours, the first 8 are regular pay, and the remaining 2 hours are overtime.  Total regular hours: $8 \times 5 = 40$   Total overtime hours: $2 \
```


## truncated_before_final

### gsm8k-train-000004 sample 0

- failure_reason: `no_boxed_answer`
- answer: `624`
- boxed_answers: ``
- estimated_token_count: `125`
- reached_max_new_tokens: `True`

```text
To find out how many pages James writes in a year, we can break it down step by step:  1. James writes a **3-page letter** to **2 different friends**. 2. He does this **twice a week**. 3. There are **52 weeks** in a year.  So, the total number of pages he writes in a year is:  $$ 3 \text{ pages} \times 2 \text{ friends} \times 2 \text{ times/week} \times 52 \text{ weeks} = 624 \text{ pages} $$  $$
```

### gsm8k-train-000004 sample 1

- failure_reason: `no_boxed_answer`
- answer: `624`
- boxed_answers: ``
- estimated_token_count: `127`
- reached_max_new_tokens: `True`

```text
To find how many pages James writes in a year, we break it down step by step:  - He writes a **3-page letter** to **2 different friends**. - He does this **twice a week**. - There are **52 weeks** in a year.  So, the total number of pages per week is:  $$ 3 \text{ pages} \times 2 \text{ friends} \times 2 \text{ times/week} = 12 \text{ pages/week} $$  Then, for the whole year:  $$ 12 \text{ pages/week}
```

### gsm8k-train-000004 sample 2

- failure_reason: `no_boxed_answer`
- answer: `624`
- boxed_answers: ``
- estimated_token_count: `125`
- reached_max_new_tokens: `True`

```text
To find how many pages James writes in a year, we break it down step by step:  - He writes a **3-page letter** to **2 different friends**. - He does this **twice a week**. - There are **52 weeks** in a year.  So, the total number of pages per week is: $$ 3 \text{ pages/letter} \times 2 \text{ friends} \times 2 \text{ times/week} = 12 \text{ pages/week} $$  Then, for a year: $$ 12 \text{ pages/week
```

### gsm8k-train-000004 sample 5

- failure_reason: `no_boxed_answer`
- answer: `624`
- boxed_answers: ``
- estimated_token_count: `127`
- reached_max_new_tokens: `True`

```text
To find how many pages James writes in a year, we can break it down step by step:  - He writes a **3-page letter** to **2 different friends**. - He does this **twice a week**.  So, per week: - Pages per friend: 3 - Number of friends: 2 - Total pages per week: $ 3 \times 2 = 6 $ pages  He does this **twice a week**, so: - Total pages per week: $ 6 \times 2 = 12 $ pages  There are **52 weeks** in a year
```

### gsm8k-train-000007 sample 0

- failure_reason: `no_boxed_answer`
- answer: `16`
- boxed_answers: ``
- estimated_token_count: `131`
- reached_max_new_tokens: `True`

```text
Let's break down the problem step by step.  1. **Start with a box on a scale.** 2. **Add enough jelly beans to bring the weight to 2 pounds.**      - Weight = 2 pounds.  3. **Add enough brownies to cause the weight to triple.**      - Triple of 2 pounds = $ 2 \times 3 = 6 $ pounds.  4. **Add another 2 pounds of jelly beans.**      - New weight = $ 6 + 2 = 8 $ pounds.  5. **Add enough gummy worms to double the weight
```

### gsm8k-train-000007 sample 1

- failure_reason: `no_boxed_answer`
- answer: `16`
- boxed_answers: ``
- estimated_token_count: `127`
- reached_max_new_tokens: `True`

```text
Let's break down the problem step by step:  1. **Start with a box on a scale.**      - Ken adds **2 pounds of jelly beans**, so the weight is **2 pounds**.  2. **He adds enough brownies to **triple** the weight.**      - Tripling 2 pounds gives:        $ 2 \times 3 = 6 $ pounds.  3. **He adds another **2 pounds of jelly beans**.**      - Adding 2 pounds to 6 pounds gives:        $ 6 + 2 = 8 $ pounds.  4
```

### gsm8k-train-000007 sample 3

- failure_reason: `no_boxed_answer`
- answer: `16`
- boxed_answers: ``
- estimated_token_count: `134`
- reached_max_new_tokens: `True`

```text
Let's break down the problem step by step:  1. Ken starts with a box on a scale. 2. He adds **2 pounds of jelly beans**, so the weight is **2 pounds**. 3. Then he adds **enough brownies to triple** the weight.      Triple of 2 pounds is $ 2 \times 3 = 6 $ pounds. 4. Next, he adds **another 2 pounds of jelly beans**, so the weight becomes $ 6 + 2 = 8 $ pounds. 5. Finally, he adds **enough gummy worms to double** the weight.
```

### gsm8k-train-000007 sample 4

- failure_reason: `no_boxed_answer`
- answer: `16`
- boxed_answers: ``
- estimated_token_count: `127`
- reached_max_new_tokens: `True`

```text
Let's track the weight step by step.  1. Ken starts with a box on a scale. 2. He adds enough jelly beans to bring the weight to **2 pounds**. 3. He adds enough brownies to **tripple** the weight:      $ 2 \times 3 = 6 $ pounds. 4. He adds another **2 pounds of jelly beans**:      $ 6 + 2 = 8 $ pounds. 5. He adds enough gummy worms to **double** the weight:      $ 8 \times 2 = 16 $ pounds.  Final weight:
```

### gsm8k-train-000007 sample 5

- failure_reason: `no_boxed_answer`
- answer: `16`
- boxed_answers: ``
- estimated_token_count: `151`
- reached_max_new_tokens: `True`

```text
We are given a series of steps involving adding different items to a box and tracking the weight changes. Let's follow each step carefully.  1. **Start with a box on a scale.**      - Initial weight: 0 pounds (assumed, as no weight is given initially).  2. **Pour enough jelly beans to bring the weight to 2 pounds.**      - Weight = 2 pounds.  3. **Add enough brownies to cause the weight to triple.**      - New weight = 2 × 3 = 6 pounds.  4. **Add another 2 pounds of jelly beans.
```

### gsm8k-train-000007 sample 6

- failure_reason: `no_boxed_answer`
- answer: `16`
- boxed_answers: ``
- estimated_token_count: `124`
- reached_max_new_tokens: `True`

```text
Let's break down the problem step by step:  1. **Start with a box on a scale.** 2. **Add jelly beans to bring the weight to 2 pounds.**      - Weight = 2 pounds. 3. **Add brownies to triple the weight.**      - New weight = 2 × 3 = 6 pounds. 4. **Add another 2 pounds of jelly beans.**      - New weight = 6 + 2 = 8 pounds. 5. **Add gummy worms to double the weight.**      - Final weight = 8 × 2
```


## final_answer_unboxed

### gsm8k-train-000005 sample 1

- failure_reason: `no_boxed_answer`
- answer: `35`
- boxed_answers: ``
- estimated_token_count: `99`
- reached_max_new_tokens: `False`

```text
To solve the problem, we break it down step by step:  1. **Yellow flowers**: 10 2. **Purple flowers**: 80% more than yellow      $ 10 + 0.8 \times 10 = 10 + 8 = 18 $      So, 18 purple flowers.  3. **Green flowers**: 25% as many as yellow and purple combined      $ 10 + 18 = 28 $      $ 25\% \text{ of } 28 = 0.25
```

### gsm8k-train-000005 sample 2

- failure_reason: `no_boxed_answer`
- answer: `35`
- boxed_answers: ``
- estimated_token_count: `100`
- reached_max_new_tokens: `False`

```text
To solve the problem, we break it down step by step:  - **Yellow flowers**: 10 - **Purple flowers**: 80% more than yellow     $$   10 + 0.8 \times 10 = 10 + 8 = 18   $$ - **Green flowers**: 25% as many as yellow and purple combined     $$   \text{Yellow + Purple} = 10 + 18 = 28 \\   \text{Green} = 0.25 \times 28 = 7
```

### gsm8k-train-000005 sample 3

- failure_reason: `no_boxed_answer`
- answer: `35`
- boxed_answers: ``
- estimated_token_count: `108`
- reached_max_new_tokens: `False`

```text
To solve the problem, we'll calculate the number of flowers for each color and then add them up.  - Yellow flowers: 10   - Purple flowers: 80% more than yellow = $10 + 0.8 \times 10 = 18$   - Green flowers: 25% as many as yellow and purple combined = $0.25 \times (10 + 18) = 0.25 \times 28 = 7$  Now, add them all together:   $10 + 18 + 7 = 35
```

### gsm8k-train-000005 sample 6

- failure_reason: `no_boxed_answer`
- answer: `35`
- boxed_answers: ``
- estimated_token_count: `99`
- reached_max_new_tokens: `False`

```text
To solve the problem, we break it down step by step:  - **Yellow flowers**: 10 - **Purple flowers**: 80% more than yellow     $ 10 + (0.8 \times 10) = 10 + 8 = 18 $  - **Green flowers**: 25% as many as yellow and purple combined     $ \text{Yellow} + \text{Purple} = 10 + 18 = 28 $     $ 25\% \text{ of } 28 = 0.25
```

### gsm8k-train-000005 sample 7

- failure_reason: `no_boxed_answer`
- answer: `35`
- boxed_answers: ``
- estimated_token_count: `119`
- reached_max_new_tokens: `False`

```text
To solve the problem, we need to find the total number of flowers in Mark's garden by adding up the number of yellow, purple, and green flowers.  1. **Yellow flowers:** 10   2. **Purple flowers:** 80% more than yellow      $$    10 + 0.8 \times 10 = 10 + 8 = 18    $$  3. **Green flowers:** 25% as many as yellow and purple combined      $$    \text{Yellow + Purple} = 10 + 18 = 2
```

### gsm8k-train-000008 sample 6

- failure_reason: `no_boxed_answer`
- answer: `41`
- boxed_answers: ``
- estimated_token_count: `117`
- reached_max_new_tokens: `False`

```text
To find out how much Alexis paid for the shoes, we start by calculating the total amount she spent on all the other items:  - Button-up shirt: $30   - Suit pants: $46   - Suit coat: $38   - Socks: $11   - Belt: $18    Total spent on these items:   $30 + $46 + $38 + $11 + $18 = $143    Alexis had a budget of $200 and has $16 left. So the amount she spent in total is:   $20
```

### gsm8k-train-000009 sample 7

- failure_reason: `no_boxed_answer`
- answer: `990`
- boxed_answers: ``
- estimated_token_count: `96`
- reached_max_new_tokens: `False`

```text
Tina's hourly wage is $18.00.   Overtime is paid at 1.5 times the hourly wage, so $18.00 × 1.5 = $27.00 per hour for overtime.  She works 10 hours per day for 5 days:   - 8 hours are regular pay.   - 2 hours are overtime.  Daily earnings:   - Regular: 8 × $18.00 = $144.00   - Overtime: 2 × $27.00 = $54.00
```

### gsm8k-train-000011 sample 5

- failure_reason: `no_boxed_answer`
- answer: `5`
- boxed_answers: ``
- estimated_token_count: `108`
- reached_max_new_tokens: `False`

```text
Let's break down the problem step by step.  1. **Total cost of shoes**: $95   2. **Allowance for 3 months**: $3 \times 5 = 15$   3. **Money from mowing lawns**: $15 \times 4 = 60$   4. **Let $ x $ be the number of driveways he shoveled**.      - Money from shoveling: $7x$   5. **Total money before buying shoes**:      $$    15 + 60 + 7x = 75
```

### gsm8k-train-000011 sample 6

- failure_reason: `no_boxed_answer`
- answer: `5`
- boxed_answers: ``
- estimated_token_count: `119`
- reached_max_new_tokens: `False`

```text
We are given:  - Shoes cost: $95   - Allowance: $5/month for 3 months → $15   - Mows lawns: $15 per lawn   - Shovels driveways: $7 per driveway   - After buying shoes, he has $15 left   - Mows 4 lawns    We need to find how many driveways he shoveled.  Let $ x $ be the number of driveways he shoveled.  Total money earned:  - Allowance: $15   - From mowing lawns: $15 × 4 = $60
```

### gsm8k-train-000014 sample 1

- failure_reason: `no_boxed_answer`
- answer: `5`
- boxed_answers: ``
- estimated_token_count: `118`
- reached_max_new_tokens: `False`

```text
To find how many hours it will take Joy to read 120 pages, we start by determining her reading rate:  - Joy reads 8 pages in 20 minutes. - So, her rate is $ \frac{8 \text{ pages}}{20 \text{ minutes}} = \frac{2 \text{ pages}}{5 \text{ minutes}} $.  Now, find how many minutes it takes to read 120 pages:  $$ \text{Time} = \frac{120 \text{ pages}}{2 \text{ pages/5 minutes}} = 1
```


## malformed_boxed

### gsm8k-train-000004 sample 3

- failure_reason: `malformed_boxed_answer`
- answer: `624`
- boxed_answers: ``
- estimated_token_count: `110`
- reached_max_new_tokens: `False`

```text
To find how many pages James writes in a year:  - He writes a 3-page letter to 2 friends **twice a week**. - So, per week: $ 3 \text{ pages} \times 2 \text{ friends} \times 2 \text{ times} = 12 \text{ pages} $. - There are 52 weeks in a year. - Total pages per year: $ 12 \text{ pages/week} \times 52 \text{ weeks} = 624 \text{ pages} $.  $$ \boxed{
```

### gsm8k-train-000016 sample 0

- failure_reason: `malformed_boxed_answer`
- answer: `800`
- boxed_answers: ``
- estimated_token_count: `91`
- reached_max_new_tokens: `False`

```text
First, find Mike's share of the profit. The ratio is 2:5, and Johnson's share is $2500, which corresponds to 5 parts.  $$ 1 \text{ part} = \frac{2500}{5} = 500 $$  $$ \text{Mike's share} = 2 \times 500 = 1000 $$  After buying a shirt for $200, Mike has:  $$ 1000 - 200 = 800 $$  $$ \boxed{8
```

### gsm8k-train-000019 sample 0

- failure_reason: `malformed_boxed_answer`
- answer: `16`
- boxed_answers: ``
- estimated_token_count: `107`
- reached_max_new_tokens: `False`

```text
To find the total time Tim spends biking in a week, we calculate the total distance he bikes and divide by his speed.  - **Weekdays**: 5 days × 2 round trips × 20 miles = 5 × 2 × 20 = 200 miles - **Weekend**: 200 miles  **Total distance**: 200 + 200 = 400 miles  **Speed**: 25 mph  **Total time**: $ \frac{400}{25} = 16 $ hours  $$ \boxed{16
```


## parser_too_strict

### gsm8k-train-000001 sample 0

- failure_reason: `boxed_not_final_only`
- answer: `10`
- boxed_answers: `10`
- estimated_token_count: `101`
- reached_max_new_tokens: `False`

```text
To find out how much Weng earned, we need to calculate her earnings for 50 minutes at a rate of $12 per hour.  First, convert 50 minutes to hours: $$ 50 \text{ minutes} = \frac{50}{60} = \frac{5}{6} \text{ hours} $$  Now, multiply by her hourly rate: $$ \frac{5}{6} \times 12 = 10 $$  So, Weng earned $\boxed{10}$ dollars.
```

### gsm8k-train-000001 sample 4

- failure_reason: `boxed_not_final_only`
- answer: `10`
- boxed_answers: `10`
- estimated_token_count: `79`
- reached_max_new_tokens: `False`

```text
To find out how much Weng earned, we first convert 50 minutes to hours:  $$ 50 \text{ minutes} = \frac{50}{60} = \frac{5}{6} \text{ hours} $$  Then, multiply by her hourly rate:  $$ \frac{5}{6} \times 12 = 10 $$  So, Weng earned $\boxed{10}$ dollars.
```

### gsm8k-train-000001 sample 6

- failure_reason: `boxed_not_final_only`
- answer: `10`
- boxed_answers: `10`
- estimated_token_count: `78`
- reached_max_new_tokens: `False`

```text
To find out how much Weng earned, we first convert 50 minutes to hours:  $$ 50 \text{ minutes} = \frac{50}{60} = \frac{5}{6} \text{ hours} $$  Then multiply by her hourly rate:  $$ \frac{5}{6} \times 12 = 10 $$  So, Weng earned $\boxed{10}$ dollars.
```

### gsm8k-train-000001 sample 7

- failure_reason: `boxed_not_final_only`
- answer: `10`
- boxed_answers: `10`
- estimated_token_count: `83`
- reached_max_new_tokens: `False`

```text
To find out how much Weng earned, we first convert 50 minutes to hours:  $$ 50 \text{ minutes} = \frac{50}{60} \text{ hours} = \frac{5}{6} \text{ hours} $$  Then multiply by her hourly rate:  $$ \frac{5}{6} \times 12 = 10 $$  So, Weng earned $\boxed{10}$ dollars.
```

### gsm8k-train-000003 sample 7

- failure_reason: `boxed_not_final_only`
- answer: `42`
- boxed_answers: `42`
- estimated_token_count: `101`
- reached_max_new_tokens: `False`

```text
To solve the problem:  1. Total pages in the book: 120   2. Pages read yesterday: 12   3. Pages read today: $ 2 \times 12 = 24 $   4. Total pages read so far: $ 12 + 24 = 36 $   5. Remaining pages: $ 120 - 36 = 84 $   6. Half of the remaining pages: $ \frac{84}{2} = 42 $  So, Julie should read $\boxed{42}$ pages tomorrow.
```
