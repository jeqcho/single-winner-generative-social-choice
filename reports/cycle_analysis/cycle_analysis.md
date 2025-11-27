# Preference Cycle Analysis

Analysis of preference cycles found in abortion pairwise preference data.

## Summary

All three persona files contain preference cycles, meaning a Directed Acyclic Graph (DAG) cannot be formed. This indicates fundamental inconsistencies in the pairwise preferences.

## pairwise_100_abortion_persona_1.csv

### Persona

```
age: 29
sex: Female
race: Two or More Races
ancestry: Hispanic
household language: Spanish
education: 1 or more years of college credit, no degree
employment status: Civilian employed, at work
class of worker: Employee of a private not-for-profit, tax-exempt, or charitable organization
industry category: MED-General Medical And Surgical Hospitals, And Specialty (Except Psychiatric And Substance Abuse) Hospitals
occupation category: CMS-Other Community and Social Service Specialists
detailed job description: Provides social services to patients and their families in a hospital setting, including counseling and referral services.
income: 62000.0
marital status: Married
household type: Married couple household with children of the householder less than 18
family presence and age: With related children 5 to 17 years only
place of birth: Texas/TX
citizenship: Born in the United States
veteran status: Non-veteran
disability: No disability
health insurance: With health insurance coverage
fertility: Has children
hearing difficulty: No hearing difficulty
vision difficulty: No vision difficulty
cognitive difficulty: No cognitive difficulty
ability to speak english: Very well
big five scores: Openness: Low, Conscientiousness: High, Extraversion: Average, Agreeableness: Average, Neuroticism: Extremely High
defining quirks: Has a deep passion for music and entertainment, often attending live concerts and playing the guitar in her free time.
mannerisms: Tends to be thoughtful and analytical in her approach to situations, often taking time to consider all options before making a decision.
personal time: Spends her personal time with her family, exploring the outdoors, practicing her religion, and playing the guitar.
lifestyle: Leads a busy and fulfilling lifestyle, balancing work, family, and personal interests.
ideology: Believes strongly in social justice and equality, often advocating for these issues in her professional and personal life.
political views: Liberal
religion: Muslim

```

**Cycle:** 13 → 7 → 90 → 13

### Statement 13

I think laws about abortion should try to protect babies because I believe every life is special and made by God. But I also think they should care about the mom and make sure she is safe and has help, because God wants us to be kind and loving to everyone.

### Statement 7

As a Christian and a conservative, I believe our laws on abortion should prioritize protecting unborn life while still recognizing rare, tragic situations where the mother’s life is at serious risk. I think decisions about abortion should be guided by moral principles rooted in faith, respect for life, and personal responsibility, rather than treating it as just another medical choice. I support policies that encourage adoption, provide support for women in crisis pregnancies, and promote education about alternatives to abortion.

### Statement 90

Laws concerning abortion should be guided by the belief that life is sacred from conception and deserves legal protection, while still recognizing the gravity and complexity of situations where the mother’s life is at risk. As a conservative Protestant and a father whose kids are now grown, I see this as both a moral and a community issue: our laws should encourage responsibility, support mothers and families in crisis, and reflect a respect for the unborn that aligns with biblical principles.

### Cycle Explanation

- Statement 13 is preferred to Statement 7
- Statement 7 is preferred to Statement 90
- Statement 90 is preferred to Statement 13

---

## pairwise_100_abortion_persona_40.csv

### Persona

```
age: 16
sex: Female
race: White alone
ancestry: German
household language: English only
education: Grade 10
employment status: Civilian employed, at work
class of worker: Employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions
industry category: AGR-Animal Production And Aquaculture
occupation category: FFF-Other Agricultural Workers
detailed job description: Assists in the care and breeding of farm animals
income: 81000.0
marital status: Never married or under 15 years old
household type: Married couple household with children of the householder less than 18
family presence and age: With related children under 5 years and 5 to 17 years
place of birth: Colorado/CO
citizenship: Born in the United States
veteran status: Non-Veteran
disability: None
health insurance: With health insurance coverage
big five scores: Openness: Low, Conscientiousness: Extremely High, Extraversion: Average, Agreeableness: High, Neuroticism: Low
defining quirks: Has a deep connection with animals
mannerisms: Often seen wearing farm boots and a hat
personal time: Spends free time caring for animals or reading
lifestyle: Rural, outdoorsy, and active
ideology: Conservative
political views: Republican
religion: Jewish

```

**Cycle:** 0 → 51 → 12 → 0

### Statement 0

Laws concerning abortion should prioritize a woman’s right to make decisions about her own body, supported by sound medical science and respect for individual conscience rather than religious doctrine. As someone who works in public education and values both personal freedom and social responsibility, I believe policy should focus on comprehensive sex education, accessible contraception, and support services so fewer people face crisis pregnancies, while keeping the ultimate decision in the hands of the pregnant person and their healthcare provider.

### Statement 51

I think laws about abortion should try to protect the life of the baby because I believe every life is created by God and is important, but they should also care about the mom’s health and safety. I don’t think it’s an easy decision, so the laws should make sure people think very carefully, talk with doctors and family, and only allow it in very serious situations, like when the mother’s life is in danger.

### Statement 12

Laws concerning abortion should balance a woman’s personal freedom with our shared responsibility to protect life, especially as a pregnancy progresses. I believe decisions early in pregnancy should be left mostly to the woman, her doctor, and her conscience, while later-term abortions should be more carefully regulated with strong medical and ethical oversight.

### Cycle Explanation

- Statement 0 is preferred to Statement 51
- Statement 51 is preferred to Statement 12
- Statement 12 is preferred to Statement 0

---

## pairwise_100_abortion_persona_7.csv

### Persona

```
age: 68
sex: Male
race: White alone
ancestry: American
household language: English only
education: Associate's degree
employment status: Not in labor force
class of worker: Retired
industry category: Retired
occupation category: Retired Professional
detailed job description: Retired professional
income: 25000.0
marital status: Divorced
household type: Male householder, no spouse/partner present, living alone
family presence and age: No family
place of birth: New York/NY
citizenship: Born in the United States
veteran status: Non-Veteran
disability: With a disability
health insurance: With health insurance coverage
big five scores: Openness: Extremely High, Conscientiousness: Low, Extraversion: High, Agreeableness: Extremely Low, Neuroticism: Extremely High
defining quirks: Dedicated to maintaining a healthy lifestyle despite his disability
mannerisms: Strong New York accent and expressive hand gestures
personal time: Reading, Doing puzzles
lifestyle: Relaxed and leisurely
ideology: Liberal
political views: Democratic
religion: Religiously Unaffiliated

```

**Cycle:** 0 → 1 → 22 → 0

### Statement 0

Laws concerning abortion should prioritize a woman’s right to make decisions about her own body, supported by sound medical science and respect for individual conscience rather than religious doctrine. As someone who works in public education and values both personal freedom and social responsibility, I believe policy should focus on comprehensive sex education, accessible contraception, and support services so fewer people face crisis pregnancies, while keeping the ultimate decision in the hands of the pregnant person and their healthcare provider.

### Statement 1

I think laws about abortion should mainly protect a person’s right to make decisions about their own body, especially in tough situations like rape, health risks, or not being able to support a child. My faith matters to me, but I don’t think the government should force one religious view on everyone, because people have different beliefs and life circumstances. Instead, we should focus on making sure people have access to good sex education, birth control, and healthcare so they can prevent unwanted pregnancies and make informed choices.

### Statement 22

Laws about abortion should respect a woman’s right to make decisions about her own body, while ensuring access to safe, affordable medical care regardless of income. As a Democrat and a Catholic, I believe faith can shape personal choices, but it shouldn’t be imposed through government policy, especially on people who don’t share the same beliefs. Instead, laws should focus on protecting health, privacy, and dignity, and we should invest more in sex education, contraception, and social support so fewer people are put in desperate situations in the first place.

### Cycle Explanation

- Statement 0 is preferred to Statement 1
- Statement 1 is preferred to Statement 22
- Statement 22 is preferred to Statement 0

---

