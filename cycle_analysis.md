# Preference Cycle Analysis

Analysis of preference cycles found in abortion pairwise preference data.

## Summary

All three persona files contain preference cycles, meaning a Directed Acyclic Graph (DAG) cannot be formed. This indicates fundamental inconsistencies in the pairwise preferences.

## pairwise_100_abortion_persona_1.csv

**Cycle:** 13 → 7 → 90 → 13

### Statement 13

{'persona': 'age: 9\nsex: Female\nrace: White alone\nancestry: American\nhousehold language: English only\neducation: Grade 2\nemployment status: Unemployed\nclass of worker: Not applicable\nindustry category: Not applicable\noccupation category: Not applicable\ndetailed job description: Student\nincome: Not applicable\nmarital status: Never married or under 15 years old\nhousehold type: Married couple household with children of the householder less than 18\nfamily presence and age: With related children under 5 years and 5 to 17 years\nplace of birth: Missouri/MO\ncitizenship: Born in the United States\nveteran status: Not applicable\ndisability: None\nhealth insurance: With health insurance coverage\nbig five scores: Openness: Low, Conscientiousness: High, Extraversion: Average, Agreeableness: Extremely High, Neuroticism: Extremely Low\ndefining quirks: Loves to play pretend doctor\nmannerisms: Often lost in her own world of imagination\npersonal time: Spends free time playing with toys or drawing\nlifestyle: Active and playful\nideology: Not applicable\npolitical views: Not applicable\nreligion: Protestant\n', 'statement': 'I think laws about abortion should try to protect babies because I believe every life is special and made by God. But I also think they should care about the mom and make sure she is safe and has help, because God wants us to be kind and loving to everyone.'}

### Statement 7

{'persona': "age: 54\nsex: Female\nrace: White alone\nancestry: Irish\nhousehold language: English only\neducation: Associate's degree\nemployment status: Civilian employed, at work\nclass of worker: Employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions\nindustry category: MED-Offices Of Physicians\noccupation category: HLS-Medical Assistants\ndetailed job description: Assists physicians by performing administrative and clinical tasks\nincome: 244000.0\nmarital status: Married\nhousehold type: Married couple household, no children of the householder less than 18\nfamily presence and age: No related children\nplace of birth: New York/NY\ncitizenship: Born in the United States\nveteran status: Non-Veteran\ndisability: None\nhealth insurance: With health insurance coverage\nbig five scores: Openness: Low, Conscientiousness: Extremely High, Extraversion: Extremely High, Agreeableness: Low, Neuroticism: Low\ndefining quirks: Enjoys painting in her free time\nmannerisms: Often hums a tune\npersonal time: Spends free time painting or gardening\nlifestyle: Active and health-conscious\nideology: Conservative\npolitical views: Republican\nreligion: Other Christian\n", 'statement': 'As a Christian and a conservative, I believe our laws on abortion should prioritize protecting unborn life while still recognizing rare, tragic situations where the mother’s life is at serious risk. I think decisions about abortion should be guided by moral principles rooted in faith, respect for life, and personal responsibility, rather than treating it as just another medical choice. I support policies that encourage adoption, provide support for women in crisis pregnancies, and promote education about alternatives to abortion.'}

### Statement 90

{'persona': 'age: 71\nsex: Male\nrace: White alone\nancestry: American\nhousehold language: English only\neducation: Regular high school diploma\nemployment status: Not in labor force\nclass of worker: Employee of a private not-for-profit, tax-exempt, or charitable organization\nindustry category: FIN-Savings Institutions, Including Credit Unions\noccupation category: MGR-Chief Executives And Legislators\ndetailed job description: Oversaw the operations of a non-profit savings institution, including strategic planning and fundraising\nincome: 94900.0\nmarital status: Married\nhousehold type: Married couple household, no children of the householder less than 18\nfamily presence and age: No related children\nplace of birth: Tennessee/TN\ncitizenship: Born in the United States\nveteran status: Non-Veteran\ndisability: None\nhealth insurance: With health insurance coverage\nbig five scores: Openness: Extremely High, Conscientiousness: Average, Extraversion: Low, Agreeableness: High, Neuroticism: Extremely High\ndefining quirks: Plays a musical instrument in his spare time\nmannerisms: Tends to fidget when nervous\npersonal time: Spends time with family and engaging in hobbies\nlifestyle: Relaxed and leisurely\nideology: Conservative\npolitical views: Republican\nreligion: Protestant\n', 'statement': 'Laws concerning abortion should be guided by the belief that life is sacred from conception and deserves legal protection, while still recognizing the gravity and complexity of situations where the mother’s life is at risk. As a conservative Protestant and a father whose kids are now grown, I see this as both a moral and a community issue: our laws should encourage responsibility, support mothers and families in crisis, and reflect a respect for the unborn that aligns with biblical principles.'}

### Cycle Explanation

- Statement 13 is preferred to Statement 7
- Statement 7 is preferred to Statement 90
- Statement 90 is preferred to Statement 13

This creates a logical inconsistency where preferences form a cycle rather than a clear hierarchy.

---

## pairwise_100_abortion_persona_40.csv

**Cycle:** 0 → 51 → 12 → 0

### Statement 0

{'persona': "age: 53\nsex: Male\nrace: White alone\nancestry: English\nhousehold language: English only\neducation: Master's degree\nemployment status: Not in labor force\nclass of worker: Local government employee (city, county, etc.)\nindustry category: EDU-Elementary And Secondary Schools\noccupation category: MGR-Education And Childcare Administrators\ndetailed job description: Oversees educational programs and childcare services in local area\nincome: 156000.0\nmarital status: Married\nhousehold type: Married couple household, no children of the householder less than 18\nfamily presence and age: No related children\nplace of birth: Illinois/IL\ncitizenship: Born in the United States\nveteran status: Non-Veteran\ndisability: With a disability\nhealth insurance: With health insurance coverage\nbig five scores: Openness: Extremely Low, Conscientiousness: Average, Extraversion: Average, Agreeableness: Average, Neuroticism: Extremely Low\ndefining quirks: Passionate advocate for outdoor education and sustainable practices in schools\nmannerisms: Takes a little longer to process information and respond\npersonal time: Volunteering, Participating in local sustainability initiatives\nlifestyle: Active and community-oriented\nideology: Progressive\npolitical views: Liberal\nreligion: Protestant\n", 'statement': 'Laws concerning abortion should prioritize a woman’s right to make decisions about her own body, supported by sound medical science and respect for individual conscience rather than religious doctrine. As someone who works in public education and values both personal freedom and social responsibility, I believe policy should focus on comprehensive sex education, accessible contraception, and support services so fewer people face crisis pregnancies, while keeping the ultimate decision in the hands of the pregnant person and their healthcare provider.'}

### Statement 51

{'persona': 'age: 12\nsex: Male\nrace: White alone\nancestry: Irish\nhousehold language: English only\neducation: Grade 5\nemployment status: Unemployed\ndetailed job description: Student\nmarital status: Never married or under 15 years old\nhousehold type: Married couple household with children of the householder less than 18\nfamily presence and age: With related children 5 to 17 years only\nplace of birth: Pennsylvania/PA\ncitizenship: Born in the United States\nveteran status: Non-Veteran\ndisability: With a disability\nhealth insurance: With health insurance coverage\ncognitive difficulty: Yes\nbig five scores: Openness: High, Conscientiousness: Extremely High, Extraversion: Extremely Low, Agreeableness: Extremely Low, Neuroticism: High\ndefining quirks: Loves to draw and create stories\nmannerisms: Often hums tunes while working\npersonal time: Spends free time drawing or playing video games\nlifestyle: Active and creative\nideology: Too young to form an ideology\npolitical views: Too young to form political views\nreligion: Protestant\n', 'statement': 'I think laws about abortion should try to protect the life of the baby because I believe every life is created by God and is important, but they should also care about the mom’s health and safety. I don’t think it’s an easy decision, so the laws should make sure people think very carefully, talk with doctors and family, and only allow it in very serious situations, like when the mother’s life is in danger.'}

### Statement 12

{'persona': 'age: 30\nsex: Female\nrace: White alone\nancestry: Pennsylvania German\nhousehold language: English only\neducation: Regular high school diploma\nemployment status: Civilian employed, at work\nclass of worker: Self-employed in own not incorporated business, professional practice, or farm\nindustry category: SRV-Private Households\noccupation category: PRS-Childcare Workers\ndetailed job description: Provides care and supervision for children in her care\nincome: 118700.0\nmarital status: Never married\nhousehold type: Cohabiting couple household, no children of the householder less than 18\nfamily presence and age: No family\nplace of birth: Pennsylvania/PA\ncitizenship: Born in the United States\nveteran status: Non-Veteran\ndisability: None\nhealth insurance: With health insurance coverage\nbig five scores: Openness: Average, Conscientiousness: Low, Extraversion: Extremely Low, Agreeableness: High, Neuroticism: Extremely High\ndefining quirks: Passionate about health and wellness, often shares tips with parents of children in her care\nmannerisms: Speaks with a distinct Pennsylvania Dutch accent\npersonal time: Spends free time in her garden or reading\nlifestyle: Healthy and active, enjoys outdoor activities\nideology: Believes in personal freedom and social responsibility\npolitical views: Independent\nreligion: Catholic\n', 'statement': 'Laws concerning abortion should balance a woman’s personal freedom with our shared responsibility to protect life, especially as a pregnancy progresses. I believe decisions early in pregnancy should be left mostly to the woman, her doctor, and her conscience, while later-term abortions should be more carefully regulated with strong medical and ethical oversight.'}

### Cycle Explanation

- Statement 0 is preferred to Statement 51
- Statement 51 is preferred to Statement 12
- Statement 12 is preferred to Statement 0

This creates a logical inconsistency where preferences form a cycle rather than a clear hierarchy.

---

## pairwise_100_abortion_persona_7.csv

**Cycle:** 0 → 1 → 22 → 0

### Statement 0

{'persona': "age: 53\nsex: Male\nrace: White alone\nancestry: English\nhousehold language: English only\neducation: Master's degree\nemployment status: Not in labor force\nclass of worker: Local government employee (city, county, etc.)\nindustry category: EDU-Elementary And Secondary Schools\noccupation category: MGR-Education And Childcare Administrators\ndetailed job description: Oversees educational programs and childcare services in local area\nincome: 156000.0\nmarital status: Married\nhousehold type: Married couple household, no children of the householder less than 18\nfamily presence and age: No related children\nplace of birth: Illinois/IL\ncitizenship: Born in the United States\nveteran status: Non-Veteran\ndisability: With a disability\nhealth insurance: With health insurance coverage\nbig five scores: Openness: Extremely Low, Conscientiousness: Average, Extraversion: Average, Agreeableness: Average, Neuroticism: Extremely Low\ndefining quirks: Passionate advocate for outdoor education and sustainable practices in schools\nmannerisms: Takes a little longer to process information and respond\npersonal time: Volunteering, Participating in local sustainability initiatives\nlifestyle: Active and community-oriented\nideology: Progressive\npolitical views: Liberal\nreligion: Protestant\n", 'statement': 'Laws concerning abortion should prioritize a woman’s right to make decisions about her own body, supported by sound medical science and respect for individual conscience rather than religious doctrine. As someone who works in public education and values both personal freedom and social responsibility, I believe policy should focus on comprehensive sex education, accessible contraception, and support services so fewer people face crisis pregnancies, while keeping the ultimate decision in the hands of the pregnant person and their healthcare provider.'}

### Statement 1

{'persona': 'age: 15\nsex: Female\nrace: Two or More Races\nancestry: Hispanic and White\nhousehold language: Spanish\neducation: Grade 9\nemployment status: Unemployed\nclass of worker: Not applicable\nindustry category: Not applicable\noccupation category: Not applicable\ndetailed job description: Student\nincome: 0\nmarital status: Never married or under 15 years old\nhousehold type: Married couple household with children of the householder less than 18\nfamily presence and age: With related children 5 to 17 years only\nplace of birth: New Jersey/NJ\ncitizenship: Born in the United States\nveteran status: Non-Veteran\ndisability: None\nhealth insurance: With health insurance coverage\nbig five scores: Openness: Extremely Low, Conscientiousness: Average, Extraversion: Low, Agreeableness: High, Neuroticism: Average\ndefining quirks: Enjoys exploring new technology and gadgets\nmannerisms: Tends to use regional slang and expressions\npersonal time: Spends free time studying, hanging out with friends, or exploring new tech gadgets\nlifestyle: Active and social, with a strong focus on school and family\nideology: Progressive\npolitical views: Too young to vote, but interested in social justice issues\nreligion: Other Christian\n', 'statement': 'I think laws about abortion should mainly protect a person’s right to make decisions about their own body, especially in tough situations like rape, health risks, or not being able to support a child. My faith matters to me, but I don’t think the government should force one religious view on everyone, because people have different beliefs and life circumstances. Instead, we should focus on making sure people have access to good sex education, birth control, and healthcare so they can prevent unwanted pregnancies and make informed choices.'}

### Statement 22

{'persona': 'age: 24\nsex: Male\nrace: Black or African American alone\nancestry: African\nhousehold language: English\neducation: 1 or more years of college credit, no degree\nemployment status: Not in labor force\nclass of worker: Employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions\nindustry category: TRN-Warehousing And Storage\noccupation category: TRN-Laborers And Freight, Stock, And Material Movers, Hand\ndetailed job description: Moving and organizing goods in a warehouse\nincome: 20000-30000\nmarital status: Never married or under 15 years old\nhousehold type: One-person household\nfamily presence and age: No family\nplace of birth: California/CA\ncitizenship: Born in the United States\nveteran status: Non-Veteran\ndisability: None\nhealth insurance: No health insurance coverage\nbig five scores: Openness: Average, Conscientiousness: Average, Extraversion: High, Agreeableness: Average, Neuroticism: Extremely Low\ndefining quirks: Has a unique sense of style and often wears colorful, unconventional outfits\nmannerisms: Often hums or sings to himself\npersonal time: Spends free time exploring new music or practicing a musical instrument\nlifestyle: Active and social\nideology: Liberal\npolitical views: Democrat\nreligion: Catholic\n', 'statement': 'Laws about abortion should respect a woman’s right to make decisions about her own body, while ensuring access to safe, affordable medical care regardless of income. As a Democrat and a Catholic, I believe faith can shape personal choices, but it shouldn’t be imposed through government policy, especially on people who don’t share the same beliefs. Instead, laws should focus on protecting health, privacy, and dignity, and we should invest more in sex education, contraception, and social support so fewer people are put in desperate situations in the first place.'}

### Cycle Explanation

- Statement 0 is preferred to Statement 1
- Statement 1 is preferred to Statement 22
- Statement 22 is preferred to Statement 0

This creates a logical inconsistency where preferences form a cycle rather than a clear hierarchy.

---

