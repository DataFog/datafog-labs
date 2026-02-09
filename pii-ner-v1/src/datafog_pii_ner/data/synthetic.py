"""Synthetic data generation for zero-occurrence entity types.

Generates training examples with known span offsets for entity types
that have no coverage in the existing open-source datasets.
"""

import random
import re
import string

# --- Value generators / curated lists per entity type ---

NATIONALITIES = [
    "American", "British", "Canadian", "Australian", "German", "French",
    "Italian", "Spanish", "Portuguese", "Dutch", "Belgian", "Swiss",
    "Austrian", "Swedish", "Norwegian", "Danish", "Finnish", "Polish",
    "Czech", "Hungarian", "Romanian", "Bulgarian", "Greek", "Turkish",
    "Russian", "Ukrainian", "Japanese", "Chinese", "Korean", "Indian",
    "Pakistani", "Bangladeshi", "Indonesian", "Malaysian", "Thai",
    "Vietnamese", "Filipino", "Nigerian", "Ghanaian", "Kenyan",
    "South African", "Egyptian", "Moroccan", "Algerian", "Tunisian",
    "Brazilian", "Mexican", "Argentine", "Colombian", "Chilean",
    "Peruvian", "Venezuelan", "Cuban", "Jamaican", "Trinidadian",
    "Saudi", "Emirati", "Qatari", "Kuwaiti", "Omani", "Bahraini",
    "Iraqi", "Iranian", "Afghan", "Nepali", "Sri Lankan", "Burmese",
    "Cambodian", "Laotian", "Mongolian", "Uzbek", "Kazakh", "Georgian",
    "Armenian", "Azerbaijani", "Israeli", "Lebanese", "Jordanian",
    "Syrian", "Palestinian", "Yemeni", "Somali", "Ethiopian", "Sudanese",
    "Tanzanian", "Ugandan", "Rwandan", "Congolese", "Cameroonian",
    "Senegalese", "Malian", "Ivorian", "Liberian", "Sierra Leonean",
    "Zambian", "Zimbabwean", "Mozambican", "Angolan", "Namibian",
    "Botswanan", "Malagasy", "Mauritian", "Seychellois", "Comoran",
    "New Zealander", "Fijian", "Samoan", "Tongan", "Papua New Guinean",
    "Ecuadorian", "Bolivian", "Paraguayan", "Uruguayan", "Guatemalan",
    "Honduran", "Salvadoran", "Nicaraguan", "Costa Rican", "Panamanian",
    "Dominican", "Puerto Rican", "Haitian", "Barbadian", "Bahamian",
    "Irish", "Scottish", "Welsh", "Icelandic", "Luxembourgish",
    "Maltese", "Cypriot", "Estonian", "Latvian", "Lithuanian",
    "Slovenian", "Croatian", "Serbian", "Bosnian", "Montenegrin",
    "Macedonian", "Albanian", "Kosovar", "Moldovan", "Belarusian",
    "Singaporean", "Bruneian", "Timorese", "Maldivian", "Bhutanese",
    "Tajik", "Turkmen", "Kyrgyz",
]

RELIGIONS = [
    "Christian", "Catholic", "Protestant", "Orthodox", "Baptist",
    "Methodist", "Lutheran", "Presbyterian", "Anglican", "Pentecostal",
    "Evangelical", "Mormon", "Adventist", "Quaker",
    "Muslim", "Sunni", "Shia", "Sufi",
    "Jewish", "Hindu", "Buddhist", "Sikh", "Jain",
    "Taoist", "Shinto", "Confucian", "Bahai", "Zoroastrian",
    "Atheist", "Agnostic",
]

MARITAL_STATUSES = [
    "Single", "Married", "Divorced", "Widowed", "Separated",
    "Domestic Partnership",
]

POLITICAL_AFFILIATIONS = [
    "Democrat", "Republican", "Independent", "Libertarian", "Green Party",
    "Labour", "Conservative", "Liberal Democrat", "Social Democrat",
    "Progressive", "Centrist", "Socialist", "Communist",
    "Nationalist", "Populist", "Anarchist", "Moderate",
    "Christian Democrat", "Pirate Party", "Reform Party",
]

SEXUAL_ORIENTATIONS = [
    "heterosexual", "homosexual", "bisexual", "pansexual",
    "asexual", "queer", "demisexual", "questioning",
]

HEALTH_CONDITIONS = [
    "diabetes", "hypertension", "asthma", "depression", "anxiety disorder",
    "bipolar disorder", "schizophrenia", "PTSD", "ADHD", "autism",
    "Alzheimer's disease", "Parkinson's disease", "multiple sclerosis",
    "epilepsy", "migraine", "chronic fatigue syndrome", "fibromyalgia",
    "rheumatoid arthritis", "osteoarthritis", "lupus",
    "Crohn's disease", "ulcerative colitis", "celiac disease",
    "Type 1 diabetes", "Type 2 diabetes", "gestational diabetes",
    "coronary artery disease", "heart failure", "atrial fibrillation",
    "chronic kidney disease", "liver cirrhosis", "hepatitis B",
    "hepatitis C", "HIV", "tuberculosis", "pneumonia",
    "breast cancer", "lung cancer", "prostate cancer", "leukemia",
    "lymphoma", "melanoma", "colon cancer", "pancreatic cancer",
    "ovarian cancer", "thyroid cancer", "bladder cancer",
    "sickle cell disease", "cystic fibrosis", "Down syndrome",
    "hemophilia", "muscular dystrophy", "spina bifida",
    "sleep apnea", "insomnia", "eating disorder", "OCD",
    "substance use disorder", "alcoholism", "opioid addiction",
    "chronic pain syndrome", "irritable bowel syndrome",
    "endometriosis", "polycystic ovary syndrome", "infertility",
    "glaucoma", "macular degeneration", "cataracts",
    "hearing loss", "tinnitus", "vertigo",
    "eczema", "psoriasis", "dermatitis", "rosacea",
    "osteoporosis", "gout", "anemia", "thyroid disorder",
    "Graves' disease", "Hashimoto's thyroiditis",
    "stroke", "traumatic brain injury", "concussion",
    "carpal tunnel syndrome", "sciatica", "herniated disc",
]

CRIMINAL_RECORD_PHRASES = [
    "felony conviction 2019", "misdemeanor assault 2020",
    "DUI charge 2018", "theft conviction 2017",
    "fraud conviction 2021", "drug possession charge 2016",
    "burglary conviction 2015", "arson charge 2019",
    "embezzlement conviction 2020", "vandalism charge 2018",
    "shoplifting conviction 2022", "tax evasion charge 2017",
    "forgery conviction 2019", "identity theft charge 2021",
    "assault conviction 2016", "domestic violence charge 2020",
    "reckless driving conviction 2018", "trespassing charge 2022",
    "disorderly conduct conviction 2017", "weapons charge 2019",
    "probation violation 2021", "parole violation 2020",
    "warrant for arrest 2022", "restraining order violation 2019",
    "public intoxication charge 2018", "hit and run conviction 2017",
    "money laundering charge 2021", "extortion conviction 2020",
    "bribery charge 2019", "perjury conviction 2018",
]

# --- Templates per entity type (20-30 each) ---

TEMPLATES = {
    "HEALTH_CONDITION": [
        "The patient was diagnosed with {value}.",
        "Medical history: {value}.",
        "Primary diagnosis: {value}.",
        "The individual has been treated for {value}.",
        "Chart notes indicate {value}.",
        "The patient presents with {value}.",
        "Condition: {value}.",
        "The member was referred for {value} treatment.",
        "Ongoing management of {value}.",
        "The claimant reports {value}.",
        "Family history of {value}.",
        "Screening positive for {value}.",
        "The employee disclosed {value}.",
        "Insurance claim for {value}.",
        "The resident has a history of {value}.",
        "Differential diagnosis includes {value}.",
        "Treatment plan for {value} initiated.",
        "The child was diagnosed with {value} at age five.",
        "Chronic condition: {value}.",
        "Lab results consistent with {value}.",
        "The patient is managing {value} with medication.",
        "Hospitalized for {value}.",
        "Discharge diagnosis: {value}.",
        "Follow-up visit for {value}.",
        "The applicant disclosed a diagnosis of {value}.",
    ],
    "NATIONALITY": [
        "The applicant is {value} by nationality.",
        "Patient identified as {value}.",
        "Citizenship: {value}.",
        "The employee is a {value} citizen.",
        "Records show the individual is {value}.",
        "Country of origin: {value}.",
        "The passport holder is {value}.",
        "Nationality listed as {value} in the database.",
        "The student is {value} and enrolled in the exchange program.",
        "Background check confirms {value} nationality.",
        "The suspect is reportedly {value}.",
        "Immigration status: {value} national.",
        "The customer indicated they are {value}.",
        "National origin: {value}.",
        "The claimant has {value} citizenship.",
        "Born in the country, {value} by descent.",
        "Declared nationality is {value}.",
        "The {value} national applied for residency.",
        "Cultural background: {value}.",
        "The witness is a {value} citizen residing abroad.",
        "Filed under {value} nationality.",
        "The respondent claims to be {value}.",
        "Ethnic and national heritage: {value}.",
        "The defendant is {value}.",
        "The job applicant listed {value} as their nationality.",
    ],
    "RELIGION": [
        "The patient identifies as {value}.",
        "Religious affiliation: {value}.",
        "The individual practices {value} faith.",
        "Spiritual background: {value}.",
        "The employee requested {value} holiday accommodation.",
        "Listed religion is {value}.",
        "The student follows {value} traditions.",
        "Faith community: {value}.",
        "The applicant self-identified as {value}.",
        "Religious preference: {value}.",
        "The member belongs to the {value} community.",
        "Denomination: {value}.",
        "The deceased was {value}.",
        "Chaplain noted {value} affiliation.",
        "The client follows {value} practices.",
        "Place of worship indicates {value} faith.",
        "The registrant declared their religion as {value}.",
        "Belief system: {value}.",
        "The respondent identifies with {value} beliefs.",
        "Dietary restrictions consistent with {value} practice.",
        "Census record shows {value} religious affiliation.",
        "The family practices {value} traditions at home.",
        "The counselor noted {value} spiritual orientation.",
        "Funeral arrangements per {value} customs.",
        "The teacher requested accommodation for {value} observance.",
    ],
    "MARITAL_STATUS": [
        "Marital status: {value}.",
        "The applicant is currently {value}.",
        "Filed as {value} on the tax return.",
        "The patient reported being {value}.",
        "Relationship status listed as {value}.",
        "Census data shows {value}.",
        "The employee is {value}.",
        "Current marital status: {value}.",
        "Registered as {value} in the system.",
        "Insurance beneficiary status: {value}.",
        "The individual's status changed to {value}.",
        "Court records indicate {value}.",
        "Status updated to {value} as of last month.",
        "The claimant declared {value} status.",
        "Family status: {value}.",
        "Housing application lists status as {value}.",
        "Emergency contact form shows {value}.",
        "HR records indicate the employee is {value}.",
        "Welfare eligibility: {value} household.",
        "Personal declaration: {value}.",
        "Medical intake form indicates {value}.",
        "The respondent is {value} according to the filing.",
        "Legal status: {value} per court decree.",
        "Background check shows {value}.",
        "The donor is {value}.",
    ],
    "STUDENT_ID": [
        "Student ID: {value}.",
        "Enrolled under student number {value}.",
        "The student's ID is {value}.",
        "Academic record for {value}.",
        "Student identifier: {value}.",
        "Library card issued to student {value}.",
        "Transcript requested for student {value}.",
        "Dormitory assigned to {value}.",
        "Financial aid application for student {value}.",
        "Exam results for {value} are now available.",
        "The university assigned ID {value}.",
        "Course registration confirmed for {value}.",
        "Student {value} has been placed on the dean's list.",
        "Meal plan activated for student {value}.",
        "ID badge number: {value}.",
        "Parking permit issued to {value}.",
        "Graduation clearance for student {value}.",
        "The scholarship was awarded to student {value}.",
        "Lab access granted to {value}.",
        "Student health records for {value}.",
        "Advising appointment for student {value}.",
        "Disciplinary record: student {value}.",
        "Internship application by {value}.",
        "Research assistant: student {value}.",
        "Class roster includes {value}.",
    ],
    "CRYPTO_WALLET": [
        "Payment sent to wallet {value}.",
        "Crypto wallet address: {value}.",
        "The transaction was received by {value}.",
        "Withdrawal to {value} confirmed.",
        "Bitcoin address: {value}.",
        "Ethereum wallet: {value}.",
        "Funds transferred to {value}.",
        "The wallet {value} received the deposit.",
        "Blockchain ledger shows transfer to {value}.",
        "Refund issued to wallet {value}.",
        "Customer wallet: {value}.",
        "Send payment to address {value}.",
        "The donor's crypto address is {value}.",
        "Audit trail points to wallet {value}.",
        "Cold storage address: {value}.",
        "Hot wallet: {value}.",
        "The exchange withdrew to {value}.",
        "DeFi contract interacted with {value}.",
        "Mining rewards deposited to {value}.",
        "NFT minted to wallet {value}.",
        "Staking rewards sent to {value}.",
        "Token airdrop to {value}.",
        "The suspect's wallet is {value}.",
        "Forfeiture order for wallet {value}.",
        "Gas fees paid from {value}.",
    ],
    "INSURANCE_NUMBER": [
        "Insurance number: {value}.",
        "Policy ID: {value}.",
        "The insured party's number is {value}.",
        "Claim filed under policy {value}.",
        "Health insurance ID: {value}.",
        "Beneficiary number: {value}.",
        "Coverage verified for {value}.",
        "The plan number is {value}.",
        "Premium payment for policy {value}.",
        "Insurance card shows {value}.",
        "Dental plan ID: {value}.",
        "Vision coverage number: {value}.",
        "The member's insurance number is {value}.",
        "Group policy: {value}.",
        "Auto insurance policy: {value}.",
        "Home insurance number: {value}.",
        "Life insurance policy: {value}.",
        "Disability insurance: {value}.",
        "Workers compensation claim: {value}.",
        "Travel insurance policy number {value}.",
        "Liability coverage ID: {value}.",
        "The provider billed insurance {value}.",
        "Pre-authorization for policy {value}.",
        "Referral under insurance {value}.",
        "Copay applied for {value}.",
    ],
    "SALARY": [
        "Annual salary: {value}.",
        "The employee earns {value} per year.",
        "Compensation package: {value}.",
        "Base pay is {value}.",
        "The offer includes a salary of {value}.",
        "Gross income: {value}.",
        "Current salary: {value}.",
        "The position pays {value} annually.",
        "Salary range: {value}.",
        "Hourly rate equivalent of {value}.",
        "Monthly compensation: {value}.",
        "The contractor invoiced {value}.",
        "Take-home pay: {value}.",
        "Salary before taxes: {value}.",
        "Total compensation is {value}.",
        "The raise brings salary to {value}.",
        "Starting salary: {value}.",
        "Negotiated pay: {value}.",
        "The W-2 shows earnings of {value}.",
        "Performance bonus and salary: {value}.",
        "Payroll records show {value}.",
        "Severance package: {value}.",
        "Commission-based earnings: {value}.",
        "The executive's salary is {value}.",
        "Pension contribution based on {value}.",
    ],
    "CRIMINAL_RECORD": [
        "Criminal history includes {value}.",
        "Background check revealed {value}.",
        "Prior offense: {value}.",
        "Criminal record: {value}.",
        "The defendant has a record of {value}.",
        "Conviction history: {value}.",
        "Past offenses include {value}.",
        "Legal history: {value}.",
        "Court records show {value}.",
        "The applicant disclosed {value}.",
        "Rap sheet includes {value}.",
        "Prior conviction: {value}.",
        "The individual has a history of {value}.",
        "Sentencing records: {value}.",
        "Parole report mentions {value}.",
        "The screening flagged {value}.",
        "Arrest record: {value}.",
        "Juvenile record: {value}.",
        "The FBI report lists {value}.",
        "State records show {value}.",
        "Federal conviction: {value}.",
        "Pending case: {value}.",
        "The tenant screening found {value}.",
        "Employment check uncovered {value}.",
        "Fingerprint match: {value}.",
    ],
    "POLITICAL_AFFILIATION": [
        "Political affiliation: {value}.",
        "Registered as {value}.",
        "The voter is affiliated with {value}.",
        "Party membership: {value}.",
        "The candidate ran as a {value}.",
        "Political party: {value}.",
        "Donor records show {value} affiliation.",
        "The individual supports {value}.",
        "Voting registration: {value}.",
        "Campaign contributions to {value}.",
        "The activist is aligned with {value}.",
        "Election records: {value} party member.",
        "The official is a member of {value}.",
        "Political orientation: {value}.",
        "Union endorsement: {value}.",
        "The lobbyist represents {value} interests.",
        "Survey response: {value} leaning.",
        "The representative is {value}.",
        "Primary ballot: {value}.",
        "Coalition member: {value}.",
        "The delegate supports {value} platform.",
        "Political stance: {value}.",
        "The constituent voted {value}.",
        "Party registration: {value}.",
        "Ideological alignment: {value}.",
    ],
    "SEXUAL_ORIENTATION": [
        "Sexual orientation: {value}.",
        "The patient identifies as {value}.",
        "Self-reported orientation: {value}.",
        "The individual is {value}.",
        "Demographic survey: {value}.",
        "Personal profile indicates {value}.",
        "The client identifies as {value}.",
        "Health intake form: {value}.",
        "The respondent reported being {value}.",
        "Census classification: {value}.",
        "Identity: {value}.",
        "Counseling notes: identifies as {value}.",
        "The member reported {value} orientation.",
        "Social services record: {value}.",
        "The youth identifies as {value}.",
        "Support group: {value} community.",
        "Psychological assessment: {value}.",
        "The student disclosed being {value}.",
        "Diversity survey: {value}.",
        "The employee disclosed {value} orientation.",
        "Medical record notes: {value}.",
        "The applicant's orientation is {value}.",
        "Wellness screening: {value}.",
        "The participant identifies as {value}.",
        "Intake form indicates {value}.",
    ],
}


def _generate_student_id(rng: random.Random) -> str:
    return f"STU-{rng.randint(100000, 999999)}"


def _generate_crypto_wallet(rng: random.Random) -> str:
    if rng.random() < 0.5:
        # ETH-style address
        return "0x" + "".join(rng.choices(string.hexdigits[:16], k=40))
    else:
        # BTC-style address (simplified)
        prefix = rng.choice(["1", "3", "bc1q"])
        chars = string.ascii_letters + string.digits
        length = 33 if prefix in ("1", "3") else 39
        return prefix + "".join(rng.choices(chars, k=length - len(prefix)))


def _generate_insurance_number(rng: random.Random) -> str:
    patterns = [
        lambda: f"INS-{rng.randint(1000000, 9999999)}",
        lambda: f"HP{rng.randint(100000000, 999999999)}",
        lambda: f"{rng.choice(string.ascii_uppercase)}{rng.randint(10000000, 99999999)}",
        lambda: f"POL-{rng.randint(100000, 999999)}-{rng.randint(10, 99)}",
    ]
    return rng.choice(patterns)()


def _generate_salary(rng: random.Random) -> str:
    patterns = [
        lambda: f"${rng.randint(30, 250):,},000/year",
        lambda: f"${rng.randint(30000, 250000):,}",
        lambda: f"${rng.randint(30, 250):,},000 annually",
        lambda: f"EUR {rng.randint(25000, 200000):,}",
        lambda: f"${rng.randint(2500, 20000):,}/month",
        lambda: f"${rng.randint(15, 150)}/hour",
    ]
    return rng.choice(patterns)()


VALUE_GENERATORS = {
    "HEALTH_CONDITION": lambda rng: rng.choice(HEALTH_CONDITIONS),
    "NATIONALITY": lambda rng: rng.choice(NATIONALITIES),
    "RELIGION": lambda rng: rng.choice(RELIGIONS),
    "MARITAL_STATUS": lambda rng: rng.choice(MARITAL_STATUSES),
    "STUDENT_ID": _generate_student_id,
    "CRYPTO_WALLET": _generate_crypto_wallet,
    "INSURANCE_NUMBER": _generate_insurance_number,
    "SALARY": _generate_salary,
    "CRIMINAL_RECORD": lambda rng: rng.choice(CRIMINAL_RECORD_PHRASES),
    "POLITICAL_AFFILIATION": lambda rng: rng.choice(POLITICAL_AFFILIATIONS),
    "SEXUAL_ORIENTATION": lambda rng: rng.choice(SEXUAL_ORIENTATIONS),
}


def generate_example(
    entity_type: str,
    rng: random.Random,
) -> dict:
    """Generate a single synthetic training example for the given entity type.

    Returns:
        {"text": str, "spans": [{"start": int, "end": int, "label": str}]}
    """
    if entity_type not in VALUE_GENERATORS:
        raise ValueError(f"No generator for entity type: {entity_type}")
    if entity_type not in TEMPLATES:
        raise ValueError(f"No templates for entity type: {entity_type}")

    value = VALUE_GENERATORS[entity_type](rng)
    template = rng.choice(TEMPLATES[entity_type])

    # Find where {value} appears in template to compute span offset
    placeholder = "{value}"
    idx = template.index(placeholder)
    text = template.replace(placeholder, value, 1)

    start = idx
    end = idx + len(value)

    return {
        "text": text,
        "spans": [{"start": start, "end": end, "label": entity_type}],
    }


def generate_dataset(
    entity_type: str,
    n: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """Generate n synthetic examples for an entity type."""
    rng = random.Random(seed)
    return [generate_example(entity_type, rng) for _ in range(n)]


SYNTHETIC_ENTITY_TYPES = list(VALUE_GENERATORS.keys())
