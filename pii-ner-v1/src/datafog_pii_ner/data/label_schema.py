"""Canonical PII entity type taxonomy and BIO label mapping.

Defines the unified label set across all training datasets (AI4Privacy,
Nemotron-PII, Gretel) and generates BIO tag IDs for token classification.
"""

# Tier 1: Critical PII — target >=0.98 recall
TIER_1 = [
    "SSN",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "PASSPORT_NUMBER",
    "DRIVERS_LICENSE",
    "TAX_ID",
]

# Tier 2: High sensitivity — target >=0.95 recall
TIER_2 = [
    "PERSON",
    "EMAIL",
    "PHONE",
    "DATE_OF_BIRTH",
    "STREET_ADDRESS",
    "IP_ADDRESS",
]

# Tier 3: Moderate sensitivity — target >=0.90 recall
TIER_3 = [
    "USERNAME",
    "DATE",
    "LOCATION",
    "ORGANIZATION",
    "URL",
    "LICENSE_PLATE",
    "AGE",
    "NATIONALITY",
    "GENDER",
    "ETHNICITY",
    "RELIGION",
    "MARITAL_STATUS",
]

# Tier 4: Domain-specific — target >=0.85 recall
TIER_4 = [
    "MEDICAL_RECORD",
    "EMPLOYEE_ID",
    "STUDENT_ID",
    "ACCOUNT_NUMBER",
    "PIN",
    "PASSWORD",
    "BIOMETRIC",
    "VEHICLE_ID",
    "DEVICE_ID",
    "CRYPTO_WALLET",
    "IBAN",
    "SWIFT_CODE",
    "INSURANCE_NUMBER",
    "SALARY",
    "CRIMINAL_RECORD",
    "POLITICAL_AFFILIATION",
    "SEXUAL_ORIENTATION",
    "HEALTH_CONDITION",
    "GENETIC_DATA",
    "TRADE_UNION",
]

ALL_ENTITY_TYPES = TIER_1 + TIER_2 + TIER_3 + TIER_4

TIER_MAP = {}
for t in TIER_1:
    TIER_MAP[t] = 1
for t in TIER_2:
    TIER_MAP[t] = 2
for t in TIER_3:
    TIER_MAP[t] = 3
for t in TIER_4:
    TIER_MAP[t] = 4


def build_bio_label_list() -> list[str]:
    """Build the full BIO label list: O, B-TYPE, I-TYPE for each entity type."""
    labels = ["O"]
    for etype in ALL_ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    return labels


BIO_LABELS = build_bio_label_list()
LABEL_TO_ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(BIO_LABELS)}
NUM_LABELS = len(BIO_LABELS)

# Default tier weights for loss scaling.
# Sequences containing higher-tier entities get proportionally higher loss.
DEFAULT_TIER_WEIGHTS = {1: 3.0, 2: 2.0, 3: 1.5, 4: 1.0}


def build_label_weights(tier_weights: dict[int, float] | None = None) -> list[float]:
    """Build per-label weight vector from tier weights.

    Returns a list of length NUM_LABELS where each label ID maps to a weight.
    O tag gets weight 1.0. Entity tags get their tier's weight.
    """
    if tier_weights is None:
        tier_weights = DEFAULT_TIER_WEIGHTS

    weights = [1.0] * NUM_LABELS  # O tag = 1.0
    for etype in ALL_ENTITY_TYPES:
        tier = TIER_MAP[etype]
        w = tier_weights.get(tier, 1.0)
        b_id = LABEL_TO_ID[f"B-{etype}"]
        i_id = LABEL_TO_ID[f"I-{etype}"]
        weights[b_id] = w
        weights[i_id] = w
    return weights

# --- Dataset-specific label mappings ---
# Maps each source dataset's entity type names to our canonical names.

AI4PRIVACY_LABEL_MAP = {
    "FIRSTNAME": "PERSON",
    "LASTNAME": "PERSON",
    "FULLNAME": "PERSON",
    "NAME": "PERSON",
    "EMAIL": "EMAIL",
    "PHONENUMBER": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "SOCIALSECURITYNUMBER": "SSN",
    "SSN": "SSN",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "DATE": "DATE",
    "DATEOFBIRTH": "DATE_OF_BIRTH",
    "DATE_OF_BIRTH": "DATE_OF_BIRTH",
    "STREET_ADDRESS": "STREET_ADDRESS",
    "STREETADDRESS": "STREET_ADDRESS",
    "ADDRESS": "STREET_ADDRESS",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTY": "LOCATION",
    "COUNTRY": "LOCATION",
    "ZIPCODE": "STREET_ADDRESS",
    "IP_ADDRESS": "IP_ADDRESS",
    "IPADDRESS": "IP_ADDRESS",
    "USERNAME": "USERNAME",
    "URL": "URL",
    "ORGANIZATION": "ORGANIZATION",
    "COMPANY": "ORGANIZATION",
    "GENDER": "GENDER",
    "AGE": "AGE",
    "NATIONALITY": "NATIONALITY",
    "ETHNICITY": "ETHNICITY",
    "RELIGION": "RELIGION",
    "MARITAL_STATUS": "MARITAL_STATUS",
    "POLITICAL_AFFILIATION": "POLITICAL_AFFILIATION",
    "IBAN": "IBAN",
    "SWIFT": "SWIFT_CODE",
    "PASSPORT_NUMBER": "PASSPORT_NUMBER",
    "PASSPORTNUMBER": "PASSPORT_NUMBER",
    "DRIVERS_LICENSE": "DRIVERS_LICENSE",
    "DRIVERSLICENSE": "DRIVERS_LICENSE",
    "LICENSE_PLATE": "LICENSE_PLATE",
    "LICENSEPLATE": "LICENSE_PLATE",
    "BANK_ACCOUNT": "BANK_ACCOUNT",
    "BANKACCOUNTNUMBER": "BANK_ACCOUNT",
    "TAX_ID": "TAX_ID",
    "TAXID": "TAX_ID",
    "PASSWORD": "PASSWORD",
    "PIN": "PIN",
    "SALARY": "SALARY",
    "CRYPTO_WALLET": "CRYPTO_WALLET",
    "VEHICLE_ID": "VEHICLE_ID",
    "MEDICAL_RECORD": "MEDICAL_RECORD",
    "INSURANCE_NUMBER": "INSURANCE_NUMBER",
}

NEMOTRON_LABEL_MAP = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "EMAIL": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "PHONE": "PHONE",
    "FAX_NUMBER": "PHONE",
    "SSN": "SSN",
    "SOCIAL_SECURITY_NUMBER": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "CREDIT_DEBIT_CARD": "CREDIT_CARD",
    "DATE_OF_BIRTH": "DATE_OF_BIRTH",
    "DOB": "DATE_OF_BIRTH",
    "DATE": "DATE",
    "DATE_TIME": "DATE",
    "TIME": "DATE",
    "ADDRESS": "STREET_ADDRESS",
    "STREET_ADDRESS": "STREET_ADDRESS",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTY": "LOCATION",
    "COUNTRY": "LOCATION",
    "POSTCODE": "STREET_ADDRESS",
    "COORDINATE": "LOCATION",
    "IP_ADDRESS": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "USERNAME": "USERNAME",
    "USER_NAME": "USERNAME",
    "URL": "URL",
    "ORGANIZATION": "ORGANIZATION",
    "ORG": "ORGANIZATION",
    "COMPANY_NAME": "ORGANIZATION",
    "LOCATION": "LOCATION",
    "LOC": "LOCATION",
    "PASSPORT": "PASSPORT_NUMBER",
    "PASSPORT_NUMBER": "PASSPORT_NUMBER",
    "DRIVERS_LICENSE": "DRIVERS_LICENSE",
    "DRIVER_LICENSE": "DRIVERS_LICENSE",
    "CERTIFICATE_LICENSE_NUMBER": "DRIVERS_LICENSE",
    "BANK_ACCOUNT": "BANK_ACCOUNT",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "CUSTOMER_ID": "ACCOUNT_NUMBER",
    "TAX_ID": "TAX_ID",
    "AGE": "AGE",
    "GENDER": "GENDER",
    "NATIONALITY": "NATIONALITY",
    "LICENSE_PLATE": "LICENSE_PLATE",
    "VEHICLE_IDENTIFIER": "VEHICLE_ID",
    "IBAN": "IBAN",
    "SWIFT": "SWIFT_CODE",
    "PASSWORD": "PASSWORD",
    "BIOMETRIC_IDENTIFIER": "BIOMETRIC",
    "EMPLOYEE_ID": "EMPLOYEE_ID",
}

GRETEL_LABEL_MAP = {
    "person": "PERSON",
    "name": "PERSON",
    "first_name": "PERSON",
    "last_name": "PERSON",
    "email": "EMAIL",
    "phone": "PHONE",
    "phone_number": "PHONE",
    "ssn": "SSN",
    "credit_card": "CREDIT_CARD",
    "credit_card_number": "CREDIT_CARD",
    "credit_card_security_code": "CREDIT_CARD",
    "date_of_birth": "DATE_OF_BIRTH",
    "dob": "DATE_OF_BIRTH",
    "date": "DATE",
    "date_time": "DATE",
    "time": "DATE",
    "address": "STREET_ADDRESS",
    "street_address": "STREET_ADDRESS",
    "ip_address": "IP_ADDRESS",
    "ipv4": "IP_ADDRESS",
    "ipv6": "IP_ADDRESS",
    "username": "USERNAME",
    "user_name": "USERNAME",
    "url": "URL",
    "organization": "ORGANIZATION",
    "company": "ORGANIZATION",
    "location": "LOCATION",
    "local_latlng": "LOCATION",
    "passport": "PASSPORT_NUMBER",
    "passport_number": "PASSPORT_NUMBER",
    "drivers_license": "DRIVERS_LICENSE",
    "driver_license_number": "DRIVERS_LICENSE",
    "bank_account": "BANK_ACCOUNT",
    "bank_routing_number": "BANK_ACCOUNT",
    "bban": "BANK_ACCOUNT",
    "iban": "IBAN",
    "swift_bic_code": "SWIFT_CODE",
    "license_plate": "LICENSE_PLATE",
    "password": "PASSWORD",
    "api_key": "PASSWORD",
    "account_pin": "PIN",
    "customer_id": "ACCOUNT_NUMBER",
    "employee_id": "EMPLOYEE_ID",
}


def map_label(source_label: str, dataset_name: str) -> str | None:
    """Map a dataset-specific label to our canonical label.

    Returns None if the label is not recognized (will be mapped to O).
    """
    label_maps = {
        "ai4privacy": AI4PRIVACY_LABEL_MAP,
        "nemotron": NEMOTRON_LABEL_MAP,
        "gretel": GRETEL_LABEL_MAP,
    }

    label_map = label_maps.get(dataset_name)
    if label_map is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Try exact match, then uppercase, then title case
    normalized = source_label.strip()
    return (
        label_map.get(normalized)
        or label_map.get(normalized.upper())
        or label_map.get(normalized.lower())
    )
