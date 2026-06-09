import pandas as pd


def preprocess_data(df):

    # Fill missing age values
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    # Encode Sex
    df['Sex'] = df['Sex'].replace({
        'male': 0,
        'female': 1
    }).astype(int)

    return df


def create_features(df):

    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # IsAlone
    df['IsAlone'] = df['FamilySize'].apply(
        lambda x: 1 if x == 1 else 0
    )

    return df


def encode_gender(sex):

    sex = sex.lower()

    if sex == "male":
        return 0

    elif sex == "female":
        return 1

    else:
        raise ValueError("Invalid gender")


def encode_isalone(value):

    value = value.lower()

    if value == "alone":
        return 1

    elif value == "not alone":
        return 0

    else:
        raise ValueError("Invalid value")