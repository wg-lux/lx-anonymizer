import gender_guesser.detector as gender

def determine_gender(first_name, detector):
    '''
    The result will be one of unknown (name not found), andy (androgynous), male, female, mostly_male, or mostly_female. \
    The difference between andy and unknown is that the former is found to have the same probability \
    to be male than to be female, while the later means that the name wasn't found in the database.
    '''
    d = gender.Detector()
    country = "germany"

    return d.get_gender(first_name, country)