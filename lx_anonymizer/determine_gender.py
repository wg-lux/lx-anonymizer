import gender_guesser.detector as gender_detector

# Initialize the detector
detector = gender_detector.Detector(case_sensitive=False)

def determine_gender(first_name):
    """
    Determines the gender of a given first name.
    
    Args:
        first_name (str): First name to determine gender for
        
    Returns:
        str: Gender ('male', 'female', 'unknown')
    """
    if not first_name:
        return "unknown" 
        
    # Get the gender prediction
    gender_prediction = detector.get_gender(first_name)
     
    # Map the prediction to a more simplified set of categories
    if gender_prediction in ["male", "mostly_male"]:
        return "male"
    elif gender_prediction in ["female", "mostly_female"]:
        return "female"
    else:
        return "unknown"  # Includes 'unknown' and 'andy' (androgynous)