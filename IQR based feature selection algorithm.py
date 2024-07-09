import numpy as np

def detect_consecutive_outliers_with_iqr(data):
    """
    Detect for continuous outliers (i.e. numerical spikes) in the list and automatically set a threshold.

    Parameters:
    Data (list): A list of input values.

    return:
    Bool: If there are consecutive outliers, return True; Otherwise, return False.
    """
    if len(data) < 2:
        # If the length of the list is less than 2, there are no consecutive elements
        return False

        # Calculate IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Set an IQR based threshold
    threshold = Q3 + 1.5 * IQR

    # Initialize a variable to track whether the previous element is also an outlier
    previous_was_outlier = False
    outlier_number = 0

    for i in range(1, len(data)):
        # Check if the current value exceeds the IQR threshold
        if data[i] > threshold:
            outlier_number += 1
            if previous_was_outlier and outlier_number > 4:
                # If the previous element is also an outlier,
                # there are consecutive outliers that last for more than 5 minutes
                return True
            else:
                # Mark the current element as a possible outlier start
                previous_was_outlier = True
        else:
            # Reset tags
            previous_was_outlier = False
            outlier_number = 0

            # After traversing the entire list, if no consecutive outliers are found, return False
    return False