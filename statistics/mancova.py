import numpy as np
import statsmodels.api as sm


def mancova(data, dependent, independent):
    """
    Perform a MANCOVA test.

    Args:
    data: pandas DataFrame
    dependent: str
    independent: str

    Returns:
    result: statsmodels.multivariate.manova.py.MANOVA
    """
    # Add a constant to the DataFrame
    data = sm.add_constant(data)
    
    # Fit the model
    model = sm.MANOVA.from_formula(f"{dependent} ~ {independent}", data=data)


    # Perform the MANCOVA test
    result = model.mv_test()

    return result

