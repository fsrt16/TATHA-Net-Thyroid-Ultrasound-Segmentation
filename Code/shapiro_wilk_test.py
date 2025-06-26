from scipy.stats import shapiro

def shapiro_wilk_test(scores):
    """
    Perform Shapiro-Wilk test for normality on a given array of scores.

    Parameters:
    - scores (array-like): A 1D array or list of numerical values (e.g., accuracy scores from CV).

    Returns:
    - p_value (float): The p-value of the test
    - is_normal (bool): True if data is normally distributed (p > 0.05), else False
    - stat (float): The test statistic (W)
    """
    stat, p_value = shapiro(scores)
    is_normal = p_value > 0.05
    return p_value, is_normal, stat
