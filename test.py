# import numpy as np
# from scipy import stats

# d1 = np.random.random((100, 5))
# d2 = np.random.random((100, 5)) * 1.3 + 0.7

# t_statistic, p_values = stats.ttest_ind(d1, d2, axis=0)
# print("T-statistics:", t_statistic)
# print("P-values:", p_values)


import numpy as np
from hyppo.ksample import MMD

# Generate some example data
# Two different 100x2 datasets
data1 = np.random.randn(100, 5)
data2 = np.random.randn(100, 5) * 1.02 + 0.01  # Add an offset to make them different

# Create an instance of the MMD test class
mmd_test = MMD()

# Run the test
# The .test() method returns a tuple: (test_statistic, p_value)
stat, pvalue = mmd_test.test(data1, data2)

# Print the results
print(f"Test statistic: {stat}")
print(f"p-value: {pvalue}")

# Interpret the result
alpha = 0.05
if pvalue < alpha:
    print("The null hypothesis is rejected: The two distributions are different.")
else:
    print("The null hypothesis is not rejected: The two distributions are likely the same.")
