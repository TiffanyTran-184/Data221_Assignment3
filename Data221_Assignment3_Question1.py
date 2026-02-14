import pandas as pd
import numpy as np


crime_data = pd.read_csv("crime1.csv")
violentCrimesPerPop = crime_data.loc[:,["ViolentCrimesPerPop"]]
mean = np.mean(violentCrimesPerPop)
median = np.median(violentCrimesPerPop)
stardard_deviation = np.std(violentCrimesPerPop)
minimum_value = np.min(violentCrimesPerPop)
maximum_value = np.max(violentCrimesPerPop)

print("The mean of the crime data is: ", mean)
print("The median of the crime data is: ", median)
print("The minimum value of the crime data is: ", minimum_value)
print("The maximum value of the crime data is: ", maximum_value)

# Question 1: Compare the mean and median. Does the distribution look symmetric or skewed? Explain briefly
# Since the mean value is greater than the median value (0.441191 > 0.39), the distribution is likely right-skewed

#Question 2:  If there are extreme values (very large or very small), which statistic is more affected: mean or median? Explain why
#If there are extreme values, I believe the mean value is more affected because it uses all data values in its calculation.
#The median is more robust since it depends only on the middle value of the sorted data.