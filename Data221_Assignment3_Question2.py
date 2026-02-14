import matplotlib.pyplot as plt
import pandas as pd


crime_data = pd.read_csv("crime1.csv")
violentCrimesPerPop = crime_data[["ViolentCrimesPerPop"]]


#-----------------Histogram-----------------
plt.figure("Figure1")
plt.hist(violentCrimesPerPop, bins= 10, edgecolor = "white" )
plt.title("Histogram of Violent Crimes Per Population")
plt.xlabel("Ratio of Violent Crimes per Population")
plt.ylabel("Frequency")
plt.show()

#-----------------Box Plot-----------------
plt.figure("Figure2")
plt.boxplot(violentCrimesPerPop)
plt.title("Box Plot of Violent Crimes Per Population")
plt.xlabel("Ratio of Violent Crimes per Population")
plt.ylabel("Values")
plt.show()

#-----------------Questions-----------------
#After generating the plots, write comments in your code describing:
# • What the histogram shows about how the data values are spread
# • What the box plot shows about the median
# • Whether the box plot suggests the presence of outliers
# Your explanations should be 5–7 sentences total, written as comments.

#-----------------Answers-----------------
#The histogram shows the frequency of violent crimes per population ratios within specific ranges.
# It indicates a likely right-skewed distribution because the tail of the data is on the right side.
# The box plot supports this, with the median positioned closer to the bottom of the box, reflecting the same slight right skew.
# Additionally, the box plot suggests there are no outliers present in the data.