import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

# Define global variables
data_frame = None

# Reading the data set and extracting GOP for year 3, calculating and printing the 95% confidence interval
def read_dataset():
    global data_frame
    file_path = 'Dataset.xlsx'
    data_frame = pd.read_excel(file_path, dtype={'NIC_CODE': str})

    # Handle missing data
    data_frame.dropna(subset=['GOP_Year3'], inplace=True)

    # Check the data type and convert them if necessary
    data_frame['GOP_Year3'] = pd.to_numeric(data_frame['GOP_Year3'], errors='coerce')

    gop_year_3 = data_frame['GOP_Year3']
    confidence_interval = stats.norm.interval(0.95, loc=np.mean(gop_year_3), scale=stats.sem(gop_year_3))
    print("\nQuestion 1:")
    print("95% Confidence Interval for GOP_Year3:")
    print(confidence_interval)


# Define and calculate performance measures
def performance_measures():
    global data_frame
    # Define two performance measures
    data_frame['GOP_Per_Employee'] = data_frame['GOP_Year3'] / data_frame['EMP_TOTAL']
    data_frame['Asset_Utilization_Ratio'] = data_frame['GOP_Year3'] / data_frame['MKT_VAL_FA']

    # Handle missing data in 'GOP_Per_Employee' and 'Asset_Utilization_Ratio'
    data_frame.dropna(subset=['GOP_Per_Employee', 'Asset_Utilization_Ratio'], inplace=True)

    # Check data type and convert if necessary
    data_frame['GOP_Per_Employee'] = pd.to_numeric(data_frame['GOP_Per_Employee'], errors='coerce')
    data_frame['Asset_Utilization_Ratio'] = pd.to_numeric(data_frame['Asset_Utilization_Ratio'], errors='coerce')

    # Explain the measures
    measure1_desc = "GOP Per Employee is a measure of profitability, indicating the percentage of revenue that is profit."
    measure2_desc = "Asset_Utilization_Ratio measures the efficiency of units in terms of output per employee."

    print("\n\n\nQuestion 2:")
    print("\n1. ", measure1_desc)
    print("\n2. ", measure2_desc)
    # Calculate 99% confidence intervals for both performance measures
    confidence_interval_measure1 = stats.norm.interval(0.99, loc=np.mean(data_frame['GOP_Per_Employee']),
                                                       scale=stats.sem(data_frame['GOP_Per_Employee']))
    confidence_interval_measure2 = stats.norm.interval(0.99, loc=np.mean(data_frame['Asset_Utilization_Ratio']),
                                                       scale=stats.sem(data_frame['Asset_Utilization_Ratio']))

    print("\n\n\nQuestion 3:")
    print("\n99% Confidence Interval for GOP_Per_Employee:")
    print(confidence_interval_measure1)
    print("\n99% Confidence Interval for Asset_Utilization_Ratio:")
    print(confidence_interval_measure2)


def sssbe_unit_performance():
    global data_frame
    # a. Probability that a firm is a SSSBE unit
    probability_sssbe = len(data_frame[data_frame['UNIT_TYPE'] == 2]) / len(data_frame)
    print("\n\n\nQuestion 4:")
    print("\nProbability that a firm is a SSSBE unit: ", probability_sssbe)

    # b. Calculate the average GOP_Per_Employee
    average_measure1 = np.mean(data_frame['GOP_Per_Employee'])
    # Probability that a firm is GOOD in performance
    probability_good_performance = len(data_frame[data_frame['GOP_Per_Employee'] > average_measure1]) / len(data_frame)
    print("\nProbability that firm is Good in Performance:", (probability_good_performance))

    # c. Probability that a firm is a SSSBE Unit and ALSO GOOD in performance
    probability_sssbe_and_good = len(
        data_frame[(data_frame['UNIT_TYPE'] == 2) & (data_frame['GOP_Per_Employee'] > average_measure1)]) / len(data_frame)
    print("\nProbability that a firm is both a SSSBE unit and also GOOD in Performance:", (probability_sssbe_and_good))

    # d. Conclusion on SSSBE unit performance
    if probability_sssbe_and_good > probability_good_performance:
        print("\nSSSBE units have a higher probability of being GOOD in performance.")
    else:
        print("\nSSSBE units have a higher probability of being BAD in performance.")



def null_and_alternate_hypotheses():
    global data_frame
    # Performing the Null and alternate hypotheses
    null_hypothesis = "The population average of VOE_Year3 is 87,300."
    alternate_hypothesis = "The population average of VOE_Year3 is greater than 87,300."

    # Perform a one-sided t-test
    t_statistic, p_value = ttest_1samp(data_frame['VOE_Year3'], 87300)

    # Check if null hypothesis can be rejected or not
    print("\n\n\nQuestion 5:")
    if p_value < 0.05:
        print("\nReject the null hypothesis:", alternate_hypothesis)
    else:
        print("\nFail to reject the null hypothesis:", null_hypothesis)



def proportion_of_sssbe_in_population():
    global data_frame
    # Calculate the proportion of SSSBE units in the population
    proportion_sssbe = len(data_frame[data_frame['UNIT_TYPE'] == 1]) / len(data_frame)
    proportion_ssi = len(data_frame[data_frame['UNIT_TYPE'] == 2]) / len(data_frame)

    # Check if the proportion is less than 25%
    print("\n\n\nQuestion 6:")
    if proportion_sssbe < 0.25:
        print("\nRecommend providing special incentives for SSSBE units.")
    else:
        print("\nNo need for special incentives for SSSBE units based on the proportion.")

    if proportion_ssi < 0.25:
        print("\nRecommend providing special incentives for SSI units.")
    else:
        print("\nNo need for special incentives for SSI units based on the proportion.")

    if proportion_sssbe > proportion_ssi:
        print("\nThese incentives will be provided for SSSBE Units.")
    else:
        print("\nThese incentives will be provided for SSI units")




def gender_distribution():
    global data_frame
    # Create a contingency table
    contingency_table = pd.crosstab(data_frame['UNIT_TYPE'], data_frame['MAN_BY'])

    # Perform a chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)

    # Get the counts of males and females
    male_count = contingency_table[1].sum()
    female_count = contingency_table[2].sum()
    print("\n\n\nQuestion 7:")
    print("\nNumber of Males:", male_count)
    print("\nNumber of Females:", female_count)

    # Check the p-value to make a conclusion
    if p < 0.05:
        print("\nThere is evidence to suggest a significant difference in gender distribution.")
        print("p-value:", p)
    else:
        print("\nThere is no significant difference in gender distribution.")
        print("p-value:", p)


def sssbe_and_ssi():
    global data_frame
    # Separating data into SSSBE and SSI groups
    ssssbe_data = data_frame[data_frame['UNIT_TYPE'] == 1]
    ssi_data = data_frame[data_frame['UNIT_TYPE'] == 2]

    # Performing a t-test for GOP_Per_Employee
    t_statistic, p_value = ttest_ind(ssssbe_data['GOP_Per_Employee'], ssi_data['GOP_Per_Employee'])

    # Checking if there is a significant difference
    print("\n\n\nQuestion 8:")
    if p_value < 0.05:
        print("\nThere is a significant difference in performance between SSSBE and SSI.")
        if t_statistic < 0:
            print("\nSSSBE has better performance.")
        else:
            print("\nSSI has better performance.")
    else:
        print("\nThere is no significant difference in performance between SSSBE and SSI.")


    print("\n\n\nQuestion 9:")
    # Calculating the proportion of better performing units for both groups based on measure 1
    average_measure1_sssbe = np.mean(ssssbe_data['GOP_Per_Employee'])
    average_measure1_ssi = np.mean(ssi_data['GOP_Per_Employee'])

    # Define better performing units based on measure 1
    ssssbe_better = ssssbe_data['GOP_Per_Employee'] > average_measure1_sssbe
    ssi_better = ssi_data['GOP_Per_Employee'] > average_measure1_ssi

    # Calculate the proportions
    proportion_better_sssbe = sum(ssssbe_better) / len(ssssbe_data)
    proportion_better_ssi = sum(ssi_better) / len(ssi_data)
    print("\nProportion of units performing better than SSSBE:")
    print(proportion_better_sssbe)
    print("\nProportion of units performing better than SSI:")
    print(proportion_better_ssi)

    # Perform a hypothesis test to compare proportions for measure 1
    # Define better performing units based on measure 1
    ssssbe_better_measure1 = ssssbe_data['GOP_Per_Employee'] > average_measure1_sssbe
    ssi_better_measure1 = ssi_data['GOP_Per_Employee'] > average_measure1_ssi

    # Check if there are valid observations for measure 1
    if ssssbe_better_measure1.any() and ssi_better_measure1.any():
        # Create a contingency table for measure 1
        contingency_table_measure1 = pd.crosstab(ssssbe_better_measure1, ssi_better_measure1)

        # Check if the contingency table has data
        if contingency_table_measure1.values.sum() > 0:
            # Perform a chi-squared test for measure 1
            chi2_measure1, p_measure1, _, _ = chi2_contingency(contingency_table_measure1)

            # Check p-value and draw conclusions
            alpha = 0.05  # Significance level

            if p_measure1 < alpha:
                print("\nFor GOP_Per_Employee:")
                print("\nReject the null hypothesis: There is a significant difference in proportions.")
            else:
                print("\nFor GOP_Per_Employee:")
                print("\nFail to reject the null hypothesis: There is no significant difference in proportions.")
        else:
            print("\nNo valid observations for GOP_Per_Employee.")
    else:
        print("\nNo valid observations for GOP_Per_Employee.")

    # Calculate the proportion of better performing units for both groups based on measure 2
    average_measure2_sssbe = np.mean(ssssbe_data['Asset_Utilization_Ratio'])
    average_measure2_ssi = np.mean(ssi_data['Asset_Utilization_Ratio'])
    # Perform a hypothesis test (e.g., chi-squared test) to compare proportions for measure 2
    # Define better performing units based on measure 2
    ssssbe_better_measure2 = ssssbe_data['Asset_Utilization_Ratio'] > average_measure2_sssbe
    ssi_better_measure2 = ssi_data['Asset_Utilization_Ratio'] > average_measure2_ssi

    # Check if there are valid observations for measure 2
    if ssssbe_better_measure2.any() and ssi_better_measure2.any():
        # Create a contingency table for measure 2
        contingency_table_measure2 = pd.crosstab(ssssbe_better_measure2, ssi_better_measure2)

        # Check if the contingency table has data
        if contingency_table_measure2.values.sum() > 0:
            # Perform a chi-squared test for measure 2
            chi2_measure2, p_measure2, _, _ = chi2_contingency(contingency_table_measure2)

            # Check p-value and draw conclusions
            alpha = 0.05  # Significance level

            if p_measure2 < alpha:
                print("\nFor Asset_Utilization_Ratio:")
                print("Reject the null hypothesis: There is a significant difference in proportions.")
            else:
                print("\nFor Asset_Utilization_Ratio:")
                print("Fail to reject the null hypothesis: There is no significant difference in proportions.")
        else:
            print("\nNo valid observations for Asset_Utilization_Ratio.")
    else:
        print("\nNo valid observations for Asset_Utilization_Ratio.")


def plot_histogram():
    global data_frame

    # Calculate the 99th percentile to exclude extreme outliers for both columns
    percentile_99_margin = np.percentile(data_frame['GOP_Per_Employee'], 99)
    percentile_99_Asset_Utilization_Ratio = np.percentile(data_frame['Asset_Utilization_Ratio'], 99)

    # Create a finite range for the histogram for both columns
    min_value_margin = data_frame['GOP_Per_Employee'].min()
    max_value_margin = percentile_99_margin
    min_value_Asset_Utilization_Ratio = data_frame['Asset_Utilization_Ratio'].min()
    max_value_Asset_Utilization_Ratio = percentile_99_Asset_Utilization_Ratio

    # Define the number of bins
    num_bins = 20

    # Create an array of evenly spaced bin edges for both columns
    bins_margin = np.linspace(min_value_margin, max_value_margin, num_bins)
    bins_Asset_Utilization_Ratio = np.linspace(min_value_Asset_Utilization_Ratio, max_value_Asset_Utilization_Ratio, num_bins)


    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(data_frame['GOP_Per_Employee'], bins=bins_margin, alpha=0.5, color='blue')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of GOP Per Employee')

    plt.subplot(1, 2, 2)
    plt.hist(data_frame['Asset_Utilization_Ratio'], bins=bins_Asset_Utilization_Ratio, alpha=0.5, color='green')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Asset Utilization Ratio')

    plt.tight_layout()
    plt.show()


read_dataset()
performance_measures()
sssbe_unit_performance()
null_and_alternate_hypotheses()
proportion_of_sssbe_in_population()
gender_distribution()
sssbe_and_ssi()
plot_histogram()