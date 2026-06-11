# Property-Market-Simulator


This project looked at creating a visual and interactive property value calculator with a synthetic market simulation to go alongside the users result to allow for an analysis between the results.
This was done by downloading and engineering the dataset which can be seen in the 'ames_housing_dataset_manipulation' before writing the interactive application in 'Simulator_app'

## Data Engineering 
The dataset came from the Ames Housing Dataset from Kaggle (https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)
The dataset was filtered into specific columns that were to be included in the project ('SalePrice', 'Neighborhood', 'Year Built', 'TotRms AbvGrd', 'Bldg Type', 'Garage Cars', 'Paved Drive')
Year built was turned into house age by subtracting it from 2010 (the last year the data was collected). 
Neighborhoods were then sorted by their mean property price and merged into 3 groups of either budget (<$140,000), average ($140,000-$220,000), or premium (>$220,000).
Structural classifications were remapped to fit a UK based terminology ('1Fam' -> Detached, 'TwnhsE' -> Semi-Detached, 'TwnHs' -> Terraced, 'Duplex / 2fmCon' -> Flat).
Categorical variables such as 'Bldg Type' and 'Location Tier' were transformed using one hot encoding with one column dropped to be used as the baseline.

## Predicted Machine Learning Model
The feature matrix is split into independent features (everything but the sale price) and dependent target value (sale price)
A linear regression model is used to extract the true base intercept and coefficients:

Baseline Price (Intercept): $40,386.64

Attribute Weights (Coefficients):

TotRms AbvGrd: $16,252.76

Garage Cars: $25,594.13

Paved Drive: $5,766.77

House Age: $-506.64

Bldg Type_2fmCon: $-18,199.37

Bldg Type_Duplex: $-54,091.89

Bldg Type_Twnhs: $-35,294.54

Bldg Type_TwnhsE: $-10,443.59

Location Tier_Budget Tier: $-10,768.94

Location Tier_Premium Tier: $65,201.46

The model then finds the standard deviation ($79,873) and residual standard error ($45,155)

# Interactive Simulation

The application layer takes user property inputs through terminal prompt and creates a synthetic database and triggers an analysis of both

## Valuation Interface
Using the weights from the configuration matrix, the app calculates the expected price of the users custom house:
user_predicted_price = (
    MODEL_CONFIG["BASELINE_PRICE"] +
    (user_rooms * MODEL_CONFIG["WEIGHTS"]["Rooms"]) +
    (user_garages * MODEL_CONFIG["WEIGHTS"]["Garages"]) +
    (user_driveway * MODEL_CONFIG["WEIGHTS"]["Driveway"]) +
    (user_age * MODEL_CONFIG["WEIGHTS"]["House_Age"]) +
    MODEL_CONFIG["TIER_WEIGHTS"][user_tier] +
    MODEL_CONFIG["PROPERTY_TYPE_WEIGHTS"][user_bldg]
)

## Synthetic Database Generation
The script simulates a real estate market by constructing a city wide market of 3,000 properties split equally into the three tiers of budget, average, and premium.
Each individual property is assigned unique structural attributes through probability based decisions (so that structural attributes more closely match building types, ie. a flat being less likely to have a garage than a detached house).
This is then used to calculate the property value before adding a Gaussian noise which is scaled to the residual standard error to simulate market variation and anomalies.

# Statistics

The users house is evaluated against their targeted subset tier and the city wide market simultaneously using a statistical summary dashboard which compares mean price, standard deviation, standard error of the mean, and dual z-score comparisons.
<img width="405" height="237" alt="image" src="https://github.com/user-attachments/assets/4cb7322b-1a2c-4702-8c10-ad0534283150" />

# Visual Reporting

Matplotlib was used for visualising the report with
1. A multi-tier density analysis which plots three distinct overlapping bell curves representing value distribution of each location tier, with a vertical dashed marker to represent the users property value and total city average.
2. A market segment boxplot analysis showing boxplots for each location tier to demonstrate median bands and market outliers, with a marker indicating the users absolute asset placement within its chosen location tier.
<img width="1584" height="569" alt="image" src="https://github.com/user-attachments/assets/bf77e6dd-1f61-4152-8c4c-947cb601e057" />




# Limitations & Future Work

While a linear regression model provides an interpretable baseline for attribute weights it comes with the limitations of calculating negative property values
Because the model combines features additively, a property with multiple compounding penalty coefficients can force the output to be below 0 which is not a realistic output.
Future work could focus on:
1. Switching to a log-linear model: Instead of predicting raw price directly, the model can be configured to predict the logarithm of the price which prevents the output from being a negative number
2. Changing fixed adjustments to percentage based impacts: Rather than flat hardcoded values such as how a garage adds $25,594 to a house's value, these values would be turned into percentages, such as adding a dynamic 15% increase to the value for example, allowing the tool to automatically scale the value of a feature based on the size and location of a house (such as how a garage on a larger house may be worth more than one on a flat)
