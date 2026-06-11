import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Model values

MODEL_CONFIG = {
    "BASELINE_PRICE": 40386.64,
    "MARKET_NOISE_SCALE": 45155.56, # Residual Standard Error
    "WEIGHTS": {
        "Rooms": 16252.76,
        "Garages": 25594.13,
        "Driveway": 5766.77,
        "House_Age": -506.64
    },
    "TIER_WEIGHTS": {
        "Budget": -10768.94,
        "Average": 0.00,
        "Premium": 65201.46
    },
    "PROPERTY_TYPE_WEIGHTS": {
        "Detached": 0.00,
        "Semi-Detached": -10443.59,  # Mapped from TwnhsE
        "Terraced": -35294.54,       # Mapped from Twnhs
        "Flat": -54091.89  # Mapped from Duplex
    }
}

# User input section

print("Property Valuation & Market Dynamics Simulator")

# Structural inputs
user_rooms = int(input("Enter number of rooms/bedrooms: "))
user_garages = int(input("Enter garage size (0 for none, 1 for single, 2 for double): "))
user_driveway = int(input("Has private driveway? (1 for yes, 0 for no): "))
user_age = int(input("Enter property age in years: "))

# Categorical inputs
print("\nProperty Types: Detached, Semi-Detached, Terraced, Flat")
user_bldg = input("Select property type: ").strip()
print("\nLocation Type: Budget, Average, Premium:")
user_tier = input("Select location type: ").strip()

# Valuation calculation

user_predicted_price = (
    MODEL_CONFIG["BASELINE_PRICE"] +
    (user_rooms * MODEL_CONFIG["WEIGHTS"]["Rooms"]) +
    (user_garages * MODEL_CONFIG["WEIGHTS"]["Garages"]) +
    (user_driveway * MODEL_CONFIG["WEIGHTS"]["Driveway"]) +
    (user_age * MODEL_CONFIG["WEIGHTS"]["House_Age"]) +
    MODEL_CONFIG["TIER_WEIGHTS"][user_tier] +
    MODEL_CONFIG["PROPERTY_TYPE_WEIGHTS"][user_bldg]
)

# Synthetic Database Generator

tiers = ['Budget', 'Average', 'Premium'] # location tiers grouping various neighbourhoods based on their mean price
properties_per_tier = 1000 # Generate a local comparison market of 3,000 properties evenly split between the location tiers (budget, average, premium)
all_market_data = [] #create dictionary for storing generated properties

property_types = list(MODEL_CONFIG["PROPERTY_TYPE_WEIGHTS"].keys())

for tier in tiers:
    for _ in range(properties_per_tier):
        # pick a building type
        bldg_type = np.random.choice(property_types, p=[0.2, 0.3, 0.3, 0.2]) # probabilty of each building type

        # correlate attributes to the builidng type to avoid unrealistic builds
        if bldg_type == "Flat":
            fake_rooms = np.random.randint(1, 4)       # flats: 1-3 rooms
            fake_garages = np.random.choice([0, 1], p=[0.9, 0.1]) #less chance of having a garage or driveway
            fake_driveway = np.random.choice([0, 1], p=[0.8, 0.2])
        elif bldg_type == "Terraced":
            fake_rooms = np.random.randint(2, 5)       # terraced: 2-4 rooms
            fake_garages = np.random.choice([0, 1], p=[0.7, 0.3])
            fake_driveway = np.random.choice([0, 1], p=[0.5, 0.5])
        elif bldg_type == "Semi-Detached":
            fake_rooms = np.random.randint(3, 6)       # semi: 3-5 rooms
            fake_garages = np.random.randint(0, 3)
            fake_driveway = np.random.choice([0, 1], p=[0.2, 0.8])
        else: # (Detached)
            fake_rooms = np.random.randint(4, 8)       # detached: 4-7 rooms
            fake_garages = np.random.randint(1, 4)
            fake_driveway = np.random.choice([0, 1], p=[0.05, 0.95])

        fake_age = np.random.randint(1, 80)


        structural_value = (
            (fake_rooms * MODEL_CONFIG["WEIGHTS"]["Rooms"]) +
            (fake_garages * MODEL_CONFIG["WEIGHTS"]["Garages"]) +
            (fake_driveway * MODEL_CONFIG["WEIGHTS"]["Driveway"]) +
            (fake_age * MODEL_CONFIG["WEIGHTS"]["House_Age"]) +
            MODEL_CONFIG["PROPERTY_TYPE_WEIGHTS"][bldg_type]
        )

        # apply noise from the RSE and calculate final price
        noise = np.random.normal(loc=0, scale=MODEL_CONFIG["MARKET_NOISE_SCALE"])
        final_price = MODEL_CONFIG["BASELINE_PRICE"] + MODEL_CONFIG["TIER_WEIGHTS"][tier] + structural_value + noise

        all_market_data.append({ #add each generated property to dictionary with price and location
            'Price': final_price,
            'Location_Tier': tier
        })

# turn synthetic database into a pandas dataframe
city_df = pd.DataFrame(all_market_data)

# Cross tier statistical calculations

# Overall city stats
city_mean = city_df['Price'].mean() #average price of homes
city_std = city_df['Price'].std() #standard deviation of homes (measure of variation/spread in prices)
city_sem = city_std / np.sqrt(len(city_df)) #standard error of the mean (estimate of how much the sample mean is likely to shift between random samples)
city_z_score = (user_predicted_price - city_mean) / city_std #Z-Score shows the number of standard deviations the user price is away from the city average (95% usually fall within -2 and +2)

# Specific chosen tier stats
tier_subset = city_df[city_df['Location_Tier'] == user_tier]['Price']
tier_mean = tier_subset.mean()
tier_std = tier_subset.std()
tier_sem = tier_std / np.sqrt(len(tier_subset))
tier_z_score = (user_predicted_price - tier_mean) / tier_std

# Calculate means for other tiers for comparison
tier_means = city_df.groupby('Location_Tier')['Price'].mean()


# Statistical Reporting
print("\nAnalytical Report")
print(f"Your property valuation:  ${user_predicted_price:,.2f} ({user_tier}) Tier")
print(f"Local market for ({user_tier}) tier report:")
print(f"Local tier mean: ${tier_mean:,.2f}")
print(f"Local standard deviation: ${tier_std:,.2f}")
print(f"Local standard error: ${tier_sem:,.2f}")
print(f"Your local Z-Score: {tier_z_score:.4f}")

print("\nFull city report:")
print(f"City mean: ${city_mean:,.2f}")
print(f"City standard deviation: ${city_std:,.2f}")
print(f"City standard error: ${city_sem:,.2f}")
print(f"Your city-wide Z-Score: {city_z_score:.4f}")


# Matplolib dashboard
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Multi-tier distribution density plot
for tier_name in tiers: #loops through the different tiers
  subset = city_df[city_df['Location_Tier'] == tier_name]['Price'] #filters the main database to the specific tier and the price column
  sns.kdeplot(subset, label=f"{tier_name} Tier Market", fill=True, alpha=0.2, ax=ax1)

#benchmark lines to highlight where users property value and overall mean sits
ax1.axvline(user_predicted_price, color="red", linestyle="--", linewidth=2.5, label=f"Your property (${user_predicted_price:,.0f})")
ax1.axvline(city_mean, color="black", linestyle=":", linewidth=1.5, label=f"Total city average (${city_mean:,.0f})")

ax1.set_title("Property Location on City-wide Regional Market Curves", fontsize=12, fontweight="bold")
ax1.set_xlabel("Property Value ($)")
ax1.set_ylabel("Density")
ax1.legend()

# Plot 2: Market group comparisons showing the individual tier distribution and their means and outliers.

sns.boxplot(x='Location_Tier', y='Price', data=city_df, order=tiers, palette="pastel", ax=ax2, width=0.5)
ax2.scatter(x=user_tier, y=user_predicted_price, color='red', s=200, marker='*', zorder=5, label=f"Your Asset Position")

ax2.set_title("Property Asset Placement vs Market Segments", fontsize=12, fontweight="bold")
ax2.set_xlabel("Market Tier Location")
ax2.set_ylabel("Property Value ($)")
ax2.legend()

plt.tight_layout()
plt.show()
