import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path

# Use non-interactive backend for script plotting
matplotlib.use("Agg")
sns.set(style="whitegrid")

# ---------------------------
# Configuration / constants
# ---------------------------
sample_fraction = 0.01  # <-- keep this value; used to scale up counts later
scale_factor = 1.0 / sample_fraction
data_dir = Path("/Users/bhavukmathur/Desktop/Bhavuk_Mathur/Education/upgrad/assign1/trip_records")
shapefile_path = Path("/Users/bhavukmathur/Desktop/Bhavuk_Mathur/Education/upgrad/assign1/taxi_zones/taxi_zones.shp")

# Output folder
out_dir = data_dir
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load & stratified sample
# ---------------------------
dfs = []
for month in range(1, 13):
    file_name = data_dir / f"2023-{month}.parquet"
    print(f"Loading {file_name}")
    month_df = pd.read_parquet(file_name)
    # parse datetimes
    month_df['tpep_pickup_datetime'] = pd.to_datetime(month_df['tpep_pickup_datetime'])
    month_df['tpep_dropoff_datetime'] = pd.to_datetime(month_df['tpep_dropoff_datetime'])
    month_df['pickup_date'] = month_df['tpep_pickup_datetime'].dt.date
    month_df['pickup_hour'] = month_df['tpep_pickup_datetime'].dt.hour

    # stratified sample per date-hour
    sampled = (month_df.groupby(['pickup_date', 'pickup_hour'], group_keys=False)
                       .apply(lambda x: x.sample(frac=sample_fraction, random_state=42) 
                              if (len(x) > 0) else x))
    dfs.append(sampled)

sampled_data = pd.concat(dfs, ignore_index=True)
print("Sampled rows:", sampled_data.shape)

# Save sampled snapshot
sample_csv = out_dir / "sample_2023.csv"
sample_parquet = out_dir / "sample_2023.parquet"
sampled_data.to_csv(sample_csv, index=False)
sampled_data.to_parquet(sample_parquet, index=False, engine="pyarrow")
print("Saved sampled files.")

# ---------------------------
# Load sampled file for cleaning & analysis
# ---------------------------
df = pd.read_parquet(sample_parquet)

# basic cleanup
df.reset_index(drop=True, inplace=True)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# combine duplicate airport fee columns if present
if "airport_fee" in df.columns and "Airport_fee" in df.columns:
    df["airport_fee"] = df["airport_fee"].fillna(0) + df["Airport_fee"].fillna(0)
    df.drop(columns=["Airport_fee"], inplace=True)

# clip monetary negatives to 0
monetary_columns = [
    "fare_amount", "extra", "mta_tax", "tip_amount",
    "tolls_amount", "improvement_surcharge",
    "total_amount", "congestion_surcharge", "airport_fee"
]
for col in monetary_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

# impute passenger_count, RatecodeID, congestion_surcharge
if "passenger_count" in df.columns:
    df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")
    df["passenger_count"].fillna(df["passenger_count"].median(), inplace=True)
if "RatecodeID" in df.columns:
    df["RatecodeID"] = pd.to_numeric(df["RatecodeID"], errors="coerce")
    if not df["RatecodeID"].mode().empty:
        df["RatecodeID"].fillna(df["RatecodeID"].mode()[0], inplace=True)
if "congestion_surcharge" in df.columns:
    df["congestion_surcharge"].fillna(0, inplace=True)

# remaining missing: numeric -> 0, object -> 'Unknown'
for col in df.columns:
    if df[col].dtype.kind in "iufc":  # numeric
        df[col].fillna(0, inplace=True)
    else:
        df[col].fillna("Unknown", inplace=True)

# Outlier handling (drop clearly invalid rows)
df = df[~((df.get("trip_distance", 0) < 0.1) & (df.get("fare_amount", 0) > 300))]
df = df[~((df.get("trip_distance", 0) == 0) & (df.get("fare_amount", 0) == 0) &
          (df.get("PULocationID", -1) != df.get("DOLocationID", -1)))]
df = df[df.get("trip_distance", 0) <= 250]
df = df[df.get("passenger_count", 0) <= 6]

# Keep valid payment types 1-4
if "payment_type" in df.columns:
    df["payment_type"] = pd.to_numeric(df["payment_type"], errors="coerce").fillna(0).astype(int)
    df = df[df["payment_type"].isin([1,2,3,4])]

# ensure datetimes parsed
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# derive time columns
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek  # 0=Mon
df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
df['is_weekend'] = df['pickup_dayofweek'].isin([5,6])

# compute trip duration in minutes and average speed (mph)
df['trip_duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()/60.0
# avoid zero/negative durations
df['trip_duration_min'] = pd.to_numeric(df['trip_duration_min'], errors='coerce')
df = df[df['trip_duration_min'] > 0]
# speed: miles per hour
df['trip_distance'] = pd.to_numeric(df['trip_distance'], errors='coerce').fillna(0)
df['speed_mph'] = df['trip_distance'] / (df['trip_duration_min'] / 60.0)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Merge zone data (shapefile) and prepare mapping for zone names
gdf_zones = gpd.read_file(shapefile_path)
gdf_zones['LocationID'] = pd.to_numeric(gdf_zones['LocationID'], errors='coerce')
gdf_zones = gdf_zones[['LocationID', 'zone', 'borough', 'geometry']]

# Map zone names onto df for pickups and dropoffs (optional columns)
df['PULocationID'] = pd.to_numeric(df['PULocationID'], errors='coerce').astype('Int64')
df['DOLocationID'] = pd.to_numeric(df['DOLocationID'], errors='coerce').astype('Int64')

# create mapping dicts
locid_to_zone = gdf_zones.set_index('LocationID')['zone'].to_dict()
locid_to_borough = gdf_zones.set_index('LocationID')['borough'].to_dict()

df['PU_zone'] = df['PULocationID'].map(locid_to_zone).fillna("Unknown")
df['PU_borough'] = df['PULocationID'].map(locid_to_borough).fillna("Unknown")
df['DO_zone'] = df['DOLocationID'].map(locid_to_zone).fillna("Unknown")
df['DO_borough'] = df['DOLocationID'].map(locid_to_borough).fillna("Unknown")

# ---------------------------
# 1. Analyze variations by time of day and location (bottlenecks)
#    We'll compute average speed per route and per time-of-day bucket
# ---------------------------
# define time-of-day buckets
def tod_bucket(hour):
    if 5 <= hour < 9:
        return "Morning_peak"
    if 9 <= hour < 16:
        return "Midday"
    if 16 <= hour < 20:
        return "Evening_peak"
    if 20 <= hour < 23:
        return "Evening"
    return "Night"  # 23-5

df['tod_bucket'] = df['pickup_hour'].apply(tod_bucket)

# define route id
df['route'] = df['PULocationID'].astype(str) + "_" + df['DOLocationID'].astype(str)

# avg speed per route x tod
route_speed = (df.groupby(['route', 'tod_bucket'])
                 .agg(avg_speed_mph=('speed_mph','mean'),
                      trip_count=('speed_mph','count'),
                      median_speed=('speed_mph','median'))
                 .reset_index())
# filter routes with at least some number of trips to be meaningful (e.g., >=5)
route_speed_filtered = route_speed[route_speed['trip_count'] >= 5]

# identify slowest routes per bucket (lowest avg_speed)
slowest_by_bucket = (route_speed_filtered.sort_values(['tod_bucket','avg_speed_mph'])
                     .groupby('tod_bucket').head(10))

print("\nSlowest routes by time-of-day bucket (sample):")
print(slowest_by_bucket[['tod_bucket','route','avg_speed_mph','trip_count']].head(20))

# ---------------------------
# 2. Find routes with slowest speeds at different times (already computed)
#     Save to CSV
# ---------------------------
slowest_by_bucket.to_csv(out_dir / "slowest_routes_by_tod.csv", index=False)

# ---------------------------
# 3. Visualize number of trips per hour and busiest hour
# ---------------------------
trips_per_hour = df.groupby('pickup_hour').size().reset_index(name='trip_count')
trips_per_hour['scaled_trip_count'] = (trips_per_hour['trip_count'] * scale_factor).round().astype(int)

plt.figure(figsize=(10,5))
sns.lineplot(data=trips_per_hour, x='pickup_hour', y='scaled_trip_count', marker='o')
plt.title("Scaled number of trips per hour (scaled by 1/sample_fraction)")
plt.xlabel("Hour of day")
plt.ylabel("Estimated number of trips (scaled)")
plt.xticks(range(0,24))
plt.grid(True)
plt.savefig(out_dir / "trips_per_hour_scaled.png")
plt.close()

busiest_hour_row = trips_per_hour.loc[trips_per_hour['trip_count'].idxmax()]
busiest_hour = int(busiest_hour_row['pickup_hour'])
busiest_hour_scaled = int(busiest_hour_row['scaled_trip_count'])
print(f"\nBusiest hour (sample): {busiest_hour} (scaled est. trips: {busiest_hour_scaled})")

# ---------------------------
# 4. Scale up the number of trips
#    done above with 'scaled_trip_count' = count * scale_factor
#    We'll also compute scaled trip counts per zone
# ---------------------------
# pickups per PU zone
pickups_per_zone = df.groupby('PULocationID').size().reset_index(name='sample_trip_count')
pickups_per_zone['estimated_trips'] = (pickups_per_zone['sample_trip_count'] * scale_factor).round().astype(int)

# dropoffs per zone
dropoffs_per_zone = df.groupby('DOLocationID').size().reset_index(name='sample_dropoff_count')
dropoffs_per_zone['estimated_dropoffs'] = (dropoffs_per_zone['sample_dropoff_count'] * scale_factor).round().astype(int)

# merge into gdf_zones
gdf_merged = gdf_zones.merge(pickups_per_zone, left_on='LocationID', right_on='PULocationID', how='left')
gdf_merged = gdf_merged.merge(dropoffs_per_zone, left_on='LocationID', right_on='DOLocationID', how='left')
gdf_merged['sample_trip_count'] = gdf_merged['sample_trip_count'].fillna(0).astype(int)
gdf_merged['sample_dropoff_count'] = gdf_merged['sample_dropoff_count'].fillna(0).astype(int)
gdf_merged['estimated_trips'] = gdf_merged['estimated_trips'].fillna(0).astype(int)
gdf_merged['estimated_dropoffs'] = gdf_merged['estimated_dropoffs'].fillna(0).astype(int)

# ---------------------------
# 6. Compare traffic trends for weekdays and weekends
# ---------------------------
weekday_df = df[df['is_weekend'] == False]
weekend_df = df[df['is_weekend'] == True]

weekday_counts = weekday_df.groupby('pickup_hour').size().reset_index(name='count').set_index('pickup_hour')
weekend_counts = weekend_df.groupby('pickup_hour').size().reset_index(name='count').set_index('pickup_hour')

# scale
weekday_counts['scaled'] = (weekday_counts['count'] * scale_factor).round().astype(int)
weekend_counts['scaled'] = (weekend_counts['count'] * scale_factor).round().astype(int)

plt.figure(figsize=(10,6))
sns.lineplot(data=weekday_counts['scaled'], label='Weekday', marker='o')
sns.lineplot(data=weekend_counts['scaled'], label='Weekend', marker='o')
plt.title("Weekday vs Weekend: Scaled Trips per Hour")
plt.xlabel("Hour")
plt.ylabel("Estimated trips (scaled)")
plt.legend()
plt.savefig(out_dir / "weekday_vs_weekend_trips_per_hour.png")
plt.close()

# ---------------------------
# 7. Top 10 pickup and dropoff zones
# ---------------------------
top10_pickups = pickups_per_zone.sort_values('sample_trip_count', ascending=False).head(10)
top10_dropoffs = dropoffs_per_zone.sort_values('sample_dropoff_count', ascending=False).head(10)

# attach zone names
top10_pickups['zone'] = top10_pickups['PULocationID'].map(locid_to_zone)
top10_dropoffs['zone'] = top10_dropoffs['DOLocationID'].map(locid_to_zone)

print("\nTop 10 pickup zones (sample counts, estimated counts):")
print(top10_pickups[['PULocationID','zone','sample_trip_count','estimated_trips']])

print("\nTop 10 dropoff zones (sample counts, estimated counts):")
print(top10_dropoffs[['DOLocationID','zone','sample_dropoff_count','estimated_dropoffs']])

# ---------------------------
# 8. Ratio of pickups to dropoffs per zone
#    ratio = estimated_pickups / estimated_dropoffs (avoid div by zero)
# ---------------------------
zone_counts = pd.DataFrame({
    'LocationID': gdf_merged['LocationID'],
    'zone': gdf_merged['zone'],
    'estimated_pickups': gdf_merged['estimated_trips'],
    'estimated_dropoffs': gdf_merged['estimated_dropoffs']
})
zone_counts['estimated_dropoffs_adj'] = zone_counts['estimated_dropoffs'].replace({0: np.nan})
zone_counts['pickup_dropoff_ratio'] = zone_counts['estimated_pickups'] / zone_counts['estimated_dropoffs_adj']
# where dropoffs are zero, ratio will be NaN; fill with large number to indicate imbalance if desired
zone_counts['pickup_dropoff_ratio'] = zone_counts['pickup_dropoff_ratio'].replace([np.inf], np.nan)

top10_ratio = zone_counts.sort_values('pickup_dropoff_ratio', ascending=False).head(10)
bottom10_ratio = zone_counts.sort_values('pickup_dropoff_ratio', ascending=True).dropna().head(10)

print("\nTop 10 pickup/dropoff ratio (highest):")
print(top10_ratio[['LocationID','zone','estimated_pickups','estimated_dropoffs','pickup_dropoff_ratio']])

print("\nTop 10 pickup/dropoff ratio (lowest non-zero):")
print(bottom10_ratio[['LocationID','zone','estimated_pickups','estimated_dropoffs','pickup_dropoff_ratio']])

# ---------------------------
# 9. Zones with high pickup and dropoff traffic during night hours (23:00-05:00)
# ---------------------------
night_df = df[(df['pickup_hour'] >= 23) | (df['pickup_hour'] <= 5)]
night_pickups = night_df.groupby('PULocationID').size().reset_index(name='sample_night_pickups')
night_pickups['estimated_night_pickups'] = (night_pickups['sample_night_pickups'] * scale_factor).round().astype(int)
night_pickups = night_pickups.sort_values('sample_night_pickups', ascending=False)

night_dropoffs = df[(df['tpep_dropoff_datetime'].dt.hour >= 23) | (df['tpep_dropoff_datetime'].dt.hour <= 5)]
night_dropoffs = night_dropoffs.groupby('DOLocationID').size().reset_index(name='sample_night_dropoffs')
night_dropoffs['estimated_night_dropoffs'] = (night_dropoffs['sample_night_dropoffs'] * scale_factor).round().astype(int)
night_dropoffs = night_dropoffs.sort_values('sample_night_dropoffs', ascending=False)

# show top night zones
night_pickups['zone'] = night_pickups['PULocationID'].map(locid_to_zone)
night_dropoffs['zone'] = night_dropoffs['DOLocationID'].map(locid_to_zone)
print("\nTop night pickup zones (sample -> estimated):")
print(night_pickups.head(10)[['PULocationID','zone','sample_night_pickups','estimated_night_pickups']])

print("\nTop night dropoff zones (sample -> estimated):")
print(night_dropoffs.head(10)[['DOLocationID','zone','sample_night_dropoffs','estimated_night_dropoffs']])

# ---------------------------
# 10. Revenue share for nighttime vs daytime
# ---------------------------
df['is_night'] = df['pickup_hour'].apply(lambda h: True if (h >= 23 or h <= 5) else False)
revenue_by_night = df.groupby('is_night')['total_amount'].sum().reset_index()
revenue_by_night['scaled_revenue'] = revenue_by_night['total_amount'] * scale_factor
print("\nRevenue share (sample totals and scaled estimates):")
print(revenue_by_night)

# also compute proportion
total_rev = revenue_by_night['scaled_revenue'].sum()
revenue_by_night['proportion'] = revenue_by_night['scaled_revenue'] / total_rev
print("\nRevenue proportions (night vs day):")
print(revenue_by_night[['is_night','proportion']])

# ---------------------------
# Save/plot helpful outputs
# ---------------------------
# 1) save slowest routes by bucket (already saved)
# 2) save trips per hour (sample + scaled)
trips_per_hour.to_csv(out_dir / "trips_per_hour_scaled.csv", index=False)
trips_per_hour.to_csv(out_dir / "trips_per_hour_sample.csv", index=False)

# 3) save pickups/dropoffs per zone
pickups_per_zone.to_csv(out_dir / "pickups_per_zone_sample.csv", index=False)
dropoffs_per_zone.to_csv(out_dir / "dropoffs_per_zone_sample.csv", index=False)
gdf_merged.to_file(out_dir / "zones_with_counts.geojson", driver="GeoJSON")

# 4) plot map of zones colored by estimated trips (uses gdf_merged created earlier)
fig, ax = plt.subplots(1,1, figsize=(14,12))
gdf_merged.plot(column='estimated_trips', cmap='OrRd', legend=True, ax=ax,
                legend_kwds={'label': "Estimated pickups (scaled)"},
                edgecolor='black', linewidth=0.3)
ax.set_title("Estimated Pickups per Zone (scaled by 1/sample_fraction)")
ax.axis('off')
plt.savefig(out_dir / "zones_estimated_pickups_map.png", bbox_inches='tight')
plt.close()

# 5) weekday vs weekend plot already saved earlier as weekday_vs_weekend_trips_per_hour.png

# ---------------------------
# Final save of cleaned sampled df (for transparency)
# ---------------------------
# convert object cols to string for safe parquet write
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str)

final_clean_csv = out_dir / "sample_2023_clean_full_pipeline.csv"
final_clean_parquet = out_dir / "sample_2023_clean_full_pipeline.parquet"
df.to_csv(final_clean_csv, index=False)
df.to_parquet(final_clean_parquet, index=False, engine="pyarrow")

print("\nAll outputs saved to", out_dir)


# ---------------------------
# PREP: fare per mile per passenger & tip percentage
# ---------------------------
# Avoid divide-by-zero: where trip_distance==0, set NaN for fare-per-mile
df['fare_per_mile'] = np.where(df['trip_distance'] > 0, df['fare_amount'] / df['trip_distance'], np.nan)
# fare per mile per passenger: divide further by passenger_count where passenger_count>0
df['fare_per_mile_per_passenger'] = np.where(
    (df['trip_distance'] > 0) & (df['passenger_count'] > 0),
    df['fare_amount'] / df['trip_distance'] / df['passenger_count'],
    np.nan
)

# Tip percentage (relative to fare_amount). Avoid divide by zero.
df['tip_pct'] = np.where(df['fare_amount'] > 0, (df['tip_amount'] / df['fare_amount']) * 100.0, np.nan)

# bucket time-of-day for comparisons
def tod_bucket(hour):
    if 5 <= hour < 9:
        return "Morning_peak"
    if 9 <= hour < 16:
        return "Midday"
    if 16 <= hour < 20:
        return "Evening_peak"
    if 20 <= hour < 23:
        return "Evening"
    return "Night"  # 23-5

df['tod_bucket'] = df['pickup_hour'].apply(tod_bucket)

# ---------------------------
# 1. Analyse fare per mile per passenger for different passenger counts
# ---------------------------
farepm_passenger = (df.groupby('passenger_count', as_index=False)
                     .agg(
                          count_trips=('fare_per_mile_per_passenger','count'),
                          mean_farepmpp=('fare_per_mile_per_passenger','mean'),
                          median_farepmpp=('fare_per_mile_per_passenger','median'),
                          std_farepmpp=('fare_per_mile_per_passenger','std')
                     )
                     .sort_values('passenger_count'))

farepm_passenger.to_csv(out_dir / "fare_per_mile_per_passenger_by_passenger_count.csv", index=False)
print("\nFare-per-mile-per-passenger by passenger_count:")
print(farepm_passenger.head(20))

plt.figure(figsize=(8,5))
sns.barplot(data=farepm_passenger[farepm_passenger['count_trips']>0],
            x='passenger_count', y='mean_farepmpp', palette='viridis')
plt.title("Mean Fare per Mile per Passenger by Passenger Count")
plt.xlabel("Passenger Count")
plt.ylabel("Mean Fare per Mile per Passenger ($/mile/passenger)")
plt.savefig(out_dir / "mean_farepmpp_by_passenger_count.png")
plt.close()

# ---------------------------
# 2. Compare avg fare per mile across days and times of day
# ---------------------------
# by day of week
farepm_by_day = (df.groupby('pickup_dayofweek', as_index=False)
                   .agg(mean_farepm=('fare_per_mile','mean'),
                        med_farepm=('fare_per_mile','median'),
                        trips=('fare_per_mile','count'))
                 .sort_values('pickup_dayofweek'))

farepm_by_day.to_csv(out_dir / "fare_per_mile_by_day.csv", index=False)
print("\nFare per mile by day of week:")
print(farepm_by_day)

plt.figure(figsize=(8,5))
sns.barplot(data=farepm_by_day, x='pickup_dayofweek', y='mean_farepm', palette='coolwarm')
plt.title("Mean Fare per Mile by Day of Week (0=Mon)")
plt.xlabel("Day of Week")
plt.ylabel("Mean Fare per Mile ($/mile)")
plt.savefig(out_dir / "mean_farepm_by_day.png")
plt.close()

# by time-of-day bucket
farepm_by_tod = (df.groupby('tod_bucket', as_index=False)
                   .agg(mean_farepm=('fare_per_mile','mean'),
                        med_farepm=('fare_per_mile','median'),
                        trips=('fare_per_mile','count')))

farepm_by_tod.to_csv(out_dir / "fare_per_mile_by_tod_bucket.csv", index=False)
print("\nFare per mile by time-of-day bucket:")
print(farepm_by_tod)

plt.figure(figsize=(8,5))
order = ["Morning_peak","Midday","Evening_peak","Evening","Night"]
sns.barplot(data=farepm_by_tod, x='tod_bucket', y='mean_farepm', order=order, palette='magma')
plt.title("Mean Fare per Mile by Time-of-day Bucket")
plt.xlabel("Time of Day")
plt.ylabel("Mean Fare per Mile ($/mile)")
plt.savefig(out_dir / "mean_farepm_by_tod.png")
plt.close()

# ---------------------------
# 3. Compare fare per mile for different vendors
# ---------------------------
if 'VendorID' in df.columns:
    farepm_by_vendor = (df.groupby('VendorID', as_index=False)
                         .agg(mean_farepm=('fare_per_mile','mean'),
                              med_farepm=('fare_per_mile','median'),
                              trips=('fare_per_mile','count')))
    farepm_by_vendor.to_csv(out_dir / "fare_per_mile_by_vendor.csv", index=False)
    print("\nFare per mile by Vendor:")
    print(farepm_by_vendor)

    plt.figure(figsize=(7,5))
    sns.barplot(data=farepm_by_vendor, x='VendorID', y='mean_farepm', palette='Set2')
    plt.title("Mean Fare per Mile by Vendor")
    plt.xlabel("VendorID")
    plt.ylabel("Mean Fare per Mile ($/mile)")
    plt.savefig(out_dir / "mean_farepm_by_vendor.png")
    plt.close()

# ---------------------------
# 4. Tiered fare-per-mile analysis by distance buckets
#    - <=2 miles
#    - >2 and <=5
#    - >5
# ---------------------------
def dist_bucket(d):
    if d <= 2:
        return "0-2"
    if d <= 5:
        return "2-5"
    return ">5"

df['dist_bucket'] = df['trip_distance'].apply(dist_bucket)

farepm_by_distbucket = (df.groupby('dist_bucket', as_index=False)
                          .agg(mean_farepm=('fare_per_mile','mean'),
                               median_farepm=('fare_per_mile','median'),
                               trips=('fare_per_mile','count')))
farepm_by_distbucket.to_csv(out_dir / "fare_per_mile_by_distance_bucket.csv", index=False)
print("\nFare per mile by distance bucket:")
print(farepm_by_distbucket)

plt.figure(figsize=(7,5))
sns.barplot(data=farepm_by_distbucket.sort_values('dist_bucket'), x='dist_bucket', y='mean_farepm', palette='viridis')
plt.title("Mean Fare per Mile by Distance Bucket")
plt.xlabel("Distance bucket (miles)")
plt.ylabel("Mean Fare per Mile ($/mile)")
plt.savefig(out_dir / "mean_farepm_by_distance_bucket.png")
plt.close()

# ---------------------------
# 5. Analyze tip percentages based on distances, passenger counts and pickup times
# ---------------------------
# tip vs distance: aggregate by distance buckets (use same dist_bucket)
tip_by_dist = (df.groupby('dist_bucket', as_index=False)
                 .agg(mean_tip_pct=('tip_pct','mean'),
                      median_tip_pct=('tip_pct','median'),
                      trips=('tip_pct','count')))
tip_by_dist.to_csv(out_dir / "tip_pct_by_distance_bucket.csv", index=False)
print("\nTip pct by distance bucket:")
print(tip_by_dist)

plt.figure(figsize=(8,5))
sns.boxplot(x='dist_bucket', y='tip_pct', data=df[df['tip_pct'].notna()], order=["0-2","2-5",">5"])
plt.title("Tip Percentage by Distance Bucket")
plt.xlabel("Distance bucket")
plt.ylabel("Tip %")
plt.savefig(out_dir / "tip_pct_box_by_distance_bucket.png")
plt.close()

# tip vs passenger_count
tip_by_pax = (df.groupby('passenger_count', as_index=False)
                .agg(mean_tip_pct=('tip_pct','mean'),
                     median_tip_pct=('tip_pct','median'),
                     trips=('tip_pct','count')))
tip_by_pax.to_csv(out_dir / "tip_pct_by_passenger_count.csv", index=False)
print("\nTip pct by passenger_count:")
print(tip_by_pax.head(15))

plt.figure(figsize=(10,5))
sns.lineplot(data=tip_by_pax, x='passenger_count', y='mean_tip_pct', marker='o')
plt.title("Mean Tip % by Passenger Count")
plt.xlabel("Passenger Count")
plt.ylabel("Mean Tip %")
plt.savefig(out_dir / "mean_tippct_by_passenger_count.png")
plt.close()

# tip vs time-of-day bucket
tip_by_tod = (df.groupby('tod_bucket', as_index=False)
                .agg(mean_tip_pct=('tip_pct','mean'),
                     median_tip_pct=('tip_pct','median'),
                     trips=('tip_pct','count')))
tip_by_tod.to_csv(out_dir / "tip_pct_by_tod_bucket.csv", index=False)
print("\nTip pct by time-of-day bucket:")
print(tip_by_tod)

plt.figure(figsize=(8,5))
sns.barplot(data=tip_by_tod, x='tod_bucket', y='mean_tip_pct', order=order)
plt.title("Mean Tip % by Time-of-day Bucket")
plt.xlabel("Time of Day")
plt.ylabel("Mean Tip %")
plt.savefig(out_dir / "mean_tippct_by_tod_bucket.png")
plt.close()

# ---------------------------
# 6. Compare trips with tip% < 10% to trips with tip% > 25%
# ---------------------------
df['tip_group'] = np.where(df['tip_pct'] < 10, '<10%', np.where(df['tip_pct'] > 25, '>25%', '10-25%'))

tipgroup_stats = (df.groupby('tip_group', as_index=False)
                   .agg(trips=('tip_pct','count'),
                        mean_fare=('fare_amount','mean'),
                        median_fare=('fare_amount','median'),
                        mean_dist=('trip_distance','mean'),
                        mean_passengers=('passenger_count','mean')))
tipgroup_stats.to_csv(out_dir / "tip_group_comparison.csv", index=False)
print("\nTip group comparison (<10%, 10-25%, >25%):")
print(tipgroup_stats)

plt.figure(figsize=(8,5))
sns.barplot(data=tipgroup_stats, x='tip_group', y='trips', palette='Set1')
plt.title("Trip counts by Tip Group")
plt.xlabel("Tip group")
plt.ylabel("Number of trips (sample)")
plt.savefig(out_dir / "trip_counts_by_tip_group.png")
plt.close()

# ---------------------------
# 7. How passenger count varies across hours and days
# ---------------------------
pax_by_hour = (df.groupby('pickup_hour', as_index=False)
                 .agg(mean_pax=('passenger_count','mean'),
                      median_pax=('passenger_count','median'),
                      trips=('passenger_count','count')))
pax_by_hour.to_csv(out_dir / "passenger_count_by_hour.csv", index=False)

plt.figure(figsize=(10,5))
sns.lineplot(data=pax_by_hour, x='pickup_hour', y='mean_pax', marker='o')
plt.title("Mean Passenger Count by Hour")
plt.xlabel("Hour")
plt.ylabel("Mean Passenger Count")
plt.xticks(range(0,24))
plt.savefig(out_dir / "mean_passenger_count_by_hour.png")
plt.close()

pax_by_day = (df.groupby('pickup_dayofweek', as_index=False)
                .agg(mean_pax=('passenger_count','mean'),
                     trips=('passenger_count','count')))
pax_by_day.to_csv(out_dir / "passenger_count_by_day.csv", index=False)

plt.figure(figsize=(8,5))
sns.barplot(data=pax_by_day, x='pickup_dayofweek', y='mean_pax', palette='cool')
plt.title("Mean Passenger Count by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Mean Passenger Count")
plt.savefig(out_dir / "mean_passenger_count_by_day.png")
plt.close()

# ---------------------------
# 8. How passenger count varies across zones
# ---------------------------
pax_by_zone = (df.groupby('PU_zone', as_index=False)
                 .agg(mean_pax=('passenger_count','mean'),
                      median_pax=('passenger_count','median'),
                      trips=('passenger_count','count'))
                 .sort_values('trips', ascending=False))
pax_by_zone.to_csv(out_dir / "passenger_count_by_zone.csv", index=False)
print("\nTop zones by number of trips and mean passenger count:")
print(pax_by_zone.head(15))

# ---------------------------
# 9. Create new column: average passenger count in each zone (PU_zone)
# ---------------------------
zone_mean_pax = pax_by_zone.set_index('PU_zone')['mean_pax'].to_dict()
df['avg_passenger_count_zone'] = df['PU_zone'].map(zone_mean_pax).fillna(df['passenger_count'].mean())

# Save a CSV of df with avg_passenger_count_zone for inspection
df[['PULocationID','PU_zone','passenger_count','avg_passenger_count_zone']].head(20).to_csv(out_dir / "sample_with_avg_passenger_zone.csv", index=False)

# ---------------------------
# 10. Analyze extra charges / surcharges
#    Count how often improvement_surcharge, congestion_surcharge, tolls_amount, airport_fee, extra are applied
# ---------------------------
surcharge_cols = ['improvement_surcharge', 'congestion_surcharge', 'tolls_amount', 'airport_fee', 'extra']
surcharge_stats = []
for col in surcharge_cols:
    if col in df.columns:
        applied = (df[col] > 0).sum()
        total = len(df)
        pct = applied / total * 100.0
        mean_val = df.loc[df[col] > 0, col].mean() if applied>0 else 0
        surcharge_stats.append({'surcharge': col, 'applied_count': int(applied), 'applied_pct': pct, 'mean_when_applied': mean_val})
surcharge_df = pd.DataFrame(surcharge_stats).sort_values('applied_count', ascending=False)
surcharge_df.to_csv(out_dir / "surcharge_application_stats.csv", index=False)
print("\nSurcharge application statistics:")
print(surcharge_df)

# Visualize surcharge application frequency
plt.figure(figsize=(8,5))
sns.barplot(data=surcharge_df, x='surcharge', y='applied_count', palette='rocket')
plt.title("Surcharge application counts (sample)")
plt.xlabel("Surcharge")
plt.ylabel("Count of trips where surcharge > 0")
plt.savefig(out_dir / "surcharge_application_counts.png")
plt.close()

# Also show surcharge frequency by time-of-day bucket (how often applied per bucket)
surcharge_by_tod = []
for col in surcharge_cols:
    if col in df.columns:
        stats = (df.groupby('tod_bucket').apply(lambda g: (g[col]>0).sum()).reset_index(name='applied_count'))
        stats['surcharge'] = col
        stats['bucket_total'] = df.groupby('tod_bucket').size().values
        stats['applied_pct'] = stats['applied_count'] / stats['bucket_total'] * 100.0
        surcharge_by_tod.append(stats)
if surcharge_by_tod:
    surcharge_by_tod_df = pd.concat(surcharge_by_tod, ignore_index=True)
    surcharge_by_tod_df.to_csv(out_dir / "surcharge_by_tod_bucket.csv", index=False)

# ---------------------------
# Save outputs and final cleaned df sample
# ---------------------------
farepm_passenger.to_csv(out_dir / "farepm_passenger_summary.csv", index=False)
farepm_by_day.to_csv(out_dir / "farepm_by_day.csv", index=False)
farepm_by_tod.to_csv(out_dir / "farepm_by_tod.csv", index=False)
if 'farepm_by_vendor' in locals():
    farepm_by_vendor.to_csv(out_dir / "farepm_by_vendor.csv", index=False)
farepm_by_distbucket.to_csv(out_dir / "farepm_by_distbucket.csv", index=False)

# Save final sample (with added derived columns)
# convert object cols to str for safe parquet write
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str)

final_csv = out_dir / "sample_2023_clean_analysis_ready.csv"
final_parquet = out_dir / "sample_2023_clean_analysis_ready.parquet"
df.to_csv(final_csv, index=False)
df.to_parquet(final_parquet, index=False, engine="pyarrow")

print("\nAll analysis outputs saved under:", out_dir)