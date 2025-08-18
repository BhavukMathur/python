- The goal is to analyse patterns in the data that can inform strategic decisions to improve service efficiency, maximise revenue, and enhance passenger experience.

- TODO:
- Data Loading
- Data Cleaning
- Exploratory Analysis: Bivariate and Multivariate
- Creating Visualisations to Support the Analysis
- Deriving Insights and Stating Conclusions


Data:
trip_records - .parquet files

#	VendorID	tpep_pickup_datetime	tpep_dropoff_datetime	passenger_count	trip_distance	RatecodeID	store_and_fwd_flag	PULocationID	DOLocationID	payment_type	fare_amount	extra	mta_tax	tip_amount	tolls_amount	improvement_surcharge	total_amount	congestion_surcharge	airport_fee	__index_level_0__
1	2	2023-01-01T00:32:10.000Z	2023-01-01T00:40:36.000Z	1	0.97	1	N	161	141	2	9.3	1	0.5	0	0	1	14.3	2.5	0	0
2	2	2023-01-01T00:55:08.000Z	2023-01-01T01:01:27.000Z	1	1.1	1	N	43	237	1	7.9	1	0.5	4	0	1	16.9	2.5	0	1
3	2	2023-01-01T00:25:04.000Z	2023-01-01T00:37:49.000Z	1	2.51	1	N	48	238	1	14.9	1	0.5	15	0	1	34.9	2.5	0	2

taxi_zones - .dbf file

OBJECTID	SHAPE_LENG	SHAPE_AREA	ZONE	LOCATIONID	BOROUGH	CPU
1	0.116357453189	0.0007823067885	Newark Airport	1	EWR	
2	0.43346966679	0.00486634037837	Jamaica Bay	2	Queens	
3	0.0843411059012	0.000314414156821	Allerton/Pelham Gardens	3	Bronx	
4	0.0435665270921	0.000111871946192	Alphabet City	4	Manhattan	

TODO:
1. Why payment_type only 1
2. 