import csv
import pandas as pd

# # Read in the first CSV file
# df1 = pd.read_csv('geolocation.csv')

# # Read in the second CSV file
# df2 = pd.read_csv('states.csv')

# # Merge the data frames based on a common key
# merged_df = pd.merge(df1, df2, on=['country_name','state_name'])

# # Write the merged data frame to a new CSV file
# merged_df.to_csv('geolocation_merged.csv', index=False)

with open('geolocation_merged.csv', 'r', encoding='UTF-8') as csvfile:
    reader = csv.DictReader(csvfile)
    selected_columns = ["city_name", "city_latitude","city_longitude","state_code", "state_name","state_latitude","state_longitude","country_code", "country_name","country_latitude","country_longitude"]
    with open('geolocation_final.csv', 'w', newline='', encoding='UTF-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(selected_columns)
        for row in reader:
            selected_row = [row[field] for field in selected_columns]
            writer.writerow(selected_row)
