import csv
import numpy as np
from yacana import ToolError

def load_csv():
    points = []
    country_names = []
    filename = 'data/countries.csv'
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            if row['latitude'] != "":
                latitude = float(row['latitude'])
                longitude = float(row['longitude'])
                points.append([longitude, latitude])  # Swap order for compatibility
                country_names.append(row['name'])  # Store country names
    return np.array(points), country_names

global_points, global_country_names = load_csv()

def get_location_with_type(lat : float, lon : float):
    if not (isinstance(lat, float)) or lat is None:
        raise ToolError("Latitude was not provided or None.")
    if not (isinstance(lon, float)) or lat is None:
        raise ToolError("Longitude was not provided or None.")

    country = get_nearest_country(lat, lon)
    return {"type": "country", "name": country}

def get_nearest_country(latitude, longitude):
    flight_point = np.array([longitude, latitude])
    distances = np.linalg.norm(global_points - flight_point, axis=1)
    closest_idx = np.argmin(distances)
    return global_country_names[closest_idx]

# Usage Example

# flight_lat = 46.780
# flight_lon = -33.349
# nearest_country = get_location_with_type(flight_lat, flight_lon)
# print(f"The flight is likely flying over: {nearest_country}")
