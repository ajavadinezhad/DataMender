"""Generate sample messy dataset for testing"""
import polars as pl
import random
from datetime import datetime, timedelta


def generate_ride_sharing_data(n_rows: int = 10000) -> pl.DataFrame:
    """
    Generate messy ride-sharing dataset with intentional issues
    
    Args:
        n_rows: Number of rows to generate
        
    Returns:
        Polars DataFrame
    """
    random.seed(42)
    
    data = {
        "ride_id": list(range(1, n_rows + 1)),
        "driver_age": [],
        "passenger_age": [],
        "distance_km": [],
        "duration_minutes": [],
        "fare_amount": [],
        "tip_percentage": [],
        "rating": [],
        "pickup_time": [],
        "dropoff_time": [],
        "vehicle_type": [],
    }
    
    vehicle_types = ["sedan", "suv", "bike", "luxury", None]  # Include some nulls
    
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    
    for i in range(n_rows):
        # Driver age (with some invalid values)
        if random.random() < 0.05:  # 5% invalid
            driver_age = random.choice([-25, 200, None])
        else:
            driver_age = random.randint(22, 65)
        data["driver_age"].append(float(driver_age) if driver_age is not None else None)
        
        # Passenger age (with some invalid values)
        if random.random() < 0.03:  # 3% invalid
            passenger_age = random.choice([-10, 150, None])
        else:
            passenger_age = random.randint(18, 80)
        data["passenger_age"].append(float(passenger_age) if passenger_age is not None else None)
        
        # Distance (with some negative values)
        if random.random() < 0.08:  # 8% negative
            distance = -random.uniform(1, 50)
        else:
            distance = random.uniform(1, 50)
        data["distance_km"].append(distance)
        
        # Duration (with some negative values)
        if random.random() < 0.06:  # 6% negative
            duration = -random.uniform(5, 120)
        else:
            duration = random.uniform(5, 120)
        data["duration_minutes"].append(duration)
        
        # Fare (with some negative values and nulls)
        if random.random() < 0.04:
            fare = random.choice([None, -random.uniform(10, 100)])
        else:
            fare = abs(distance) * 2 + random.uniform(5, 20)
        data["fare_amount"].append(fare)
        
        # Tip percentage (should be 0-100, but has invalid values)
        if random.random() < 0.10:  # 10% invalid
            tip = random.choice([-5, 150, None])
        else:
            tip = random.uniform(0, 30)
        data["tip_percentage"].append(tip)
        
        # Rating (1-5, with some invalid)
        if random.random() < 0.07:
            rating = random.choice([0, 6, None])
        else:
            rating = random.randint(1, 5)
        data["rating"].append(float(rating) if rating is not None else None)
        
        # Pickup time
        pickup = base_time + timedelta(minutes=i * 5)
        data["pickup_time"].append(pickup)
        
        # Dropoff time (sometimes before pickup - invalid!)
        if random.random() < 0.05:  # 5% invalid order
            dropoff = pickup - timedelta(minutes=random.randint(10, 60))
        else:
            dropoff = pickup + timedelta(minutes=abs(duration) if duration else 30)
        data["dropoff_time"].append(dropoff)
        
        # Vehicle type (with some nulls)
        data["vehicle_type"].append(random.choice(vehicle_types))
    
    return pl.DataFrame(data, strict=False)


if __name__ == "__main__":
    print("Generating sample ride-sharing dataset...")
    df = generate_ride_sharing_data(25000)
    
    # Save as CSV
    output_csv = "sample_rides.csv"
    df.write_csv(output_csv)
    print(f"✅ Saved {len(df)} rows to {output_csv}")
    
    # Save as Parquet
    output_parquet = "sample_rides.parquet"
    df.write_parquet(output_parquet)
    print(f"✅ Saved {len(df)} rows to {output_parquet}")
    
    # Show sample
    print("\nSample data (first 5 rows):")
    print(df.head())
    
    # Show issues
    print("\n⚠️  Intentional data issues:")
    print(f"  - Negative ages: {(df['driver_age'] < 0).sum()}")
    print(f"  - Negative distances: {(df['distance_km'] < 0).sum()}")
    print(f"  - Negative fares: {(df['fare_amount'] < 0).sum()}")
    print(f"  - Invalid ratings: {((df['rating'] < 1) | (df['rating'] > 5)).sum()}")
    print(f"  - Null vehicle types: {df['vehicle_type'].null_count()}")

