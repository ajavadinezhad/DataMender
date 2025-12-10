
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
    
    vehicle_types = ["sedan", "suv", "bike", "luxury", None]
    
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    
    for i in range(n_rows):
        if random.random() < 0.05:
            driver_age = random.choice([-25, 200, None])
        else:
            driver_age = random.randint(22, 65)
        data["driver_age"].append(float(driver_age) if driver_age is not None else None)
        
        if random.random() < 0.03:
            passenger_age = random.choice([-10, 150, None])
        else:
            passenger_age = random.randint(18, 80)
        data["passenger_age"].append(float(passenger_age) if passenger_age is not None else None)
        
        if random.random() < 0.08:
            distance = -random.uniform(1, 50)
        else:
            distance = random.uniform(1, 50)
        data["distance_km"].append(distance)
        
        if random.random() < 0.06:
            duration = -random.uniform(5, 120)
        else:
            duration = random.uniform(5, 120)
        data["duration_minutes"].append(duration)
        
        if random.random() < 0.04:
            fare = random.choice([None, -random.uniform(10, 100)])
        else:
            fare = abs(distance) * 2 + random.uniform(5, 20)
        data["fare_amount"].append(fare)
        
        if random.random() < 0.10:
            tip = random.choice([-5, 150, None])
        else:
            tip = random.uniform(0, 30)
        data["tip_percentage"].append(tip)
        
        if random.random() < 0.07:
            rating = random.choice([0, 6, None])
        else:
            rating = random.randint(1, 5)
        data["rating"].append(float(rating) if rating is not None else None)
        
        pickup = base_time + timedelta(minutes=i * 5)
        data["pickup_time"].append(pickup)
        
        if random.random() < 0.05:
            dropoff = pickup - timedelta(minutes=random.randint(10, 60))
        else:
            dropoff = pickup + timedelta(minutes=abs(duration) if duration else 30)
        data["dropoff_time"].append(dropoff)
        
        data["vehicle_type"].append(random.choice(vehicle_types))
    
    return pl.DataFrame(data, strict=False)
