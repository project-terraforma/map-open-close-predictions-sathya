import overturemaps

def test_fetch(resource_name):
    print(f"Testing resource: {resource_name}")
    try:
        bbox = [-122.42, 37.77, -122.41, 37.78]
        iterator = overturemaps.record_batch_reader(resource_name, bbox=bbox)
        table = iterator.read_all()
        print(f"Success! Fetched {table.num_rows} rows.")
    except Exception as e:
        print(f"Failed with {e}")

if __name__ == "__main__":
    test_fetch("places")
    test_fetch("place")
    test_fetch("building")
    test_fetch("buildings")
