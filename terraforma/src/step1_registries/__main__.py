import sys
from src.step1_registries import run, ALIASES

args = sys.argv[1:]

if args and args[0] == "geocode":
    from src.step1_registries.geocode import geocode_city
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    cities = args[1:] or ["singapore", "mumbai"]
    for city in cities:
        key = ALIASES.get(city, city)
        geocode_city(key)
else:
    run(args or None)
