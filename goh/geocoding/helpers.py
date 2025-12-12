import math

from pandas import isna

# Coordenadas aproximadas dos limites de Goiânia, GO, Brasil
BOUNDS = {
    "north": -16.5080,
    "south": -16.7740,
    "east": -49.0160,
    "west": -49.33370,
}

# Raio da Terra em metros para fórmula de Haversine
EARTH_RADIUS = 6371000

# Marco Zero de Goiânia (Praça Cívica)
GOIANIA_CENTER_LATITUDE = -16.6799
GOIANIA_CENTER_LONGITUDE = -49.2550

# Conversão aproximada de graus para metros
METERS_PER_DEGREE_LATITUDE = 111132.954 
METERS_PER_DEGREE_LONGITUDE = 111132.954 * math.cos(math.radians(GOIANIA_CENTER_LATITUDE))


def is_in_goiania(result: dict) -> bool:
    """
    Verifica se o resultado de geocodificação está localizado em Goiânia, Goiás, Brasil.

    Parameters:
        result (dict): Um dicionário representando o resultado de geocodificação do Google Maps.

    Returns:
        bool: True se o resultado estiver localizado em Goiânia, Goiás, Brasil; False caso contrário.
    """
    address_components = result.get("address_components", [])
    
    has_goiania = False
    has_goias = False
    has_brazil = False

    for component in address_components:
        types = component.get("types", [])
        name = component.get("long_name", "").lower()
        short_name = component.get("short_name", "").lower()

        if "administrative_area_level_2" in types or "locality" in types:
            if "goiânia" in name or "goiania" in name:
                has_goiania = True

        if "administrative_area_level_1" in types:
            if "goiás" in name or "goias" in name or short_name == "go":
                has_goias = True

        if "country" in types:
            if "brasil" in name or "brazil" in name or short_name == "br":
                has_brazil = True

    location = result.get("geometry", {}).get("location", {})
    latitude = location.get("lat", 0)
    longitude = location.get("lng", 0)

    in_bounds = (
        BOUNDS["south"] <= latitude <= BOUNDS["north"] and
        BOUNDS["west"] <= longitude <= BOUNDS["east"]
    )

    return (has_goiania or in_bounds) and has_goias and has_brazil


def is_valid_neighborhood(result: dict) -> bool:
    """
    Verifica se o resultado de geocodificação corresponde a um tipo válido de bairro.

    Parameters:
        result (dict): Um dicionário representando o resultado de geocodificação do Google Maps.
        neighborhood_name (str): O nome do bairro (mantido para compatibilidade, mas não usado na validação estrita).

    Returns:
        bool: True se o resultado for de um tipo compatível com bairro; False caso contrário.
    """
    types = result.get("types", [])

    valid_types = [
        "sublocality",
        "sublocality_level_1",
        "sublocality_level_2",
        "neighborhood",
        "locality",
        "political",
        "route",
        "establishment",
        "point_of_interest"
    ]

    if not any(t in types for t in valid_types):
        return False
    
    return True


def is_valid_street(result: dict) -> bool:
    """
    Verifica se o resultado de geocodificação corresponde a um tipo válido de rua.

    Parameters:
        result (dict): Um dicionário representando o resultado de geocodificação do Google Maps.
        street_name (str): O nome da rua (mantido para compatibilidade, mas não usado na validação estrita).

    Returns:
        bool: True se o resultado for de um tipo compatível com rua; False caso contrário.
    """
    types = result.get("types", [])

    valid_types = [
        "route",
        "street_address",
        "premise",
        "subpremise",
        "point_of_interest",
        "establishment",
        "intersection"
    ]

    if not any(t in types for t in valid_types):
        return False
    
    return True


def calculate_metrics(latitude: float, longitude: float) -> dict[str, float | None]:
    """
    Calcula métricas espaciais lineares a partir de coordenadas geográficas.
    Utiliza a fórmula de Haversine para distância ao centro de Goiânia
    e uma transformação simples para coordenadas cartesianas locais.
    
    Parameters:
        latitude (float): Latitude do ponto.
        longitude (float): Longitude do ponto.
    
    Returns:
        dict[str, float | None]: Dicionário com as métricas calculadas:
            - distance_to_center_m (float | None): Distância ao centro de Goiânia em metros.
            - coordinate_x_m (float | None): Coordenada X local em metros (leste-oeste).
            - coordinate_y_m (float | None): Coordenada Y local em metros (norte-sul).
    """
    if isna(latitude) or isna(longitude):
        return {
            "distance_to_center_m": None,
            "coordinate_x_m": None,
            "coordinate_y_m": None
        }
    # Distância ao centro de Goiânia usando a fórmula de Haversine
    phi1 = math.radians(latitude)
    phi2 = math.radians(GOIANIA_CENTER_LATITUDE)
    delta_phi = math.radians(GOIANIA_CENTER_LATITUDE - latitude)
    delta_lambda = math.radians(GOIANIA_CENTER_LONGITUDE - longitude)

    a = math.sin(delta_phi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist_center = EARTH_RADIUS * c

    # Coordenadas cartesianas locais em metros (aproximação)
    y_meters = (latitude - GOIANIA_CENTER_LATITUDE) * METERS_PER_DEGREE_LATITUDE
    x_meters = (longitude - GOIANIA_CENTER_LONGITUDE) * METERS_PER_DEGREE_LONGITUDE

    return {
        "distance_to_center_m": round(dist_center, 2),
        "coordinate_x_m": round(x_meters, 2),
        "coordinate_y_m": round(y_meters, 2)
    }