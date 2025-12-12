import logging

import requests

from .config import GEOCODING_CONFIG
from .helpers import is_in_goiania, is_valid_street, BOUNDS


logger = logging.getLogger(__name__)


def geocode_street(street_name: str) -> dict | None:

    address = f"{street_name}, Goiânia, Goiás, Brasil"

    params = {
        "address": address,
        "key": GEOCODING_CONFIG.GOOGLE_API_KEY,
        "language": "pt-BR",
        "components": "administrative_area:GO|country:BR",
        "bounds": f"{BOUNDS['south']},{BOUNDS['west']}|{BOUNDS['north']},{BOUNDS['east']}",
    }

    try:
        response = requests.get(GEOCODING_CONFIG.GEOCODING_API_URL, params=params, timeout=GEOCODING_CONFIG.GEOCODING_API_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "OK":
            logger.debug(f"A API retornou status {data.get('status')} para o endereço: {address}")
            return None
        
        if not data.get("results"):
            logger.debug(f"Nenhum resultado encontrado para a rua: {street_name}")
            return None
        
        result = data["results"][0]

        if not is_in_goiania(result):
            logger.debug(f"O resultado não está em Goiânia para a rua: {street_name}")
            return None
        
        if not is_valid_street(result):
            logger.debug(f"O resultado não corresponde a uma rua válida: {street_name}")
            return None
        
        location = result.get("geometry", {}).get("location", {})

        logger.debug(f"Geocodificação bem-sucedida para a rua: {street_name} -> {location}")

        return {"latitude": location.get("lat"), "longitude": location.get("lng")}
    
    except requests.RequestException as e:
        logger.error(f"Erro na requisição para a API de geocodificação: {e}")
        return None
    
    except (KeyError, IndexError) as e:
        logger.error(f"Erro ao processar os dados de geocodificação para a rua {street_name}: {e}")
        return None