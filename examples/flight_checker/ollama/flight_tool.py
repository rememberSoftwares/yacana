"""
OpenSky Network Flight Data Function for AI Agent Function Calling

This module provides a function suitable for AI agent function calling that
retrieves flight data from the OpenSky Network API based on flight number.
"""

import os
import json
from typing import Dict, Any
from datetime import datetime, timedelta
import requests
from relative_location_tool import get_location_with_type

from yacana import ToolError

time_window_hours = 48


def get_flight_data_by_callsign(
    flight_number: str
) -> Dict[str, Any]:
    """
    Retrieves comprehensive flight data from OpenSky Network API based on flight number/callsign.

    This function searches for flights matching the provided flight number or callsign and returns
    detailed information including current position, altitude, velocity, and route information.

    Parameters
    ----------
    flight_number: str
        The flight number or callsign to search for (e.g., "DLH441", "UAL123")

    Returns
    -------
    Dict[str, Any]
        A dictionary containing flight information with the following structure:
        {
            "flight_found": bool,
            "callsign": str,
            "icao24": str,
            "current_position": {
                "latitude": float,
                "longitude": float,
                "altitude_meters": float,
                "on_ground": bool
            },
            "velocity_info": {
                "velocity_ms": float,
                "true_track_degrees": float,
                "vertical_rate_ms": float
            },
            "aircraft_info": {
                "origin_country": str,
                "squawk": str,
                "spi": bool
            },
            "timestamps": {
                "last_contact": int,
                "time_position": int
            },
            "historical_flights": List[Dict],
            "error": str or None
        }

    Raises
    ------
    requests.RequestException
        If API request fails
    ValueError
        If flight_number is empty or invalid

    Examples
    --------
        >>> flight_data = get_flight_data_by_callsign("DLH441")
        >>> print(f"Flight {flight_data['callsign']} is at altitude {flight_data['current_position']['altitude_meters']}m")
    """

    # Input validation
    if not flight_number or not isinstance(flight_number, str):
        raise ToolError("Flight number was not provided or None.")

    # Clean up the flight number (remove spaces, convert to uppercase)
    callsign = flight_number.strip().upper()

    try:
        # Step 1: Search for current state vectors with the callsign
        base_url = "https://opensky-network.org/api/states/all"

        # Prepare authentication if provided
        headers = {
            "Accept": "application/json",
            "Authorization": "Bearer {}".format(os.getenv("TOKEN_OPENSKY"))
        }


        # Make request to get current states
        response = requests.get(base_url, headers=headers, timeout=30)
        response.raise_for_status()

        states_data = response.json()

        # Find flights matching the callsign
        matching_flights = []
        if states_data.get('states'):
            for state in states_data['states']:
                if state[1] and callsign in state[1].strip().upper():
                    matching_flights.append(state)

        if not matching_flights:
            # Step 2: If no current flights found, search historical data
            end_time = int(datetime.now().timestamp())
            begin_time = int((datetime.now() - timedelta(hours=time_window_hours)).timestamp())

            # Try to find historical flights with this callsign
            flights_url = "https://opensky-network.org/api/flights/all"
            params = {
                'begin': begin_time,
                'end': end_time
            }

            hist_response = requests.get(flights_url, params=params, headers=headers, timeout=30)

            historical_flights = []
            if hist_response.status_code == 200:
                hist_data = hist_response.json()
                if hist_data:
                    for flight in hist_data:
                        if flight.get('callsign') and callsign in flight['callsign'].strip().upper():
                            historical_flights.append({
                                'callsign': flight.get('callsign'),
                                'icao24': flight.get('icao24'),
                                'departure_airport': flight.get('estDepartureAirport'),
                                'arrival_airport': flight.get('estArrivalAirport'),
                                'first_seen': flight.get('firstSeen'),
                                'last_seen': flight.get('lastSeen')
                            })

            return {
                "flight_found": False,
                "callsign": callsign,
                "message": f"No current flights found for {callsign}",
                "historical_flights": historical_flights,
                "search_time_window_hours": time_window_hours,
                "error": None
            }

        # Process the first matching flight
        flight_state = matching_flights[0]
        icao24 = flight_state[0]
        found_callsign = flight_state[1].strip() if flight_state[1] else callsign

        # Step 3: Get additional flight information using ICAO24
        end_time = int(datetime.now().timestamp())
        begin_time = int((datetime.now() - timedelta(hours=time_window_hours)).timestamp())

        flights_url = "https://opensky-network.org/api/flights/aircraft"
        params = {
            'icao24': icao24,
            'begin': begin_time,
            'end': end_time
        }

        hist_response = requests.get(flights_url, params=params,headers=headers, timeout=30)
        historical_flights = []

        if hist_response.status_code == 200:
            hist_data = hist_response.json()
            if hist_data:
                for flight in hist_data:
                    historical_flights.append({
                        'callsign': flight.get('callsign'),
                        'departure_airport': flight.get('estDepartureAirport'),
                        'arrival_airport': flight.get('estArrivalAirport'),
                        'first_seen': flight.get('firstSeen'),
                        'last_seen': flight.get('lastSeen'),
                        'departure_time': datetime.fromtimestamp(flight['firstSeen']).isoformat() if flight.get('firstSeen') else None,
                        'arrival_time': datetime.fromtimestamp(flight['lastSeen']).isoformat() if flight.get('lastSeen') else None
                    })

        concrete_location = get_location_with_type(flight_state[6], flight_state[5])

        # Format the comprehensive response
        result = {
            "flight_found": True,
            "callsign": found_callsign,
            "icao24": icao24,
            "current_position": {
                "latitude": flight_state[6],
                "longitude": flight_state[5],
                "altitude_meters": flight_state[7],
                "geometric_altitude_meters": flight_state[13],
                "on_ground": flight_state[8],
                "concrete_location": concrete_location
            },
            "velocity_info": {
                "velocity_ms": flight_state[9],
                "true_track_degrees": flight_state[10],
                "vertical_rate_ms": flight_state[11]
            },
            "aircraft_info": {
                "origin_country": flight_state[2],
                "squawk": flight_state[14],
                "spi": flight_state[15],
                "position_source": flight_state[16]
            },
            "timestamps": {
                "last_contact": flight_state[4],
                "time_position": flight_state[3],
                "last_contact_datetime": datetime.fromtimestamp(flight_state[4]).isoformat() if flight_state[4] else None,
                "time_position_datetime": datetime.fromtimestamp(flight_state[3]).isoformat() if flight_state[3] else None
            },
            "historical_flights": historical_flights,
            "search_time_window_hours": time_window_hours,
            "api_timestamp": states_data.get('time'),
            "total_matching_flights": len(matching_flights),
            "error": None
        }

        return result

    except requests.RequestException as e:
        return {
            "flight_found": False,
            "callsign": callsign,
            "error": f"API request failed: {str(e)}",
            "error_type": "network_error"
        }
    except Exception as e:
        return {
            "flight_found": False,
            "callsign": callsign,
            "error": f"Unexpected error: {str(e)}",
            "error_type": "general_error"
        }


# JSON Schema for AI Function Calling
FLIGHT_DATA_FUNCTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_flight_data_by_callsign",
        "description": "Retrieves comprehensive real-time and historical flight data from OpenSky Network API based on flight number or callsign. Returns current position, altitude, velocity, route information, and historical flights.",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_number": {
                    "type": "string",
                    "description": "The flight number or callsign to search for (e.g., 'DLH441', 'UAL123', 'AF1234'). Can be with or without airline prefix."
                },
                "time_window_hours": {
                    "type": "integer",
                    "description": "Time window in hours to search for historical flights (default: 24 hours)",
                    "default": 24,
                    "minimum": 1,
                    "maximum": 168
                }
            },
            "required": ["flight_number"],
            "additionalProperties": False
        }
    }
}

if __name__ == "__main__":
    # Example usage
    print("OpenSky Flight Data Function")
    print("=" * 40)

    # Test with a sample flight number
    result1 = get_flight_data_by_callsign("VJT716")
    print("Sample call result:")
    print(json.dumps(result1, indent=2))

    # print("\nFunction schema for AI agents:")
    # print(json.dumps(FLIGHT_DATA_FUNCTION_SCHEMA, indent=2))
