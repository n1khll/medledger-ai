"""Hospital Search Module - Real-time web search for hospitals by pincode using Serper"""
import json
import os
import re
from typing import List, Dict, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

logger = get_logger(__name__)

def _sanitize_for_logging(text: str) -> str:
    """
    Remove emojis and problematic Unicode characters for Windows console logging.
    
    Args:
        text: Text that may contain emojis or special Unicode characters
        
    Returns:
        Sanitized text safe for Windows console (cp1252 encoding)
    """
    if not text:
        return text
    
    # Remove emojis and other problematic Unicode characters
    # Keep only ASCII and common Latin-1 characters
    try:
        # Try to encode as ASCII first (safest)
        text.encode('ascii')
        return text
    except UnicodeEncodeError:
        # If it contains non-ASCII, remove emojis and keep only printable ASCII
        # Remove emojis (Unicode ranges for emojis)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # Remove any remaining non-ASCII characters that can't be encoded
        try:
            return text.encode('ascii', 'ignore').decode('ascii')
        except:
            return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

# Try to import httpx for direct API calls
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available. Install: pip install httpx")

# Try to import Serper tool (fallback)
try:
    from crewai_tools import SerperDevTool
    SERPER_TOOL_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.tools import SerperDevTool
        SERPER_TOOL_AVAILABLE = True
    except ImportError:
        SERPER_TOOL_AVAILABLE = False


def _extract_phone_number(text: str) -> Optional[str]:
    """
    Extract phone number from text using regex patterns.
    Handles Indian phone number formats and international formats.
    
    Args:
        text: Text that may contain a phone number
        
    Returns:
        Phone number string or None if not found
    """
    if not text:
        return None
    
    # Indian phone number patterns
    patterns = [
        r'\+91[-\s]?[6-9]\d{9}',  # +91 9876543210
        r'\+91[-\s]?\d{2,4}[-\s]?\d{6,8}',  # +91 80 12345678
        r'[6-9]\d{9}',             # 9876543210 (10 digits starting with 6-9)
        r'\d{11}',                 # 11 digits (STD code + number)
        r'\(\d{2,4}\)[-\s]?\d{6,8}',  # (080) 12345678
        r'\d{3,4}[-\s]?\d{6,8}',  # 080-12345678
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Clean up the phone number
            phone = matches[0].strip()
            # Remove common separators but keep + if present
            phone = re.sub(r'[-\s()]', '', phone)
            
            # Format Indian numbers
            if phone.startswith('91') and len(phone) >= 12:
                phone = '+' + phone[:12]  # +91 + 10 digits
            elif phone.startswith('+91') and len(phone) >= 13:
                phone = phone[:13]  # +91 + 10 digits
            elif len(phone) == 10 and phone[0] in '6789':
                phone = '+91' + phone
            elif len(phone) >= 10:
                phone = '+91' + phone[-10:]  # Take last 10 digits
            
            logger.info(f"Extracted phone: {phone} from text: {text[:100]}")
            return phone
    
    return None


def _search_serper_maps(api_key: str, query: str, pincode: str) -> List[Dict]:
    """
    Search using Serper Google Maps API for more structured results with phone numbers.
    
    Args:
        api_key: Serper API key
        query: Search query string
        pincode: Pincode for location context
        
    Returns:
        List of hospital dictionaries with phone numbers
    """
    try:
        url = "https://google.serper.dev/maps"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "location": f"{pincode}, India"
        }
        
        response = httpx.post(url, headers=headers, json=payload, timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            places = data.get("places", [])
            logger.info(f"Serper Maps API returned {len(places)} places")
            
            hospitals = []
            for i, place in enumerate(places[:10]):
                title = place.get("title", "").strip()
                address = place.get("address", "").strip()
                phone = place.get("phoneNumber", "").strip()
                rating = place.get("rating")
                
                # Extract phone if not directly available
                if not phone and address:
                    phone = _extract_phone_number(address)
                
                hospital = {
                    "id": f"HOSP_{i + 1:03d}",
                    "name": title[:100],
                    "address": address[:200] if address else f"Near pincode {pincode}, India",
                    "pincode": pincode,
                    "specialties": ["Emergency"] if "emergency" in title.lower() else ["General Medicine"],
                    "phone": phone if phone else "Contact hospital for details",
                    "availability": "24/7",
                    "distance_km": None,
                    "rating": rating,
                    "source": "serper_maps",
                    "link": place.get("website", "")
                }
                hospitals.append(hospital)
                safe_title = _sanitize_for_logging(title[:50])
                logger.info(f"Added from Maps: {safe_title} - Phone: {phone if phone else 'N/A'}")
            
            return hospitals
        else:
            logger.error(f"Serper Maps API error: Status {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error in Serper Maps API call: {str(e)}", exc_info=True)
        return []


def search_hospitals_by_pincode(pincode: str, specialty: Optional[str] = None) -> List[Dict]:
    """
    Search hospitals by pincode using Serper web search, optionally filtered by specialty.
    
    Args:
        pincode: Pincode/ZIP code to search
        specialty: Optional specialty to filter by (e.g., "Cardiology", "Neurology")
        
    Returns:
        List of hospital dictionaries matching the criteria
    """
    logger.info(f"Searching hospitals for pincode: {pincode}, specialty: {specialty}")
    
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        logger.error("SERPER_API_KEY not found in environment variables")
        return []
    
    # Build search query - improved for Indian pincodes
    if specialty:
        search_query = f"{specialty} hospital near {pincode} India"
    else:
        search_query = f"best hospitals near {pincode} India"
    
    logger.info(f"Executing Serper web search: {search_query}")
    logger.info(f"Serper API Key present: {bool(serper_api_key)}")
    
    try:
        # METHOD 1: Try Serper Google Maps API first (has phone numbers)
        if HTTPX_AVAILABLE:
            logger.info("Trying Serper Google Maps API for structured data with phone numbers")
            hospitals_maps = _search_serper_maps(serper_api_key, search_query, pincode)
            
            # If we got good results from Maps with phone numbers, use them
            hospitals_with_phones = [h for h in hospitals_maps if h.get("phone") and h["phone"] != "Contact hospital for details"]
            if hospitals_with_phones:
                logger.info(f"Got {len(hospitals_with_phones)} hospitals with phone numbers from Maps API")
                return hospitals_maps  # Return all Maps results
            
            # If Maps didn't return good results, fall back to web search
            logger.info("Maps API returned no results with phones, falling back to web search")
            search_results = _search_serper_direct(serper_api_key, search_query)
        # METHOD 2: Fallback to SerperDevTool if available
        elif SERPER_TOOL_AVAILABLE:
            logger.info("Using SerperDevTool (fallback)")
            search_tool = SerperDevTool(api_key=serper_api_key)
            search_results = search_tool.run(search_query)
        else:
            logger.error("Neither httpx nor SerperDevTool available. Cannot search hospitals.")
            return []
        
        logger.info(f"=== SERPER DEBUG INFO ===")
        logger.info(f"Serper search completed. Result type: {type(search_results)}")
        logger.info(f"Result is None: {search_results is None}")
        
        # Log full result for debugging
        if search_results is not None:
            result_str = str(search_results)
            logger.info(f"Result string length: {len(result_str)}")
            logger.info(f"Result preview (first 2000 chars): {result_str[:2000]}")
            
            # Try to get repr for better debugging
            try:
                result_repr = repr(search_results)
                logger.info(f"Result repr (first 1000 chars): {result_repr[:1000]}")
            except:
                pass
            
            if isinstance(search_results, dict):
                logger.info(f"Result keys: {list(search_results.keys())}")
                if "organic" in search_results:
                    organic = search_results.get("organic", [])
                    logger.info(f"Organic results count: {len(organic)}")
                    if organic:
                        logger.info(f"First organic result: {organic[0]}")
                if "answerBox" in search_results:
                    logger.info(f"Answer box present: {search_results.get('answerBox')}")
                if "knowledgeGraph" in search_results:
                    logger.info(f"Knowledge graph present: {search_results.get('knowledgeGraph')}")
            elif isinstance(search_results, str):
                logger.info(f"Result is string, length: {len(search_results)}")
                # Try to parse as JSON
                try:
                    parsed_test = json.loads(search_results)
                    logger.info(f"String is valid JSON, type: {type(parsed_test)}")
                    if isinstance(parsed_test, dict):
                        logger.info(f"Parsed JSON keys: {list(parsed_test.keys())}")
                except:
                    logger.info("String is NOT valid JSON")
        else:
            logger.error("Serper returned None!")
        
        logger.info(f"=== END SERPER DEBUG ===")
        
        # Parse search results into structured format
        hospitals = _parse_serper_results(search_results, pincode, specialty)
        
        logger.info(f"After parsing, found {len(hospitals)} hospitals for pincode: {pincode}")
        
        return hospitals
    
    except Exception as e:
        logger.error(f"Error searching hospitals with Serper: {str(e)}", exc_info=True)
        return []


def _parse_serper_results(search_results: any, pincode: str, specialty: Optional[str] = None) -> List[Dict]:
    """
    Parse Serper search results into structured hospital format.
    
    Args:
        search_results: Raw search results from SerperDevTool
        pincode: Pincode used in search
        specialty: Optional specialty filter
        
    Returns:
        List of hospital dictionaries
    """
    hospitals = []
    
    try:
        logger.info(f"Parsing Serper results. Type: {type(search_results)}")
        
        # Handle None case
        if search_results is None:
            logger.error("Serper returned None - search may have failed")
            return []
        
        # First, try if it's already a dict (JSON response)
        if isinstance(search_results, dict):
            logger.info("Serper returned dict format")
            logger.info(f"Dict keys: {list(search_results.keys())}")
            
            # Try multiple possible keys for results
            organic_results = search_results.get("organic", [])
            if not organic_results:
                organic_results = search_results.get("results", [])
            if not organic_results:
                organic_results = search_results.get("items", [])
            if not organic_results:
                organic_results = search_results.get("data", [])
            
            logger.info(f"Found {len(organic_results)} organic results")
            
            # If no organic results, log what we have
            if not organic_results:
                logger.warning(f"No 'organic' results found. Available keys: {list(search_results.keys())}")
                # Try to extract from other possible structures
                for key in ["answerBox", "knowledgeGraph", "places", "localPack"]:
                    if key in search_results:
                        logger.info(f"Found {key}: {search_results[key]}")
            
            for i, result in enumerate(organic_results[:10]):
                if not isinstance(result, dict):
                    logger.warning(f"Result {i} is not a dict: {type(result)}")
                    continue
                    
                title = result.get("title", result.get("name", "")).strip()
                snippet = result.get("snippet", result.get("description", result.get("text", ""))).strip()
                link = result.get("link", result.get("url", ""))
                
                # Extract phone number from snippet or other fields
                phone = None
                if snippet:
                    phone = _extract_phone_number(snippet)
                
                # If not found in snippet, check other possible fields
                if not phone:
                    phone_field = result.get("phone", result.get("phoneNumber", result.get("telephone", "")))
                    if phone_field:
                        phone = _extract_phone_number(str(phone_field))
                
                # Fallback to default if still not found
                if not phone:
                    phone = "Contact hospital for details"
                
                safe_title = _sanitize_for_logging(title[:50])
                safe_phone = _sanitize_for_logging(phone[:20] if phone else 'None')
                logger.info(f"Processing result {i}: title='{safe_title}', phone={safe_phone}")
                
                # More lenient matching - accept any result that might be a hospital
                if title:
                    # Check if it's a hospital-related result
                    is_hospital = any(keyword in title.lower() for keyword in [
                        "hospital", "medical", "clinic", "healthcare", "health", 
                        "care", "center", "centre", "institute", "facility"
                    ])
                    
                    # Also accept if snippet contains hospital keywords
                    if not is_hospital and snippet:
                        is_hospital = any(keyword in snippet.lower() for keyword in [
                            "hospital", "medical", "clinic", "healthcare", "doctor", "patient"
                        ])
                    
                    if is_hospital or len(organic_results) <= 5:  # Accept all if few results
                        hospital = {
                            "id": f"HOSP_{i + 1:03d}",
                            "name": title[:100],
                            "address": snippet.split(".")[0][:200] if snippet else f"Near pincode {pincode}, India",
                            "pincode": pincode,
                            "specialties": [specialty] if specialty else ["General Medicine", "Emergency"],
                            "phone": phone,
                            "availability": "24/7",
                            "distance_km": None,
                            "source": "serper_web_search",
                            "link": link
                        }
                        hospitals.append(hospital)
                        safe_title = _sanitize_for_logging(title[:50])
                        logger.info(f"Added hospital: {safe_title} - Phone: {phone}")
        
        # Try parsing as JSON string
        elif isinstance(search_results, str):
            logger.info("Serper returned string format, attempting JSON parse")
            try:
                # Try to parse as JSON first
                parsed = json.loads(search_results)
                logger.info(f"Successfully parsed JSON string. Type: {type(parsed)}")
                
                if isinstance(parsed, dict):
                    logger.info(f"Parsed dict keys: {list(parsed.keys())}")
                    # Try multiple possible keys
                    organic_results = parsed.get("organic", parsed.get("results", parsed.get("items", [])))
                    
                    logger.info(f"Found {len(organic_results)} results in parsed JSON")
                    
                    for i, result in enumerate(organic_results[:10]):
                        if not isinstance(result, dict):
                            continue
                            
                        title = result.get("title", result.get("name", "")).strip()
                        snippet = result.get("snippet", result.get("description", "")).strip()
                        link = result.get("link", result.get("url", ""))
                        
                        # Extract phone number
                        phone = None
                        if snippet:
                            phone = _extract_phone_number(snippet)
                        if not phone:
                            phone_field = result.get("phone", result.get("phoneNumber", ""))
                            if phone_field:
                                phone = _extract_phone_number(str(phone_field))
                        if not phone:
                            phone = "Contact hospital for details"
                        
                        if title:
                            is_hospital = any(keyword in title.lower() for keyword in [
                                "hospital", "medical", "clinic", "healthcare", "health"
                            ]) or any(keyword in snippet.lower() for keyword in [
                                "hospital", "medical", "clinic"
                            ]) if snippet else False
                            
                            if is_hospital or len(organic_results) <= 5:
                                hospital = {
                                    "id": f"HOSP_{i + 1:03d}",
                                    "name": title[:100],
                                    "address": snippet.split(".")[0][:200] if snippet else f"Near pincode {pincode}",
                                    "pincode": pincode,
                                    "specialties": [specialty] if specialty else ["General Medicine", "Emergency"],
                                    "phone": phone,
                                    "availability": "24/7",
                                    "distance_km": None,
                                    "source": "serper_web_search",
                                    "link": link
                                }
                                hospitals.append(hospital)
                                safe_title = _sanitize_for_logging(title[:50])
                                logger.info(f"Added hospital from JSON string: {safe_title} - Phone: {phone}")
            except json.JSONDecodeError:
                # Not JSON, try text parsing
                logger.info("Not JSON format, parsing as text")
                results_str = str(search_results)
                lines = results_str.split("\n")
                
                for line in lines:
                    line = line.strip()
                    if not line or len(line) < 5:
                        continue
                    
                    # Look for hospital names
                    if any(keyword in line.lower() for keyword in ["hospital", "medical", "clinic", "healthcare", "health"]):
                        # Extract hospital name and phone
                        hospital_name = line.split(",")[0].split("(")[0].split("-")[0].strip()
                        phone = _extract_phone_number(line)
                        
                        if len(hospital_name) > 3 and hospital_name.lower() not in ["hospital", "medical", "clinic"]:
                            hospital = {
                                "id": f"HOSP_{len(hospitals) + 1:03d}",
                                "name": hospital_name[:100],
                                "address": f"Near pincode {pincode}, India",
                                "pincode": pincode,
                                "specialties": [specialty] if specialty else ["General Medicine", "Emergency"],
                                "phone": phone if phone else "Contact hospital for details",
                                "availability": "24/7",
                                "distance_km": None,
                                "source": "serper_web_search",
                                "link": ""
                            }
                            hospitals.append(hospital)
                            if len(hospitals) >= 10:
                                break
        
        logger.info(f"Parsed {len(hospitals)} hospitals from Serper results")
        
        # Remove duplicates
        unique_hospitals = []
        seen_names = set()
        for hosp in hospitals:
            name_lower = hosp["name"].lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_hospitals.append(hosp)
        
        logger.info(f"Returning {len(unique_hospitals)} unique hospitals")
        return unique_hospitals[:10]
        
    except Exception as e:
        logger.error(f"Error parsing Serper results: {str(e)}", exc_info=True)
        logger.error(f"Search results type: {type(search_results)}, value: {str(search_results)[:500]}")
        return []


def _search_serper_direct(api_key: str, query: str) -> Optional[Dict]:
    """
    Direct Serper API call using httpx.
    This is more reliable than using SerperDevTool wrapper.
    
    Args:
        api_key: Serper API key
        query: Search query string
        
    Returns:
        Dictionary with search results or None on error
    """
    try:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query
        }
        
        response = httpx.post(url, headers=headers, json=payload, timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Direct Serper API call successful. Got {len(data.get('organic', []))} results")
            return data
        else:
            logger.error(f"Serper API error: Status {response.status_code}, {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error in direct Serper API call: {str(e)}", exc_info=True)
        return None


def get_hospital_by_id(hospital_id: str) -> Optional[Dict]:
    """
    Get hospital details by ID.
    Note: In web search mode, we don't have persistent IDs.
    This is a placeholder for future database integration.
    
    Args:
        hospital_id: Hospital identifier
        
    Returns:
        Hospital dictionary or None if not found
    """
    logger.warning(f"get_hospital_by_id called with {hospital_id} - web search mode doesn't support persistent ID lookup")
    return None


def get_available_specialties(pincode: str) -> List[str]:
    """
    Get all available specialties for hospitals in a given pincode.
    For web search, returns common specialties (can be enhanced with search).
    
    Args:
        pincode: Pincode to search
        
    Returns:
        List of common specialties
    """
    return [
        "Cardiology", "Neurology", "Orthopedics", "Oncology",
        "Pediatrics", "General Medicine", "Emergency", "Dermatology",
        "Gynecology", "Urology", "Psychiatry", "Dentistry"
    ]
