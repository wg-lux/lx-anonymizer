import requests
from typing import Dict, Union  # Added Dict, Union

base_url = 'https://www.namenforschung.net/dfd/woerterbuch/liste/'
current_offset = 4  # Start from the offset provided
limit = 20  # Hypothetical limit, you will need to adjust this based on the actual limit

while True:
    query_parameters: Dict[str, Union[str, int]] = {  # Changed here
        'tx_dfd_names[currentSelectedFacets]': '',
        'tx_dfd_names[query]': '',
        'tx_dfd_names[offset]': current_offset,
        'tx_dfd_names[limit]': limit,
        'tx_dfd_names[action]': 'list',
        'tx_dfd_names[controller]': 'Names',
        'cHash': '59cc1988a3c45cdb6a75e5d71a848be7'
    }
    response = requests.get(base_url, params=query_parameters)
    # ... process the response and then increment the offset for the next page
    current_offset += limit