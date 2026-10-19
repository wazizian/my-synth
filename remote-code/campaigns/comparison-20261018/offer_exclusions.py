"""Read authoritative stale rental IDs immediately before allocation/projection."""
import json
from pathlib import Path

REGISTRY = Path('/home/jump/campaign-control/stale_offer_ids.json')
KNOWN = {53061818, 48448052, 55917102, 56219704, 55701183, 46384002, 28780319}


def excluded_offer_ids(registry=REGISTRY):
    values = json.loads(Path(registry).read_text())
    if not isinstance(values, list) or any(type(value) is not int or value <= 0 for value in values):
        raise RuntimeError('Stale offer registry must contain positive integer IDs')
    return KNOWN | set(values)


def exclusion_arguments():
    from wallet_guard import assert_funding_available
    assert_funding_available()
    return [part for offer in sorted(excluded_offer_ids()) for part in ('--exclude-offer', str(offer))]
