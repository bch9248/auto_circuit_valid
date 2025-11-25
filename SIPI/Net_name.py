"""
Net name definitions and platform-specific mappings for WWAN connections
"""

# WWAN Power net names
WWAN_power_names = ['P3V3DS_WWAN', 'PVSIM', 'PVSIM_DB', 'P3V3DS_WWAN']

# WWAN Ground net names
WWAN_gnd_names = ['DGND', 'DGND_DB']

# Platform-to-WWAN component mapping
# Maps platform name to the WWAN connector component REFDES
PLATFORM_WWAN_COMPONENT = {
    'G12_MACHU14_TLD_1217': 'CN1401',
    'SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602': 'CN1400',
}

# Helper function to get WWAN component for a platform
def get_wwan_component(platform: str) -> str:
    """
    Get the WWAN connector component REFDES for a given platform.
    
    Args:
        platform (str): Platform name
    
    Returns:
        str: Component REFDES (e.g., 'CN1401', 'CN1400')
             Returns 'CN1401' as default if platform not found
    
    Example:
        >>> get_wwan_component('G12_MACHU14_TLD_1217')
        'CN1401'
        >>> get_wwan_component('SVTP804 2nd Release Candidate_Cashmere_AMD_DB_CM01_250602')
        'CN1400'
    """
    return PLATFORM_WWAN_COMPONENT.get(platform, 'CN1401')