# Adoppted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/utils/env_options.py

"""Global environment options for LLM serving."""
import enum
import os


class Options(enum.Enum):
    """Environment variables for SkyPilot."""
    # Use these variables in your enviorment such as
    # export SHOW_DEBUG_INFO=0 or LLM_DISABLE_USAGE_COLLECTION=1
    SHOW_DEBUG_INFO = 'LLM_SHOW_DEBUG_INFO'
    DISABLE_LOGGING = 'LLM_DISABLE_USAGE_COLLECTION' # Decide whether to send the data to Grafana or not
    MINIMIZE_LOGGING = 'LLM_MINIMIZE_LOGGING'

    def get(self):
        """Check if an environment variable is set to True."""
        return os.getenv(self.value, 'False').lower() in ('true', '1')
